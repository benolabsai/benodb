use anyhow::Result;
use object_store::{path::Path, ObjectStore, PutMode, PutOptions, UpdateVersion};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::oneshot;
use uuid::Uuid;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LockPayload {
    pub owner: String,
    pub expires_at: u64,
    #[serde(default)]
    pub generation: u64,
}

pub struct FileBasedLock {
    store: Arc<dyn ObjectStore>,
    path: Path,
    owner: String,
    ttl_seconds: u64,
    clock_skew_ms: u64,
}

pub struct LockGuard {
    store: Arc<dyn ObjectStore>,
    path: Path,
    owner: String,
    generation: u64,
    abort_tx: Option<oneshot::Sender<()>>,
    released: bool,
}

impl LockGuard {
    pub fn owner(&self) -> &str {
        &self.owner
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub async fn verify_valid(&self) -> Result<()> {
        let res = self.store.get(&self.path).await?;
        let bytes = res.bytes().await?;
        let payload: LockPayload = serde_json::from_slice(&bytes)?;
        if payload.owner != self.owner {
            anyhow::bail!(
                "Lease fencing violation: lock '{}' is now owned by '{}', expected '{}'",
                self.path,
                payload.owner,
                self.owner
            );
        }
        if payload.generation != self.generation {
            anyhow::bail!(
                "Lease fencing violation: lock '{}' generation mismatch (current {}, expected {})",
                self.path,
                payload.generation,
                self.generation
            );
        }
        let now_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
        let expires_ms = payload.expires_at * 1000;
        if now_ms > expires_ms {
            anyhow::bail!(
                "Lease expired: lock '{}' expired at {} ms, current time is {} ms",
                self.path,
                expires_ms,
                now_ms
            );
        }
        Ok(())
    }

    /// Release the lock synchronously, waiting for the delete to complete.
    ///
    /// `Drop` spawns the delete on the runtime, so a caller that immediately
    /// re-acquires the same lock (for example, rebuilding two columns in a row)
    /// can observe the still-present lock file and fail. Call this before
    /// returning from a short critical section.
    pub async fn release(mut self) {
        if let Some(tx) = self.abort_tx.take() {
            let _ = tx.send(());
        }
        self.released = true;
        Self::release_internal(
            self.store.clone(),
            self.path.clone(),
            self.owner.clone(),
            self.generation,
        )
        .await;
    }

    async fn release_internal(
        store: Arc<dyn ObjectStore>,
        path: Path,
        owner: String,
        generation: u64,
    ) {
        match store.get(&path).await {
            Ok(res) => {
                let meta = res.meta.clone();
                if let Ok(bytes) = res.bytes().await {
                    if let Ok(payload) = serde_json::from_slice::<LockPayload>(&bytes) {
                        if payload.owner != owner
                            || (payload.generation != 0 && payload.generation != generation)
                        {
                            tracing::warn!(
                                "Lock release skipped: lock is now owned by '{}' (gen {}), not '{}' (gen {}).",
                                payload.owner,
                                payload.generation,
                                owner,
                                generation
                            );
                            return;
                        }

                        // Ownership matches. Atomically mark as released/expired with
                        // UpdateVersion to fence against any concurrent steal before releasing.
                        // Leaving the released tombstone (owner="", expires_at=0) allows subsequent
                        // try_acquire to observe it as expired and atomically claim it via conditional put_opts,
                        // completely eliminating the race where another worker claims the lock before delete.
                        let released_payload = LockPayload {
                            owner: String::new(),
                            expires_at: 0,
                            generation,
                        };
                        let update_opts = PutOptions {
                            mode: PutMode::Update(UpdateVersion {
                                e_tag: meta.e_tag,
                                version: meta.version,
                            }),
                            ..Default::default()
                        };

                        if let Ok(bytes) = serde_json::to_vec(&released_payload) {
                            match store.put_opts(&path, bytes.into(), update_opts).await {
                                Ok(_) => {
                                    // Successfully updated to released tombstone without unconditional delete.
                                }
                                Err(object_store::Error::NotImplemented)
                                | Err(object_store::Error::NotSupported { .. }) => {
                                    // Fallback for stores lacking conditional update (e.g. local FS).
                                    let _ = store.delete(&path).await;
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "Conditional lock release aborted (lock stolen by another process): {}",
                                        e
                                    );
                                }
                            }
                        }
                    }
                }
            }
            Err(_) => {} // Already deleted
        }
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        if let Some(tx) = self.abort_tx.take() {
            let _ = tx.send(());
        }
        let store = self.store.clone();
        let path = self.path.clone();
        let owner = self.owner.clone();
        let generation = self.generation;

        tokio::spawn(async move {
            Self::release_internal(store, path, owner, generation).await;
        });
    }
}

impl FileBasedLock {
    pub fn new(store: Arc<dyn ObjectStore>, path: Path, ttl_seconds: u64) -> Self {
        Self {
            store,
            path,
            owner: Uuid::new_v4().to_string(),
            ttl_seconds,
            clock_skew_ms: 5000, // 5 seconds default NTP drift allowance
        }
    }

    pub fn with_clock_skew(mut self, ms: u64) -> Self {
        self.clock_skew_ms = ms;
        self
    }

    pub async fn try_acquire(&self) -> Result<Option<LockGuard>> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let initial_generation = 1;
        let payload = LockPayload {
            owner: self.owner.clone(),
            expires_at: now + self.ttl_seconds,
            generation: initial_generation,
        };
        let bytes = serde_json::to_vec(&payload)?;

        let opts = PutOptions {
            mode: PutMode::Create,
            ..Default::default()
        };

        match self
            .store
            .put_opts(&self.path, bytes.clone().into(), opts.clone())
            .await
        {
            Ok(_) => Ok(Some(self.spawn_heartbeat(initial_generation))),
            Err(object_store::Error::AlreadyExists { .. }) => {
                // Check if expired
                match self.store.get(&self.path).await {
                    Ok(get_res) => {
                        let meta = get_res.meta.clone();
                        let current_bytes = get_res.bytes().await?;
                        let (is_expired, next_generation) = if let Ok(current_payload) =
                            serde_json::from_slice::<LockPayload>(&current_bytes)
                        {
                            let current_time_ms =
                                SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
                            let expires_ms =
                                (current_payload.expires_at * 1000) + self.clock_skew_ms;
                            (
                                current_time_ms > expires_ms,
                                current_payload.generation.saturating_add(1),
                            )
                        } else {
                            tracing::warn!("Unparseable lock payload at {}. Treating as active (fail-closed) to preserve mutual exclusion.", self.path);
                            (false, 1)
                        };

                        if is_expired {
                            let steal_payload = LockPayload {
                                owner: self.owner.clone(),
                                expires_at: now + self.ttl_seconds,
                                generation: next_generation,
                            };
                            let steal_bytes = serde_json::to_vec(&steal_payload)?;

                            // It's expired. Try to steal it using UpdateVersion if supported (S3/GCS)
                            let update_opts = PutOptions {
                                mode: PutMode::Update(UpdateVersion {
                                    e_tag: meta.e_tag,
                                    version: meta.version,
                                }),
                                ..Default::default()
                            };

                            match self
                                .store
                                .put_opts(&self.path, steal_bytes.clone().into(), update_opts)
                                .await
                            {
                                Ok(_) => return Ok(Some(self.spawn_heartbeat(next_generation))),
                                Err(object_store::Error::NotImplemented)
                                | Err(object_store::Error::NotSupported { .. }) => {
                                    tracing::warn!("Lock steal using non-atomic fallback (TOCTOU risk). Consider using S3/GCS/Azure for production locking.");
                                    let _ = self.store.delete(&self.path).await;
                                    let jitter = rand::random::<u64>() % 50;
                                    tokio::time::sleep(std::time::Duration::from_millis(jitter))
                                        .await;

                                    match self
                                        .store
                                        .put_opts(&self.path, steal_bytes.into(), opts)
                                        .await
                                    {
                                        Ok(_) => {
                                            return Ok(Some(self.spawn_heartbeat(next_generation)))
                                        }
                                        Err(_) => return Ok(None),
                                    }
                                }
                                Err(_) => return Ok(None),
                            }
                        }
                    }
                    Err(_) => {} // File deleted between our PutMode::Create and get
                }
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }

    fn spawn_heartbeat(&self, generation: u64) -> LockGuard {
        let (tx, mut rx) = oneshot::channel();
        let store = self.store.clone();
        let path = self.path.clone();
        let owner = self.owner.clone();
        let ttl_seconds = self.ttl_seconds;

        // Heartbeat wakes up at TTL/2 to renew
        let interval_ms = (ttl_seconds * 1000) / 2;

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut rx => {
                        // Drop guard signaled cancellation
                        break;
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_millis(interval_ms)) => {
                        let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                        let payload = LockPayload {
                            owner: owner.clone(),
                            expires_at: now + ttl_seconds,
                            generation,
                        };
                        if let Ok(bytes) = serde_json::to_vec(&payload) {
                            // Fetch meta to do an atomic PutMode::Update if supported
                            if let Ok(get_res) = store.get(&path).await {
                                let meta = get_res.meta.clone();
                                if let Ok(current_bytes) = get_res.bytes().await {
                                    if let Ok(current_payload) = serde_json::from_slice::<LockPayload>(&current_bytes) {
                                        if current_payload.owner != owner || current_payload.generation != generation {
                                            tracing::error!(
                                                "Lock heartbeat detected ownership/generation change! Current owner: '{}' (gen {}), our owner: '{}' (gen {}). Aborting heartbeat renewal.",
                                                current_payload.owner,
                                                current_payload.generation,
                                                owner,
                                                generation
                                            );
                                            break;
                                        }
                                    }
                                }
                                let update_opts = PutOptions {
                                    mode: PutMode::Update(UpdateVersion {
                                        e_tag: meta.e_tag,
                                        version: meta.version,
                                    }),
                                    ..Default::default()
                                };
                                if let Err(e) = store.put_opts(&path, bytes.into(), update_opts).await {
                                    tracing::warn!("Failed to renew lock heartbeat: {}", e);
                                }
                            }
                        }
                    }
                }
            }
        });

        LockGuard {
            store: self.store.clone(),
            path: self.path.clone(),
            owner: self.owner.clone(),
            generation,
            abort_tx: Some(tx),
            released: false,
        }
    }

    pub async fn acquire(&self) -> Result<LockGuard> {
        let max_retries = 100;
        for attempt in 0..max_retries {
            if let Some(guard) = self.try_acquire().await? {
                return Ok(guard);
            }
            let base_delay = 50 * (2u64.pow(attempt.min(6) as u32));
            let jitter = rand::random::<u64>() % base_delay;
            tokio::time::sleep(std::time::Duration::from_millis(base_delay + jitter)).await;
        }
        Err(anyhow::anyhow!(
            "Failed to acquire distributed lock after {} attempts",
            max_retries
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_lock_acquire_and_release() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("test.lock");

        {
            let lock = FileBasedLock::new(store.clone(), path.clone(), 30);
            let _guard = lock
                .try_acquire()
                .await
                .unwrap()
                .expect("Should acquire lock");

            assert!(
                FileBasedLock::new(store.clone(), path.clone(), 30)
                    .try_acquire()
                    .await
                    .unwrap()
                    .is_none(),
                "Second acquire should fail"
            );
        } // guard drops here

        // Give the spawned drop task a tiny bit of time to execute
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        assert!(
            FileBasedLock::new(store.clone(), path.clone(), 30)
                .try_acquire()
                .await
                .unwrap()
                .is_some(),
            "Should acquire after release"
        );
    }

    #[tokio::test]
    async fn test_lock_expiration_steal() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("test_expire.lock");

        // Lock with 0 TTL (expires immediately)
        let lock1 = FileBasedLock::new(store.clone(), path.clone(), 0).with_clock_skew(0);
        let _guard = lock1.try_acquire().await.unwrap().expect("Should acquire");

        // Even though we have a guard, the heartbeat sleeps for TTL/2 (0ms), so it tries to renew
        // but the expiration is immediate. We sleep past the TTL + clock_skew.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let lock2 = FileBasedLock::new(store.clone(), path.clone(), 30).with_clock_skew(0);
        assert!(
            lock2.try_acquire().await.unwrap().is_some(),
            "Should steal expired lock"
        );
    }

    #[tokio::test]
    async fn test_stale_release_does_not_delete_stolen_lock() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("stolen.lock");

        // 1. Process A acquires lock with 0 TTL (expires immediately)
        let lock_a = FileBasedLock::new(store.clone(), path.clone(), 0).with_clock_skew(0);
        let guard_a = lock_a.try_acquire().await.unwrap().expect("A acquires");

        // 2. Process B steals the expired lock
        let lock_b = FileBasedLock::new(store.clone(), path.clone(), 30).with_clock_skew(0);
        let _guard_b = lock_b.try_acquire().await.unwrap().expect("B steals");

        // 3. Process A calls release on its stale guard
        guard_a.release().await;

        // 4. Assert that B's lock still exists and cannot be acquired by C
        let lock_c = FileBasedLock::new(store.clone(), path.clone(), 30).with_clock_skew(0);
        assert!(
            lock_c.try_acquire().await.unwrap().is_none(),
            "C should not acquire because B still holds the lock"
        );

        // Verify the payload in storage still has B as owner
        let res = store.get(&path).await.unwrap();
        let bytes = res.bytes().await.unwrap();
        let payload: LockPayload = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload.owner, lock_b.owner);
    }

    #[tokio::test]
    async fn test_released_tombstone_reacquired_without_deletion_race() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("tombstone.lock");

        // 1. Process A acquires lock
        let lock_a = FileBasedLock::new(store.clone(), path.clone(), 30).with_clock_skew(0);
        let guard_a = lock_a.try_acquire().await.unwrap().expect("A acquires");
        assert_eq!(guard_a.generation(), 1);

        // 2. Process A releases the lock (payload becomes released tombstone with owner="", expires_at=0)
        guard_a.release().await;

        // Verify tombstone payload in storage
        let res = store.get(&path).await.unwrap();
        let bytes = res.bytes().await.unwrap();
        let payload: LockPayload = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload.owner, "");
        assert_eq!(payload.expires_at, 0);

        // 3. Process B acquires the released tombstone
        let lock_b = FileBasedLock::new(store.clone(), path.clone(), 30).with_clock_skew(0);
        let guard_b = lock_b
            .try_acquire()
            .await
            .unwrap()
            .expect("B should acquire released tombstone");
        assert_eq!(guard_b.generation(), 2);
        assert_eq!(guard_b.owner(), &lock_b.owner);
        assert!(guard_b.verify_valid().await.is_ok());

        // 4. Process C cannot acquire while B holds it
        let lock_c = FileBasedLock::new(store.clone(), path.clone(), 30).with_clock_skew(0);
        assert!(lock_c.try_acquire().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_lock_guard_generation_fencing() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("fencing.lock");

        // 1. Worker A acquires with immediate expiry
        let lock_a = FileBasedLock::new(store.clone(), path.clone(), 0).with_clock_skew(0);
        let guard_a = lock_a.try_acquire().await.unwrap().expect("A acquires");
        assert_eq!(guard_a.generation(), 1);

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // 2. Worker B steals the expired lock
        let lock_b = FileBasedLock::new(store.clone(), path.clone(), 30).with_clock_skew(0);
        let guard_b = lock_b.try_acquire().await.unwrap().expect("B steals");
        assert_eq!(guard_b.generation(), 2);

        // 3. Worker A checks validity and gets fenced out
        let verify_res = guard_a.verify_valid().await;
        assert!(verify_res.is_err(), "Worker A must be fenced out");
        assert!(verify_res
            .unwrap_err()
            .to_string()
            .contains("Lease fencing violation"));

        // 4. Worker B is valid
        assert!(guard_b.verify_valid().await.is_ok());
    }
}
