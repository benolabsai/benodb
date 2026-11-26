// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Partition specification management: updating partition specs and batch partitioning.

use anyhow::Result;
use std::sync::Arc;

use super::super::types::*;
use super::ManifestManager;
use crate::core::storage::is_already_exists;

impl ManifestManager {
    /// Update partition specification
    pub async fn update_partition_spec(&self, new_spec: PartitionSpec) -> Result<Manifest> {
        let max_retries = 10;
        let mut attempt = 0;
        loop {
            let (current_manifest, current_ver) = if attempt == 0 {
                self.load_latest().await?
            } else {
                self.load_latest_direct().await?
            };
            let new_ver = current_ver + 1;

            let mut new_manifest = Manifest::new_with_spec(
                new_ver,
                current_manifest.entries.clone(),
                Some(current_ver),
                current_manifest.schemas.clone(),
                current_manifest.current_schema_id,
                new_spec.clone(),
            );
            // Preserve the tiered manifest list: for a tiered table the segment
            // entries live in the manifest list, not inline in `entries` (which
            // is empty). Dropping `manifest_list_path` here would make the new
            // manifest reference NO segments — i.e. a partition-spec update would
            // silently lose all data.
            new_manifest.manifest_list_path = current_manifest.manifest_list_path.clone();
            new_manifest.properties = current_manifest.properties.clone();
            new_manifest.format_version = current_manifest.format_version;
            // Carry the authoritative, partition-scoped delete list forward.
            new_manifest.delete_files = current_manifest.delete_files.clone();

            let filename = format!("v{}.json", new_ver);
            let path = self.manifest_dir.child(filename);
            let bytes = serde_json::to_vec_pretty(&new_manifest)?;

            use object_store::{PutMode, PutOptions};
            let opts = PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            };

            match self.store.put_opts(&path, bytes.into(), opts).await {
                Ok(_) => {
                    tracing::info!("Committed Manifest v{} (Partition Spec Update)", new_ver);
                    let dir_key = self.get_dir_cache_key();
                    crate::core::cache::LATEST_VERSION_CACHE
                        .invalidate(&dir_key)
                        .await;
                    crate::core::cache::LATEST_VERSION_CACHE
                        .insert(dir_key, new_ver)
                        .await;
                    let file_key = self.get_cache_key(&path);
                    crate::core::cache::MANIFEST_CACHE
                        .insert(file_key, Arc::new(new_manifest.clone()))
                        .await;
                    return Ok(new_manifest);
                }
                Err(e) if is_already_exists(&e) => {
                    let dir_key = self.get_dir_cache_key();
                    crate::core::cache::LATEST_VERSION_CACHE
                        .invalidate(&dir_key)
                        .await;
                    attempt += 1;
                    if attempt >= max_retries {
                        break;
                    }
                    let base_delay = 10 * (2u64.pow(attempt.min(5) as u32));
                    let jitter = rand::random::<u64>() % base_delay;
                    tokio::time::sleep(std::time::Duration::from_millis(base_delay + jitter)).await;
                    continue;
                }
                Err(e) => return Err(e.into()),
            }
        }
        Err(anyhow::anyhow!("Failed to commit partition spec update"))
    }
}
