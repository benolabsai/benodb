// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use anyhow::{Context, Result};
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use crate::core::error::BenoStreamError;
use crate::core::manifest::ManifestManager;
use crate::telemetry::metrics::INGEST_ROWS_TOTAL;
use crate::SegmentConfig;

use super::{CommitPhase, PendingCommit, PendingIndexBuild, PendingWrite, Table};
use crate::core::index::memory::InMemoryVectorIndex;
use crate::core::segment::HybridSegmentWriter;
use arrow::datatypes::Schema;
use futures::StreamExt;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// Resident set size in bytes, or `0` when the platform cannot report it.
///
/// Delegates to [`crate::core::memory::rss_bytes`] so the back-pressure
/// high-water mark and the heap-trim policy read the same number. A `0` result
/// (unknown) makes the back-pressure check fail-open rather than block forever.
fn current_rss_bytes() -> usize {
    crate::core::memory::rss_bytes().unwrap_or(0) as usize
}

/// Upper bound on how long the ingest back-pressure waits when RSS is stuck
/// above the cap for reasons the current write cannot relieve (pre-existing
/// process memory the engine will never free), before it fails open.
///
/// Overridable via `BDB_INGEST_BACKPRESSURE_GRACE_SECS`; a non-positive or
/// unparseable value falls back to the 60-second default.
fn ingest_backpressure_grace() -> std::time::Duration {
    std::time::Duration::from_secs_f64(backpressure_grace_secs(
        std::env::var("BDB_INGEST_BACKPRESSURE_GRACE_SECS")
            .ok()
            .as_deref(),
    ))
}

/// Parse the fail-open grace (seconds). A missing, non-positive, or unparseable
/// value falls back to the 60-second default so a typo cannot disable the
/// liveness safety net. Pure so it can be asserted without process-global env.
pub(crate) fn backpressure_grace_secs(raw: Option<&str>) -> f64 {
    raw.and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(60.0)
}

impl Table {
    /// Write Arrow RecordBatches to the table (Buffered)
    ///
    /// Data is written to an in-memory buffer. It is NOT persisted to disk until:
    /// 1. The buffer exceeds `BDB_CACHE_GB`
    /// 2. `commit()` is called explicitly
    pub fn write(&self, batches: Vec<RecordBatch>) -> Result<()> {
        // A pinned handle is a read view. Refuse before entering the runtime
        // so the caller learns the view is frozen rather than racing a commit.
        self.ensure_writable()?;
        self.runtime().block_on(self.write_async(batches))
    }

    /// Commit buffered writes to disk
    pub fn commit(&self) -> Result<()> {
        self.ensure_writable()?;
        self.runtime().block_on(self.flush_async())
    }

    /// Async commit
    #[tracing::instrument(skip(self))]
    pub async fn commit_async(&self) -> Result<()> {
        self.ensure_writable()?;
        self.flush_async().await?;
        Ok(())
    }

    /// Truncate the table (metadata-only operation)
    pub fn truncate(&self) -> Result<()> {
        self.runtime().block_on(self.truncate_async())
    }

    /// Async implementation of truncate
    #[tracing::instrument(skip(self))]
    pub async fn truncate_async(&self) -> Result<()> {
        self.ensure_writable()?;
        let _maintenance_guard = self.maintenance_lock.write().await;
        let _commit_guard = self.commit_lock.lock().await;

        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);

        // Step 1: Get all current entry paths
        // We must load THE WHOLE current manifest to know what to remove.
        let (_, entries, _) = manifest_manager.load_latest_full().await?;
        let remove_paths: Vec<String> = entries.iter().map(|e| e.file_path.clone()).collect();

        // Step 2: Commit with all paths in remove_paths and empty add_entries
        // This is a single atomic snapshot swap that points to an empty segment set.
        let new_manifest = manifest_manager
            .commit(
                &[],
                &remove_paths,
                crate::core::manifest::CommitMetadata::default(),
            )
            .await?;

        // Step 3: Clear in-memory buffers & indexes in canonical lock order before WAL reset.
        {
            let mut pc = self.pending_commit.lock();
            let mut pending = self.pending_writes.write();
            let mut idx = self.indexing.memory_index.write();
            *pc = None;
            pending.clear();
            *idx = None;
        }
        self.committed_wal_txs.lock().clear();
        self.indexing.cluster_indexes.write().clear();
        self.indexing.vector_routers.write().clear();
        self.bump_buffer_epoch();

        // Step 4: Truncate WAL cleanly via the WAL worker.
        {
            let wal = self.wal.lock().await;
            wal.truncate_all_async()
                .await
                .context("Failed to truncate WAL during table truncate")?;
        }
        self.recovered_wal_paths.lock().clear();

        // Manifest commits are invisible to external readers until the snapshot
        // is published in the table metadata.
        self.publish_snapshot_async(&new_manifest, "delete").await?;

        tracing::info!(
            "Table truncated: metadata reset, all {} segments removed, buffers cleared.",
            remove_paths.len()
        );
        Ok(())
    }

    /// Compact the WAL (consolidate log entries)
    pub fn checkpoint(&self) -> Result<()> {
        self.block_on_io(self.checkpoint_async())
            .and_then(|res| res)
    }

    /// Async checkpoint: flush the WAL, then publish a metadata version that
    /// pins the current index artifacts.
    ///
    /// The metadata commit is the point. A WAL checkpoint alone leaves the index
    /// artifacts without a version to belong to, so a rollback has no artifact
    /// set to resolve, and time travel can only recover the data. Committing a
    /// version with no data change records `Manifest::table_indexes` and the
    /// segment `index_files` as they stand, which is the set that belongs to
    /// this point in time.
    pub async fn checkpoint_async(&self) -> Result<()> {
        self.ensure_writable()?;
        {
            let wal = self.wal.lock().await;
            wal.compact_async().await?;
        }
        self.checkpoint_index_async().await
    }

    /// Publish a metadata version that pins the current index artifacts.
    ///
    /// Safe to call when the table has no indexes: it returns without writing a
    /// version, so an empty checkpoint does not advance time for tables that
    /// only carry data.
    pub async fn checkpoint_index_async(&self) -> Result<()> {
        self.ensure_writable()?;
        let manager =
            crate::core::manifest::ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, entries, _version) = manager.load_latest_full().await?;
        let has_indexes =
            !manifest.table_indexes.is_empty() || entries.iter().any(|e| !e.index_files.is_empty());
        if !has_indexes {
            return Ok(());
        }
        manager
            .commit(&[], &[], crate::core::manifest::CommitMetadata::default())
            .await?;
        Ok(())
    }

    // Schema evolution logic moved to schema.rs

    /// Async implementation of write using the table's configured durability mode.
    #[tracing::instrument(skip(self, batches))]
    pub async fn write_async(&self, batches: Vec<RecordBatch>) -> Result<()> {
        self.ensure_writable()?;
        self.write_with_durability_async(batches, self.durability)
            .await
    }

    /// Write batches with asynchronous durability (buffered in WAL worker without waiting for fsync).
    /// Best for maximum streaming ingestion throughput.
    #[tracing::instrument(skip(self, batches))]
    pub async fn write_buffered_async(&self, batches: Vec<RecordBatch>) -> Result<()> {
        self.write_with_durability_async(batches, crate::core::table::WalDurability::Async)
            .await
    }

    /// Explicitly flush and sync pending WAL writes to durable storage.
    pub async fn flush_wal_async(&self) -> Result<()> {
        let wal = self.wal.lock().await;
        wal.flush_async().await
    }

    /// Internal write implementation with explicit durability specification.
    #[tracing::instrument(skip(self, batches))]
    pub async fn write_with_durability_async(
        &self,
        batches: Vec<RecordBatch>,
        durability: crate::core::table::WalDurability,
    ) -> Result<()> {
        self.ensure_writable()?;
        // Reject vectors that do not match the table's declared dimension.
        // This is the shared choke point for Python, SQL/Flight SQL, MCP, and
        // any other connector that writes through the engine. The pre-check
        // keeps non-vector ingestion on the previous fast path: no property
        // read and no schema walk unless a vector-named column is present.
        if crate::core::table::properties::batches_have_vector_columns(&batches) {
            let embedding_properties = self.properties_async().await?;
            crate::core::table::properties::validate_embedding_batches(
                &self.uri,
                &embedding_properties,
                &batches,
            )?;
        }

        if let Some(max_gb) = self.max_ingest_ram_gb {
            let max_bytes = (max_gb * 1_000_000_000.0) as usize;
            // The caller may hold memory outside the engine (e.g. a Python
            // DataFrame). Count only the engine's own footprint against the
            // budget, otherwise a large caller-side frame blocks the write
            // forever (the caller cannot free it until the write returns).
            let caller_reserved =
                self.caller_reserved_bytes
                    .load(std::sync::atomic::Ordering::Relaxed) as usize;
            let engine_rss = || current_rss_bytes().saturating_sub(caller_reserved);

            // Bound *growth* since this write began, not the absolute process
            // RSS. Memory that already existed when the write started (a prior
            // test in the same process, caches, the runtime itself) is not
            // attributable to this write; because no background task owns it, it
            // can never be reclaimed, so an absolute check would block forever
            // once the process baseline is already at/above the cap.
            let baseline = engine_rss();
            let grace = ingest_backpressure_grace();
            let mut logged = false;
            let mut paused: Option<std::time::Instant> = None;
            // How long RSS has been above the cap *without* the engine growing
            // (i.e. nothing this write allocated, so waiting cannot help).
            let mut over_absolute: Option<std::time::Instant> = None;
            // Set when a background task signalled a reclaim while stalled.
            let mut reclaimed_while_stalled = false;
            let mut failed_open = false;

            // Back-pressure on the ingest RAM high-water mark. Wait on a
            // notification from background tasks that release memory (index
            // builds, heap trims) with a bounded fallback poll so we still
            // re-check RSS if nothing fires. The previous implementation slept a
            // flat 500 ms per iteration, which both wasted time when memory was
            // freed promptly and delayed resumption when it was not.
            loop {
                if max_bytes == 0 {
                    break;
                }
                let rss = engine_rss();
                let grew_over_budget = rss.saturating_sub(baseline) >= max_bytes;
                let over_absolute_rss = rss >= max_bytes;
                if !grew_over_budget && !over_absolute_rss {
                    break;
                }

                if !logged {
                    tracing::warn!(
                        "Engine RSS ({:.2} GB, excluding {:.2} GB caller-reserved; baseline {:.2} GB) is at/above the max ingest RAM limit ({:.2} GB). Pausing ingestion until background tasks reclaim memory...",
                        rss as f64 / 1_000_000_000.0,
                        caller_reserved as f64 / 1_000_000_000.0,
                        baseline as f64 / 1_000_000_000.0,
                        max_gb
                    );
                    logged = true;
                }
                if paused.is_none() {
                    paused = Some(std::time::Instant::now());
                }

                if over_absolute_rss && !grew_over_budget {
                    // RSS is above the cap purely from memory this write did not
                    // allocate and the engine is not growing. Waiting only helps
                    // if a background task frees memory: if one already ran (or
                    // the grace elapsed) and RSS is still over the cap, this write
                    // can never clear it — fail open rather than hang forever.
                    if reclaimed_while_stalled {
                        failed_open = true;
                        break;
                    }
                    let start = *over_absolute.get_or_insert_with(std::time::Instant::now);
                    if start.elapsed() >= grace {
                        failed_open = true;
                        break;
                    }
                } else {
                    over_absolute = None;
                    reclaimed_while_stalled = false;
                }

                tokio::select! {
                    _ = self.memory_reclaimed.notified() => {
                        reclaimed_while_stalled = true;
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_millis(250)) => {}
                }
            }
            crate::telemetry::metrics::INGEST_RSS_BYTES_GAUGE.set(current_rss_bytes() as i64);
            if let Some(start) = paused {
                let paused_secs = start.elapsed().as_secs_f64();
                crate::telemetry::metrics::INGEST_BACKPRESSURE_PAUSES_TOTAL.inc();
                crate::telemetry::metrics::INGEST_BACKPRESSURE_PAUSE_SECONDS.observe(paused_secs);
                if failed_open {
                    tracing::error!(
                        paused_seconds = paused_secs,
                        baseline_gb = baseline as f64 / 1_000_000_000.0,
                        max_gb,
                        "Ingest RAM limit exceeded by pre-existing process memory that no background task can reclaim (paused {:.1}s). Proceeding without further back-pressure (fail-open) so the writer does not block forever.",
                        paused_secs
                    );
                } else {
                    tracing::info!(
                        paused_seconds = paused_secs,
                        "RSS back under the ingest RAM limit; resuming ingestion"
                    );
                }
            }
        }

        // Disk admission: refuse a flush when a local table's filesystem is below
        // the free-space threshold (default `BDB_MIN_FREE_DISK_GB`), so the
        // failure is a clear error instead of a partial segment written mid-way.
        // Fail-open when the free space is unknown.
        if let Some(local) = self.uri.strip_prefix("file://") {
            let path = std::path::Path::new(local);
            if let Some(free) = crate::core::resources::free_disk_bytes_for_new_file(path) {
                crate::telemetry::metrics::FREE_DISK_BYTES_GAUGE.set(free as i64);
            }
            crate::core::resources::check_min_free_disk(path).map_err(|e| anyhow::anyhow!(e))?;
        }

        let _maintenance_guard = self.maintenance_lock.read().await;

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        INGEST_ROWS_TOTAL.inc_by(total_rows as u64);

        if batches.is_empty() {
            return Ok(());
        }

        let mut batches = batches;

        let mut is_empty_schema = false;
        if let Some(first_batch) = batches.first() {
            let mut lock = self.schema.write();
            is_empty_schema = lock.fields().is_empty();
            if is_empty_schema {
                let incoming_schema = first_batch.schema();
                let table_pattern = self.label_pattern;

                let mut fields = Vec::with_capacity(incoming_schema.fields().len());
                let mut renamed = false;
                for (i, field) in incoming_schema.fields().iter().enumerate() {
                    let mut new_name = (*field).name().clone();

                    // Check if name is numeric or empty, and apply pattern
                    let is_numeric = new_name.chars().all(|c| c.is_ascii_digit());
                    if is_numeric || new_name.is_empty() {
                        new_name = match table_pattern {
                            crate::core::table::LabelPattern::ExcelAlpha => {
                                crate::core::table::excel_column_label(i)
                            }
                            crate::core::table::LabelPattern::Polars => format!("column_{}", i + 1),
                            crate::core::table::LabelPattern::Pandas => i.to_string(),
                        };
                        if new_name != *field.name() {
                            renamed = true;
                        }
                    }

                    let new_field = (**field).clone().with_name(new_name);
                    fields.push(new_field);
                }
                let auto_schema = Arc::new(arrow::datatypes::Schema::new(fields));
                *lock = auto_schema.clone();
                drop(lock);

                if renamed {
                    batches = batches
                        .into_iter()
                        .map(|b| {
                            RecordBatch::try_new(auto_schema.clone(), b.columns().to_vec())
                                .map_err(|e| anyhow::anyhow!(e))
                        })
                        .collect::<Result<Vec<RecordBatch>>>()?;
                }
            } else {
                drop(lock);
            }
        }

        // 2. Primary Key Uniqueness Validation
        let t_pk = std::time::Instant::now();
        let pk_cols = self.primary_key.read().clone();
        let _pk_validation_guard = if !pk_cols.is_empty() {
            Some(self.pk_validation_lock.lock().await)
        } else {
            None
        };

        if !pk_cols.is_empty() {
            let mut seen_keys = std::collections::HashSet::new();

            // 1. Pre-populate seen keys from any Staging commit + active write buffer
            {
                let pc_guard = self.pending_commit.lock();
                let pending = self.pending_writes.read();
                let staging_iter = pc_guard
                    .as_ref()
                    .filter(|pc| pc.phase == CommitPhase::Staging)
                    .into_iter()
                    .flat_map(|pc| pc.batches.iter());
                for b_batch in staging_iter.chain(pending.iter().map(|p| &p.batch)) {
                    let key_cols: Vec<_> = pk_cols
                        .iter()
                        .filter_map(|name| b_batch.column_by_name(name))
                        .collect();
                    if key_cols.len() == pk_cols.len() {
                        for j in 0..b_batch.num_rows() {
                            let tuple_key = if key_cols.len() == 1 {
                                crate::core::manifest::ManifestValue::from_array(key_cols[0], j)
                                    .to_string()
                            } else {
                                let mut parts = Vec::with_capacity(key_cols.len());
                                for col in &key_cols {
                                    let val =
                                        crate::core::manifest::ManifestValue::from_array(col, j);
                                    parts.push(val.to_string());
                                }
                                crate::core::manifest::encode_composite_key(
                                    parts.iter().map(|s| s.as_str()),
                                )
                            };
                            seen_keys.insert(tuple_key);
                        }
                    }
                }
            }

            // 2. Check incoming batches against nulls, buffer/batch duplicates, and committed storage
            for batch in &batches {
                let key_cols: Vec<_> = pk_cols
                    .iter()
                    .filter_map(|name| batch.column_by_name(name))
                    .collect();
                if key_cols.len() != pk_cols.len() {
                    continue;
                }

                for i in 0..batch.num_rows() {
                    let mut tuple_values = Vec::with_capacity(pk_cols.len());
                    let mut val_strings = Vec::with_capacity(pk_cols.len());

                    for (col_idx, col) in key_cols.iter().enumerate() {
                        let m_val = crate::core::manifest::ManifestValue::from_array(col, i);
                        if matches!(m_val, crate::core::manifest::ManifestValue::Null) {
                            return Err(anyhow::Error::from(
                                BenoStreamError::NullConstraintViolation {
                                    column: pk_cols[col_idx].clone(),
                                },
                            ));
                        }
                        val_strings.push(m_val.to_string());
                        let val_json =
                            serde_json::to_value(&m_val).unwrap_or(serde_json::Value::Null);
                        tuple_values.push(val_json);
                    }

                    let tuple_key = crate::core::manifest::encode_composite_key(
                        val_strings.iter().map(|s| s.as_str()),
                    );

                    if seen_keys.contains(&tuple_key) {
                        return Err(anyhow::Error::from(BenoStreamError::PrimaryKeyViolation {
                            key: tuple_key,
                        }));
                    }

                    if self
                        ._check_composite_pk_in_storage_async(&pk_cols, &tuple_values)
                        .await?
                    {
                        return Err(anyhow::Error::from(BenoStreamError::PrimaryKeyViolation {
                            key: tuple_key,
                        }));
                    }

                    seen_keys.insert(tuple_key);
                }
            }
        }

        let pk_ms = t_pk.elapsed().as_millis();

        let t_coerce = std::time::Instant::now();
        let mut target_schema = self.arrow_schema();

        // Columns with a **declared vector index** must be stored as Float32.
        // An index declared before any data exists otherwise sees a
        // `list<double>` column and silently never builds (every query falls
        // back to a brute-force scan). Coercing the declared column here makes
        // incoming data flow into the declared index, whatever the input dtype.
        let vector_cols: std::collections::HashSet<String> =
            self.indexing.index_columns.read().iter().cloned().collect();
        let f32_field = |field: &arrow::datatypes::Field| -> arrow::datatypes::Field {
            use arrow::datatypes::{DataType, Field};
            match field.data_type() {
                DataType::List(inner) if inner.data_type() == &DataType::Float64 => Field::new(
                    field.name(),
                    DataType::List(std::sync::Arc::new(Field::new(
                        "item",
                        DataType::Float32,
                        inner.is_nullable(),
                    ))),
                    field.is_nullable(),
                ),
                DataType::FixedSizeList(inner, n) if inner.data_type() == &DataType::Float64 => {
                    Field::new(
                        field.name(),
                        DataType::FixedSizeList(
                            std::sync::Arc::new(Field::new(
                                "item",
                                DataType::Float32,
                                inner.is_nullable(),
                            )),
                            *n,
                        ),
                        field.is_nullable(),
                    )
                }
                _ => field.clone(),
            }
        };

        if let Some(first_batch) = batches.first() {
            let incoming_schema = first_batch.schema();
            let mut evolved_schema = (*target_schema).clone();
            let mut changed = false;

            for field in incoming_schema.fields() {
                let existing_field_info = evolved_schema
                    .field_with_name(field.name())
                    .ok()
                    .map(|f| (f.data_type().clone(), f.is_nullable()));

                if let Some((existing_dtype, existing_nullable)) = existing_field_info {
                    // Check if we need to widen the type (e.g. Int32 -> Int64)
                    if existing_dtype != *field.data_type() {
                        tracing::info!(
                            "Schema Evolution: Widening column '{}' from {:?} to {:?}",
                            field.name(),
                            existing_dtype,
                            field.data_type()
                        );

                        // `field` came from `evolved_schema`, so `index_of` cannot
                        // fail; skip defensively rather than panicking the writer.
                        if let Ok(idx) = evolved_schema.index_of(field.name()) {
                            let mut fields: Vec<arrow::datatypes::Field> = evolved_schema
                                .fields()
                                .iter()
                                .map(|f| (**f).clone())
                                .collect();
                            fields[idx] = if vector_cols.contains(field.name()) {
                                f32_field(field)
                            } else {
                                (**field).clone()
                            };
                            evolved_schema = Schema::new(fields);
                            changed = true;
                        }
                    }

                    // Check if we need to change Nullability (Required -> Nullable)
                    if !existing_nullable && field.is_nullable() {
                        tracing::info!(
                            "Schema Evolution: Changing column '{}' to nullable",
                            field.name()
                        );
                        if let Ok(idx) = evolved_schema.index_of(field.name()) {
                            let mut fields: Vec<arrow::datatypes::Field> = evolved_schema
                                .fields()
                                .iter()
                                .map(|f| (**f).clone())
                                .collect();
                            let mut new_field = (**field).clone();
                            new_field.set_nullable(true);
                            fields[idx] = new_field;
                            evolved_schema = Schema::new(fields);
                            changed = true;
                        }
                    }
                } else {
                    // New column added
                    tracing::info!("Schema Evolution: Adding new column '{}'", field.name());
                    let mut fields: Vec<arrow::datatypes::Field> = evolved_schema
                        .fields()
                        .iter()
                        .map(|f| (**f).clone())
                        .collect();
                    fields.push(if vector_cols.contains(field.name()) {
                        f32_field(field)
                    } else {
                        (**field).clone()
                    });
                    evolved_schema = Schema::new(fields);
                    changed = true;
                }
            }

            if changed {
                let mut lock = self.schema.write();
                *lock = Arc::new(evolved_schema);
                drop(lock);
                target_schema = self.arrow_schema();
            }
        }

        // 1. Strict Nullability Validation
        // Ensure no required (NOT NULL) columns contain nulls in the incoming batches,
        // using the potential EVOLVED schema.
        for batch in &batches {
            for field in target_schema.fields() {
                if !field.is_nullable() {
                    if let Some(col) = batch.column_by_name(field.name()) {
                        if col.null_count() > 0 {
                            return Err(anyhow::anyhow!("Null constraint violation: column '{}' is NOT NULL but batch contains {} nulls", field.name(), col.null_count()));
                        }
                    }
                }
            }
        }

        let batches: Vec<RecordBatch> = batches
            .into_iter()
            .map(|b| -> Result<RecordBatch> {
                if b.schema() != target_schema {
                    let mut cols = Vec::with_capacity(target_schema.fields().len());
                    for field in target_schema.fields() {
                        let col = if let Some(c) = b.column_by_name(field.name()) {
                            if c.data_type() == field.data_type() {
                                c.clone()
                            } else {
                                // Cast to the declared type (e.g. the Float32 a
                                // declared vector index forces) instead of
                                // failing the write.
                                arrow::compute::cast(c, field.data_type())?
                            }
                        } else {
                            arrow::array::new_null_array(field.data_type(), b.num_rows())
                        };
                        cols.push(col);
                    }
                    RecordBatch::try_new(target_schema.clone(), cols)
                        .map_err(|e| anyhow::anyhow!(e))
                } else {
                    Ok(b)
                }
            })
            .collect::<Result<Vec<RecordBatch>>>()?;

        if is_empty_schema {
            // Schema was empty, now we have detected column types from the data
            // This enables schema-on-write behavior
        }

        let coerce_ms = t_coerce.elapsed().as_millis();

        // 1. Write-Ahead Log (Durability) & 2. Indexing (In-Memory) in PARALLEL
        let t_wal = std::time::Instant::now();
        let wal = self.wal.clone();
        let memory_index = self.indexing.memory_index.clone();
        // Only use explicitly configured index columns — do NOT auto-detect by column name.
        // Auto-detecting "embedding" caused silent 15s HNSW builds on every write.
        let mut target_col = self.indexing.index_columns.read().first().cloned();
        if target_col.is_none() && self.indexing.index_all {
            if let Some(first) = batches.first() {
                target_col = first
                    .schema()
                    .fields()
                    .iter()
                    .find(|f| {
                        matches!(
                            f.data_type(),
                            arrow::datatypes::DataType::List(_)
                                | arrow::datatypes::DataType::FixedSizeList(_, _)
                        )
                    })
                    .map(|f| f.name().clone());
            }
        }

        // 1. Vector extraction task — validated and prepared BEFORE appending to the WAL
        // to guarantee that no write failing in-memory index preparation enters the WAL.
        let t_idx_task = std::time::Instant::now();
        let mut prepared_vectors: Option<crate::core::index::memory::PreparedVectors> = None;
        if let Some(ref col_name) = target_col {
            let is_time_col = batches
                .first()
                .and_then(|b| b.column_by_name(col_name))
                .is_some_and(|col| {
                    matches!(
                        col.data_type(),
                        arrow::datatypes::DataType::Time32(_)
                            | arrow::datatypes::DataType::Time64(_)
                    )
                });
            if is_time_col {
                // Time datatypes are not vectorizable; safely skip in-memory vector indexing.
                tracing::debug!(
                    "Skipping in-memory vector index for Time datatype column '{}'",
                    col_name
                );
            } else {
                let existing_dim = memory_index.read().as_ref().map(|idx| idx.dim());
                let inferred_dim = batches
                    .iter()
                    .find_map(|b| InMemoryVectorIndex::infer_column_dim(b, col_name));
                // Prefer the batch's own vector dimension so dim mismatches fail
                // cleanly in append_prepared instead of mis-slicing FixedSizeList values.
                let target_dim = inferred_dim.or(existing_dim);
                prepared_vectors =
                    InMemoryVectorIndex::extract_prepared(&batches, col_name, target_dim)?;
            }
        }
        let idx_task_ms = t_idx_task.elapsed().as_millis();

        // 2. WAL Append Task — appends according to durability configuration (Sync waits for disk sync, Async batches)
        let tx_id = uuid::Uuid::new_v4();
        let mut batches_for_wal: Vec<RecordBatch> = Vec::with_capacity(batches.len());
        for (i, b) in batches.iter().enumerate() {
            batches_for_wal.push(crate::core::wal::tag_batch_with_wal_tx(b, tx_id, i as u64)?);
        }

        let t_wal_task = std::time::Instant::now();
        let wal_lock = wal.lock().await;
        let t_wal_lock = t_wal_task.elapsed().as_millis();
        // WS2 crash boundary: process dies after the batch is handed to
        // the WAL but before it is fsynced.
        crate::core::fault_injection::check(crate::core::fault_injection::CrashPoint::WalAppend)?;
        for batch in batches_for_wal {
            match durability {
                crate::core::table::WalDurability::Sync => {
                    wal_lock.append_sync(batch).await?;
                }
                crate::core::table::WalDurability::Async => {
                    wal_lock.append_fire_and_forget(batch).await?;
                }
            }
        }
        wal_lock.should_compact()?;
        drop(wal_lock);
        let t_wal_total = t_wal_task.elapsed().as_millis();
        tracing::debug!(
            wal_lock_ms = t_wal_lock,
            wal_total_ms = t_wal_total,
            idx_task_ms,
            "wal_task and idx timing"
        );

        // WS2 crash boundary: WAL is durable, but the manifest has not been
        // committed yet. On reopen the WAL replay must recover these rows.
        crate::core::fault_injection::check(crate::core::fault_injection::CrashPoint::WalFlush)?;
        let wal_idx_ms = t_wal.elapsed().as_millis();
        // -----------------------------
        tracing::debug!(
            rows = total_rows,
            pk_ms,
            coerce_ms,
            wal_idx_ms,
            "write_with_durability_async phase timing"
        );

        // Canonical lock order: pending_writes -> indexing.memory_index.
        {
            let mut pending = self.pending_writes.write();
            let mut idx_lock = self.indexing.memory_index.write();
            if let Some(prep) = prepared_vectors {
                match idx_lock.as_mut() {
                    Some(existing) => {
                        if let Err(e) = existing.append_prepared(prep) {
                            tracing::warn!(
                                "Skipping in-memory vector index append due to error: {}",
                                e
                            );
                        }
                    }
                    None => {
                        *idx_lock = Some(InMemoryVectorIndex::from_prepared(prep));
                    }
                }
            }
            pending.extend(
                batches
                    .into_iter()
                    .map(|batch| PendingWrite { batch, tx_id }),
            );
        }
        drop(_pk_validation_guard);
        self.bump_buffer_epoch();

        // Release the maintenance read guard now that the WAL, memory index, and
        // pending_writes have been updated. This prevents deadlocks when autocommit/spillover
        // calls `commit_async()` -> `flush_async()`, and ensures queued maintenance
        // writers are not blocked across downstream I/O.
        drop(_maintenance_guard);

        // Check if we should flush (spillover)
        let should_flush = {
            let pending = self.pending_writes.read();

            // Calculate size in bytes (approximate)
            let total_bytes: usize = pending
                .iter()
                .map(|p| p.batch.get_array_memory_size())
                .sum();

            let cache_gb: usize = std::env::var("BDB_CACHE_GB")
                .unwrap_or_else(|_| "1".to_string())
                .parse()
                .unwrap_or(1);
            let limit_bytes = cache_gb * 1024 * 1024 * 1024;

            total_bytes > limit_bytes
        };

        if should_flush || self.get_autocommit() {
            if should_flush {
                tracing::info!("Write buffer exceeded limit. Flushing to disk (Spillover)...");
            }
            self.commit_async().await?;
        }

        Ok(())
    }

    /// Flush buffer to disk (protected by maintenance read barrier)
    #[tracing::instrument(skip(self))]
    pub async fn flush_async(&self) -> Result<()> {
        self.ensure_writable()?;
        let _maintenance_guard = self.maintenance_lock.read().await;
        self.flush_unlocked_async().await
    }

    /// Internal flush implementation without acquiring `maintenance_lock.read()`.
    /// Used by operations that already hold `maintenance_lock.write()` (such as compaction).
    pub(crate) async fn flush_unlocked_async(&self) -> Result<()> {
        let _commit_guard = self.commit_lock.lock().await;

        // Resume or clean up any interrupted in-flight PendingCommit before starting a new flush.
        let prior = {
            let pc_guard = self.pending_commit.lock();
            pc_guard.as_ref().map(|pc| (pc.id, pc.phase))
        };
        if let Some((prior_id, prior_phase)) = prior {
            match prior_phase {
                CommitPhase::Staging => {
                    self.restore_staging_pending_commit_to_buffer(prior_id);
                }
                CommitPhase::ManifestCommitted | CommitPhase::IcebergPublished => {
                    self.finish_pending_commit_tail_async().await?;
                }
            }
        }

        let commit_id = uuid::Uuid::new_v4();
        // Canonical sync lock order: pending_commit -> pending_writes -> indexing.memory_index.
        let (batches_to_write, wal_tx_ids): (Vec<RecordBatch>, Vec<uuid::Uuid>) = {
            let mut pc_guard = self.pending_commit.lock();
            let mut pending = self.pending_writes.write();
            let mut mem_idx_guard = self.indexing.memory_index.write();

            if pending.is_empty() {
                return Ok(());
            }

            let taken = std::mem::take(&mut *pending);
            let frozen_memory_index = mem_idx_guard.take();
            let mut batches = Vec::with_capacity(taken.len());
            let mut tx_ids = Vec::with_capacity(taken.len());
            for p in taken {
                batches.push(p.batch);
                tx_ids.push(p.tx_id);
            }

            *pc_guard = Some(PendingCommit {
                id: commit_id,
                phase: CommitPhase::Staging,
                batches: batches.clone(),
                wal_tx_ids: tx_ids.clone(),
                frozen_memory_index,
                segment_entries: Vec::new(),
                v3_row_id_base: 0,
                added_rows_total: 0,
                committed_manifest: None,
                metadata_location: None,
                published_snapshot: None,
                should_update_schema: false,
                pending_builds: Vec::new(),
                subscribers_notified: false,
            });

            (batches, tx_ids)
        };
        self.bump_buffer_epoch();

        // Seal the active WAL segment so subsequent concurrent writes land in a fresh
        // segment file and post-commit watermark pruning never drops un-flushed rows.
        let rotate_res = {
            let wal = self.wal.lock().await;
            wal.rotate_async().await
        };
        if let Err(e) = rotate_res {
            self.restore_staging_pending_commit_to_buffer(commit_id);
            return Err(e);
        }
        if let Err(e) =
            crate::core::fault_injection::check(crate::core::fault_injection::CrashPoint::WalRotate)
        {
            self.restore_staging_pending_commit_to_buffer(commit_id);
            return Err(e);
        }

        // Stage Parquet files and atomically commit the manifest. On pre-manifest failure,
        // restore the Staging buffer so in-memory visibility and retryability remain intact.
        let staged = match self
            .stage_and_commit_manifest_async(&batches_to_write, &wal_tx_ids, commit_id)
            .await
        {
            Ok(res) => res,
            Err(e) => {
                self.restore_staging_pending_commit_to_buffer(commit_id);
                return Err(e);
            }
        };

        // Transition Staging -> ManifestCommitted atomically under pending_commit,
        // clearing cluster artifacts before bumping buffer_epoch so concurrent readers
        // force-reload the newly committed manifest instead of serving stale indexes.
        {
            let mut pc_guard = self.pending_commit.lock();
            if let Some(pc) = pc_guard.as_mut() {
                if pc.id == commit_id {
                    pc.phase = CommitPhase::ManifestCommitted;
                    pc.frozen_memory_index = None;
                    pc.segment_entries = staged.segment_entries;
                    pc.v3_row_id_base = staged.v3_row_id_base;
                    pc.added_rows_total = staged.added_rows_total;
                    pc.committed_manifest = Some(staged.new_manifest);
                    pc.should_update_schema = staged.should_update_schema;
                    pc.pending_builds = staged.pending_builds;
                }
            }
        }
        {
            let mut committed = self.committed_wal_txs.lock();
            for tx in &wal_tx_ids {
                committed.insert(*tx);
            }
        }
        self.indexing.cluster_indexes.write().clear();
        self.indexing.vector_routers.write().clear();
        self.bump_buffer_epoch();

        // WS2 crash boundary: the manifest is committed (visible) but tail publication
        // (metadata/catalog/WAL pruning/subscriber notification) has not completed yet.
        crate::core::fault_injection::check(
            crate::core::fault_injection::CrashPoint::ManifestVisible,
        )?;

        self.finish_pending_commit_tail_async().await
    }

    /// Restore a `Staging` `PendingCommit` back into `pending_writes` and `memory_index`
    /// after a pre-manifest failure or cancellation, preserving arrival order when
    /// newer concurrent writes have arrived while staging was in flight.
    fn restore_staging_pending_commit_to_buffer(&self, expected_id: uuid::Uuid) {
        let restored_any = {
            let mut pc_guard = self.pending_commit.lock();
            let mut pending = self.pending_writes.write();
            let mut idx_guard = self.indexing.memory_index.write();

            let matches_staging = pc_guard
                .as_ref()
                .is_some_and(|pc| pc.id == expected_id && pc.phase == CommitPhase::Staging);
            if !matches_staging {
                false
            } else if let Some(pc) = pc_guard.take() {
                let mut restored: Vec<PendingWrite> = pc
                    .batches
                    .into_iter()
                    .zip(pc.wal_tx_ids)
                    .map(|(batch, tx_id)| PendingWrite { batch, tx_id })
                    .collect();
                let current_pending = std::mem::take(&mut *pending);
                restored.extend(current_pending);
                *pending = restored;

                let current_idx = idx_guard.take();
                match (pc.frozen_memory_index, current_idx) {
                    (Some(mut frozen), Some(current)) => {
                        if let Err(e) = frozen.extend_from(&current) {
                            tracing::warn!(
                                "Failed to merge concurrent memory_index during staging rollback: {}",
                                e
                            );
                        }
                        *idx_guard = Some(frozen);
                    }
                    (Some(frozen), None) => {
                        *idx_guard = Some(frozen);
                    }
                    (None, Some(current)) => {
                        *idx_guard = Some(current);
                    }
                    (None, None) => {
                        *idx_guard = None;
                    }
                }
                true
            } else {
                false
            }
        };

        if restored_any {
            self.bump_buffer_epoch();
        }
    }

    async fn cleanup_staged_segment_files(&self, entries: &[crate::core::manifest::ManifestEntry]) {
        let base_local = self.uri.strip_prefix("file://").unwrap_or(&self.uri);
        let is_remote = self.uri.contains("://") && !self.uri.starts_with("file://");
        let root_prefix = self.uri.trim_end_matches('/');
        for entry in entries {
            let rel = entry
                .file_path
                .strip_prefix(root_prefix)
                .unwrap_or(&entry.file_path)
                .trim_start_matches('/');
            if !rel.is_empty() {
                let _ = self
                    .store
                    .delete(&object_store::path::Path::from(rel))
                    .await;
            }
            if !is_remote {
                let full = std::path::Path::new(base_local).join(&entry.file_path);
                let _ = tokio::fs::remove_file(&full).await;
            }
        }
    }

    async fn spawn_pending_index_builds(
        &self,
        pending_builds: Vec<PendingIndexBuild>,
    ) -> Result<()> {
        if pending_builds.is_empty() {
            return Ok(());
        }

        let base_path_cap = self
            .uri
            .strip_prefix("file://")
            .unwrap_or(&self.uri)
            .to_string();
        let index_cols_cap = self.indexing.index_columns.read().clone();
        let index_all_flag = self.indexing.index_all;
        let index_configs_cap = self.indexing.index_configs.read().clone();
        let default_device_cap = self.indexing.default_device.read().clone();
        let pk_cap = self.primary_key.read().clone();
        let table_store_cap = self.store.clone();
        let spec_cap = self.partition_spec.clone();
        let memory_reclaimed_cap = self.memory_reclaimed.clone();
        let manifest_manager_cap = ManifestManager::new(self.store.clone(), "", &self.uri);

        for (entry, batch, partition_values, segment_id) in pending_builds {
            let index_cols_c = index_cols_cap.clone();
            let base_path_c = base_path_cap.clone();
            let segment_id_c = segment_id.clone();
            let batch_c = batch.clone();
            let partition_values_c = partition_values.clone();
            let index_configs_c = index_configs_cap.clone();
            let default_device_c = default_device_cap.clone();
            let entry_c = entry.clone();
            let pk_c = pk_cap.clone();
            let table_store_c = table_store_cap.clone();
            let spec_c = spec_cap.clone();
            let memory_reclaimed_c = memory_reclaimed_cap.clone();
            let manifest_manager_c = manifest_manager_cap.clone();

            // Bound concurrent index builds (see `Table::index_build_gate`).
            let permit = self.acquire_index_build_permit().await?;

            let handle = tokio::spawn(async move {
                let _permit = permit;
                let hive_path = spec_c.partition_to_path(&partition_values_c);
                let full_base_path = if hive_path.is_empty() {
                    base_path_c
                } else {
                    format!("{}/{}", base_path_c, hive_path)
                };
                let parquet_path_rel = if hive_path.is_empty() {
                    format!("{}.parquet", segment_id_c)
                } else {
                    format!("{}/{}.parquet", hive_path, segment_id_c)
                };
                let config_index = SegmentConfig::new(&full_base_path, &segment_id_c)
                    .with_index_all(index_all_flag)
                    .with_columns_to_index(index_cols_c)
                    .with_partition_values(partition_values_c.clone())
                    .with_column_devices(
                        index_configs_c
                            .iter()
                            .filter_map(|(c, cfg)| {
                                cfg.device.as_ref().map(|d| (c.clone(), d.clone()))
                            })
                            .collect(),
                    )
                    .with_default_device(default_device_c)
                    .with_parquet_path(parquet_path_rel);

                let index_res = tokio::spawn(async move {
                    let mut index_writer =
                        HybridSegmentWriter::new(config_index).with_index_configs(index_configs_c);
                    index_writer.primary_key = pk_c;
                    index_writer.set_store(table_store_c);
                    index_writer.build_indexes(&batch_c, 0)?;
                    index_writer.finish_indexing().await?;
                    crate::core::fault_injection::check(
                        crate::core::fault_injection::CrashPoint::IndexUpload,
                    )?;
                    index_writer.upload_to_store().await?;
                    let files = index_writer.get_generated_files();
                    let updated_entry_info = index_writer.to_manifest_entry();
                    Ok::<(crate::core::manifest::ManifestEntry, Vec<String>), anyhow::Error>((
                        updated_entry_info,
                        files,
                    ))
                })
                .await;

                match index_res {
                    Ok(Ok((updated_entry, _files))) => {
                        let mut merged_entry = entry_c;
                        let mut updated_index_files = updated_entry.index_files;
                        let hive_path = spec_c.partition_to_path(&partition_values_c);
                        if !hive_path.is_empty() {
                            for idx in &mut updated_index_files {
                                idx.file_path =
                                    crate::core::segment::ensure_prefix(&hive_path, &idx.file_path);
                            }
                        }
                        merged_entry.index_files = updated_index_files;
                        // This is an in-place update: remove the unindexed entry and
                        // add the indexed one. If a concurrent compaction already
                        // removed this segment while its indexes were building, the
                        // removal is a no-op and the add would *resurrect* the file —
                        // whose rows the compacted replacement already contains, so
                        // every row in it becomes a duplicate. Commit no-op instead.
                        let commit_metadata = crate::core::manifest::CommitMetadata {
                            discard_add_on_missing_remove: true,
                            ..Default::default()
                        };
                        let file_path = merged_entry.file_path.clone();
                        let index_count = merged_entry.index_files.len();
                        let remove_paths = vec![merged_entry.file_path.clone()];
                        match manifest_manager_c
                            .commit(&[merged_entry], &remove_paths, commit_metadata)
                            .await
                        {
                            Ok(_) => tracing::info!(
                                "Successfully attached {} indexes to manifest for segment {}",
                                index_count,
                                file_path
                            ),
                            Err(e) => tracing::error!(
                                "Failed to attach indexes for segment {}: {}",
                                file_path,
                                e
                            ),
                        }
                    }
                    _ => {
                        tracing::error!("Index building failed for segment {}", segment_id_c);
                    }
                }
                memory_reclaimed_c.notify_waiters();
            });
            self.background_tasks.lock().await.push(handle);
        }

        Ok(())
    }

    async fn stage_and_commit_manifest_async(
        &self,
        batches_to_write: &[RecordBatch],
        wal_tx_ids: &[uuid::Uuid],
        commit_id: uuid::Uuid,
    ) -> Result<StagedManifestResult> {
        type PartitionSegment = (
            crate::core::manifest::ManifestEntry,
            Vec<String>,
            String,
            RecordBatch,
            HashMap<String, Value>,
        );

        const MAX_V3_RETRIES: usize = 64;
        let spec = self.partition_spec.clone();
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let is_local_uri = !self.uri.contains("://") || self.uri.starts_with("file://");
        let base_path = self.uri.strip_prefix("file://").unwrap_or(&self.uri);
        if is_local_uri {
            std::fs::create_dir_all(base_path)?;
        }

        let pk_cols = self.primary_key.read().clone();
        let _cross_process_guard = if !pk_cols.is_empty() {
            let lock_path = self.pk_lock_path();
            let lock =
                crate::core::lock::FileBasedLock::new(self.store.clone(), lock_path.clone(), 30);
            let mut acquired = None;
            for _ in 0..100 {
                if let Some(guard) = lock.try_acquire().await? {
                    acquired = Some(guard);
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
            match acquired {
                Some(guard) => Some(guard),
                None => {
                    return Err(anyhow::Error::from(
                        crate::core::error::BenoStreamError::LockAcquireFailed {
                            resource: lock_path.to_string(),
                            retries: 100,
                        },
                    ));
                }
            }
        } else {
            None
        };

        for v3_attempt in 0..MAX_V3_RETRIES {
            let (head_manifest, head_ver) = manifest_manager.load_latest_direct().await?;
            let existing_entries = if head_ver > 0 {
                manifest_manager.load_all_entries(&head_manifest).await?
            } else {
                Vec::new()
            };

            let head_committed: HashSet<uuid::Uuid> = head_manifest
                .properties
                .get("benodb.committed_wal_tx")
                .map(|s| {
                    s.split(',')
                        .filter_map(|t| uuid::Uuid::parse_str(t.trim()).ok())
                        .collect()
                })
                .unwrap_or_default();

            let row_id_base: i64 = existing_entries
                .iter()
                .filter_map(|e| e.first_row_id.map(|f| f + e.record_count))
                .max()
                .unwrap_or(0);

            // Idempotency guard: if every WAL tx in this Staging commit is already
            // recorded in the durable manifest head (e.g., retry after an ambiguous
            // commit ack), converge on head_manifest without writing duplicate files.
            let uncommitted_batches: Vec<RecordBatch> =
                if !wal_tx_ids.is_empty() && wal_tx_ids.len() == batches_to_write.len() {
                    batches_to_write
                        .iter()
                        .zip(wal_tx_ids.iter())
                        .filter(|(_, tx)| !head_committed.contains(tx))
                        .map(|(b, _)| b.clone())
                        .collect()
                } else {
                    batches_to_write.to_vec()
                };

            if !wal_tx_ids.is_empty() && uncommitted_batches.is_empty() {
                tracing::debug!(
                    "All {} WAL transaction(s) already committed in manifest v{}; converging without rewrite",
                    wal_tx_ids.len(),
                    head_ver
                );
                manifest_manager
                    .update_version_cache_monotonic(&head_manifest, head_ver)
                    .await;
                return Ok(StagedManifestResult {
                    new_manifest: head_manifest,
                    segment_entries: existing_entries,
                    v3_row_id_base: row_id_base,
                    added_rows_total: 0,
                    should_update_schema: false,
                    pending_builds: Vec::new(),
                });
            }

            let raw_fv = self.get_format_version().max(head_manifest.format_version);
            let effective_fv = if raw_fv == 0 { 2 } else { raw_fv };
            let sequence_number = (head_ver + 1) as i64;

            // Consolidate batches of compatible schemas before partitioning/writing
            let consolidated_batches: Vec<RecordBatch> = {
                let mut groups: Vec<(arrow::datatypes::SchemaRef, Vec<RecordBatch>)> = Vec::new();
                for b in &uncommitted_batches {
                    if b.num_rows() == 0 {
                        continue;
                    }
                    let mut placed = false;
                    for (schema, list) in groups.iter_mut() {
                        if schema == &b.schema() {
                            list.push(b.clone());
                            placed = true;
                            break;
                        }
                    }
                    if !placed {
                        groups.push((b.schema(), vec![b.clone()]));
                    }
                }
                let mut res = Vec::new();
                for (schema, list) in groups {
                    if list.len() == 1 {
                        if let Some(first) = list.into_iter().next() {
                            res.push(first);
                        }
                    } else if let Ok(merged) = arrow::compute::concat_batches(&schema, &list) {
                        res.push(merged);
                    } else {
                        res.extend(list);
                    }
                }
                res
            };

            // Align every batch to the table's full evolved schema so that
            // schema-evolved columns are present in EVERY segment.
            let table_schema = self.arrow_schema();
            let aligned_batches: Vec<RecordBatch> = consolidated_batches
                .into_iter()
                .map(|b| {
                    if b.schema() == table_schema {
                        b
                    } else {
                        let mut cols = Vec::with_capacity(table_schema.fields().len());
                        for field in table_schema.fields() {
                            let col = if let Some(c) = b.column_by_name(field.name()) {
                                c.clone()
                            } else {
                                arrow::array::new_null_array(field.data_type(), b.num_rows())
                            };
                            cols.push(col);
                        }
                        match RecordBatch::try_new(table_schema.clone(), cols) {
                            Ok(aligned) => aligned,
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    "flush: failed to align batch to table schema; writing as-is"
                                );
                                b
                            }
                        }
                    }
                })
                .collect();

            let mut partitioned_batches = Vec::new();
            for batch in &aligned_batches {
                let sorted_batch = self.apply_sort_order(batch)?;
                let mut pb = spec.partition_batch(&sorted_batch)?;
                partitioned_batches.append(&mut pb);
            }

            let mut grouped_partitions: Vec<(HashMap<String, Value>, Vec<RecordBatch>)> =
                Vec::new();
            for (pv, b) in partitioned_batches {
                if b.num_rows() == 0 {
                    continue;
                }
                let mut placed = false;
                for (p_key, b_list) in grouped_partitions.iter_mut() {
                    if p_key == &pv {
                        b_list.push(b.clone());
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    grouped_partitions.push((pv, vec![b]));
                }
            }
            let coalesced_partitions: Vec<(HashMap<String, Value>, RecordBatch)> = {
                let mut res = Vec::new();
                for (pv, list) in grouped_partitions {
                    if list.len() == 1 {
                        if let Some(first) = list.into_iter().next() {
                            res.push((pv, first));
                        }
                    } else {
                        let schema = list[0].schema();
                        if let Ok(merged) = arrow::compute::concat_batches(&schema, &list) {
                            res.push((pv, merged));
                        } else {
                            for b in list {
                                res.push((pv.clone(), b));
                            }
                        }
                    }
                }
                res
            };

            let mut next_row_id = row_id_base;
            let mut partitions_with_lineage: Vec<(
                HashMap<String, Value>,
                RecordBatch,
                Option<i64>,
            )> = Vec::with_capacity(coalesced_partitions.len());
            for (pv, batch) in coalesced_partitions {
                if effective_fv >= 3 {
                    let first_row_id = next_row_id;
                    let batch =
                        self.add_v3_metadata_columns(&batch, sequence_number, first_row_id)?;
                    next_row_id += batch.num_rows() as i64;
                    partitions_with_lineage.push((pv, batch, Some(first_row_id)));
                } else {
                    partitions_with_lineage.push((pv, batch, None));
                }
            }

            let mut all_new_entries = Vec::new();
            let mut files_to_upload: Vec<(String, String)> = Vec::new();
            let mut pending_builds: Vec<PendingIndexBuild> = Vec::new();
            let index_cols = self.indexing.index_columns.read().clone();
            let index_all_flag = self.indexing.index_all;
            let default_device = self.indexing.default_device.read().clone();

            let concurrency = self
                .query_config
                .max_parallel_segments
                .unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(|p| p.get())
                        .unwrap_or(16)
                })
                .min(64);
            let stream = futures::stream::iter(partitions_with_lineage.into_iter().map(
                |(partition_values, batch, first_row_id)| {
                    let base_path = base_path.to_string();
                    let spec = spec.clone();
                    let default_device_inner = default_device.clone();
                    async move {
                        let segment_id = format!("seg_{}", uuid::Uuid::new_v4());
                        let hive_path = spec.partition_to_path(&partition_values);
                        let full_base_path = if hive_path.is_empty() {
                            base_path
                        } else {
                            format!("{}/{}", base_path, hive_path)
                        };
                        if is_local_uri {
                            let _ = std::fs::create_dir_all(&full_base_path);
                        }

                        let config_write = SegmentConfig::new(&full_base_path, &segment_id)
                            .with_index_all(false)
                            .with_columns_to_index(Vec::new())
                            .with_partition_values(partition_values.clone())
                            .with_default_device(default_device_inner);
                        let mut writer_write = HybridSegmentWriter::new(config_write);
                        writer_write.primary_key = self.primary_key.read().clone();
                        writer_write.set_store(self.store.clone());

                        let batch_inner = batch.clone();
                        let (mut entry, generated_files) =
                            tokio::task::spawn_blocking(move || {
                                writer_write.write_batch(&batch_inner)?;
                                let entry = writer_write.to_manifest_entry();
                                let files = writer_write.get_generated_files();
                                Ok::<
                                    (crate::core::manifest::ManifestEntry, Vec<String>),
                                    anyhow::Error,
                                >((entry, files))
                            })
                            .await
                            .context("Flush task panicked")??;
                        entry.first_row_id = first_row_id;

                        Ok::<PartitionSegment, anyhow::Error>((
                            entry,
                            generated_files,
                            segment_id,
                            batch,
                            partition_values,
                        ))
                    }
                },
            ))
            .buffer_unordered(concurrency);

            let results: Vec<Result<PartitionSegment>> = stream.collect().await;
            let has_pks = !self.primary_key.read().is_empty();
            for res in results {
                let (mut entry, generated_files, segment_id, batch, partition_values): PartitionSegment =
                    res?;

                let hive_path = spec.partition_to_path(&partition_values);
                if !hive_path.is_empty() {
                    entry.file_path =
                        crate::core::segment::ensure_prefix(&hive_path, &entry.file_path);
                    for idx in &mut entry.index_files {
                        idx.file_path =
                            crate::core::segment::ensure_prefix(&hive_path, &idx.file_path);
                    }
                }

                for local_path in generated_files {
                    let filename = local_path
                        .split('/')
                        .next_back()
                        .unwrap_or(&local_path)
                        .to_string();
                    let remote_path = if hive_path.is_empty() {
                        filename
                    } else {
                        format!("{}/{}", hive_path, filename)
                    };
                    files_to_upload.push((local_path, remote_path));
                }
                all_new_entries.push(entry.clone());

                if index_all_flag || !index_cols.is_empty() || has_pks {
                    pending_builds.push((entry, batch, partition_values, segment_id));
                }
            }

            // WS2 crash boundary: data files are staged on disk before manifest commit.
            if let Err(e) = crate::core::fault_injection::check(
                crate::core::fault_injection::CrashPoint::DataUpload,
            ) {
                self.cleanup_staged_segment_files(&all_new_entries).await;
                return Err(e);
            }

            // Upload data files synchronously BEFORE committing manifest/metadata.
            if self.uri.contains("://") && !self.uri.starts_with("file://") {
                let store_clone = self.store.clone();
                for (local_path_str, remote_path_str) in files_to_upload {
                    let local_path = std::path::Path::new(&local_path_str);
                    let mut file = tokio::fs::File::open(&local_path).await.with_context(|| {
                        format!(
                            "Failed to open local staged file for upload: {}",
                            local_path_str
                        )
                    })?;
                    let remote_path = object_store::path::Path::from(remote_path_str.as_str());
                    let mut upload =
                        store_clone
                            .put_multipart(&remote_path)
                            .await
                            .with_context(|| {
                                format!(
                                    "Failed to initiate multipart upload to {}",
                                    remote_path_str
                                )
                            })?;
                    use tokio::io::AsyncReadExt;
                    // Accumulate to a legal part size before emitting, as in
                    // `stage::upload_staged_file`: `read()` returns short chunks
                    // (tokio caps each call at 2 MiB), and a non-final part under
                    // the provider floor is rejected as `EntityTooSmall`.
                    let object_len = file.metadata().await.map(|m| m.len()).unwrap_or(0);
                    let part_bytes = crate::core::storage::multipart_part_bytes(object_len);
                    let mut part: Vec<u8> = Vec::with_capacity(part_bytes);
                    let mut buf = vec![0; 1024 * 1024];
                    let mut total_uploaded = 0;
                    loop {
                        let n = file.read(&mut buf).await.with_context(|| {
                            format!("Failed to read local staged file: {}", local_path_str)
                        })?;
                        if n == 0 {
                            break;
                        }
                        total_uploaded += n;
                        part.extend_from_slice(&buf[..n]);
                        if part.len() >= part_bytes {
                            let full = std::mem::replace(&mut part, Vec::with_capacity(part_bytes));
                            upload.put_part(full.into()).await.with_context(|| {
                                format!("Failed to upload part to {}", remote_path_str)
                            })?;
                        }
                    }
                    if !part.is_empty() {
                        upload.put_part(part.into()).await.with_context(|| {
                            format!("Failed to upload the final part to {}", remote_path_str)
                        })?;
                    }
                    upload.complete().await.with_context(|| {
                        format!("Failed to complete multipart upload to {}", remote_path_str)
                    })?;
                    tracing::info!(
                        "Successfully uploaded staged file {} ({} bytes) to {}",
                        local_path_str,
                        total_uploaded,
                        remote_path_str
                    );
                    let _ = tokio::fs::remove_file(&local_path).await;
                }
            }

            let current_schema = self.arrow_schema();
            let should_update_schema = if let Some(last_schema) = head_manifest.schemas.last() {
                let latest_schema = last_schema.to_arrow();
                latest_schema != *current_schema
            } else {
                true
            };

            let (final_schemas, final_schema_id) = if should_update_schema {
                let mut new_schemas = head_manifest.schemas.clone();
                let new_id = if head_manifest.schemas.is_empty() {
                    0
                } else {
                    head_manifest.current_schema_id + 1
                };
                let mut manifest_schema =
                    crate::core::manifest::Schema::from_arrow(&current_schema, new_id);
                {
                    let cfgs = self.indexing.index_configs.read();
                    for field in &mut manifest_schema.fields {
                        if let Some(cfg) = cfgs.get(&field.name) {
                            if !cfg.algorithms.is_empty() {
                                field.indexes = cfg.algorithms.clone();
                            }
                        }
                    }
                }
                new_schemas.push(manifest_schema);
                (Some(new_schemas), Some(new_id))
            } else {
                (None, None)
            };

            let (final_sort_orders, final_sort_order_id) =
                if let Some(order) = self.get_sort_order() {
                    (Some(vec![order.clone()]), Some(order.order_id))
                } else {
                    (None, None)
                };

            let append_committed_wal_tx: Vec<String> =
                wal_tx_ids.iter().map(|t| t.to_string()).collect();

            let commit_metadata = crate::core::manifest::CommitMetadata {
                updated_schemas: final_schemas,
                updated_schema_id: final_schema_id,
                updated_partition_specs: None,
                updated_default_spec_id: None,
                updated_properties: None,
                removed_properties: None,
                updated_sort_orders: final_sort_orders,
                updated_default_sort_order_id: final_sort_order_id,
                updated_last_column_id: None,
                format_version: Some(effective_fv),
                is_fast_append: false,
                expected_version: if effective_fv >= 3 || !pk_cols.is_empty() {
                    Some(head_ver)
                } else {
                    None
                },
                append_committed_wal_tx,
                ..Default::default()
            };

            let added_rows_total: i64 = all_new_entries.iter().map(|e| e.record_count).sum();

            if !pk_cols.is_empty() {
                if let Err(e) = self
                    .validate_staged_entries_primary_keys(
                        &all_new_entries,
                        &pk_cols,
                        Some(commit_id),
                    )
                    .await
                {
                    self.cleanup_staged_segment_files(&all_new_entries).await;
                    return Err(e);
                }
            }

            // WS2 crash boundary: immediately before the atomic manifest publish point.
            if let Err(e) = crate::core::fault_injection::check(
                crate::core::fault_injection::CrashPoint::ManifestCommit,
            ) {
                self.cleanup_staged_segment_files(&all_new_entries).await;
                return Err(e);
            }

            match manifest_manager
                .commit(&all_new_entries, &[], commit_metadata)
                .await
            {
                Ok(new_manifest) => {
                    return Ok(StagedManifestResult {
                        new_manifest,
                        segment_entries: all_new_entries,
                        v3_row_id_base: row_id_base,
                        added_rows_total,
                        should_update_schema,
                        pending_builds,
                    });
                }
                Err(err)
                    if (effective_fv >= 3 || !pk_cols.is_empty())
                        && matches!(
                            err.downcast_ref::<BenoStreamError>(),
                            Some(BenoStreamError::ManifestVersionMismatch { .. })
                        ) =>
                {
                    tracing::debug!(
                        v3_attempt,
                        "Concurrent manifest publication detected; reloading manifest and re-validating primary keys"
                    );
                    self.cleanup_staged_segment_files(&all_new_entries).await;
                    if v3_attempt + 1 == MAX_V3_RETRIES {
                        return Err(err);
                    }
                    let delay_ms = 2 + (rand::random::<u64>() % 14);
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                }
                Err(err) => return Err(err),
            }
        }

        anyhow::bail!(
            "Failed to stage and commit V3 manifest after {} concurrent attempts",
            MAX_V3_RETRIES
        )
    }

    async fn finish_pending_commit_tail_async(&self) -> Result<()> {
        let (
            phase,
            new_manifest,
            v3_row_id_base,
            added_rows_total,
            segment_entries,
            existing_meta_loc,
            existing_snap,
            should_update_schema,
            wal_tx_ids,
            pending_builds,
        ) = {
            let mut pc_guard = self.pending_commit.lock();
            let Some(pc) = pc_guard.as_mut() else {
                return Ok(());
            };
            if pc.phase == CommitPhase::Staging {
                return Ok(());
            }
            let Some(manifest) = pc.committed_manifest.clone() else {
                anyhow::bail!("PendingCommit in {:?} missing committed_manifest", pc.phase);
            };
            let builds = std::mem::take(&mut pc.pending_builds);
            (
                pc.phase,
                manifest,
                pc.v3_row_id_base,
                pc.added_rows_total,
                pc.segment_entries.clone(),
                pc.metadata_location.clone(),
                pc.published_snapshot.clone(),
                pc.should_update_schema,
                pc.wal_tx_ids.clone(),
                builds,
            )
        };

        // Spawn background index builds once (drained from PendingCommit above).
        self.spawn_pending_index_builds(pending_builds).await?;

        if phase < CommitPhase::IcebergPublished {
            let effective_fv = if new_manifest.format_version == 0 {
                self.get_format_version().max(2)
            } else {
                new_manifest.format_version
            };
            let v3_first_row_id = if effective_fv >= 3 {
                segment_entries
                    .iter()
                    .filter_map(|e| e.first_row_id)
                    .min()
                    .or(Some(v3_row_id_base))
            } else {
                None
            };
            let v3_added_rows = if effective_fv >= 3 {
                Some(added_rows_total)
            } else {
                None
            };
            let v3_next_row_id = if effective_fv >= 3 {
                Some(
                    segment_entries
                        .iter()
                        .filter_map(|e| e.first_row_id.map(|f| f + e.record_count))
                        .max()
                        .unwrap_or(v3_row_id_base + added_rows_total),
                )
            } else {
                None
            };

            let catalog_tuple = match (
                &self.catalog_state.catalog,
                &self.catalog_state.namespace,
                &self.catalog_state.table_name,
            ) {
                (Some(cat), Some(ns), Some(tbl)) => Some((cat.clone(), ns.clone(), tbl.clone())),
                _ => None,
            };

            let prior_catalog_meta = if let Some((ref cat, ref ns, ref tbl)) = catalog_tuple {
                cat.load_table(ns, tbl).await.ok()
            } else {
                None
            };

            let snapshot_id = new_manifest.version as i64;
            let already_in_catalog = prior_catalog_meta.as_ref().and_then(|cm| {
                cm.snapshots
                    .iter()
                    .find(|s| s.snapshot_id == snapshot_id)
                    .cloned()
            });

            if let Some(catalog_snap) = already_in_catalog {
                let mut pc_guard = self.pending_commit.lock();
                if let Some(pc) = pc_guard.as_mut() {
                    if pc.published_snapshot.is_none() {
                        pc.published_snapshot = Some(catalog_snap);
                    }
                    pc.phase = CommitPhase::IcebergPublished;
                }
            } else {
                let meta_location_root = prior_catalog_meta
                    .as_ref()
                    .map(|m| m.location.clone())
                    .unwrap_or_else(|| self.uri.clone());
                let meta_store_arc = self.resolve_metadata_store(&meta_location_root)?;

                let (written_loc, published_snap) =
                    if let (Some(loc), Some(snap)) = (existing_meta_loc, existing_snap) {
                        (loc, snap)
                    } else {
                        crate::core::fault_injection::check(
                            crate::core::fault_injection::CrashPoint::MetadataSave,
                        )?;
                        let (loc, snap) = self
                            .publish_metadata_step_async(
                                &meta_location_root,
                                meta_store_arc.as_ref(),
                                &new_manifest,
                                "append",
                                prior_catalog_meta.as_ref().map(|m| m.table_uuid.as_str()),
                                prior_catalog_meta
                                    .as_ref()
                                    .and_then(|m| m.current_snapshot_id),
                                v3_first_row_id,
                                v3_added_rows,
                                v3_next_row_id,
                                should_update_schema,
                            )
                            .await?;
                        {
                            let mut pc_guard = self.pending_commit.lock();
                            if let Some(pc) = pc_guard.as_mut() {
                                pc.metadata_location = Some(loc.clone());
                                pc.published_snapshot = Some(snap.clone());
                            }
                        }
                        (loc, snap)
                    };

                if let Some((ref cat, ref ns, ref tbl)) = catalog_tuple {
                    crate::core::fault_injection::check(
                        crate::core::fault_injection::CrashPoint::CatalogCommit,
                    )?;
                    let (cat_loc, cat_snap) = self
                        .commit_catalog_step_async(
                            cat.as_ref(),
                            ns,
                            tbl,
                            &meta_location_root,
                            meta_store_arc.as_ref(),
                            &new_manifest,
                            "append",
                            v3_first_row_id,
                            v3_added_rows,
                            v3_next_row_id,
                            should_update_schema,
                            written_loc,
                            published_snap,
                        )
                        .await?;
                    let mut pc_guard = self.pending_commit.lock();
                    if let Some(pc) = pc_guard.as_mut() {
                        pc.metadata_location = Some(cat_loc);
                        pc.published_snapshot = Some(cat_snap);
                    }
                }

                let mut pc_guard = self.pending_commit.lock();
                if let Some(pc) = pc_guard.as_mut() {
                    pc.phase = CommitPhase::IcebergPublished;
                }
            }
        }

        self.indexing.cluster_indexes.write().clear();
        self.indexing.vector_routers.write().clear();

        // WS2 crash boundary: manifest + metadata committed, WAL segments not yet pruned.
        crate::core::fault_injection::check(crate::core::fault_injection::CrashPoint::WalTruncate)?;
        let mut committed_set = {
            let mut guard = self.committed_wal_txs.lock();
            for tx in &wal_tx_ids {
                guard.insert(*tx);
            }
            guard.clone()
        };
        if let Some(csv) = new_manifest.properties.get("benodb.committed_wal_tx") {
            for token in csv.split(',') {
                if let Ok(uuid) = uuid::Uuid::parse_str(token.trim()) {
                    committed_set.insert(uuid);
                }
            }
        }
        let pruned_paths = {
            let wal = self.wal.lock().await;
            wal.prune_committed_segments(&committed_set)
                .context("Failed to prune committed WAL segments")?
        };
        if !pruned_paths.is_empty() {
            let pruned_set: HashSet<String> = pruned_paths
                .iter()
                .filter_map(|p| p.to_str().map(str::to_string))
                .collect();
            let mut recovered = self.recovered_wal_paths.lock();
            recovered.retain(|p| !pruned_set.contains(p));
        }

        // Publish committed batches to `Table::subscribe()` consumers at most once.
        let notify_batches: Option<Vec<RecordBatch>> = {
            let mut pc_guard = self.pending_commit.lock();
            if let Some(pc) = pc_guard.as_mut() {
                if !pc.subscribers_notified {
                    pc.subscribers_notified = true;
                    Some(pc.batches.clone())
                } else {
                    None
                }
            } else {
                None
            }
        };
        if let Some(ref batches_to_notify) = notify_batches {
            self.publish_committed(batches_to_notify);
        }

        {
            let mut pc_guard = self.pending_commit.lock();
            *pc_guard = None;
        }
        self.bump_buffer_epoch();

        Ok(())
    }
}

struct StagedManifestResult {
    new_manifest: crate::core::manifest::Manifest,
    segment_entries: Vec<crate::core::manifest::ManifestEntry>,
    v3_row_id_base: i64,
    added_rows_total: i64,
    should_update_schema: bool,
    pending_builds: Vec<PendingIndexBuild>,
}
