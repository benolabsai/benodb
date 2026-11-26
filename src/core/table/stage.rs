// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Staged data-file writes for distributed connectors.
//!
//! Spark's `BatchWrite` contract splits a write in two: the **task** writes data
//! files, the **driver** commits their metadata (`SparkWrite$TaskCommit` carries
//! `DataFile[]`, not rows). That is the only shape that keeps the driver out of
//! the data path and makes the commit metadata-sized regardless of write volume.
//!
//! So this module exposes those two halves:
//!
//!  * [`Table::stage_data_files_async`] — write batches to data files (and their
//!    indexes) and return the manifest entries. Nothing is published: a reader,
//!    engine or external, cannot see the files yet.
//!  * [`Table::commit_staged_entries_async`] — commit entries produced anywhere
//!    in one snapshot. This is a metadata-only operation, so it is safe to run on
//!    the driver.
//!
//! A retried task attempt writes new files at new paths; only the entries the
//! driver accepted are ever committed, and the uncommitted files become orphans
//! for `remove_orphan_files`.
//!
//! `Table::write_mode` says whether a table supports this: `format_version = 3`
//! tables do not, because `_row_id` is assigned from the manifest head and two
//! tasks staging concurrently would hand out overlapping ids.

use anyhow::{Context, Result};
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::Arc;

use super::Table;
use crate::core::manifest::{IndexFile, ManifestEntry, ManifestManager, ManifestValue};
use crate::core::segment::HybridSegmentWriter;
use crate::SegmentConfig;

impl Table {
    /// How a connector should perform writes against this table.
    ///
    /// * `"files"` — tasks may write data files, the driver commits their
    ///   entries ([`Table::stage_data_files_async`] /
    ///   [`Table::commit_staged_entries_async`]).
    /// * `"buffered"` — the table needs commit-time coordination a task cannot
    ///   provide (`format_version = 3` row-id assignment), so writes must go
    ///   through `write_async` + `commit_async`.
    pub fn write_mode(&self) -> &'static str {
        if self.get_format_version() >= 3 {
            "buffered"
        } else {
            "files"
        }
    }

    /// Write `batches` as data files with their indexes and return the manifest
    /// entries. Does **not** commit anything.
    ///
    /// The returned entries are byte-for-byte what `flush` would commit (paths
    /// relative to the table root, hive-partition prefixed, stats populated,
    /// index files attached), so committing them publishes exactly the write the
    /// engine would have made itself.
    pub async fn stage_data_files_async(
        &self,
        batches: Vec<RecordBatch>,
    ) -> Result<Vec<ManifestEntry>> {
        if batches.is_empty() {
            return Ok(Vec::new());
        }
        if self.write_mode() != "files" {
            anyhow::bail!(
                "staged writes are not supported on format_version {} tables (use write_mode() to choose)",
                self.get_format_version()
            );
        }
        let _maintenance_guard = self.locks.maintenance_lock.read().await;

        let table_schema = self.arrow_schema();
        let aligned = self.align_staged_batches(batches, &table_schema)?;
        self.validate_staged_primary_keys(&aligned).await?;

        let spec = self.schema_state.partition_spec.clone();
        let base_path = self
            .uri
            .strip_prefix("file://")
            .unwrap_or(&self.uri)
            .to_string();

        // Sort + partition, coalescing batches that land in the same partition so
        // a partition becomes one file rather than one per input batch.
        let mut grouped: Vec<(HashMap<String, serde_json::Value>, Vec<RecordBatch>)> = Vec::new();
        for batch in &aligned {
            let sorted = self.apply_sort_order(batch)?;
            for (partition_values, split) in spec.partition_batch(&sorted)? {
                if split.num_rows() == 0 {
                    continue;
                }
                match grouped.iter_mut().find(|(key, _)| key == &partition_values) {
                    Some((_, list)) => list.push(split),
                    None => grouped.push((partition_values, vec![split])),
                }
            }
        }

        let index_columns = self.indexing.index_columns.read().clone();
        let index_all = self.indexing.index_all;
        let index_configs: HashMap<String, crate::core::table::ColumnIndexConfig> =
            self.indexing.index_configs.read().clone();
        let default_device = self.indexing.default_device.read().clone();
        let primary_key = self.schema_state.primary_key.read().clone();
        let build_indexes = index_all || !index_columns.is_empty() || !primary_key.is_empty();

        let mut entries = Vec::with_capacity(grouped.len());
        for (partition_values, list) in grouped {
            let batch = match list.len() {
                0 => continue,
                1 => match list.into_iter().next() {
                    Some(batch) => batch,
                    None => continue,
                },
                _ => arrow::compute::concat_batches(&list[0].schema(), &list)?,
            };
            let hive_path = spec.partition_to_path(&partition_values);
            let full_base_path = if hive_path.is_empty() {
                base_path.clone()
            } else {
                format!("{}/{}", base_path, hive_path)
            };
            let segment_id = format!("seg_{}", uuid::Uuid::new_v4());

            let config = SegmentConfig::new(&full_base_path, &segment_id)
                .with_index_all(false)
                .with_columns_to_index(Vec::new())
                .with_partition_values(partition_values.clone())
                .with_default_device(default_device.clone());
            let mut writer = HybridSegmentWriter::new(config);
            writer.primary_key = primary_key.clone();
            writer.set_store(self.store.clone());

            let batch_for_write = batch.clone();
            let (mut entry, generated) = tokio::task::spawn_blocking(move || {
                writer.write_batch(&batch_for_write)?;
                Ok::<(ManifestEntry, Vec<String>), anyhow::Error>((
                    writer.to_manifest_entry(),
                    writer.get_generated_files(),
                ))
            })
            .await
            .context("staged write task panicked")??;

            if !hive_path.is_empty() {
                entry.file_path = crate::core::segment::ensure_prefix(&hive_path, &entry.file_path);
                for index_file in &mut entry.index_files {
                    index_file.file_path =
                        crate::core::segment::ensure_prefix(&hive_path, &index_file.file_path);
                }
            }

            self.upload_staged_files(&generated, &hive_path).await?;

            // Build the indexes here, not later: the driver only commits
            // metadata, so there is no post-commit hook on the table that could
            // attach them afterwards.
            if build_indexes {
                let index_files = self
                    .build_staged_indexes(
                        &batch,
                        &partition_values,
                        &segment_id,
                        &hive_path,
                        &index_columns,
                        index_all,
                        &index_configs,
                        default_device.clone(),
                        &primary_key,
                    )
                    .await?;
                if !index_files.is_empty() {
                    entry.index_files = index_files;
                }
            }

            entries.push(entry);
        }

        tracing::info!(
            "Staged {} data file(s) ({} rows) for {}",
            entries.len(),
            entries.iter().map(|e| e.record_count).sum::<i64>(),
            self.uri
        );
        Ok(entries)
    }

    /// Commit entries produced by [`Table::stage_data_files_async`] (possibly in
    /// another process) as a single snapshot.
    ///
    /// Metadata-only: the data files are already durable, so this is cheap,
    /// independent of the write's size, and safe to run on a connector driver.
    pub async fn commit_staged_entries_async(&self, entries: Vec<ManifestEntry>) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let _maintenance_guard = self.locks.maintenance_lock.read().await;
        let _commit_guard = self.locks.commit_lock.lock().await;

        let pk_columns = self.schema_state.primary_key.read().clone();
        let _pk_val_guard = if !pk_columns.is_empty() {
            Some(self.locks.pk_validation_lock.lock().await)
        } else {
            None
        };

        // Cross-process serialization lock for primary key validation and publication
        let _cross_process_guard = self.acquire_pk_publication_lock(&pk_columns).await?;

        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let root_prefix = self.uri.trim_end_matches('/');
        let normalize_rel = |path: &str| -> String {
            path.strip_prefix(root_prefix)
                .unwrap_or(path)
                .trim_start_matches('/')
                .to_string()
        };

        let total_rows: i64 = entries.iter().map(|e| e.record_count).sum();
        let entry_count = entries.len();

        let mut new_manifest = None;
        let max_commit_attempts = 50;

        for attempt in 0..max_commit_attempts {
            let (latest_manifest, latest_ver) = manifest_manager.load_latest_direct().await?;
            let existing_entries = if latest_ver > 0 {
                manifest_manager.load_all_entries(&latest_manifest).await?
            } else {
                Vec::new()
            };

            let existing_paths: std::collections::HashSet<String> = existing_entries
                .iter()
                .map(|e| normalize_rel(&e.file_path))
                .collect();

            let pending_entries: Vec<ManifestEntry> = entries
                .iter()
                .filter(|e| !existing_paths.contains(&normalize_rel(&e.file_path)))
                .cloned()
                .collect();

            if pending_entries.is_empty() {
                tracing::debug!(
                    "All {} staged data file(s) already present in {} v{}; converging snapshot publication",
                    entry_count,
                    self.uri,
                    latest_manifest.version
                );
                new_manifest = Some(latest_manifest);
                break;
            }

            if !pk_columns.is_empty() {
                self.validate_staged_entries_primary_keys(&pending_entries, &pk_columns, None)
                    .await?;
            }

            if let Some(ref guard) = _cross_process_guard {
                guard.verify_valid().await?;
            }

            let commit_metadata = crate::core::manifest::CommitMetadata {
                expected_version: if !pk_columns.is_empty() {
                    Some(latest_ver)
                } else {
                    None
                },
                publication_fence: _cross_process_guard.as_ref().map(|g| g.fencing_verifier()),
                ..Default::default()
            };

            match manifest_manager
                .commit(&pending_entries, &[], commit_metadata)
                .await
            {
                Ok(manifest) => {
                    new_manifest = Some(manifest);
                    break;
                }
                Err(err)
                    if matches!(
                        err.downcast_ref::<crate::core::error::BenoStreamError>(),
                        Some(crate::core::error::BenoStreamError::ManifestVersionMismatch { .. })
                    ) =>
                {
                    tracing::debug!(
                        attempt,
                        "Concurrent manifest publication detected; reloading manifest and re-validating primary keys"
                    );
                    let delay_ms = 10 + (rand::random::<u64>() % 25);
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        let new_manifest = new_manifest.ok_or_else(|| {
            anyhow::anyhow!(
                "Failed to commit staged entries to {} after max retries due to concurrent updates",
                self.uri
            )
        })?;

        self.ensure_table_metadata_async(&new_manifest).await?;
        self.publish_snapshot_async(&new_manifest, "append").await?;

        tracing::info!(
            "Committed {} staged data file(s) ({} rows) to {} as snapshot v{}",
            entry_count,
            total_rows,
            self.uri,
            new_manifest.version
        );
        Ok(())
    }

    /// Coerce `batches` to the table schema: cast mismatched types, NULL-fill
    /// columns the batch does not carry, and reject columns the table does not
    /// declare (a staged commit cannot carry an evolved schema).
    fn align_staged_batches(
        &self,
        batches: Vec<RecordBatch>,
        table_schema: &arrow::datatypes::SchemaRef,
    ) -> Result<Vec<RecordBatch>> {
        let mut aligned = Vec::with_capacity(batches.len());
        for batch in batches {
            for field in batch.schema().fields() {
                if table_schema.field_with_name(field.name()).is_err() {
                    anyhow::bail!(
                        "staged write carries column '{}' which the table does not declare; \
                         staged writes cannot evolve the schema",
                        field.name()
                    );
                }
            }
            let mut columns = Vec::with_capacity(table_schema.fields().len());
            for field in table_schema.fields() {
                match batch.column_by_name(field.name()) {
                    Some(column) if column.data_type() == field.data_type() => {
                        columns.push(column.clone())
                    }
                    Some(column) => columns.push(arrow::compute::cast(column, field.data_type())?),
                    None => columns.push(arrow::array::new_null_array(
                        field.data_type(),
                        batch.num_rows(),
                    )),
                }
            }
            aligned.push(RecordBatch::try_new(table_schema.clone(), columns)?);
        }
        Ok(aligned)
    }

    /// Enforce the primary-key invariant the buffered write path enforces:
    /// no NULL keys, no duplicates inside the write, none already in storage.
    async fn validate_staged_primary_keys(&self, batches: &[RecordBatch]) -> Result<()> {
        let pk_columns = self.schema_state.primary_key.read().clone();
        if pk_columns.is_empty() {
            return Ok(());
        }

        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for batch in batches {
            let key_columns: Vec<_> = pk_columns
                .iter()
                .filter_map(|name| batch.column_by_name(name).cloned())
                .collect();
            if key_columns.len() != pk_columns.len() {
                continue;
            }
            for row in 0..batch.num_rows() {
                let mut val_strings = Vec::with_capacity(pk_columns.len());
                for (col_idx, column) in key_columns.iter().enumerate() {
                    let value = ManifestValue::from_array(column, row);
                    if matches!(value, ManifestValue::Null) {
                        return Err(anyhow::Error::from(
                            crate::core::error::BenoStreamError::NullConstraintViolation {
                                column: pk_columns[col_idx].clone(),
                            },
                        ));
                    }
                    val_strings.push(value.to_string());
                }
                let key = crate::core::manifest::encode_composite_key(
                    val_strings.iter().map(|s| s.as_str()),
                );
                if !seen.insert(key.clone()) {
                    return Err(anyhow::Error::from(
                        crate::core::error::BenoStreamError::PrimaryKeyViolation { key },
                    ));
                }
            }
        }

        for batch in batches {
            let key_columns: Vec<_> = pk_columns
                .iter()
                .filter_map(|name| batch.column_by_name(name).cloned())
                .collect();
            if key_columns.len() != pk_columns.len() {
                continue;
            }
            for row in 0..batch.num_rows() {
                let mut val_strings = Vec::with_capacity(pk_columns.len());
                let mut jsons = Vec::with_capacity(pk_columns.len());
                for column in &key_columns {
                    let value = ManifestValue::from_array(column, row);
                    val_strings.push(value.to_string());
                    jsons.push(serde_json::to_value(&value)?);
                }
                let key_str = crate::core::manifest::encode_composite_key(
                    val_strings.iter().map(|s| s.as_str()),
                );
                if self
                    ._check_composite_pk_in_storage_async(&pk_columns, &jsons)
                    .await?
                {
                    return Err(anyhow::Error::from(
                        crate::core::error::BenoStreamError::PrimaryKeyViolation { key: key_str },
                    ));
                }
            }
        }
        Ok(())
    }

    /// Enforce primary-key uniqueness across staged entries and against committed
    /// storage at the commit boundary (under commit_lock and pk_validation_lock).
    pub(crate) async fn validate_staged_entries_primary_keys(
        &self,
        entries: &[ManifestEntry],
        pk_columns: &[String],
        exclude_commit_id: Option<uuid::Uuid>,
    ) -> Result<()> {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        // 1. Pre-populate seen keys from any active in-flight write buffer or staging commit
        {
            let pc_guard = self.commit_buffer.pending_commit.lock();
            let pending = self.commit_buffer.pending_writes.read();
            let staging_iter = pc_guard
                .as_ref()
                .filter(|pc| {
                    pc.phase == super::CommitPhase::Staging && Some(pc.id) != exclude_commit_id
                })
                .into_iter()
                .flat_map(|pc| pc.batches.iter());
            for b_batch in staging_iter.chain(pending.iter().map(|p| &p.batch)) {
                let key_cols: Vec<_> = pk_columns
                    .iter()
                    .filter_map(|name| b_batch.column_by_name(name))
                    .collect();
                if key_cols.len() == pk_columns.len() {
                    for r in 0..b_batch.num_rows() {
                        let mut val_strings = Vec::with_capacity(pk_columns.len());
                        for col in &key_cols {
                            val_strings.push(ManifestValue::from_array(col, r).to_string());
                        }
                        let tuple_key = crate::core::manifest::encode_composite_key(
                            val_strings.iter().map(|s| s.as_str()),
                        );
                        seen.insert(tuple_key);
                    }
                }
            }
        }

        let table_schema = self.arrow_schema();
        let mut pk_fields = Vec::with_capacity(pk_columns.len());
        for name in pk_columns {
            match table_schema.field_with_name(name) {
                Ok(field) => pk_fields.push(field.clone()),
                Err(_) => {
                    return Err(anyhow::Error::from(
                        crate::core::error::BenoStreamError::ColumnNotFound {
                            column: name.clone(),
                            table: Some(self.uri.clone()),
                        },
                    ));
                }
            }
        }
        let proj_schema = Arc::new(arrow::datatypes::Schema::new(pk_fields));

        use futures::StreamExt;
        for entry in entries {
            let reader = self.segment_reader(entry)?;
            let mut stream = reader.stream_all(Some(proj_schema.clone())).await?;
            while let Some(batch_res) = stream.next().await {
                let batch = batch_res?;
                let key_columns: Vec<_> = pk_columns
                    .iter()
                    .filter_map(|name| batch.column_by_name(name).cloned())
                    .collect();
                if key_columns.len() != pk_columns.len() {
                    return Err(anyhow::Error::from(
                        crate::core::error::BenoStreamError::SchemaIncompatible {
                            reason: format!(
                                "Staged file batch is missing required primary key columns: expected {:?}, found {:?}",
                                pk_columns,
                                batch.schema().fields().iter().map(|f| f.name().as_str()).collect::<Vec<_>>()
                            ),
                        },
                    ));
                }
                for row in 0..batch.num_rows() {
                    let mut val_strings = Vec::with_capacity(pk_columns.len());
                    let mut jsons = Vec::with_capacity(pk_columns.len());
                    for (col_idx, column) in key_columns.iter().enumerate() {
                        let value = ManifestValue::from_array(column, row);
                        if matches!(value, ManifestValue::Null) {
                            return Err(anyhow::Error::from(
                                crate::core::error::BenoStreamError::NullConstraintViolation {
                                    column: pk_columns[col_idx].clone(),
                                },
                            ));
                        }
                        val_strings.push(value.to_string());
                        jsons.push(serde_json::to_value(&value)?);
                    }
                    let key = crate::core::manifest::encode_composite_key(
                        val_strings.iter().map(|s| s.as_str()),
                    );
                    if !seen.insert(key.clone()) {
                        return Err(anyhow::Error::from(
                            crate::core::error::BenoStreamError::PrimaryKeyViolation { key },
                        ));
                    }
                    if self
                        ._check_composite_pk_in_storage_async(pk_columns, &jsons)
                        .await?
                    {
                        return Err(anyhow::Error::from(
                            crate::core::error::BenoStreamError::PrimaryKeyViolation { key },
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Upload the local staging files produced by a segment writer.
    async fn upload_staged_files(&self, files: &[String], hive_path: &str) -> Result<()> {
        if !(self.uri.contains("://") && !self.uri.starts_with("file://")) {
            return Ok(());
        }
        for local_path in files {
            let filename = local_path
                .split('/')
                .next_back()
                .unwrap_or(local_path)
                .to_string();
            let remote = if hive_path.is_empty() {
                filename
            } else {
                format!("{}/{}", hive_path, filename)
            };
            self.upload_staged_file(local_path, &remote).await?;
        }
        Ok(())
    }

    /// Stream one local file into the object store and remove it.
    async fn upload_staged_file(&self, local_path: &str, remote_path: &str) -> Result<()> {
        use object_store::path::Path as ObjectPath;
        use tokio::io::AsyncReadExt;

        let mut file = tokio::fs::File::open(local_path)
            .await
            .with_context(|| format!("failed to open staged file {}", local_path))?;
        let remote = ObjectPath::from(remote_path);
        let mut upload = self
            .store
            .put_multipart(&remote)
            .await
            .with_context(|| format!("failed to initiate upload to {}", remote_path))?;
        // Accumulate to a legal part size before emitting. `tokio::fs::File::read`
        // returns at most 2 MiB per call, so emitting per read produced 2 MiB
        // non-final parts, which S3 and GCS reject with `EntityTooSmall`. The
        // size is derived from the object length so the part count stays within
        // the provider limits; see `storage::multipart_part_bytes`.
        let object_len = file.metadata().await.map(|m| m.len()).unwrap_or(0);
        let part_bytes = crate::core::storage::multipart_part_bytes(object_len);
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut part: Vec<u8> = Vec::with_capacity(part_bytes);
        loop {
            let read = file
                .read(&mut buffer)
                .await
                .with_context(|| format!("failed to read staged file {}", local_path))?;
            if read == 0 {
                break;
            }
            part.extend_from_slice(&buffer[..read]);
            if part.len() >= part_bytes {
                let full = std::mem::replace(&mut part, Vec::with_capacity(part_bytes));
                upload
                    .put_part(full.into())
                    .await
                    .with_context(|| format!("failed to upload a part to {}", remote_path))?;
            }
        }
        if !part.is_empty() {
            upload
                .put_part(part.into())
                .await
                .with_context(|| format!("failed to upload the final part to {}", remote_path))?;
        }
        upload
            .complete()
            .await
            .with_context(|| format!("failed to complete upload to {}", remote_path))?;
        let _ = tokio::fs::remove_file(local_path).await;
        Ok(())
    }

    /// Build one data file's indexes and return them (paths relative to the
    /// table root, hive-partition prefixed).
    #[allow(clippy::too_many_arguments)]
    async fn build_staged_indexes(
        &self,
        batch: &RecordBatch,
        partition_values: &HashMap<String, serde_json::Value>,
        segment_id: &str,
        hive_path: &str,
        index_columns: &[String],
        index_all: bool,
        index_configs: &HashMap<String, crate::core::table::ColumnIndexConfig>,
        default_device: Option<String>,
        primary_key: &[String],
    ) -> Result<Vec<IndexFile>> {
        let base_path = self
            .uri
            .strip_prefix("file://")
            .unwrap_or(&self.uri)
            .to_string();
        let full_base_path = if hive_path.is_empty() {
            base_path
        } else {
            format!("{}/{}", base_path, hive_path)
        };
        let parquet_path = if hive_path.is_empty() {
            format!("{}.parquet", segment_id)
        } else {
            format!("{}/{}.parquet", hive_path, segment_id)
        };

        let config = SegmentConfig::new(&full_base_path, segment_id)
            .with_index_all(index_all)
            .with_columns_to_index(index_columns.to_vec())
            .with_partition_values(partition_values.clone())
            .with_column_devices(
                index_configs
                    .iter()
                    .filter_map(|(column, cfg)| {
                        cfg.device.as_ref().map(|d| (column.clone(), d.clone()))
                    })
                    .collect(),
            )
            .with_default_device(default_device)
            .with_parquet_path(parquet_path);

        let mut writer = HybridSegmentWriter::new(config).with_index_configs(index_configs.clone());
        writer.primary_key = primary_key.to_vec();
        writer.set_store(self.store.clone());

        writer
            .build_indexes(batch, 0)
            .context("failed to build staged indexes")?;
        writer.finish_indexing().await?;
        writer.upload_to_store().await?;

        let mut index_files = writer.to_manifest_entry().index_files;
        if !hive_path.is_empty() {
            for index_file in &mut index_files {
                index_file.file_path =
                    crate::core::segment::ensure_prefix(hive_path, &index_file.file_path);
            }
        }
        Ok(index_files)
    }

    /// Create the Iceberg metadata skeleton if this table has none yet, so the
    /// snapshot the caller is about to publish has somewhere to land.
    async fn ensure_table_metadata_async(
        &self,
        new_manifest: &crate::core::manifest::Manifest,
    ) -> Result<()> {
        if self.catalog_state.catalog.is_some() {
            // A catalog owns the metadata location; `publish_snapshot_async`
            // resolves it and reports a genuinely missing table.
            return Ok(());
        }
        let fallback_schema = self.arrow_schema();
        let uri_clone = self.uri.clone();
        let manifest_clone = new_manifest.clone();
        let raw_fv = self.get_format_version().max(new_manifest.format_version);
        let eff_fv = if raw_fv == 0 { 2 } else { raw_fv };

        crate::core::metadata::TableMetadata::commit_update_atomic(
            self.store.as_ref(),
            new_manifest.version.max(1) as i32,
            move |existing| {
                if let Some(meta) = existing {
                    return Ok(crate::core::metadata::MetadataUpdateOutcome::AlreadyCurrent(meta));
                }
                let schema = manifest_clone.schemas.last().cloned().unwrap_or_else(|| {
                    crate::core::manifest::Schema::from_arrow(&fallback_schema, 0)
                });
                let mut metadata = crate::core::metadata::TableMetadata::new(
                    eff_fv,
                    uuid::Uuid::new_v4().to_string(),
                    uri_clone.clone(),
                    schema,
                    manifest_clone.partition_spec.clone(),
                    manifest_clone
                        .sort_orders
                        .first()
                        .cloned()
                        .unwrap_or_default(),
                );
                metadata.current_schema_id = manifest_clone.current_schema_id;
                metadata.default_spec_id = manifest_clone.default_spec_id;
                metadata.default_sort_order_id = manifest_clone.default_sort_order_id;
                metadata.last_column_id = manifest_clone.last_column_id;
                Ok(crate::core::metadata::MetadataUpdateOutcome::Updated(
                    metadata,
                ))
            },
        )
        .await?;
        Ok(())
    }
}
