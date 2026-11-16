// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::index::memory::InMemoryVectorIndex;
use crate::core::query::QueryConfig;
use crate::SegmentConfig;
/// Table state management: indexing state, catalog identity, and configuration accessors.
///
/// Contains:
/// - `TableIndexState` / `TableCatalogState` structs
/// - `ColumnIndexConfig` / `LabelPattern` types
/// - State accessor/mutator methods on `Table`
use anyhow::{Context, Result};
use arrow::datatypes::SchemaRef;
use object_store::ObjectStore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::runtime::Runtime;

use super::Table;

// ---------------------------------------------------------------------------
// Public type definitions (re-exported from mod.rs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ColumnIndexConfig {
    pub device: Option<String>,
    pub tokenizer: Option<String>,
    pub enabled: bool,
    pub algorithms: Vec<crate::core::manifest::IndexAlgorithm>,
}

/// Strategy for labeling unnamed or numerically named columns during first write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LabelPattern {
    /// Excel style: A, B, C... AA, AB... (Default for premium UX)
    #[default]
    ExcelAlpha,
    /// Polars style: column_1, column_2...
    Polars,
    /// Pandas style: 0, 1, 2...
    Pandas,
}

/// Indexing state and configuration for a Table
#[derive(Clone)]
pub(crate) struct TableIndexState {
    pub index_all: bool,
    pub index_columns: Arc<parking_lot::RwLock<Vec<String>>>,
    pub index_configs: Arc<parking_lot::RwLock<HashMap<String, ColumnIndexConfig>>>,
    pub default_device: Arc<parking_lot::RwLock<Option<String>>>,
    pub memory_index: Arc<parking_lot::RwLock<Option<InMemoryVectorIndex>>>,
    pub vector_routers:
        Arc<parking_lot::RwLock<HashMap<String, Arc<crate::core::index::TableVectorRouter>>>>,
    /// Table-level, cluster-keyed vector indexes, keyed by column.
    ///
    /// Populated lazily by `ensure_cluster_index`; cleared whenever a rebuild or
    /// a data commit invalidates the artifact.
    pub cluster_indexes:
        Arc<parking_lot::RwLock<HashMap<String, Arc<crate::core::index::ClusterIndex>>>>,
    /// Table-level centroids for the next per-segment rebuild, keyed by column.
    ///
    /// Set by `rebuild_indexes_with_global_centroids` and consumed by the
    /// backfill path so every segment in one pass assigns against the same
    /// centroid set. Empty for ordinary builds.
    pub global_centroids: Arc<parking_lot::RwLock<HashMap<String, Vec<Vec<f32>>>>>,
}

/// Catalog identity and synchronization state for a Table
#[derive(Clone)]
pub(crate) struct TableCatalogState {
    pub catalog: Option<Arc<dyn crate::core::catalog::Catalog>>,
    pub namespace: Option<String>,
    pub table_name: Option<String>,
}

// ---------------------------------------------------------------------------
// impl Table — state accessors / mutators
// ---------------------------------------------------------------------------

impl Table {
    /// Get the current SegmentConfig derived from the table's state.
    pub fn get_config(&self) -> SegmentConfig {
        SegmentConfig::new(&self.uri, "")
            .with_index_all(self.indexing.index_all)
            .with_columns_to_index(self.indexing.index_columns.read().clone())
    }

    pub fn set_index_all(&mut self, enabled: bool) {
        self.indexing.index_all = enabled;
    }

    pub fn get_index_all(&self) -> bool {
        self.indexing.index_all
    }

    pub fn set_autocommit(&self, enabled: bool) {
        // `Relaxed` is intentional: `autocommit` is a user-facing mode toggle,
        // not a synchronization primitive. It guards no other memory, and the
        // write path reads it once per operation, so plain atomicity is all
        // that is required — no acquire/release ordering.
        self.autocommit
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn get_autocommit(&self) -> bool {
        // See `set_autocommit`: `Relaxed` is sufficient — a best-effort mode
        // switch, not a happens-before edge.
        self.autocommit.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Check if the table currently has an active in-memory vector index.
    pub fn has_memory_index(&self) -> bool {
        let pc_guard = self.pending_commit.lock();
        if let Some(pc) = pc_guard.as_ref() {
            if pc.phase == super::CommitPhase::Staging && pc.frozen_memory_index.is_some() {
                return true;
            }
        }
        self.indexing.memory_index.read().is_some()
    }

    /// Get the number of rows currently in the write buffer (not yet flushed).
    pub fn write_buffer_row_count(&self) -> usize {
        let pc_guard = self.pending_commit.lock();
        let pending = self.pending_writes.read();
        let staging_rows: usize = pc_guard
            .as_ref()
            .filter(|pc| pc.phase == super::CommitPhase::Staging)
            .map(|pc| pc.batches.iter().map(|b| b.num_rows()).sum())
            .unwrap_or(0);
        let pending_rows: usize = pending.iter().map(|p| p.batch.num_rows()).sum();
        staging_rows + pending_rows
    }

    /// Get the list of currently indexed column names.
    pub fn get_index_columns(&self) -> Vec<String> {
        self.indexing.index_columns.read().clone()
    }

    pub fn set_default_device(&mut self, device: Option<String>) {
        let mut d = self.indexing.default_device.write();
        *d = device;
    }

    pub fn get_default_device(&self) -> Option<String> {
        self.indexing.default_device.read().clone()
    }

    pub fn remove_index_columns(&mut self, columns: Vec<String>) {
        let mut index_cols = self.indexing.index_columns.write();
        index_cols.retain(|c| !columns.contains(c));
    }

    pub fn remove_all_index_columns(&mut self) {
        let mut index_cols = self.indexing.index_columns.write();
        index_cols.clear();
        self.indexing.index_all = false;
    }

    pub fn object_store(&self) -> Arc<dyn ObjectStore> {
        self.store.clone()
    }

    pub fn table_uri(&self) -> &str {
        &self.uri
    }

    pub fn query_config(&self) -> &QueryConfig {
        &self.query_config
    }

    /// The Tokio runtime backing this table's synchronous APIs.
    ///
    /// `rt` is set by every public constructor; the `expect` below is the one
    /// documented structural invariant kept in NO_PANIC_POLICY.md. Returning
    /// `Option`/`Result` here would ripple through every synchronous caller to
    /// express "this cannot happen", so the invariant is asserted instead.
    #[allow(clippy::expect_used)]
    pub fn runtime(&self) -> Arc<Runtime> {
        self.rt
            .as_ref()
            .expect("Runtime not available on async Table")
            .clone()
    }

    /// Run a future to completion for a synchronous table API.
    ///
    /// Uses the table's runtime when present. Async-constructed tables
    /// ([`Table::new_async`]) have no runtime, so a temporary current-thread
    /// runtime is created. Callers must not already be inside a Tokio runtime —
    /// the synchronous merge/delete APIs are driven from `spawn_blocking`.
    pub(crate) fn block_on_io<F: std::future::Future>(&self, fut: F) -> Result<F::Output> {
        match &self.rt {
            Some(rt) => Ok(rt.block_on(fut)),
            None => {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .context("failed to build a temporary runtime for a synchronous table API")?;
                Ok(rt.block_on(fut))
            }
        }
    }

    pub fn arrow_schema(&self) -> SchemaRef {
        self.schema.read().clone()
    }

    /// A manifest manager bound to the version this handle reads.
    ///
    /// Every read path builds its manager here, so a pinned handle resolves
    /// the pinned version in the segment loads and index lookups as well as
    /// in [`Table::manifest`].
    pub(crate) fn manifest_manager(&self) -> crate::core::manifest::ManifestManager {
        crate::core::manifest::ManifestManager::new(self.store.clone(), "", &self.uri)
            .with_pinned_version(self.pinned_version())
    }

    pub async fn manifest(&self) -> Result<crate::core::manifest::Manifest> {
        // A pinned handle answers from the version it was pinned to, which is
        // what makes the index follow the data through time travel: the
        // artifact set lives in `table_indexes` and in each segment's
        // `index_files`, both carried by that version.
        let (manifest, _) = self.manifest_manager().load_latest().await?;
        Ok(manifest)
    }

    /// Pin this handle to a historical manifest version for reads.
    ///
    /// Every subsequent read resolves that version, including its index
    /// artifacts. Writes are refused while the handle is pinned. The version is
    /// verified to exist before the pin takes effect, so a typo fails here
    /// rather than at the first query.
    pub async fn at_version(&self, version: u64) -> Result<()> {
        let _ = self.manifest_at_version(version).await?;
        *self.pinned_version.write() = Some(version);
        Ok(())
    }

    /// The version this handle is pinned to, if any.
    pub fn pinned_version(&self) -> Option<u64> {
        *self.pinned_version.read()
    }

    /// Return the handle to the head.
    pub fn unpin(&self) {
        *self.pinned_version.write() = None;
    }

    /// Error unless the handle is unpinned, used by every write path.
    pub fn ensure_writable(&self) -> Result<()> {
        if let Some(version) = self.pinned_version() {
            anyhow::bail!(
                "table handle is pinned to version {} for reads; call unpin() before writing",
                version
            );
        }
        Ok(())
    }

    /// MVCC snapshot version: the latest committed manifest version.
    ///
    /// Every commit is a monotonically increasing `v{N}.json` written with
    /// `PutMode::Create`, so this value is a stable snapshot identifier. A
    /// reader can pin it and later read exactly that snapshot via
    /// [`Table::manifest_at_version`].
    pub async fn snapshot_version(&self) -> Result<u64> {
        let (_, version) = self.manifest_manager().load_latest_direct().await?;
        Ok(version)
    }

    /// Load a specific historical snapshot (MVCC read).
    ///
    /// Returns the manifest for `version`, or an error if that version does not
    /// exist. This is the read half of MVCC: a long-running reader can hold a
    /// version and observe a consistent view even while writers commit newer
    /// versions.
    pub async fn manifest_at_version(
        &self,
        version: u64,
    ) -> Result<crate::core::manifest::Manifest> {
        crate::core::manifest::ManifestManager::new(self.store.clone(), "", &self.uri)
            .load_version(version)
            .await
    }
}
