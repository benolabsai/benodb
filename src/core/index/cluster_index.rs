// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Table-level, cluster-keyed vector index artifact.
//!
//! The per-segment index ([`crate::core::index::hnsw_ivf::HnswIvfIndex`]) is
//! owned by one Parquet file: each segment trains its own k-means, so a cluster
//! id is segment-local and a query must probe every segment to cover its true
//! neighbours. This module builds the other shape: **one** Puffin artifact, not
//! owned by any data file, keyed by *global* cluster, whose graph nodes carry
//! `(segment_idx, row_id)`.
//!
//! The decoupling that makes this possible is already in the engine: a graph
//! node identifies a row by `(segment_id, row_id)`, and
//! [`crate::core::table::Table::fetch_results_by_id`] groups results by segment
//! and reads rows by local id. So a cluster's members can be contiguous in the
//! artifact while remaining scattered across the Parquet files, and the payload
//! fetch still resolves them. The Parquet files are never rewritten.
//!
//! ## Artifact layout
//!
//! One Puffin container holds:
//!
//! | Blob | Contents |
//! |---|---|
//! | `centroids` | global centroids (Parquet list column) + metric/quantizer metadata |
//! | `segments` | `segment_idx -> segment_id` table |
//! | `cluster_{c}.hnsw.graph` | Arrow IPC HNSW graph over cluster `c`'s codes |
//! | `cluster_{c}.node_meta` | `(segment_idx, row_id)` per graph node |
//!
//! `node_meta` is the only genuinely new structure; the graph and quantizer
//! payloads are the existing ones.
//!
//! ## Memory bound
//!
//! The build never holds the whole corpus. Assignment streams Parquet batches
//! into a fixed number of hash buckets on disk (`cluster_id % BUCKETS`), and
//! each bucket is then loaded, grouped by cluster, and written out one cluster
//! at a time. Working set is `corpus_bytes / BUCKETS` plus one cluster's graph.

use std::collections::HashMap;
use std::io::{Cursor, Write};
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{Array, ArrayRef, Float32Builder, ListBuilder, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use futures::StreamExt;
use object_store::ObjectStore;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

// `hnsw_ivf` defines its own L1/Hamming/Jaccard distance wrappers (the prelude
// exports same-named types from `hnsw_rs::dist`); the graph variants are typed
// against the `hnsw_ivf` ones, so import those explicitly to shadow the glob.
use crate::core::index::hnsw_ivf::{
    assign_to_centroids, DistHamming, DistJaccard, DistL1, HnswGraph,
};
use crate::core::index::hnsw_rs::prelude::*;
use crate::core::index::{global_centroids, Quantizer, QuantizerImpl, VectorMetric, VectorValue};
use crate::core::manifest::{IndexAlgorithm, ManifestEntry, TableIndexFile};
use crate::core::puffin::{
    PuffinReader, PuffinWriter, PUFFIN_BLOB_HNSW_IVF_CENTROIDS,
    PUFFIN_BLOB_VECTOR_CLUSTER_NODE_META, PUFFIN_BLOB_VECTOR_CLUSTER_SEGMENTS,
    PUFFIN_BLOB_VECTOR_CLUSTER_SHARDS, PUFFIN_BLOB_VECTOR_HNSW_GRAPH,
};
use crate::core::reader::stats;
use crate::core::search::ScoredResult;
use parquet::file::reader::FileReader;

/// Default number of global clusters when the caller does not choose one.
pub const DEFAULT_CLUSTER_COUNT: usize = 1024;

/// Default HNSW graph degree.
pub const DEFAULT_CLUSTER_M: usize = 16;

/// Default HNSW construction beam.
pub const DEFAULT_CLUSTER_EF_CONSTRUCTION: usize = 200;

/// Default target size for each sequential Puffin shard: 512 MiB (536,870,912 bytes).
///
/// Adheres to Apache Iceberg's standard default target file size (512 MiB)
/// and strictly respects cloud object storage single-PUT boundaries (e.g. AWS S3 5 GiB limit).
pub const DEFAULT_TARGET_SHARD_BYTES: usize = 536_870_912;

/// Get the target shard size in bytes from the environment (`BSDB_INDEX_TARGET_SHARD_BYTES`)
/// or fallback to [`DEFAULT_TARGET_SHARD_BYTES`].
pub fn target_shard_bytes() -> usize {
    std::env::var("BSDB_INDEX_TARGET_SHARD_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_TARGET_SHARD_BYTES)
}

/// Rows sampled for global k-means training.
///
/// 1–2 % of a 10M-row table. Training on a sample is what keeps the centroid
/// fit bounded; the full assignment pass still touches every row.
pub const DEFAULT_SAMPLE_BUDGET: usize = 200_000;

/// Number of on-disk hash buckets used to bound build memory.
#[cfg(test)]
mod bucket_count_tests {
    use super::*;

    #[test]
    fn bucket_count_bounds_each_spill_to_256_mb() {
        // 100M rows at 128 dimensions of f32 is about 51 GB of spilled records.
        let bytes = 100_000_000usize * 128 * 4;
        let buckets = build_bucket_count(bytes);
        assert!(buckets >= 64, "at least the floor");
        assert_eq!(buckets, bytes.div_ceil(256 * 1024 * 1024).clamp(64, 1024));
        assert!(buckets > 64, "100M vectors must exceed the fixed 64 of old");
        // Small tables keep the floor rather than spilling a single tiny file.
        assert_eq!(build_bucket_count(1_000_000), 64);
        // Very large tables are capped so the loop stays bounded.
        assert_eq!(build_bucket_count(1usize << 42), 1024);
    }

    #[test]
    fn bucket_parallelism_stays_within_bounds() {
        // Reading the environment in a test is racy, so this asserts only the
        // bounds the parser enforces, which is what protects the build from a
        // window that would hold the whole artifact in memory.
        let n = bucket_parallelism();
        assert!((1..=8).contains(&n), "window {n} outside 1..=8");
    }
}

/// Root directory for the bucket and shard spill of a table level build.
///
/// At 100M the builder passes about 90 GB through this directory, so the system
/// temp directory is a poor default on hosts where `/tmp` is a sized tmpfs.
/// `BSDB_CLUSTER_BUILD_TMPDIR` points it at the storage volume instead.
fn build_spill_root() -> std::path::PathBuf {
    std::env::var_os("BSDB_CLUSTER_BUILD_TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
}

/// Remove spill directories left behind by a build that was killed outright.
///
/// The index deletes its own temporary directory on drop, but an OOM kill or a
/// SIGKILL leaves it in place, and at 100M each one is tens of GB.
fn sweep_stale_spill_dirs(root: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    let now = std::time::SystemTime::now();
    for entry in entries.flatten() {
        if !entry
            .file_name()
            .to_string_lossy()
            .starts_with("bsdb-cluster-")
        {
            continue;
        }
        let age = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|m| now.duration_since(m).ok());
        if age.map(|a| a.as_secs() > 6 * 3600).unwrap_or(false) {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// Free bytes available to this user on the filesystem holding `path`.
#[cfg(unix)]
fn available_spill_bytes(path: &std::path::Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `c_path` is a valid NUL terminated path and `stat` is writable.
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    Some((stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64))
}

/// Free space is not read on non unix hosts; the caller then skips the check.
#[cfg(not(unix))]
fn available_spill_bytes(_path: &std::path::Path) -> Option<u64> {
    None
}

fn build_bucket_count(total_vector_bytes: usize) -> usize {
    if let Some(n) = std::env::var("BSDB_CLUSTER_BUILD_BUCKETS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return n;
    }
    // Pass 3 reads one bucket file whole and then groups it, which duplicates
    // its vectors on the heap. A fixed count of 64 put about 800 MB in each
    // bucket for 100M x 128 dimensions, and the grouped copy doubled that on top
    // of the graph builds. Sizing buckets to roughly 256 MB of spilled records
    // keeps that transient to a few hundred megabytes regardless of table size.
    let target = 256 * 1024 * 1024;
    total_vector_bytes.div_ceil(target).clamp(64, 1024)
}

/// Number of buckets read, grouped, and built together.
///
/// The graph build is the long pole, and a bucket holds only
/// `centroids / buckets` clusters, so one bucket at a time caps the phase at
/// that many threads. Three buckets in flight gives the cores work while
/// holding a few GB of grouped vectors, not the whole artifact.
fn bucket_parallelism() -> usize {
    std::env::var("BSDB_CLUSTER_BUILD_PARALLEL_BUCKETS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(3)
        .clamp(1, 8)
}

/// Per-node locator: which data file, and which row inside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeMeta {
    /// Index into [`ClusterIndex::segment_table`].
    pub segment_idx: u32,
    /// Row position within that segment's Parquet file.
    pub row_id: u32,
}

/// Byte slices of one cluster's graph and node metadata inside a Puffin container.
#[derive(Debug, Clone, Copy)]
pub struct ClusterSlices {
    pub graph: (usize, usize),
    pub node_meta: (usize, usize),
}

/// One shard's entry in the artifact's shard descriptor.
///
/// Shard 0 carries this list as a Puffin blob. A remote client reads it once and
/// learns which shard owns which cluster without a HEAD and two ranged reads per
/// shard, which at 100M is 78 shards and 234 requests before the first query.
/// `cluster_start` and `cluster_end` are inclusive and come from the builder,
/// which writes clusters to shards in ascending id order.
///
/// `bytes` is the shard's object size. It is recorded so a client can size a
/// range read or report artifact bytes without a HEAD per shard.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ShardDescriptor {
    path: String,
    cluster_start: usize,
    cluster_end: usize,
    bytes: u64,
}

/// The shard paths a descriptor blob names, in either of its two forms.
///
/// Returns `None` when the blob is neither form, so the caller can fall through
/// to treating the container as a single shard.
fn shard_paths_from_descriptor(bytes: &[u8]) -> Option<Vec<String>> {
    if let Ok(descriptors) = serde_json::from_slice::<Vec<ShardDescriptor>>(bytes) {
        return Some(descriptors.into_iter().map(|d| d.path).collect());
    }
    serde_json::from_slice::<Vec<String>>(bytes).ok()
}

/// Read the shard descriptor from shard 0's footer.
///
/// Returns `None` for an artifact written before the descriptor existed, in
/// which case the caller falls back to reading every shard footer. Returns an
/// error only when the blob is present and unparseable, because silently
/// treating a corrupt descriptor as absent would double the open cost without
/// saying so.
async fn read_shard_descriptors(
    store: &Arc<dyn object_store::ObjectStore>,
    path: &str,
    footer: &crate::core::puffin::PuffinFooter,
) -> Result<Option<Vec<ShardDescriptor>>> {
    let Some((offset, len)) = blob_range_in_footer(footer, PUFFIN_BLOB_VECTOR_CLUSTER_SHARDS)
    else {
        return Ok(None);
    };
    let bytes = store
        .get_range(
            &object_store::path::Path::from(path),
            offset..offset + len,
        )
        .await?;
    if let Ok(descriptors) = serde_json::from_slice::<Vec<ShardDescriptor>>(&bytes) {
        return Ok(Some(descriptors));
    }
    // A `Vec<String>` is the pre-descriptor form: paths only, no ranges, so the
    // caller has to read every footer. Recognizing it keeps older artifacts
    // loadable instead of failing their opens outright.
    if serde_json::from_slice::<Vec<String>>(&bytes).is_ok() {
        return Ok(None);
    }
    anyhow::bail!("shard descriptor in '{path}' is neither a range list nor a path list")
}

/// A sequential Puffin shard of the cluster index.
#[derive(Clone)]
pub struct ClusterShard {
    /// Zero-based sequential shard index.
    pub shard_idx: usize,
    /// Storage path if loaded from or written to an object store.
    pub file_path: Option<String>,
    /// Range of global cluster IDs contained in this shard: [start, end] inclusive.
    pub cluster_range: (usize, usize),
    /// Container bytes (can be memory-mapped or owned).
    pub container: Bytes,
    /// Byte offsets and lengths for each cluster's blobs inside this shard.
    pub cluster_slices: HashMap<usize, ClusterSlices>,
}

struct MmapOwner(Arc<memmap2::Mmap>);

/// Open one Puffin shard as a byte view the kernel can reclaim.
///
/// `load_sharded` used to read every shard into owned `Bytes`, so a 40 GB
/// table level index became 40 GB of anonymous memory before the first query.
/// Under any cache budget smaller than the artifact that is an OOM kill, which
/// is exactly what a 16 GiB scope did at 100M. Mapping the shard keeps the
/// bytes file backed and evictable, and the prefetch advice issues the
/// readahead that a graph traversal would otherwise turn into 4 KB faults.
async fn open_shard_container(
    store: Arc<dyn object_store::ObjectStore>,
    path: &str,
) -> anyhow::Result<Bytes> {
    use crate::core::cache::DiskCache;
    // No `WILLNEED` here. A 78 shard artifact is 43 GB, so prefetching every
    // shard on load reads the whole index before the first query and the memory
    // cap evicts it before the second. The probed clusters are advised instead,
    // in `prefetch_clusters`, once a query knows which ones it needs.
    match DiskCache::new(store.clone()).get_mmap(path).await {
        Ok(mmap) => Ok(Bytes::from_owner(MmapOwner(mmap))),
        Err(_) => Ok(store
            .get(&object_store::path::Path::from(path))
            .await?
            .bytes()
            .await?),
    }
}

/// Gap below which two byte ranges are merged into one read.
///
/// A cluster's graph and its node metadata are separate Puffin blobs, and
/// adjacent clusters sit next to each other inside a shard, so one query often
/// wants ranges a few KiB apart. Reading the gap is cheaper than asking for it
/// separately: locally the cost is the syscall and the queue slot, and on S3 it
/// is the per-request charge plus a round trip. 256 KiB of slack against a 2 MiB
/// cluster payload is at most 12% of extra bytes for one fewer request.
const PREFETCH_MERGE_GAP: usize = 256 * 1024;

/// Merge byte ranges into the fewest spans that cover them.
///
/// Ranges are sorted by offset and extended while the next starts within
/// `max_gap` of the current span's end; overlaps collapse to their union.
/// Zero-length ranges are dropped, since they cover nothing and would otherwise
/// produce a spurious span at offset 0.
fn coalesce_spans(mut spans: Vec<(usize, usize)>, max_gap: usize) -> Vec<(usize, usize)> {
    spans.retain(|(_, len)| *len > 0);
    spans.sort_unstable_by_key(|(offset, _)| *offset);
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    for (offset, len) in spans {
        match merged.last_mut() {
            Some((start, cur_len)) if offset <= *start + *cur_len + max_gap => {
                let end = (offset + len).max(*start + *cur_len);
                *cur_len = end - *start;
            }
            _ => merged.push((offset, len)),
        }
    }
    merged
}

/// True when `blob` lies entirely inside the span `[start, start + len)`.
fn span_covers(start: usize, len: usize, blob: (usize, usize)) -> bool {
    let (blob_off, blob_len) = blob;
    blob_off >= start && blob_off + blob_len <= start + len
}

/// Byte budget for the deserialized cluster cache, in KiB.
///
/// The cache held 64 entries regardless of the memory envelope. At 100M that is
/// 64 of 40,100 clusters, which cannot hold anything a later query wants, so
/// every probe re-deserialized its graph out of a shard mapping whose pages were
/// already being reclaimed: the 16 GiB sweep measured 250 MB/s of refaults while
/// the 8 GiB budget sat unused.
///
/// `BSDB_CLUSTER_CACHE_KIB` sets it directly. Otherwise a quarter of
/// `BSDB_CACHE_GB` when that is set, because the shard mappings share the same
/// cap and page cache is the cheaper thing to lose. 256 MiB otherwise.
fn cluster_cache_budget_kib() -> u64 {
    fn from_env(name: &str) -> Option<u64> {
        std::env::var(name).ok()?.trim().parse::<u64>().ok()
    }
    if let Some(kib) = from_env("BSDB_CLUSTER_CACHE_KIB") {
        return kib.max(1);
    }
    if let Some(gb) = from_env("BSDB_CACHE_GB") {
        return (gb.saturating_mul(1024 * 1024) / 4).max(1);
    }
    256 * 1024
}

/// One deserialized cluster, as the cache holds it.
///
/// `bytes` is the size of the two blobs the store returned for the cluster. It
/// is the weight the cache budgets for. Weighing the decoded graph instead
/// (`HnswGraph::footprint_bytes`, which reports Arrow's
/// `get_array_memory_size`) overstated a cluster by roughly 5x on the measured
/// arm: 299 resident clusters weighed 2 GiB while the whole process, Python and
/// numpy included, peaked at 529 MB. The graph is decoded with
/// `load_from_bytes_owned`, so it aliases the fetched buffer rather than copying
/// it — the fetched bytes *are* the resident cost, and they are also the cost the
/// object store bills for, so one number serves both purposes.
struct CachedCluster {
    graph: HnswGraph,
    node_meta: Vec<NodeMeta>,
    /// Bytes returned by the store for this cluster's graph and metadata blobs.
    bytes: usize,
}

/// Weighed size of one cached cluster, in KiB.
fn cluster_entry_kib(value: &Arc<CachedCluster>) -> u32 {
    ((value.bytes / 1024) as u32).max(1)
}

/// Read a shard's Puffin footer with two ranged reads.
///
/// The footer is self-locating: the trailing 12 bytes carry its length and flags,
/// and the payload sits exactly that far before them. This is what lets a remote
/// client learn a shard's cluster offsets without transferring the shard.
async fn read_remote_footer(
    store: &Arc<dyn object_store::ObjectStore>,
    path: &str,
) -> Result<(crate::core::puffin::PuffinFooter, u64)> {
    let object = object_store::path::Path::from(path);
    let size = store.head(&object).await?.size;
    let trailer_len = crate::core::puffin::PUFFIN_FOOTER_TRAILER_LEN as u64;
    if size < trailer_len {
        anyhow::bail!("'{path}' is {size} bytes, too small to be a Puffin container");
    }
    let tail = store.get_range(&object, size - trailer_len..size).await?;
    let (footer_size, flags) = crate::core::puffin::parse_footer_trailer(&tail)?;
    let footer_start = size
        .checked_sub(trailer_len + footer_size as u64)
        .ok_or_else(|| {
            anyhow::anyhow!("'{path}' declares a {footer_size} byte footer beyond its start")
        })?;
    let payload = store
        .get_range(&object, footer_start..size - trailer_len)
        .await?;
    let footer = crate::core::puffin::parse_footer_payload(&payload, flags)?;
    // Footer reads are cluster range reads: they are what lets a client learn a
    // cluster's offsets. Counting them here keeps the request total honest
    // whether the read happens at open or in the middle of a query.
    stats::CLUSTER_RANGE_GETS.fetch_add(2, AtomicOrdering::Relaxed);
    stats::CLUSTER_RANGE_BYTES
        .fetch_add((tail.len() + payload.len()) as u64, AtomicOrdering::Relaxed);
    Ok((footer, size))
}

impl AsRef<[u8]> for MmapOwner {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

/// A table-level, cluster-keyed vector index.
///
/// Holds the container shards and deserializes a cluster's graph on demand, so
/// opening the index costs one object-store read plus the centroid/segment
/// tables — not the whole graph. When the index exceeds the target shard size
/// (512 MiB default), it is sequentially partitioned into contiguous Puffin
/// shards.
pub struct ClusterIndex {
    /// Vector column this artifact indexes.
    pub column: String,
    /// Vector dimensionality.
    pub dim: usize,
    /// Distance metric.
    pub metric: VectorMetric,
    /// Global centroids, one per cluster.
    pub centroids: Vec<Vec<f32>>,
    /// Quantizer used for the graph codes, if any.
    pub quantizer: Option<QuantizerImpl>,
    /// `segment_idx -> segment_id`.
    pub segment_table: Vec<String>,
    /// Manifest version this artifact was built from.
    pub manifest_version: u64,
    /// Total indexed rows.
    pub record_count: i64,
    /// Sequential Puffin shards.
    pub shards: Vec<ClusterShard>,
    /// Mapping from global cluster_id to the containing shard index.
    pub cluster_to_shard: HashMap<usize, usize>,
    /// Local filesystem root of the store holding the shards, when it is local.
    ///
    /// The shards are mapped without prefetch, so a query advises the page cache
    /// for the byte ranges it is about to touch. That needs a file descriptor,
    /// which only a local store has; remote stores keep `None` and rely on the
    /// object store's own range reads.
    local_root: Option<std::path::PathBuf>,
    cache: moka::sync::Cache<usize, Arc<CachedCluster>>,
    /// Cluster slices resolved lazily for remote shards, keyed by shard index.
    ///
    /// A remote open reads shard 0 and the shard descriptor only, so shards 1..N
    /// have no slice map when the index is constructed. The first query to probe
    /// a shard reads that shard's footer (two ranged reads) and caches the map
    /// here; later queries reuse it. A `std::sync::Mutex` is enough because the
    /// critical sections are a lookup and an insert with no await between them.
    remote_slices: std::sync::Mutex<HashMap<usize, Arc<HashMap<usize, ClusterSlices>>>>,
    /// Optional temporary directory backing memory-mapped build shards.
    pub _temp_dir: Option<std::path::PathBuf>,
}

impl Drop for ClusterIndex {
    fn drop(&mut self) {
        if let Some(ref p) = self._temp_dir {
            let _ = std::fs::remove_dir_all(p);
        }
    }
}

impl std::fmt::Debug for ClusterIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusterIndex")
            .field("column", &self.column)
            .field("dim", &self.dim)
            .field("metric", &self.metric)
            .field("clusters", &self.centroids.len())
            .field("segments", &self.segment_table.len())
            .field("shards", &self.shards.len())
            .field("record_count", &self.record_count)
            .finish()
    }
}

/// Sample vectors across every data file in a table.
///
/// Used to train global centroids for the Phase-1 validation path
/// (`Table::rebuild_indexes_with_global_centroids`). A fixed stride over the
/// whole table keeps the sample bounded and spread across segments; a prefix
/// sample would be one segment.
pub async fn sample_table_vectors(
    store: Arc<dyn ObjectStore>,
    entries: &[ManifestEntry],
    column: &str,
    max_samples: usize,
) -> Result<Vec<Vec<f32>>> {
    let total_rows: usize = entries.iter().map(|e| e.record_count.max(0) as usize).sum();
    if total_rows == 0 || max_samples == 0 {
        return Ok(Vec::new());
    }
    let stride = (total_rows / max_samples).max(1);
    let mut sample = Vec::with_capacity(max_samples.min(total_rows));
    let mut global_row = 0usize;

    for entry in entries {
        let mut on_batch = |_offset: usize, vectors: &[Vec<f32>]| -> Result<()> {
            for v in vectors {
                if global_row.is_multiple_of(stride) {
                    sample.push(v.clone());
                }
                global_row += 1;
            }
            Ok(())
        };
        stream_vector_batches(&store, &entry.file_path, column, &mut on_batch).await?;
    }
    Ok(sample)
}

impl ClusterIndex {
    /// Number of global clusters.
    pub fn cluster_count(&self) -> usize {
        self.centroids.len()
    }

    /// Number of data files covered by the artifact.
    pub fn segment_count(&self) -> usize {
        self.segment_table.len()
    }

    /// Total bytes all container shards occupy.
    pub fn size_in_bytes(&self) -> usize {
        self.shards.iter().map(|s| s.container.len()).sum()
    }

    /// Number of sequential Puffin shards.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Read-only access to the sequential shards.
    pub fn shards(&self) -> &[ClusterShard] {
        &self.shards
    }

    /// Build the artifact from a table's data files.
    ///
    /// Two sequential passes over the vector column:
    ///
    /// 1. sample rows across every segment and train global centroids;
    /// 2. assign every row to its nearest centroid and hash it into an on-disk
    ///    bucket, then build one cluster at a time.
    ///
    /// When total serialized clusters exceed [`target_shard_bytes`], the artifact
    /// is sequentially partitioned into contiguous 512 MiB Puffin shards.
    /// The Parquet files are read, never written.
    pub async fn build_from_table(
        store: Arc<dyn ObjectStore>,
        entries: &[ManifestEntry],
        column: &str,
        algo: &IndexAlgorithm,
        k: usize,
        m: usize,
        ef_construction: usize,
    ) -> Result<Self> {
        Self::build_from_table_with_target(
            store,
            entries,
            column,
            algo,
            k,
            m,
            ef_construction,
            None,
        )
        .await
    }

    /// Build the artifact from a table's data files with an optional target shard size override.
    pub async fn build_from_table_with_target(
        store: Arc<dyn ObjectStore>,
        entries: &[ManifestEntry],
        column: &str,
        algo: &IndexAlgorithm,
        k: usize,
        m: usize,
        ef_construction: usize,
        target_shard_bytes_override: Option<usize>,
    ) -> Result<Self> {
        if entries.is_empty() {
            anyhow::bail!("build_from_table requires at least one data file");
        }

        let segment_table: Vec<String> = entries.iter().map(segment_id_of).collect();
        let total_rows: usize = entries.iter().map(|e| e.record_count.max(0) as usize).sum();
        if total_rows == 0 {
            anyhow::bail!("build_from_table: table has no rows");
        }

        // ---- Pass 1: sample across all segments, train global centroids ----
        let stride = (total_rows / DEFAULT_SAMPLE_BUDGET).max(1);
        let mut sample: Vec<Vec<f32>> = Vec::with_capacity(DEFAULT_SAMPLE_BUDGET.min(total_rows));
        let mut dim = 0usize;
        let mut global_row = 0usize;

        for entry in entries {
            let mut on_batch = |_offset: usize, vectors: &[Vec<f32>]| -> Result<()> {
                if dim == 0 {
                    if let Some(v) = vectors.first() {
                        dim = v.len();
                    }
                }
                for v in vectors {
                    if global_row.is_multiple_of(stride) {
                        sample.push(v.clone());
                    }
                    global_row += 1;
                }
                Ok(())
            };
            stream_vector_batches(&store, &entry.file_path, column, &mut on_batch).await?;
        }

        if sample.is_empty() || dim == 0 {
            anyhow::bail!(
                "build_from_table: column '{}' produced no vectors (is it a vector column?)",
                column
            );
        }

        let centroids = global_centroids::train_global(&sample, k, None)?;
        let quantizer = train_quantizer(algo, &sample, dim, metric_for_algo(algo))?;
        let metric = metric_for_algo(algo);

        // ---- Pass 2: assign every row, range-bucket to disk ----
        let buckets = build_bucket_count(total_rows.saturating_mul(dim).saturating_mul(4));
        let spill_root = build_spill_root();
        std::fs::create_dir_all(&spill_root)
            .with_context(|| format!("creating build spill root {}", spill_root.display()))?;
        sweep_stale_spill_dirs(&spill_root);

        // Buckets and the final shards both pass through the spill directory, so
        // the peak is roughly twice the raw vector bytes. Refuse before writing
        // anything: a 100M build that fills a sized tmpfs dies with EDQUOT hours
        // in, and the partial buckets are useless.
        let raw_vector_bytes = total_rows.saturating_mul(dim).saturating_mul(4);
        if let Some(free) = available_spill_bytes(&spill_root) {
            let needed = raw_vector_bytes.saturating_mul(22) / 10;
            if free < needed as u64 {
                anyhow::bail!(
                    "cluster build spills buckets and shards through {} and needs about {} \
                     bytes, but only {} is available. Set BSDB_CLUSTER_BUILD_TMPDIR to a \
                     directory on a filesystem with room, about 100 GB at 100M.",
                    spill_root.display(),
                    needed,
                    free
                );
            }
        }

        let tmp_dir = spill_root.join(format!("bsdb-cluster-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&tmp_dir)
            .with_context(|| format!("creating build temp dir {}", tmp_dir.display()))?;

        let mut bucket_files: Vec<std::fs::File> = Vec::with_capacity(buckets);
        for b in 0..buckets {
            let path = tmp_dir.join(format!("bucket_{b}.bin"));
            bucket_files.push(
                std::fs::File::create(&path)
                    .with_context(|| format!("creating bucket file {}", path.display()))?,
            );
        }

        let clusters_per_bucket = centroids.len().div_ceil(buckets);
        let mut assign_result: Result<()> = Ok(());
        for (seg_idx, entry) in entries.iter().enumerate() {
            let seg_idx = seg_idx as u32;
            let mut on_batch = |offset: usize, vectors: &[Vec<f32>]| -> Result<()> {
                let labels = assign_to_centroids(vectors, &centroids, metric)?;
                for (i, (label, v)) in labels.iter().zip(vectors.iter()).enumerate() {
                    let row_id = (offset + i) as u32;
                    let bucket = (*label / clusters_per_bucket).min(buckets - 1);
                    write_record(&mut bucket_files[bucket], *label as u32, seg_idx, row_id, v)?;
                }
                Ok(())
            };
            if let Err(e) =
                stream_vector_batches(&store, &entry.file_path, column, &mut on_batch).await
            {
                assign_result = Err(e);
                break;
            }
            // Bound dirty page cache. Assigning 100M vectors spills about 51 GB
            // through the bucket files, and without a sync the dirty pages are
            // charged to the cgroup faster than writeback retires them: at 100M
            // under a 16 GiB cap the kernel OOM killed the build while anonymous
            // memory was still under 1 GB. Syncing once per segment caps the
            // dirty window at one segment's spill.
            for file in bucket_files.iter() {
                let _ = file.sync_data();
            }
        }
        drop(bucket_files);

        if let Err(e) = assign_result {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return Err(e);
        }

        // ---- Pass 3: Build one cluster at a time, streaming into sequential 512 MiB shards ----
        let target_bytes = target_shard_bytes_override.unwrap_or_else(target_shard_bytes);
        let mut shards_out: Vec<ClusterShard> = Vec::new();

        struct ActiveShard {
            shard_idx: usize,
            cluster_start: usize,
            cluster_end: usize,
            path: std::path::PathBuf,
            writer: PuffinWriter<std::io::BufWriter<std::fs::File>>,
            cluster_count: usize,
        }

        let seal_shard = |shard: ActiveShard| -> Result<ClusterShard> {
            let mut buf_writer = shard.writer.finish()?;
            buf_writer.flush()?;
            let file = std::fs::File::open(&shard.path)
                .with_context(|| format!("opening sealed shard {}", shard.path.display()))?;
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
            let container = Bytes::from_owner(MmapOwner(Arc::new(mmap)));
            let slices = cluster_slices_from_container(&container)?;
            let c_start = if shard.cluster_start == usize::MAX {
                0
            } else {
                shard.cluster_start
            };
            Ok(ClusterShard {
                shard_idx: shard.shard_idx,
                file_path: None,
                cluster_range: (c_start, shard.cluster_end),
                container,
                cluster_slices: slices,
            })
        };

        let shard0_path = tmp_dir.join("shard_0.puffin");
        let shard0_file = std::fs::File::create(&shard0_path)
            .with_context(|| format!("creating shard file {}", shard0_path.display()))?;
        let mut current_shard = ActiveShard {
            shard_idx: 0,
            cluster_start: usize::MAX,
            cluster_end: 0,
            path: shard0_path,
            writer: PuffinWriter::new(std::io::BufWriter::new(shard0_file))?,
            cluster_count: 0,
        };
        write_header_blobs(
            &mut current_shard.writer,
            &centroids,
            metric,
            quantizer.as_ref(),
            &segment_table,
        )?;

        let mut build_result: Result<()> = Ok(());
        let parallel_buckets = bucket_parallelism();
        let mut next_bucket = 0usize;

        'windows: while next_bucket < buckets {
            let window_end = (next_bucket + parallel_buckets).min(buckets);
            let window: Vec<usize> = (next_bucket..window_end).collect();
            next_bucket = window_end;

            // Read and group a window of buckets at once. Grouping copies every
            // record's vector onto the heap, so the window, not the bucket size
            // alone, is what bounds this transient.
            let grouped: Vec<(usize, Result<GroupedBucket>)> = window
                .par_iter()
                .map(|&b| {
                    let path = tmp_dir.join(format!("bucket_{b}.bin"));
                    let res = std::fs::read(&path)
                        .map_err(|e| {
                            anyhow::Error::from(e)
                                .context(format!("reading bucket {}", path.display()))
                        })
                        .and_then(|bytes| group_bucket(&bytes, dim));
                    (b, res)
                })
                .collect();

            let mut grouped_ok: Vec<(usize, GroupedBucket)> = Vec::with_capacity(grouped.len());
            for (b, res) in grouped {
                match res {
                    Ok(g) => grouped_ok.push((b, g)),
                    Err(e) => {
                        build_result = Err(e);
                        break 'windows;
                    }
                }
            }

            // Build every cluster in the window in parallel. A 256 MB bucket
            // holds about six of 1024 clusters, so building one bucket at a time
            // capped this phase at six threads; a window restores enough work
            // for the cores without holding the whole artifact in memory.
            let mut jobs: Vec<(usize, usize, &ClusterMembers)> = Vec::new();
            for (b, g) in &grouped_ok {
                for (cluster_id, members) in g {
                    jobs.push((*b, *cluster_id, members));
                }
            }
            // Bucket order, then cluster order. A bucket owns a contiguous label
            // range, so this is globally ascending and the shard slices stay
            // ordered.
            jobs.sort_by_key(|(b, cluster_id, _)| (*b, *cluster_id));

            let built: Vec<Result<(usize, usize, Option<BuiltClusterBlobs>)>> = jobs
                .par_iter()
                .map(|(b, cluster_id, members)| {
                    build_cluster_blobs(
                        *cluster_id,
                        members,
                        quantizer.as_ref(),
                        metric,
                        m,
                        ef_construction,
                    )
                    .map(|opt| (*b, *cluster_id, opt))
                })
                .collect();

            let mut current_bucket: Option<usize> = None;
            for res in built {
                let (b, _cluster_id, blobs_opt) = match res {
                    Ok(v) => v,
                    Err(e) => {
                        build_result = Err(e);
                        break 'windows;
                    }
                };
                if current_bucket != Some(b) {
                    if let Some(prev) = current_bucket {
                        let _ = std::fs::remove_file(tmp_dir.join(format!("bucket_{prev}.bin")));
                    }
                    current_bucket = Some(b);
                }
                let Some(blobs) = blobs_opt else {
                    continue;
                };
                let cluster_id = blobs.cluster_id;

                // If current shard has reached target size, seal it and roll to the next sequential shard
                if current_shard.cluster_count > 0
                    && current_shard.writer.current_offset() as usize >= target_bytes
                {
                    let next_idx = current_shard.shard_idx + 1;
                    let next_path = tmp_dir.join(format!("shard_{next_idx}.puffin"));
                    let next_file = match std::fs::File::create(&next_path) {
                        Ok(f) => f,
                        Err(e) => {
                            build_result = Err(anyhow::Error::from(e)
                                .context(format!("creating shard file {}", next_path.display())));
                            break 'windows;
                        }
                    };
                    let new_writer = match PuffinWriter::new(std::io::BufWriter::new(next_file)) {
                        Ok(w) => w,
                        Err(e) => {
                            build_result = Err(e);
                            break 'windows;
                        }
                    };
                    let old_shard = std::mem::replace(
                        &mut current_shard,
                        ActiveShard {
                            shard_idx: next_idx,
                            cluster_start: usize::MAX,
                            cluster_end: 0,
                            path: next_path,
                            writer: new_writer,
                            cluster_count: 0,
                        },
                    );
                    let sealed = match seal_shard(old_shard) {
                        Ok(s) => s,
                        Err(e) => {
                            build_result = Err(e);
                            break 'windows;
                        }
                    };
                    shards_out.push(sealed);
                }

                if let Err(e) = write_cluster_blobs_to_puffin(&mut current_shard.writer, &blobs) {
                    build_result = Err(e);
                    break 'windows;
                }

                if current_shard.cluster_start == usize::MAX {
                    current_shard.cluster_start = cluster_id;
                }
                current_shard.cluster_end = cluster_id;
                current_shard.cluster_count += 1;
            }
            if let Some(last) = current_bucket {
                let _ = std::fs::remove_file(tmp_dir.join(format!("bucket_{last}.bin")));
            }
        }

        if let Err(e) = build_result {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return Err(e);
        }

        // Seal final shard
        let final_shard = seal_shard(current_shard)?;
        shards_out.push(final_shard);

        let mut idx = Self::from_shards(
            shards_out,
            column.to_string(),
            dim,
            metric,
            centroids,
            quantizer,
            segment_table,
            0,
            total_rows as i64,
        )?;
        idx._temp_dir = Some(tmp_dir);
        Ok(idx)
    }

    /// Build an artifact directly from `(segment_idx, row_id, vector)` records.
    ///
    /// Used by tests and small tables; the streaming path is [`Self::build_from_table`].
    pub fn build_from_records(
        records: Vec<(u32, u32, Vec<f32>)>,
        segment_table: Vec<String>,
        column: &str,
        algo: &IndexAlgorithm,
        k: usize,
        m: usize,
        ef_construction: usize,
    ) -> Result<Self> {
        Self::build_from_records_with_target(
            records,
            segment_table,
            column,
            algo,
            k,
            m,
            ef_construction,
            target_shard_bytes(),
        )
    }

    /// Build an artifact directly from records with an explicit target shard size.
    pub fn build_from_records_with_target(
        records: Vec<(u32, u32, Vec<f32>)>,
        segment_table: Vec<String>,
        column: &str,
        algo: &IndexAlgorithm,
        k: usize,
        m: usize,
        ef_construction: usize,
        target_bytes: usize,
    ) -> Result<Self> {
        if records.is_empty() {
            anyhow::bail!("build_from_records requires at least one record");
        }
        let record_count = records.len() as i64;
        let dim = records[0].2.len();
        let metric = metric_for_algo(algo);
        let vectors: Vec<Vec<f32>> = records.iter().map(|(_, _, v)| v.clone()).collect();
        let centroids = global_centroids::train_global(&vectors, k, None)?;
        let quantizer = train_quantizer(algo, &vectors, dim, metric)?;
        let labels = assign_to_centroids(&vectors, &centroids, metric)?;

        let mut grouped: HashMap<usize, Vec<(u32, u32, Vec<f32>)>> = HashMap::new();
        for ((seg, row, v), label) in records.into_iter().zip(labels) {
            grouped.entry(label).or_default().push((seg, row, v));
        }

        let mut sorted_cluster_ids: Vec<usize> = grouped.keys().copied().collect();
        sorted_cluster_ids.sort_unstable();

        let mut shards_out: Vec<ClusterShard> = Vec::new();
        struct ActiveShard {
            shard_idx: usize,
            cluster_start: usize,
            cluster_end: usize,
            writer: PuffinWriter<Cursor<Vec<u8>>>,
            cluster_count: usize,
        }

        let mut current_shard = ActiveShard {
            shard_idx: 0,
            cluster_start: usize::MAX,
            cluster_end: 0,
            writer: PuffinWriter::new(Cursor::new(Vec::new()))?,
            cluster_count: 0,
        };
        write_header_blobs(
            &mut current_shard.writer,
            &centroids,
            metric,
            quantizer.as_ref(),
            &segment_table,
        )?;

        for cluster_id in sorted_cluster_ids {
            let members = &grouped[&cluster_id];
            if members.is_empty() {
                continue;
            }

            if current_shard.cluster_count > 0
                && current_shard.writer.current_offset() as usize >= target_bytes
            {
                let next_idx = current_shard.shard_idx + 1;
                let new_writer = PuffinWriter::new(Cursor::new(Vec::new()))?;
                let old_shard = std::mem::replace(
                    &mut current_shard,
                    ActiveShard {
                        shard_idx: next_idx,
                        cluster_start: usize::MAX,
                        cluster_end: 0,
                        writer: new_writer,
                        cluster_count: 0,
                    },
                );
                let container = Bytes::from(old_shard.writer.finish()?.into_inner());
                let slices = cluster_slices_from_container(&container)?;
                shards_out.push(ClusterShard {
                    shard_idx: old_shard.shard_idx,
                    file_path: None,
                    cluster_range: (old_shard.cluster_start, old_shard.cluster_end),
                    container,
                    cluster_slices: slices,
                });
            }

            write_cluster_blobs(
                &mut current_shard.writer,
                cluster_id,
                members,
                quantizer.as_ref(),
                metric,
                m,
                ef_construction,
            )?;
            if current_shard.cluster_start == usize::MAX {
                current_shard.cluster_start = cluster_id;
            }
            current_shard.cluster_end = cluster_id;
            current_shard.cluster_count += 1;
        }

        let container = Bytes::from(current_shard.writer.finish()?.into_inner());
        let slices = cluster_slices_from_container(&container)?;
        let c_start = if current_shard.cluster_start == usize::MAX {
            0
        } else {
            current_shard.cluster_start
        };
        shards_out.push(ClusterShard {
            shard_idx: current_shard.shard_idx,
            file_path: None,
            cluster_range: (c_start, current_shard.cluster_end),
            container,
            cluster_slices: slices,
        });

        Self::from_shards(
            shards_out,
            column.to_string(),
            dim,
            metric,
            centroids,
            quantizer,
            segment_table,
            0,
            record_count,
        )
    }

    /// Assemble an index from sequential shards.
    pub fn from_shards(
        shards: Vec<ClusterShard>,
        column: String,
        dim: usize,
        metric: VectorMetric,
        centroids: Vec<Vec<f32>>,
        quantizer: Option<QuantizerImpl>,
        segment_table: Vec<String>,
        manifest_version: u64,
        record_count: i64,
    ) -> Result<Self> {
        let mut cluster_to_shard = HashMap::new();
        for (s_idx, shard) in shards.iter().enumerate() {
            if shard.cluster_slices.is_empty() {
                // A remote open leaves shards 1..N unresolved, so the shard's
                // declared cluster range is the only thing that can route a
                // probe to it. Expanding the range costs one map entry per
                // cluster — the same order as the resolved case — and an id that
                // turns out to have no blobs simply resolves to `None` later.
                let (start, end) = shard.cluster_range;
                for cid in start..=end {
                    cluster_to_shard.insert(cid, s_idx);
                }
            } else {
                for cid in shard.cluster_slices.keys() {
                    cluster_to_shard.insert(*cid, s_idx);
                }
            }
        }
        // Shards that arrived with their slices resolved (a local open, or shard 0
        // of a remote one) are seeded here so `shard_slices` has one code path.
        let resolved: HashMap<usize, Arc<HashMap<usize, ClusterSlices>>> = shards
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.cluster_slices.is_empty())
            .map(|(i, s)| (i, Arc::new(s.cluster_slices.clone())))
            .collect();
        Ok(Self {
            column,
            dim,
            metric,
            centroids,
            quantizer,
            segment_table,
            manifest_version,
            record_count,
            shards,
            cluster_to_shard,
            local_root: None,
            // Byte weighed, because an entry count says nothing about memory: one
            // 100M cluster is megabytes of graph and metadata, so `64` was either
            // 0.1% of the index or an OOM depending on the dataset.
            cache: moka::sync::Cache::builder()
                .weigher(|_key: &usize, value: &Arc<CachedCluster>| cluster_entry_kib(value))
                .max_capacity(cluster_cache_budget_kib())
                .build(),
            remote_slices: std::sync::Mutex::new(resolved),
            _temp_dir: None,
        })
    }

    /// Open a sharded artifact without materialising any shard body.
    ///
    /// The open cost is **one shard's footer**, not every shard's. Shard 0 carries
    /// the header blobs and a shard descriptor that names every shard with its
    /// cluster range and object size, so two ranged reads for its footer, two for
    /// the headers and one for the descriptor are enough to construct the index.
    /// Shards 1..N are resolved the first time a query probes one of their
    /// clusters, which reads that shard's footer and caches the slice map. Before
    /// the descriptor existed every shard cost a HEAD plus two ranged reads at
    /// open, which at 100M is 78 shards and 234 requests before the first query.
    ///
    /// An artifact written before the descriptor falls back to reading every
    /// footer, so it still opens; it just pays the old cost.
    async fn load_sharded_remote(
        store: Arc<dyn object_store::ObjectStore>,
        shard_paths: &[String],
        column: &str,
        manifest_version: u64,
    ) -> Result<Self> {
        let first = shard_paths
            .first()
            .ok_or_else(|| anyhow::anyhow!("cluster index has no shard files"))?;

        let (footer0, size0) = read_remote_footer(&store, first).await?;
        let slices0 = cluster_slices_from_footer(&footer0, size0 as usize)?;

        let object = object_store::path::Path::from(first.as_str());
        let (off, len) = blob_range_in_footer(&footer0, PUFFIN_BLOB_HNSW_IVF_CENTROIDS)
            .ok_or_else(|| anyhow::anyhow!("shard 0 '{first}' has no centroids blob"))?;
        let bytes = store.get_range(&object, off..off + len).await?;
        let (centroids, metric, quantizer) = read_centroids_from_bytes(bytes)?;

        let (soff, slen) = blob_range_in_footer(&footer0, PUFFIN_BLOB_VECTOR_CLUSTER_SEGMENTS)
            .ok_or_else(|| anyhow::anyhow!("shard 0 '{first}' has no segments blob"))?;
        let sbytes = store.get_range(&object, soff..soff + slen).await?;
        let segment_table = read_segments_from_bytes(sbytes)?;

        let descriptors = read_shard_descriptors(&store, first, &footer0).await?;

        let mut shards: Vec<ClusterShard> = Vec::with_capacity(shard_paths.len());
        // Shard 0 is already read for its headers, so its slices cost nothing.
        let s0_min = slices0.keys().copied().min().unwrap_or(0);
        let s0_max = slices0.keys().copied().max().unwrap_or(0);
        shards.push(ClusterShard {
            shard_idx: 0,
            file_path: Some(first.clone()),
            cluster_range: (s0_min, s0_max),
            container: Bytes::new(),
            cluster_slices: slices0,
        });

        for (i, path) in shard_paths.iter().enumerate().skip(1) {
            // The descriptor entry is only trusted when it names the same object
            // the manifest lists, so a stale descriptor degrades to a footer read
            // instead of routing a probe to the wrong shard.
            let described = descriptors
                .as_ref()
                .and_then(|list| list.get(i))
                .filter(|d| d.path == *path);
            if let Some(d) = described {
                shards.push(ClusterShard {
                    shard_idx: i,
                    file_path: Some(path.clone()),
                    cluster_range: (d.cluster_start, d.cluster_end),
                    container: Bytes::new(),
                    cluster_slices: HashMap::new(),
                });
                continue;
            }

            let (footer, size) = read_remote_footer(&store, path).await?;
            let slices = cluster_slices_from_footer(&footer, size as usize)?;
            let min = slices.keys().copied().min().unwrap_or(0);
            let max = slices.keys().copied().max().unwrap_or(0);
            shards.push(ClusterShard {
                shard_idx: i,
                file_path: Some(path.clone()),
                cluster_range: (min, max),
                container: Bytes::new(),
                cluster_slices: slices,
            });
        }

        let dim = centroids.first().map(|c| c.len()).unwrap_or(0);
        Self::from_shards(
            shards,
            column.to_string(),
            dim,
            metric,
            centroids,
            quantizer,
            segment_table,
            manifest_version,
            0,
        )
    }

    /// Assemble an index from an already-serialized container (single shard).
    pub fn from_container(
        container: Bytes,
        column: String,
        dim: usize,
        metric: VectorMetric,
        centroids: Vec<Vec<f32>>,
        quantizer: Option<QuantizerImpl>,
        segment_table: Vec<String>,
        manifest_version: u64,
        record_count: i64,
    ) -> Result<Self> {
        let cluster_slices = cluster_slices_from_container(&container)?;
        let min_c = cluster_slices.keys().copied().min().unwrap_or(0);
        let max_c = cluster_slices.keys().copied().max().unwrap_or(0);
        let shard = ClusterShard {
            shard_idx: 0,
            file_path: None,
            cluster_range: (min_c, max_c),
            container,
            cluster_slices,
        };
        Self::from_shards(
            vec![shard],
            column,
            dim,
            metric,
            centroids,
            quantizer,
            segment_table,
            manifest_version,
            record_count,
        )
    }

    /// Load an artifact (or sequential shard set) from the object store.
    pub async fn load(
        store: Arc<dyn ObjectStore>,
        path: &str,
        column: &str,
        manifest_version: u64,
    ) -> Result<Self> {
        let object_path = object_store::path::Path::from(path);
        let container = store
            .get(&object_path)
            .await
            .with_context(|| format!("opening cluster index artifact '{path}'"))?
            .bytes()
            .await
            .with_context(|| format!("reading cluster index artifact '{path}'"))?;

        // If this container is Shard 0 with a multi-shard descriptor blob, load all shards
        if let Ok(Some((offset, len))) = find_blob(&container, PUFFIN_BLOB_VECTOR_CLUSTER_SHARDS) {
            let shard_meta_bytes = container.slice(offset..offset + len);
            let shard_paths = shard_paths_from_descriptor(&shard_meta_bytes);
            if let Some(shard_paths) = shard_paths.filter(|paths| paths.len() > 1) {
                return Self::load_sharded(store, &shard_paths, column, manifest_version).await;
            }
        }

        let (centroids, metric, quantizer) = read_centroids_blob(&container)?;
        let segment_table = read_segments_blob(&container)?;
        let dim = centroids.first().map(|c| c.len()).unwrap_or(0);
        let record_count = 0;

        Self::from_container(
            container,
            column.to_string(),
            dim,
            metric,
            centroids,
            quantizer,
            segment_table,
            manifest_version,
            record_count,
        )
    }

    /// Load a sequential multi-shard index from the object store.
    pub async fn load_sharded(
        store: Arc<dyn ObjectStore>,
        shard_paths: &[String],
        column: &str,
        manifest_version: u64,
    ) -> Result<Self> {
        if shard_paths.is_empty() {
            anyhow::bail!("load_sharded requires at least one shard path");
        }

        // A remote store cannot be mapped, and materialising every shard just to
        // read its footer is the difference between 43 GB of egress and a few MB:
        // at 100M one shard is ~600 MB and a query needs ~2 MB of it. Open those
        // from their footers instead and let `hydrate_for_query` range-read the
        // clusters a query actually probes.
        if crate::core::cache::local_store_root(store.as_ref()).is_none() {
            return Self::load_sharded_remote(store, shard_paths, column, manifest_version).await;
        }

        // Shard 0 always contains the global centroids and segment table.
        let shard0_path = &shard_paths[0];
        let shard0_container = open_shard_container(store.clone(), shard0_path)
            .await
            .with_context(|| format!("opening shard 0 '{shard0_path}'"))?;

        let (centroids, metric, quantizer) = read_centroids_blob(&shard0_container)?;
        let segment_table = read_segments_blob(&shard0_container)?;
        let dim = centroids.first().map(|c| c.len()).unwrap_or(0);

        let mut shards: Vec<ClusterShard> = Vec::with_capacity(shard_paths.len());
        let s0_slices = cluster_slices_from_container(&shard0_container)?;
        let s0_min = s0_slices.keys().copied().min().unwrap_or(0);
        let s0_max = s0_slices.keys().copied().max().unwrap_or(0);
        shards.push(ClusterShard {
            shard_idx: 0,
            file_path: Some(shard0_path.clone()),
            cluster_range: (s0_min, s0_max),
            container: shard0_container,
            cluster_slices: s0_slices,
        });

        for (i, path) in shard_paths.iter().enumerate().skip(1) {
            let container = open_shard_container(store.clone(), path)
                .await
                .with_context(|| format!("opening shard {i} '{path}'"))?;
            let slices = cluster_slices_from_container(&container)?;
            let s_min = slices.keys().copied().min().unwrap_or(0);
            let s_max = slices.keys().copied().max().unwrap_or(0);
            shards.push(ClusterShard {
                shard_idx: i,
                file_path: Some(path.clone()),
                cluster_range: (s_min, s_max),
                container,
                cluster_slices: slices,
            });
        }

        let local_root = crate::core::cache::local_store_root(store.as_ref());
        let mut index = Self::from_shards(
            shards,
            column.to_string(),
            dim,
            metric,
            centroids,
            quantizer,
            segment_table,
            manifest_version,
            0,
        )?;
        index.local_root = local_root;
        Ok(index)
    }

    /// Persist the container shards to the object store.
    ///
    /// Single-shard indexes are written directly to `path`.
    /// Multi-shard indexes are partitioned sequentially into 512 MiB shards:
    /// `..._shard_0000.puffin`, `..._shard_0001.puffin`, etc.
    /// Returns the list of persisted shard paths in sequential order.
    pub async fn save(&self, store: Arc<dyn ObjectStore>, path: &str) -> Result<Vec<String>> {
        let mut saved_paths = Vec::new();
        if self.shards.len() <= 1 {
            let object_path = object_store::path::Path::from(path);
            let bytes = if let Some(shard) = self.shards.first() {
                shard.container.clone()
            } else {
                Bytes::new()
            };
            store
                .put(&object_path, bytes.into())
                .await
                .with_context(|| format!("writing cluster index artifact '{path}'"))?;
            saved_paths.push(path.to_string());
        } else {
            let stem = path.strip_suffix(".puffin").unwrap_or(path);
            let shard_names: Vec<String> = (0..self.shards.len())
                .map(|i| format!("{stem}_shard_{:04}.puffin", i))
                .collect();
            // The descriptor is what lets a remote client route a probe to a shard
            // without reading every footer. It carries the cluster range the
            // builder already assigned, which is the assignment itself.
            let descriptors: Vec<ShardDescriptor> = self
                .shards
                .iter()
                .enumerate()
                .map(|(i, shard)| ShardDescriptor {
                    path: shard_names[i].clone(),
                    cluster_start: shard.cluster_range.0,
                    cluster_end: shard.cluster_range.1,
                    bytes: shard.container.len() as u64,
                })
                .collect();

            for (i, shard) in self.shards.iter().enumerate() {
                let shard_path = &shard_names[i];
                let object_path = object_store::path::Path::from(shard_path.as_str());

                let bytes_to_write = if i == 0 {
                    let mut writer = PuffinWriter::new(Cursor::new(Vec::new()))?;
                    let reader = PuffinReader::new(Cursor::new(shard.container.as_ref()))?;
                    for b in reader.footer().blobs.iter() {
                        let offset = b.offset as usize;
                        let length = b.length as usize;
                        let blob_data = &shard.container[offset..offset + length];
                        writer.add_blob(
                            b.r#type.clone(),
                            b.fields.clone(),
                            b.snapshot_id,
                            b.sequence_number,
                            blob_data,
                            b.properties.clone(),
                        )?;
                    }
                    let shard_json = serde_json::to_vec(&descriptors)?;
                    writer.add_blob(
                        PUFFIN_BLOB_VECTOR_CLUSTER_SHARDS.to_string(),
                        Vec::new(),
                        0,
                        1,
                        &shard_json,
                        HashMap::from([(
                            "filename".to_string(),
                            "cluster_shards.json".to_string(),
                        )]),
                    )?;
                    Bytes::from(writer.finish()?.into_inner())
                } else {
                    shard.container.clone()
                };

                store
                    .put(&object_path, bytes_to_write.into())
                    .await
                    .with_context(|| format!("writing cluster index shard '{shard_path}'"))?;
                saved_paths.push(shard_path.clone());
            }
        }
        Ok(saved_paths)
    }

    /// Describe this artifact as a manifest [`TableIndexFile`].
    pub fn to_table_index_file(
        &self,
        path: &str,
        manifest_version: u64,
        algorithm: &str,
    ) -> TableIndexFile {
        let shard_files = if self.shards.len() <= 1 {
            vec![path.to_string()]
        } else {
            let stem = path.strip_suffix(".puffin").unwrap_or(path);
            (0..self.shards.len())
                .map(|i| format!("{stem}_shard_{:04}.puffin", i))
                .collect()
        };
        self.to_table_index_file_with_shards(path, shard_files, manifest_version, algorithm)
    }

    /// Describe this artifact as a manifest [`TableIndexFile`] with explicit shard files.
    pub fn to_table_index_file_with_shards(
        &self,
        path: &str,
        shard_files: Vec<String>,
        manifest_version: u64,
        algorithm: &str,
    ) -> TableIndexFile {
        TableIndexFile {
            format_version: 2,
            index_id: uuid::Uuid::new_v4().to_string(),
            index_category: "vector".to_string(),
            algorithm: algorithm.to_string(),
            column_name: self.column.clone(),
            source_manifest_version: manifest_version,
            source_segments: self.segment_table.clone(),
            source_record_count: self.record_count,
            index_checksum: None,
            file_path: path.to_string(),
            shard_files,
            blob_type: Some(blob_type_for(self.quantizer.as_ref()).to_string()),
            cluster_count: self.centroids.len(),
            dim: self.dim,
            metric: self.metric.as_str().to_string(),
            compression: "none".to_string(),
        }
    }

    /// Search the table-level index.
    ///
    /// Coarse-scans the global centroids, takes the top `n_probe`, traverses
    /// those cluster graphs across their respective sequential shards, and maps
    /// each node back to `(segment_id, row_id)`. The returned [`ScoredResult`]s
    /// feed the existing hydration path unchanged.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        n_probe: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>> {
        if k == 0 || self.centroids.is_empty() || query.len() != self.dim {
            return Ok(Vec::new());
        }

        // Coarse: rank global centroids, then warm exactly the set that will
        // answer the query.
        let probe_ids = self.rank_probes(query, n_probe);
        let prepared_query = self
            .quantizer
            .as_ref()
            .map(|q| q.prepare_query(query, self.metric));
        let fallback_query = VectorValue::Float32(query.to_vec());
        let ef = ef_search.unwrap_or(40);
        let cluster_k = (k * 4).max(ef.min(k * 8));

        self.prefetch_clusters(&probe_ids);

        let mut candidates: Vec<ScoredResult> = Vec::new();
        for cluster_id in probe_ids {
            let Some(cluster) = self.get_cluster(cluster_id) else {
                continue;
            };
            let graph = &cluster.graph;
            let node_meta = &cluster.node_meta;
            // The vectors this query could return from this cluster. Summed over
            // the probe set it is the coverage denominator, which is what makes
            // two clusterings comparable at equal work.
            stats::CLUSTER_CORPUS_ROWS.fetch_add(node_meta.len() as u64, AtomicOrdering::Relaxed);
            let cluster_ef = ef.min((node_meta.len() * 3) / 4).max(cluster_k);
            for neighbour in graph.search_with_quantizer(
                &fallback_query,
                prepared_query.as_ref(),
                cluster_k,
                cluster_ef,
                None,
            ) {
                let local_id = neighbour.d_id;
                let Some(meta) = node_meta.get(local_id) else {
                    continue;
                };
                let Some(segment_id) = self.segment_table.get(meta.segment_idx as usize) else {
                    continue;
                };
                candidates.push(ScoredResult {
                    segment_id: segment_id.clone(),
                    row_id: meta.row_id,
                    score: neighbour.distance,
                });
            }
        }

        candidates.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.dedup_by(|a, b| a.segment_id == b.segment_id && a.row_id == b.row_id);
        candidates.truncate(k);

        // The table level tier builds no `RowSelection`: it resolves candidates
        // from the shard graphs and the caller materialises exactly those rows.
        // Without this the counters reported zero for the tier that serves
        // 100M, so the index I/O claim could not be stated on it at all.
        crate::core::reader::stats::ROWS_SELECTED.fetch_add(
            candidates.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        crate::core::reader::stats::ROWS_DECODED.fetch_add(
            candidates.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(candidates)
    }

    /// Global cluster ids this query would probe, closest centroid first.
    ///
    /// Shared by `search` and by the remote hydration, so the clusters fetched
    /// are exactly the clusters read.
    fn rank_probes(&self, query: &[f32], n_probe: usize) -> Vec<usize> {
        let mut cluster_distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, metric_distance(self.metric, query, c)))
            .collect();
        cluster_distances
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let probes = n_probe.clamp(1, cluster_distances.len());
        cluster_distances
            .into_iter()
            .take(probes)
            .map(|(cluster_id, _)| cluster_id)
            .collect()
    }

    /// Fetch this query's probe clusters from a remote store into the cache.
    ///
    /// The table tier's `search` is synchronous, so a remote read cannot happen
    /// inside it. This runs first, on the async side: it coalesces the probe's
    /// byte ranges per shard, issues one ranged GET per merged span, slices each
    /// cluster's graph and node metadata out of the returned bytes, deserializes
    /// them and inserts them into the same cache a local probe populates. `search`
    /// then hits the cache on the first line of `get_cluster` and never learns
    /// where the bytes came from.
    ///
    /// Local stores are left alone: their shards are mapped, and
    /// `prefetch_clusters` advisories are cheaper than copying bytes into the
    /// cache.
    ///
    /// A failed range is logged and skipped rather than failing the query: the
    /// affected cluster is simply not probed, which costs recall on that query
    /// instead of returning an error.
    pub async fn hydrate_for_query(
        &self,
        store: &std::sync::Arc<dyn object_store::ObjectStore>,
        query: &[f32],
        n_probe: usize,
    ) {
        if self.local_root.is_some() || query.len() != self.dim {
            return;
        }
        let probe_ids = self.rank_probes(query, n_probe);
        self.hydrate_clusters(store, &probe_ids).await;
    }

    /// Hydrate specific clusters from a remote store into the cache.
    ///
    /// This is the only function that issues cluster range reads, so it is where
    /// [`stats::CLUSTER_RANGE_GETS`], [`stats::CLUSTER_RANGE_BYTES`] and
    /// [`stats::CLUSTER_CACHE_HITS`] are recorded. The three counts answer the
    /// question an egress figure cannot: how many requests a query actually
    /// costs, and whether the cache returned anything at all.
    pub async fn hydrate_clusters(
        &self,
        store: &std::sync::Arc<dyn object_store::ObjectStore>,
        cluster_ids: &[usize],
    ) {
        // Group by shard so one object is addressed once per merged span instead
        // of once per blob.
        let mut per_shard: HashMap<usize, Vec<usize>> = HashMap::new();
        for cluster_id in cluster_ids {
            if self.cache.get(cluster_id).is_some() {
                // A hit is the only thing that can make the request count fall
                // below one request per probe, so it is counted separately from
                // the misses that follow.
                stats::CLUSTER_CACHE_HITS.fetch_add(1, AtomicOrdering::Relaxed);
                continue;
            }
            if let Some(shard_idx) = self.cluster_to_shard.get(cluster_id) {
                per_shard.entry(*shard_idx).or_default().push(*cluster_id);
            }
        }

        for (shard_idx, ids) in per_shard {
            let Some(shard) = self.shards.get(shard_idx) else {
                continue;
            };
            let Some(path) = shard.file_path.as_deref().map(str::to_string) else {
                continue;
            };
            // A remote open resolves only shard 0, so this is where shards 1..N
            // learn their offsets: two ranged reads for the footer, cached for
            // every later query.
            let Some(slices) = self.shard_slices(store, shard_idx).await else {
                continue;
            };
            let mut spans = Vec::with_capacity(ids.len() * 2);
            for cluster_id in &ids {
                if let Some(slices) = slices.get(cluster_id) {
                    spans.push(slices.graph);
                    spans.push(slices.node_meta);
                }
            }
            for (offset, len) in coalesce_spans(spans, PREFETCH_MERGE_GAP) {
                let range = offset as u64..(offset + len) as u64;
                let bytes = match store
                    .get_range(&object_store::path::Path::from(path.as_str()), range)
                    .await
                {
                    Ok(bytes) => {
                        stats::CLUSTER_RANGE_GETS.fetch_add(1, AtomicOrdering::Relaxed);
                        stats::CLUSTER_RANGE_BYTES
                            .fetch_add(bytes.len() as u64, AtomicOrdering::Relaxed);
                        bytes
                    }
                    Err(e) => {
                        tracing::warn!(shard_idx, offset, len, error = %e, "cluster_index: ranged read failed");
                        continue;
                    }
                };
                for cluster_id in &ids {
                    if self.cache.get(cluster_id).is_some() {
                        continue;
                    }
                    let Some(slices) = slices.get(cluster_id) else {
                        continue;
                    };
                    // Both blobs must sit inside this merged span; a cluster whose
                    // graph and metadata land in different spans is picked up by
                    // the per-cluster fallback below.
                    if !span_covers(offset, len, slices.graph)
                        || !span_covers(offset, len, slices.node_meta)
                    {
                        continue;
                    }
                    let graph_bytes = bytes
                        .slice(slices.graph.0 - offset..slices.graph.0 - offset + slices.graph.1);
                    let meta_bytes = bytes.slice(
                        slices.node_meta.0 - offset
                            ..slices.node_meta.0 - offset + slices.node_meta.1,
                    );
                    self.cache_deserialized(*cluster_id, graph_bytes, meta_bytes);
                }
            }

            // A cluster whose graph and metadata fall into different merged spans
            // is not covered by any single one, so fetch those two blobs directly
            // rather than dropping the cluster from the probe.
            for cluster_id in &ids {
                if self.cache.get(cluster_id).is_some() {
                    continue;
                }
                let Some(slices) = slices.get(cluster_id) else {
                    continue;
                };
                // One `Path` for both reads: a future borrowing a temporary would
                // outlive it.
                let object = object_store::path::Path::from(path.as_str());
                let fetch = |range: (usize, usize)| {
                    store.get_range(&object, range.0 as u64..(range.0 + range.1) as u64)
                };
                let graph_bytes = match fetch(slices.graph).await {
                    Ok(bytes) => {
                        stats::CLUSTER_RANGE_GETS.fetch_add(1, AtomicOrdering::Relaxed);
                        stats::CLUSTER_RANGE_BYTES
                            .fetch_add(bytes.len() as u64, AtomicOrdering::Relaxed);
                        bytes
                    }
                    Err(e) => {
                        tracing::warn!(cluster_id, error = %e, "cluster_index: graph range read failed");
                        continue;
                    }
                };
                let meta_bytes = match fetch(slices.node_meta).await {
                    Ok(bytes) => {
                        stats::CLUSTER_RANGE_GETS.fetch_add(1, AtomicOrdering::Relaxed);
                        stats::CLUSTER_RANGE_BYTES
                            .fetch_add(bytes.len() as u64, AtomicOrdering::Relaxed);
                        bytes
                    }
                    Err(e) => {
                        tracing::warn!(cluster_id, error = %e, "cluster_index: metadata range read failed");
                        continue;
                    }
                };
                self.cache_deserialized(*cluster_id, graph_bytes, meta_bytes);
            }
        }

        // Occupancy, for the case the hit counter reports zero: it separates "the
        // budget is too small to retain these clusters" from "nothing was ever
        // inserted", and `weighted_size / entry_count` exposes what the weigher
        // thinks an entry costs.
        stats::CLUSTER_CACHE_ENTRIES.store(self.cache.entry_count(), AtomicOrdering::Relaxed);
        stats::CLUSTER_CACHE_WEIGHTED_KIB
            .store(self.cache.weighted_size(), AtomicOrdering::Relaxed);
    }

    /// Cluster slices for `shard_idx`, reading that shard's footer on first use.
    ///
    /// A remote open resolves shard 0 only, so shards 1..N arrive here. The map is
    /// cached per process, which is what keeps the footer cost to two ranged reads
    /// per shard rather than two per query.
    async fn shard_slices(
        &self,
        store: &std::sync::Arc<dyn object_store::ObjectStore>,
        shard_idx: usize,
    ) -> Option<Arc<HashMap<usize, ClusterSlices>>> {
        let cached = {
            let map = self.remote_slices.lock().ok()?;
            map.get(&shard_idx).cloned()
        };
        if let Some(slices) = cached {
            return Some(slices);
        }
        let path = self.shards.get(shard_idx)?.file_path.clone()?;
        let (footer, size) = match read_remote_footer(store, &path).await {
            Ok(resolved) => resolved,
            Err(e) => {
                tracing::warn!(shard_idx, error = %e, "cluster_index: shard footer read failed");
                return None;
            }
        };
        let Ok(slices) = cluster_slices_from_footer(&footer, size as usize) else {
            tracing::warn!(
                shard_idx,
                "cluster_index: shard footer has no cluster slices"
            );
            return None;
        };
        let slices = Arc::new(slices);
        if let Ok(mut map) = self.remote_slices.lock() {
            map.insert(shard_idx, slices.clone());
        }
        Some(slices)
    }

    /// Deserialize one cluster's blobs into the cache, matching `get_cluster`.
    fn cache_deserialized(&self, cluster_id: usize, graph_bytes: Bytes, meta_bytes: Bytes) {
        let bytes = graph_bytes.len() + meta_bytes.len();
        let Ok(graph) = deserialize_graph(graph_bytes, self.quantizer.as_ref(), self.metric) else {
            tracing::warn!(cluster_id, "cluster_index: graph deserialization failed");
            return;
        };
        let Ok(node_meta) = node_meta_from_parquet(meta_bytes) else {
            tracing::warn!(cluster_id, "cluster_index: node metadata parse failed");
            return;
        };
        self.cache.insert(
            cluster_id,
            Arc::new(CachedCluster {
                graph,
                node_meta,
                bytes,
            }),
        );
    }

    /// Advise the page cache for the clusters this query will read.
    ///
    /// The shard container is mapped without prefetch, so without this a graph
    /// deserialization faults its bytes in 4 KB at a time. The advice is grouped
    /// **by shard** and **coalesced**, because the previous per-cluster form did
    /// one `File::open` plus two `posix_fadvise` calls for each probed cluster:
    /// at `n_probe = 16` that is 16 opens and 32 advisories ahead of every query,
    /// against ranges that often sit a few KiB apart inside the same shard. Now a
    /// shard is opened once and the ranges it contributes are merged, so the
    /// count is one open per touched shard and a handful of advisory reads.
    ///
    /// Clusters already deserialized are skipped: their bytes are in the cache,
    /// so advising them re-reads pages the query will never touch.
    ///
    /// Remote stores keep `local_root == None` and are untouched here; they read
    /// through the object store's ranged GETs, which is the same coalescing
    /// applied at the request layer.
    fn prefetch_clusters(&self, cluster_ids: &[usize]) {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let Some(root) = self.local_root.as_ref() else {
                return;
            };
            let mut per_shard: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
            for cluster_id in cluster_ids {
                if self.cache.get(cluster_id).is_some() {
                    continue;
                }
                let Some(shard_idx) = self.cluster_to_shard.get(cluster_id) else {
                    continue;
                };
                let Some(shard) = self.shards.get(*shard_idx) else {
                    continue;
                };
                let Some(slices) = shard.cluster_slices.get(cluster_id) else {
                    continue;
                };
                let spans = per_shard.entry(*shard_idx).or_default();
                spans.push(slices.graph);
                spans.push(slices.node_meta);
            }
            for (shard_idx, spans) in per_shard {
                let Some(shard) = self.shards.get(shard_idx) else {
                    continue;
                };
                let Some(relative) = shard.file_path.as_deref() else {
                    continue;
                };
                let Ok(file) = std::fs::File::open(root.join(relative)) else {
                    continue;
                };
                let fd = file.as_raw_fd();
                for (offset, len) in coalesce_spans(spans, PREFETCH_MERGE_GAP) {
                    // SAFETY: the descriptor is open for the duration of the call
                    // and `posix_fadvise` neither reads nor writes through it.
                    unsafe {
                        libc::posix_fadvise(
                            fd,
                            offset as libc::off_t,
                            len as libc::off_t,
                            libc::POSIX_FADV_WILLNEED,
                        );
                    }
                }
            }
        }
        #[cfg(not(unix))]
        let _ = cluster_ids;
    }

    /// Load a cluster's graph and node metadata from its containing shard,
    /// caching the deserialized form.
    fn get_cluster(&self, cluster_id: usize) -> Option<Arc<CachedCluster>> {
        if let Some(hit) = self.cache.get(&cluster_id) {
            return Some(hit);
        }
        let shard_idx = *self.cluster_to_shard.get(&cluster_id)?;
        let shard = self.shards.get(shard_idx)?;
        // Remote shards carry no container: `hydrate_clusters` owns fetching their
        // bytes, and reaching here means the cluster was not part of the probe.
        if shard.container.is_empty() {
            return None;
        }
        let slices = shard.cluster_slices.get(&cluster_id)?;
        let graph_bytes = shard
            .container
            .slice(slices.graph.0..slices.graph.0 + slices.graph.1);
        let meta_bytes = shard
            .container
            .slice(slices.node_meta.0..slices.node_meta.0 + slices.node_meta.1);

        let graph = match deserialize_graph(graph_bytes, self.quantizer.as_ref(), self.metric) {
            Ok(graph) => graph,
            Err(e) => {
                tracing::warn!(cluster_id, shard_idx, error = %e, "cluster_index: graph deserialization failed");
                return None;
            }
        };
        let node_meta = match node_meta_from_parquet(meta_bytes) {
            Ok(meta) => meta,
            Err(e) => {
                tracing::warn!(cluster_id, shard_idx, error = %e, "cluster_index: node metadata parse failed");
                return None;
            }
        };

        let arc = Arc::new(CachedCluster {
            graph,
            node_meta,
            bytes: slices.graph.1 + slices.node_meta.1,
        });
        self.cache.insert(cluster_id, arc.clone());
        Some(arc)
    }
}

// ---------------------------------------------------------------------------
// Build helpers
// ---------------------------------------------------------------------------

/// Segment id convention shared with the query path: basename minus extension.
fn segment_id_of(entry: &ManifestEntry) -> String {
    let name = entry
        .file_path
        .rsplit('/')
        .next()
        .unwrap_or(&entry.file_path);
    let name = name.strip_suffix(".puffin").unwrap_or(name);
    name.strip_suffix(".parquet").unwrap_or(name).to_string()
}

/// Metric implied by the index algorithm.
///
/// The algorithm does not carry a metric, so the artifact defaults to L2 — the
/// same default the per-segment builder uses. Callers that need another metric
/// should build with [`ClusterIndex::build_from_records`] or extend this.
fn metric_for_algo(_algo: &IndexAlgorithm) -> VectorMetric {
    VectorMetric::L2
}

/// Train the quantizer named by `algo` over the sample.
fn train_quantizer(
    algo: &IndexAlgorithm,
    sample: &[Vec<f32>],
    dim: usize,
    metric: VectorMetric,
) -> Result<Option<QuantizerImpl>> {
    use crate::core::index::benoquant::BenoQuantEncoder;
    use crate::core::index::pq::{PqConfig, PqEncoder};
    use crate::core::index::sq::SqEncoder;

    Ok(match algo {
        IndexAlgorithm::HnswTq8 { .. } => Some(QuantizerImpl::BenoQuant(BenoQuantEncoder::train(
            sample, 8, metric,
        )?)),
        IndexAlgorithm::HnswSq8 { .. } => {
            Some(QuantizerImpl::Sq(SqEncoder::train(sample, 8, metric)?))
        }
        IndexAlgorithm::HnswTq4 { .. } => Some(QuantizerImpl::BenoQuant(BenoQuantEncoder::train(
            sample, 4, metric,
        )?)),
        IndexAlgorithm::HnswPq { compression, .. } => {
            let subspaces = (dim / (*compression).max(1)).max(1);
            Some(QuantizerImpl::Pq(PqEncoder::train(
                sample,
                PqConfig {
                    m: subspaces,
                    k: 256,
                    dim,
                },
            )?))
        }
        _ => None,
    })
}

/// Puffin blob type for the graph payload.
fn blob_type_for(_quantizer: Option<&QuantizerImpl>) -> &'static str {
    PUFFIN_BLOB_VECTOR_HNSW_GRAPH
}

/// Write the centroids and segment-table blobs.
fn write_header_blobs<W: std::io::Write + std::io::Seek>(
    writer: &mut PuffinWriter<W>,
    centroids: &[Vec<f32>],
    metric: VectorMetric,
    quantizer: Option<&QuantizerImpl>,
    segment_table: &[String],
) -> Result<()> {
    let centroids_bytes = centroids_to_parquet(centroids, metric, quantizer)?;
    writer.add_blob(
        PUFFIN_BLOB_HNSW_IVF_CENTROIDS.to_string(),
        Vec::new(),
        0,
        1,
        &centroids_bytes,
        HashMap::from([("filename".to_string(), "centroids.parquet".to_string())]),
    )?;

    let segments_bytes = segments_to_parquet(segment_table)?;
    writer.add_blob(
        PUFFIN_BLOB_VECTOR_CLUSTER_SEGMENTS.to_string(),
        Vec::new(),
        0,
        1,
        &segments_bytes,
        HashMap::from([("filename".to_string(), "segments.parquet".to_string())]),
    )?;
    Ok(())
}

/// In-memory serialized blobs for a completed cluster graph and node metadata.
/// One cluster's records: (segment index, row id, vector).
type ClusterMembers = Vec<(u32, u32, Vec<f32>)>;

/// One spill bucket's records, grouped by global cluster id.
type GroupedBucket = HashMap<usize, ClusterMembers>;

struct BuiltClusterBlobs {
    cluster_id: usize,
    graph_blob_type: &'static str,
    graph_bytes: Vec<u8>,
    meta_bytes: Vec<u8>,
}

/// Construct and serialize one cluster's HNSW graph and node metadata in memory.
///
/// Embarrassingly parallel: takes only immutable references and creates a thread-unique
/// temp dump file before returning the owned byte payloads.
fn build_cluster_blobs(
    cluster_id: usize,
    members: &[(u32, u32, Vec<f32>)],
    quantizer: Option<&QuantizerImpl>,
    metric: VectorMetric,
    m: usize,
    ef_construction: usize,
) -> Result<Option<BuiltClusterBlobs>> {
    if members.is_empty() {
        return Ok(None);
    }

    let vectors: Vec<Vec<f32>> = members.iter().map(|(_, _, v)| v.clone()).collect();
    let graph = build_graph(&vectors, quantizer, metric, m, ef_construction);

    let tmp_base = std::env::temp_dir().join(format!(
        "bsdb-cluster-graph-{}-{}",
        uuid::Uuid::new_v4(),
        cluster_id
    ));
    let tmp_str = tmp_base.to_string_lossy().to_string();
    graph
        .file_dump(&tmp_str)
        .map_err(|e| anyhow::anyhow!("cluster {cluster_id}: HNSW dump failed: {e}"))?;
    let graph_path = format!("{tmp_str}.hnsw.graph");
    let graph_bytes =
        std::fs::read(&graph_path).with_context(|| format!("reading dumped graph {graph_path}"))?;
    let _ = std::fs::remove_file(&graph_path);

    let node_meta: Vec<NodeMeta> = members
        .iter()
        .map(|(seg, row, _)| NodeMeta {
            segment_idx: *seg,
            row_id: *row,
        })
        .collect();
    let meta_bytes = node_meta_to_parquet(&node_meta)?;

    Ok(Some(BuiltClusterBlobs {
        cluster_id,
        graph_blob_type: blob_type_for(quantizer),
        graph_bytes,
        meta_bytes,
    }))
}

/// Write pre-built cluster graph and node metadata blobs to a Puffin shard.
fn write_cluster_blobs_to_puffin<W: std::io::Write + std::io::Seek>(
    writer: &mut PuffinWriter<W>,
    blobs: &BuiltClusterBlobs,
) -> Result<()> {
    let cluster_props = HashMap::from([
        ("cluster_id".to_string(), blobs.cluster_id.to_string()),
        (
            "filename".to_string(),
            format!("cluster_{}.hnsw.graph", blobs.cluster_id),
        ),
    ]);
    writer.add_blob(
        blobs.graph_blob_type.to_string(),
        Vec::new(),
        0,
        1,
        &blobs.graph_bytes,
        cluster_props,
    )?;

    let meta_props = HashMap::from([
        ("cluster_id".to_string(), blobs.cluster_id.to_string()),
        (
            "filename".to_string(),
            format!("cluster_{}.node_meta.parquet", blobs.cluster_id),
        ),
    ]);
    writer.add_blob(
        PUFFIN_BLOB_VECTOR_CLUSTER_NODE_META.to_string(),
        Vec::new(),
        0,
        1,
        &blobs.meta_bytes,
        meta_props,
    )?;
    Ok(())
}

/// Build and write one cluster's graph and node metadata sequentially.
fn write_cluster_blobs<W: std::io::Write + std::io::Seek>(
    writer: &mut PuffinWriter<W>,
    cluster_id: usize,
    members: &[(u32, u32, Vec<f32>)],
    quantizer: Option<&QuantizerImpl>,
    metric: VectorMetric,
    m: usize,
    ef_construction: usize,
) -> Result<()> {
    if let Some(blobs) =
        build_cluster_blobs(cluster_id, members, quantizer, metric, m, ef_construction)?
    {
        write_cluster_blobs_to_puffin(writer, &blobs)?;
    }
    Ok(())
}

/// Build an HNSW graph over one cluster's vectors.
fn build_graph(
    vectors: &[Vec<f32>],
    quantizer: Option<&QuantizerImpl>,
    metric: VectorMetric,
    m: usize,
    ef_construction: usize,
) -> HnswGraph {
    let max_layers = 16;
    let mut hnsw = if let Some(q) = quantizer {
        match q {
            QuantizerImpl::BenoQuant(tq) => {
                if tq.bits() == 4 {
                    HnswGraph::TurboQuant4(Hnsw::new(
                        m,
                        vectors.len(),
                        max_layers,
                        ef_construction,
                        crate::core::index::distance::DistL2u4,
                    ))
                } else {
                    HnswGraph::TurboQuant8(Hnsw::new(
                        m,
                        vectors.len(),
                        max_layers,
                        ef_construction,
                        crate::core::index::distance::DistL2u8,
                    ))
                }
            }
            QuantizerImpl::Sq(q) => HnswGraph::Sq8(Hnsw::new(
                m,
                vectors.len(),
                max_layers,
                ef_construction,
                crate::core::index::sq::DistSq8 {
                    encoder: Arc::new(q.clone()),
                },
            )),
            QuantizerImpl::Pq(pq) => {
                let dist = crate::core::index::pq::DistPqSdc {
                    pq: Arc::new(pq.clone()),
                };
                HnswGraph::Pq(Hnsw::new(
                    m,
                    vectors.len(),
                    max_layers,
                    ef_construction,
                    dist,
                ))
            }
        }
    } else {
        match metric {
            VectorMetric::L2 => HnswGraph::L2(Hnsw::new(
                m,
                vectors.len(),
                max_layers,
                ef_construction,
                DistL2,
            )),
            VectorMetric::Cosine => HnswGraph::Cosine(Hnsw::new(
                m,
                vectors.len(),
                max_layers,
                ef_construction,
                DistCosine,
            )),
            VectorMetric::InnerProduct => HnswGraph::Dot(Hnsw::new(
                m,
                vectors.len(),
                max_layers,
                ef_construction,
                DistDot::norm(),
            )),
            VectorMetric::L1 => HnswGraph::L1(Hnsw::new(
                m,
                vectors.len(),
                max_layers,
                ef_construction,
                DistL1,
            )),
            VectorMetric::Hamming => HnswGraph::Hamming(Hnsw::new(
                m,
                vectors.len(),
                max_layers,
                ef_construction,
                DistHamming,
            )),
            VectorMetric::Jaccard => HnswGraph::Jaccard(Hnsw::new(
                m,
                vectors.len(),
                max_layers,
                ef_construction,
                DistJaccard,
            )),
        }
    };

    hnsw.set_extend_candidates(true);
    hnsw.set_keeping_pruned(true);
    for (local_id, v) in vectors.iter().enumerate() {
        if let Some(q) = quantizer {
            hnsw.insert((VectorValue::Binary(q.encode(v)), local_id));
        } else {
            hnsw.insert((VectorValue::Float32(v.clone()), local_id));
        }
    }
    hnsw
}

/// Deserialize a graph payload, choosing the node type from the quantizer.
fn deserialize_graph(
    bytes: Bytes,
    quantizer: Option<&QuantizerImpl>,
    metric: VectorMetric,
) -> Result<HnswGraph> {
    use crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw;

    let graph = if let Some(q) = quantizer {
        match q {
            QuantizerImpl::BenoQuant(tq) => {
                if tq.bits() == 4 {
                    HnswGraph::TurboQuant4Arrow(
                        ArrowHnsw::load_from_bytes_owned(
                            bytes,
                            crate::core::index::distance::DistL2u4,
                        )
                        .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
                    )
                } else {
                    HnswGraph::TurboQuant8Arrow(
                        ArrowHnsw::load_from_bytes_owned(
                            bytes,
                            crate::core::index::distance::DistL2u8,
                        )
                        .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
                    )
                }
            }
            QuantizerImpl::Sq(q) => HnswGraph::Sq8Arrow(
                ArrowHnsw::load_from_bytes_owned(
                    bytes,
                    crate::core::index::sq::DistSq8 {
                        encoder: Arc::new(q.clone()),
                    },
                )
                .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
            ),
            QuantizerImpl::Pq(pq) => {
                let dist = crate::core::index::pq::DistPqSdc {
                    pq: Arc::new(pq.clone()),
                };
                HnswGraph::PqArrow(
                    ArrowHnsw::load_from_bytes_owned(bytes, dist)
                        .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
                )
            }
        }
    } else {
        match metric {
            VectorMetric::L2 => HnswGraph::L2Arrow(
                ArrowHnsw::load_from_bytes_owned(bytes, DistL2)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
            ),
            VectorMetric::Cosine => HnswGraph::CosineArrow(
                ArrowHnsw::load_from_bytes_owned(bytes, DistCosine)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
            ),
            VectorMetric::InnerProduct => HnswGraph::DotArrow(
                ArrowHnsw::load_from_bytes_owned(bytes, DistDot::norm())
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
            ),
            VectorMetric::L1 => HnswGraph::L1Arrow(
                ArrowHnsw::load_from_bytes_owned(bytes, DistL1)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
            ),
            VectorMetric::Hamming => HnswGraph::HammingArrow(
                ArrowHnsw::load_from_bytes_owned(bytes, DistHamming)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
            ),
            VectorMetric::Jaccard => HnswGraph::JaccardArrow(
                ArrowHnsw::load_from_bytes_owned(bytes, DistJaccard)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {e}"))?,
            ),
        }
    };
    Ok(graph)
}

// ---------------------------------------------------------------------------
// Bucket I/O
// ---------------------------------------------------------------------------

/// Append one `[cluster u32][segment u32][row u32][f32 * dim]` record.
fn write_record(
    file: &mut std::fs::File,
    cluster_id: u32,
    segment_idx: u32,
    row_id: u32,
    vector: &[f32],
) -> Result<()> {
    use std::io::Write;
    file.write_all(&cluster_id.to_le_bytes())?;
    file.write_all(&segment_idx.to_le_bytes())?;
    file.write_all(&row_id.to_le_bytes())?;
    let mut buf = Vec::with_capacity(vector.len() * 4);
    for v in vector {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    file.write_all(&buf)?;
    Ok(())
}

/// Group a bucket's records by cluster id.
fn group_bucket(bytes: &[u8], dim: usize) -> Result<HashMap<usize, Vec<(u32, u32, Vec<f32>)>>> {
    let header = 12usize;
    let record = header + dim * 4;
    if record == 0 || !bytes.len().is_multiple_of(record) {
        anyhow::bail!(
            "cluster build bucket is not a whole number of {record}-byte records ({} bytes)",
            bytes.len()
        );
    }
    let mut grouped: HashMap<usize, Vec<(u32, u32, Vec<f32>)>> = HashMap::new();
    for chunk in bytes.chunks_exact(record) {
        let cluster_id = u32::from_le_bytes(chunk[0..4].try_into().unwrap_or([0; 4])) as usize;
        let segment_idx = u32::from_le_bytes(chunk[4..8].try_into().unwrap_or([0; 4]));
        let row_id = u32::from_le_bytes(chunk[8..12].try_into().unwrap_or([0; 4]));
        let mut vector = Vec::with_capacity(dim);
        for i in 0..dim {
            let start = header + i * 4;
            let v = f32::from_le_bytes(chunk[start..start + 4].try_into().unwrap_or([0; 4]));
            vector.push(v);
        }
        grouped
            .entry(cluster_id)
            .or_default()
            .push((segment_idx, row_id, vector));
    }
    Ok(grouped)
}

// ---------------------------------------------------------------------------
// Parquet helpers
// ---------------------------------------------------------------------------

/// Stream a Parquet file's vector column in batches.
///
/// `on_batch(row_offset, vectors)` is called once per record batch, so the
/// caller never materializes a whole segment.
async fn stream_vector_batches<F>(
    store: &Arc<dyn ObjectStore>,
    path: &str,
    column: &str,
    on_batch: &mut F,
) -> Result<()>
where
    F: FnMut(usize, &[Vec<f32>]) -> Result<()>,
{
    let object_path = object_store::path::Path::from(path);
    let reader = ParquetObjectReader::new(store.clone(), object_path);
    let builder = ParquetRecordBatchStreamBuilder::new(reader)
        .await
        .with_context(|| format!("opening Parquet file '{path}'"))?;

    let col_idx = builder
        .schema()
        .fields()
        .iter()
        .position(|f| f.name() == column)
        .ok_or_else(|| anyhow::anyhow!("column '{column}' not found in '{path}'"))?;

    let mask = ProjectionMask::roots(builder.parquet_schema(), [col_idx]);
    let mut stream = builder
        .with_projection(mask)
        .build()
        .with_context(|| format!("building Parquet stream for '{path}'"))?;

    let mut row_offset = 0usize;
    while let Some(batch) = stream.next().await {
        let batch = batch.with_context(|| format!("reading batch from '{path}'"))?;
        if batch.num_rows() == 0 {
            continue;
        }
        let col = batch.column(0).clone();
        let vectors = extract_vectors(&col)?;
        on_batch(row_offset, &vectors)?;
        row_offset += batch.num_rows();
    }
    Ok(())
}

/// Extract `Vec<Vec<f32>>` from a FixedSizeList or List column.
fn extract_vectors(col: &ArrayRef) -> Result<Vec<Vec<f32>>> {
    match col.data_type() {
        DataType::FixedSizeList(_, _) => {
            let list = col
                .as_any()
                .downcast_ref::<arrow::array::FixedSizeListArray>()
                .context("invalid cast to FixedSizeListArray")?;
            Ok((0..list.len())
                .map(|i| values_to_f32(&list.value(i)))
                .collect())
        }
        DataType::List(_) => {
            let list = col
                .as_any()
                .downcast_ref::<arrow::array::ListArray>()
                .context("invalid cast to ListArray")?;
            Ok((0..list.len())
                .map(|i| values_to_f32(&list.value(i)))
                .collect())
        }
        other => anyhow::bail!("expected a vector column, found {other:?}"),
    }
}

fn values_to_f32(item: &ArrayRef) -> Vec<f32> {
    if let Some(floats) = item.as_any().downcast_ref::<arrow::array::Float32Array>() {
        floats.values().to_vec()
    } else if let Some(doubles) = item.as_any().downcast_ref::<arrow::array::Float64Array>() {
        doubles.values().iter().map(|&d| d as f32).collect()
    } else {
        vec![0.0; item.len()]
    }
}

fn centroids_to_parquet(
    centroids: &[Vec<f32>],
    metric: VectorMetric,
    quantizer: Option<&QuantizerImpl>,
) -> Result<Vec<u8>> {
    let mut builder = ListBuilder::new(Float32Builder::new());
    for centroid in centroids {
        for &v in centroid {
            builder.values().append_value(v);
        }
        builder.append(true);
    }
    let array = Arc::new(builder.finish());
    let schema = Arc::new(Schema::new(vec![Field::new(
        "centroid",
        DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
        false,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![array])?;

    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![
            parquet::file::metadata::KeyValue {
                key: "vector_metric".to_string(),
                value: Some(metric.as_str().to_string()),
            },
            parquet::file::metadata::KeyValue {
                key: "quantizer_config".to_string(),
                value: quantizer.and_then(|q| serde_json::to_string(q).ok()),
            },
        ]))
        .build();
    write_parquet_in_memory(schema, &batch, props)
}

fn segments_to_parquet(segment_table: &[String]) -> Result<Vec<u8>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("segment_idx", DataType::UInt32, false),
        Field::new("segment_id", DataType::Utf8, false),
    ]));
    let idx = UInt32Array::from((0..segment_table.len() as u32).collect::<Vec<_>>());
    let ids = StringArray::from(segment_table.to_vec());
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(idx), Arc::new(ids)])?;
    write_parquet_in_memory(schema, &batch, WriterProperties::builder().build())
}

fn node_meta_to_parquet(meta: &[NodeMeta]) -> Result<Vec<u8>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("segment_idx", DataType::UInt32, false),
        Field::new("row_id", DataType::UInt32, false),
    ]));
    let seg = UInt32Array::from(meta.iter().map(|m| m.segment_idx).collect::<Vec<_>>());
    let row = UInt32Array::from(meta.iter().map(|m| m.row_id).collect::<Vec<_>>());
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(seg), Arc::new(row)])?;
    write_parquet_in_memory(schema, &batch, WriterProperties::builder().build())
}

fn write_parquet_in_memory(
    schema: Arc<Schema>,
    batch: &RecordBatch,
    props: WriterProperties,
) -> Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props))?;
        writer.write(batch)?;
        writer.close()?;
    }
    Ok(buf)
}

fn node_meta_from_parquet(bytes: Bytes) -> Result<Vec<NodeMeta>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
    let reader = builder.build()?;
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch?;
        let seg = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("node_meta: segment_idx is not UInt32")?;
        let row = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("node_meta: row_id is not UInt32")?;
        for i in 0..batch.num_rows() {
            out.push(NodeMeta {
                segment_idx: seg.value(i),
                row_id: row.value(i),
            });
        }
    }
    Ok(out)
}

/// Byte range of the first blob of `blob_type`, from a footer that was read
/// without materialising the container.
fn blob_range_in_footer(
    footer: &crate::core::puffin::PuffinFooter,
    blob_type: &str,
) -> Option<(u64, u64)> {
    footer
        .blobs
        .iter()
        .find(|b| b.r#type == blob_type)
        .and_then(|b| {
            let offset = u64::try_from(b.offset).ok()?;
            let length = u64::try_from(b.length).ok()?;
            Some((offset, length))
        })
}

fn read_centroids_blob(
    container: &Bytes,
) -> Result<(Vec<Vec<f32>>, VectorMetric, Option<QuantizerImpl>)> {
    let (offset, length) = find_blob(container, PUFFIN_BLOB_HNSW_IVF_CENTROIDS)?
        .ok_or_else(|| anyhow::anyhow!("cluster index container has no centroids blob"))?;
    read_centroids_from_bytes(container.slice(offset..offset + length))
}

/// Parse the centroids blob from its own bytes, which a remote reader gets with
/// a ranged GET instead of holding the shard.
fn read_centroids_from_bytes(
    bytes: Bytes,
) -> Result<(Vec<Vec<f32>>, VectorMetric, Option<QuantizerImpl>)> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())?;
    let reader = builder.build()?;
    let mut centroids = Vec::new();
    for batch in reader {
        let batch = batch?;
        if batch.num_columns() == 0 {
            continue;
        }
        centroids.extend(extract_vectors(&batch.column(0).clone())?);
    }

    let mut metric = VectorMetric::L2;
    let mut quantizer = None;
    let file_reader = parquet::file::reader::SerializedFileReader::new(bytes)?;
    if let Some(kv) = file_reader.metadata().file_metadata().key_value_metadata() {
        for entry in kv {
            match entry.key.as_str() {
                "vector_metric" => {
                    if let Some(v) = &entry.value {
                        metric = v.parse::<VectorMetric>().unwrap_or(VectorMetric::L2);
                    }
                }
                "quantizer_config" => {
                    if let Some(v) = &entry.value {
                        if let Ok(mut q) = serde_json::from_str::<QuantizerImpl>(v) {
                            if let QuantizerImpl::Pq(ref mut pq) = q {
                                pq.init_lut();
                            }
                            quantizer = Some(q);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok((centroids, metric, quantizer))
}

fn read_segments_blob(container: &Bytes) -> Result<Vec<String>> {
    let (offset, length) = find_blob(container, PUFFIN_BLOB_VECTOR_CLUSTER_SEGMENTS)?
        .ok_or_else(|| anyhow::anyhow!("cluster index container has no segments blob"))?;
    read_segments_from_bytes(container.slice(offset..offset + length))
}

/// Parse the segment table from its own bytes.
fn read_segments_from_bytes(bytes: Bytes) -> Result<Vec<String>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
    let reader = builder.build()?;
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch?;
        let ids = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("segments: segment_id is not Utf8")?;
        for i in 0..batch.num_rows() {
            out.push(ids.value(i).to_string());
        }
    }
    Ok(out)
}

/// Find a blob's `(offset, length)` by type.
fn find_blob(container: &Bytes, blob_type: &str) -> Result<Option<(usize, usize)>> {
    let reader = PuffinReader::new(Cursor::new(container.as_ref()))?;
    for blob in reader.footer().blobs.iter() {
        if blob.r#type == blob_type {
            let offset = usize::try_from(blob.offset)
                .map_err(|_| anyhow::anyhow!("negative blob offset"))?;
            let length = usize::try_from(blob.length)
                .map_err(|_| anyhow::anyhow!("negative blob length"))?;
            if offset
                .checked_add(length)
                .is_none_or(|end| end > container.len())
            {
                anyhow::bail!("blob '{blob_type}' exceeds container size");
            }
            return Ok(Some((offset, length)));
        }
    }
    Ok(None)
}

/// Pair each cluster's graph blob with its node-metadata blob.
fn cluster_slices_from_container(container: &Bytes) -> Result<HashMap<usize, ClusterSlices>> {
    let reader = PuffinReader::new(Cursor::new(container.as_ref()))?;
    cluster_slices_from_footer(reader.footer(), container.len())
}

/// Cluster byte ranges from a footer alone.
///
/// `object_len` is the size of the container the offsets are relative to. For a
/// local shard that is the mapped length; for a remote one it comes from `head`,
/// so a footer read with a ranged GET can still validate its offsets without the
/// container being resident.
fn cluster_slices_from_footer(
    footer: &crate::core::puffin::PuffinFooter,
    object_len: usize,
) -> Result<HashMap<usize, ClusterSlices>> {
    let mut graphs: HashMap<usize, (usize, usize)> = HashMap::new();
    let mut metas: HashMap<usize, (usize, usize)> = HashMap::new();

    for blob in footer.blobs.iter() {
        let Some(cluster_id) = blob
            .properties
            .get("cluster_id")
            .and_then(|v| v.parse::<usize>().ok())
        else {
            continue;
        };
        let offset =
            usize::try_from(blob.offset).map_err(|_| anyhow::anyhow!("negative blob offset"))?;
        let length =
            usize::try_from(blob.length).map_err(|_| anyhow::anyhow!("negative blob length"))?;
        if offset
            .checked_add(length)
            .is_none_or(|end| end > object_len)
        {
            anyhow::bail!("cluster {cluster_id} blob exceeds container size");
        }
        if blob.r#type == PUFFIN_BLOB_VECTOR_CLUSTER_NODE_META {
            metas.insert(cluster_id, (offset, length));
        } else if blob.r#type == PUFFIN_BLOB_VECTOR_HNSW_GRAPH {
            graphs.insert(cluster_id, (offset, length));
        }
    }

    let mut out = HashMap::new();
    for (cluster_id, graph) in graphs {
        if let Some(node_meta) = metas.get(&cluster_id) {
            out.insert(
                cluster_id,
                ClusterSlices {
                    graph,
                    node_meta: *node_meta,
                },
            );
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Distance under the artifact's metric, "smaller is closer".
fn metric_distance(metric: VectorMetric, q: &[f32], c: &[f32]) -> f32 {
    use crate::core::index::distance;
    match metric {
        VectorMetric::L2 => distance::l2_distance(q, c),
        VectorMetric::Cosine => distance::cosine_distance(q, c),
        VectorMetric::InnerProduct => -distance::dot_product(q, c),
        VectorMetric::L1 => distance::l1_distance(q, c),
        VectorMetric::Hamming => distance::hamming_distance(q, c),
        VectorMetric::Jaccard => distance::jaccard_distance(q, c),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesce_spans_merges_adjacent_ranges() {
        // A cluster's graph immediately followed by its node metadata.
        assert_eq!(
            coalesce_spans(vec![(0, 100), (100, 100)], 0),
            vec![(0, 200)]
        );
    }

    #[test]
    fn coalesce_spans_merges_overlaps() {
        assert_eq!(coalesce_spans(vec![(0, 100), (50, 100)], 0), vec![(0, 150)]);
    }

    #[test]
    fn coalesce_spans_respects_the_gap() {
        // 50 bytes of slack with a 100 byte allowance merges; 150 does not.
        assert_eq!(
            coalesce_spans(vec![(0, 100), (150, 100)], 100),
            vec![(0, 250)]
        );
        assert_eq!(
            coalesce_spans(vec![(0, 100), (250, 100)], 100),
            vec![(0, 100), (250, 100)]
        );
    }

    #[test]
    fn coalesce_spans_orders_and_drops_empty_ranges() {
        // Gap 0 keeps the two distant ranges apart while the zero-length range
        // disappears: it covers nothing and must not become a span at offset 30.
        assert_eq!(
            coalesce_spans(vec![(250, 100), (0, 10), (30, 0)], 0),
            vec![(0, 10), (250, 100)]
        );
        assert!(coalesce_spans(vec![(0, 0), (5, 0)], 0).is_empty());
    }

    #[test]
    fn coalesce_spans_collapses_a_probe_that_lands_in_one_shard() {
        // Two clusters in one shard, each contributing a graph and metadata blob
        // a few KiB apart: the whole probe becomes one advisory read.
        let spans = vec![
            (0, 2_000_000),
            (2_000_000, 8_192),
            (2_100_000, 2_000_000),
            (4_100_000, 8_192),
        ];
        assert_eq!(
            coalesce_spans(spans, PREFETCH_MERGE_GAP),
            vec![(0, 4_108_192)]
        );
    }

    #[test]
    fn cluster_cache_budget_is_bytes_not_entries() {
        // The regression this guards: a fixed entry cap of 64 against 40,100
        // clusters. Any non-trivial budget has to be far larger than an entry
        // count, and a byte weighed entry has to be at least 1 KiB.
        assert!(cluster_cache_budget_kib() >= 1024);
    }

    #[test]
    fn cache_entry_is_weighed_by_fetched_bytes() {
        // The regression this guards: weighing the decoded graph instead of the
        // fetched blobs. On the measured arm a cluster's two blobs are ~900 KiB,
        // while Arrow's `get_array_memory_size` reported ~6.8 MiB for the same
        // graph, so the budget admitted a fifth of the clusters it could hold.
        let vectors: Vec<Vec<f32>> = blob(16, 8, 0.0).into_iter().map(|(_, _, v)| v).collect();
        let graph = build_graph(&vectors, None, VectorMetric::L2, 8, 32);
        let entry = Arc::new(CachedCluster {
            graph,
            node_meta: Vec::new(),
            bytes: 900 * 1024,
        });
        assert_eq!(cluster_entry_kib(&entry), 900);
    }

    #[test]
    fn descriptor_names_every_shard_with_its_range() {
        // The descriptor is what a remote open reads instead of every footer, so
        // both of its forms have to survive the round trip: the range form for a
        // new artifact, and the bare path list for one written before it existed.
        let descriptors = vec![
            ShardDescriptor {
                path: "idx_shard_0000.puffin".to_string(),
                cluster_start: 0,
                cluster_end: 40,
                bytes: 1024,
            },
            ShardDescriptor {
                path: "idx_shard_0001.puffin".to_string(),
                cluster_start: 41,
                cluster_end: 77,
                bytes: 2048,
            },
        ];
        let encoded = serde_json::to_vec(&descriptors).unwrap();
        let paths = shard_paths_from_descriptor(&encoded).unwrap();
        assert_eq!(
            paths,
            vec![
                "idx_shard_0000.puffin".to_string(),
                "idx_shard_0001.puffin".to_string()
            ]
        );

        let legacy = serde_json::to_vec(&paths).unwrap();
        assert_eq!(shard_paths_from_descriptor(&legacy).unwrap(), paths);
        assert!(shard_paths_from_descriptor(b"not json").is_none());
    }

    #[test]
    fn an_unresolved_shard_still_routes_its_clusters() {
        // A remote open leaves shards 1..N without slice maps, so the declared
        // range is the only routing table. Without this expansion those clusters
        // would resolve to no shard and the probes that hit them would be dropped.
        let resolved = HashMap::from([(
            0usize,
            ClusterSlices {
                graph: (0, 4),
                node_meta: (4, 8),
            },
        )]);
        let shards = vec![
            ClusterShard {
                shard_idx: 0,
                file_path: Some("shard_0.puffin".to_string()),
                cluster_range: (0, 0),
                container: Bytes::new(),
                cluster_slices: resolved,
            },
            ClusterShard {
                shard_idx: 1,
                file_path: Some("shard_1.puffin".to_string()),
                cluster_range: (1, 3),
                container: Bytes::new(),
                cluster_slices: HashMap::new(),
            },
        ];
        let index = ClusterIndex::from_shards(
            shards,
            "embedding".to_string(),
            8,
            VectorMetric::L2,
            vec![vec![0.0; 8]; 4],
            None,
            segments(),
            1,
            0,
        )
        .unwrap();
        assert_eq!(index.cluster_to_shard.get(&0), Some(&0));
        for cid in 1..=3 {
            assert_eq!(
                index.cluster_to_shard.get(&cid),
                Some(&1),
                "cluster {cid} must route to the shard whose range declares it"
            );
        }
    }

    fn blob(n: usize, dim: usize, offset: f32) -> Vec<(u32, u32, Vec<f32>)> {
        (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim)
                    .map(|d| offset + (i as f32) * 0.01 + (d as f32) * 0.001)
                    .collect();
                ((i % 3) as u32, i as u32, v)
            })
            .collect()
    }

    fn segments() -> Vec<String> {
        vec![
            "seg_0".to_string(),
            "seg_1".to_string(),
            "seg_2".to_string(),
        ]
    }

    #[test]
    fn build_and_search_round_trip() {
        let records = blob(300, 8, 0.0);
        let index = ClusterIndex::build_from_records(
            records,
            segments(),
            "embedding",
            &IndexAlgorithm::hnsw().with_complexity(16).with_quality(200),
            8,
            16,
            100,
        )
        .unwrap();

        assert_eq!(index.cluster_count(), 8);
        assert_eq!(index.dim, 8);
        assert_eq!(index.segment_count(), 3);

        let query = vec![0.0; 8];
        let results = index.search(&query, 5, 4, Some(64)).unwrap();
        assert!(!results.is_empty(), "search returned no results");
        assert!(results.len() <= 5);
        // Scores must be ascending (distance).
        for pair in results.windows(2) {
            assert!(pair[0].score <= pair[1].score + 1e-6);
        }
        // Every hit must resolve to a real segment.
        for r in &results {
            assert!(index.segment_table.contains(&r.segment_id));
        }
    }

    #[test]
    fn search_rejects_wrong_dimension() {
        let index = ClusterIndex::build_from_records(
            blob(50, 4, 0.0),
            segments(),
            "embedding",
            &IndexAlgorithm::hnsw().with_complexity(16).with_quality(100),
            4,
            8,
            50,
        )
        .unwrap();
        assert!(index.search(&[0.0; 3], 5, 2, None).unwrap().is_empty());
        assert!(index.search(&[0.0; 4], 0, 2, None).unwrap().is_empty());
    }

    #[test]
    fn node_meta_parquet_round_trip() {
        let meta = vec![
            NodeMeta {
                segment_idx: 0,
                row_id: 7,
            },
            NodeMeta {
                segment_idx: 2,
                row_id: 42,
            },
        ];
        let bytes = node_meta_to_parquet(&meta).unwrap();
        let decoded = node_meta_from_parquet(Bytes::from(bytes)).unwrap();
        assert_eq!(decoded, meta);
    }

    #[test]
    fn segments_parquet_round_trip() {
        let segs = segments();
        let bytes = segments_to_parquet(&segs).unwrap();
        let container = wrap_blob(PUFFIN_BLOB_VECTOR_CLUSTER_SEGMENTS, &bytes);
        assert_eq!(read_segments_blob(&container).unwrap(), segs);
    }

    #[test]
    fn bucket_grouping_preserves_records() {
        let dim = 3;
        let mut buf = Vec::new();
        let mut file = tempfile::tempfile().unwrap();
        write_record(&mut file, 5, 1, 9, &[1.0, 2.0, 3.0]).unwrap();
        write_record(&mut file, 5, 2, 10, &[4.0, 5.0, 6.0]).unwrap();
        write_record(&mut file, 7, 0, 1, &[7.0, 8.0, 9.0]).unwrap();
        use std::io::{Read, Seek, SeekFrom};
        file.seek(SeekFrom::Start(0)).unwrap();
        file.read_to_end(&mut buf).unwrap();

        let grouped = group_bucket(&buf, dim).unwrap();
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[&5].len(), 2);
        assert_eq!(grouped[&5][0], (1, 9, vec![1.0, 2.0, 3.0]));
        assert_eq!(grouped[&7][0], (0, 1, vec![7.0, 8.0, 9.0]));
    }

    #[test]
    fn table_index_file_covers_segments() {
        let index = ClusterIndex::build_from_records(
            blob(30, 4, 0.0),
            segments(),
            "embedding",
            &IndexAlgorithm::hnsw().with_complexity(16).with_quality(50),
            4,
            8,
            50,
        )
        .unwrap();
        let file = index.to_table_index_file("idx/cluster.puffin", 7, "cluster_hnsw");
        assert_eq!(file.cluster_count, 4);
        assert_eq!(file.dim, 4);
        assert_eq!(file.source_manifest_version, 7);
        assert!(file.covers_segments(&segments()));
        assert!(!file.covers_segments(&["seg_0".to_string()]));
    }

    #[test]
    fn table_index_file_is_valid_base_for() {
        let index = ClusterIndex::build_from_records(
            blob(30, 4, 0.0),
            segments(),
            "embedding",
            &IndexAlgorithm::hnsw().with_complexity(16).with_quality(50),
            4,
            8,
            50,
        )
        .unwrap();
        let file = index.to_table_index_file("idx/cluster.puffin", 7, "cluster_hnsw");
        // Exact match is valid
        assert!(file.is_valid_base_for(&segments()));
        // Superset (new delta segment added) is valid in LSM model
        let mut superset = segments();
        superset.push("seg_new_delta".to_string());
        assert!(file.is_valid_base_for(&superset));
        // Subset (one base segment missing/deleted) is invalid
        assert!(!file.is_valid_base_for(&["seg_0".to_string()]));
        assert!(!file.is_valid_base_for(&[]));
    }

    #[test]
    fn quantized_cluster_index_calibrated_search_round_trip() {
        let records: Vec<(u32, u32, Vec<f32>)> = (0..120usize)
            .map(|i| {
                let seg = (i % 3) as u32;
                let row = i as u32;
                let v = vec![i as f32; 8];
                (seg, row, v)
            })
            .collect();

        for algo in [IndexAlgorithm::hnsw_tq8(), IndexAlgorithm::hnsw_sq8()] {
            let index = ClusterIndex::build_from_records(
                records.clone(),
                segments(),
                "embedding",
                &algo,
                4,
                16,
                100,
            )
            .unwrap();

            let query = vec![119.0f32; 8];
            let results = index.search(&query, 5, 4, Some(64)).unwrap();
            assert!(!results.is_empty(), "quantized cluster search empty");
            assert!(
                results.iter().any(|r| r.row_id == 119),
                "expected nearest row 119 in top-5 for {algo:?}, got {results:?}"
            );
            for pair in results.windows(2) {
                assert!(pair[0].score <= pair[1].score + 1e-6);
            }
        }
    }

    /// Wrap a single blob in a minimal Puffin container for reader tests.
    fn wrap_blob(blob_type: &str, data: &[u8]) -> Bytes {
        let mut writer = PuffinWriter::new(Cursor::new(Vec::new())).unwrap();
        writer
            .add_blob(
                blob_type.to_string(),
                Vec::new(),
                0,
                1,
                data,
                HashMap::new(),
            )
            .unwrap();
        Bytes::from(writer.finish().unwrap().into_inner())
    }
}
