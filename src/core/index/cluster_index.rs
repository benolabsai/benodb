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

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::{Cursor, Read, Write};
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
use crate::core::index::{
    global_centroids, PreparedQuantizedQuery, Quantizer, QuantizerImpl, VectorMetric, VectorValue,
};
use crate::core::manifest::{IndexAlgorithm, ManifestEntry, TableIndexFile};
use crate::core::puffin::{
    PuffinReader, PuffinWriter, PUFFIN_BLOB_HNSW_IVF_CENTROIDS, PUFFIN_BLOB_VECTOR_CLUSTER_CODES,
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

/// Get the target shard size in bytes from the environment (`BDB_INDEX_TARGET_SHARD_BYTES`)
/// or fallback to [`DEFAULT_TARGET_SHARD_BYTES`].
pub fn target_shard_bytes() -> usize {
    std::env::var("BDB_INDEX_TARGET_SHARD_BYTES")
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
/// `BDB_CLUSTER_BUILD_TMPDIR` points it at the storage volume instead.
fn build_spill_root() -> std::path::PathBuf {
    std::env::var_os("BDB_CLUSTER_BUILD_TMPDIR")
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
    if let Some(n) = std::env::var("BDB_CLUSTER_BUILD_BUCKETS")
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
    std::env::var("BDB_CLUSTER_BUILD_PARALLEL_BUCKETS")
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

/// Magic header for a flat `cluster-codes` Puffin blob (`BSCD`).
const CODES_BLOB_MAGIC: [u8; 4] = *b"BSCD";

/// Current version of the `cluster-codes` binary format.
const CODES_BLOB_VERSION: u32 = 1;

/// Fixed 16-byte header preceding the contiguous codes array:
/// `[magic: 4][version: u32][num_vectors: u32][code_len: u32]`.
/// 16-byte alignment keeps SIMD loads and `f32` casts naturally aligned.
const CODES_HEADER_LEN: usize = 16;

/// Number of probed clusters advised per pipelined batch on local stores.
const LOCAL_PREFETCH_BATCH: usize = 4;

/// Number of probed clusters hydrated per pipelined batch on remote stores.
const REMOTE_PIPELINE_BATCH: usize = 8;

/// Maximum concurrent shard range GETs per remote hydration batch.
const REMOTE_SHARD_CONCURRENCY: usize = 8;

/// Execution mode for table-level cluster search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterSearchMode {
    /// Use the codes-only first stage whenever the shard carries `cluster-codes`
    /// blobs, falling back to HNSW graph traversal for legacy artifacts.
    Auto,
    /// Prefer codes-only scanning when available (same as `Auto`).
    Codes,
    /// Force full HNSW graph deserialization and beam traversal (useful for A/B
    /// benchmarking against codes-only search on the same artifact).
    Graph,
}

fn configured_search_mode() -> ClusterSearchMode {
    match std::env::var("BDB_CLUSTER_SEARCH_MODE")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some(v) if v.eq_ignore_ascii_case("graph") => ClusterSearchMode::Graph,
        Some(v) if v.eq_ignore_ascii_case("codes") => ClusterSearchMode::Codes,
        _ => ClusterSearchMode::Auto,
    }
}

/// Byte slices of one cluster's blobs inside a Puffin container.
#[derive(Debug, Clone, Copy)]
pub struct ClusterSlices {
    pub graph: (usize, usize),
    pub node_meta: (usize, usize),
    /// Optional contiguous flat codes blob (`org.apache.iceberg.vector.cluster-codes`).
    /// `None` on legacy artifacts built before the split-blob layout existed.
    pub codes: Option<(usize, usize)>,
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
        .get_range(&object_store::path::Path::from(path), offset..offset + len)
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
/// `BDB_CLUSTER_CACHE_KIB` sets it directly. Otherwise a quarter of
/// `BDB_CACHE_GB` when that is set, because the shard mappings share the same
/// cap and page cache is the cheaper thing to lose. 256 MiB otherwise.
fn cluster_cache_budget_kib() -> u64 {
    fn from_env(name: &str) -> Option<u64> {
        std::env::var(name).ok()?.trim().parse::<u64>().ok()
    }
    if let Some(kib) = from_env("BDB_CLUSTER_CACHE_KIB") {
        return kib.max(1);
    }
    if let Some(gb) = from_env("BDB_CACHE_GB") {
        return (gb.saturating_mul(1024 * 1024) / 4).max(1);
    }
    256 * 1024
}

/// Cache key unifying full graphs, codes-only blobs, and per-cluster node
/// metadata under a single byte-weighed memory budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ClusterCacheKey {
    Graph(usize),
    Codes(usize),
    NodeMeta(usize),
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

/// Zero-copy view of one cluster's contiguous flat codes blob.
struct CachedClusterCodes {
    /// Full blob bytes (including 16-byte header). Aliases the mmap on local
    /// stores or the fetched HTTP buffer on remote stores.
    bytes: Bytes,
    /// Number of vectors in the cluster (matching `node_meta.len()`).
    num_vectors: usize,
    /// Fixed byte stride per vector code.
    code_len: usize,
}

impl CachedClusterCodes {
    #[inline]
    fn data(&self) -> &[u8] {
        &self.bytes[CODES_HEADER_LEN..]
    }
}

/// Cached per-cluster `(segment_idx, row_id)` table for Phase 3 winner resolution.
struct CachedNodeMeta {
    entries: Vec<NodeMeta>,
    /// Resident weight in bytes (maximum of decoded heap size and raw blob size).
    bytes: usize,
}

#[derive(Clone)]
enum CachedClusterEntry {
    Graph(Arc<CachedCluster>),
    Codes(Arc<CachedClusterCodes>),
    NodeMeta(Arc<CachedNodeMeta>),
}

impl CachedClusterEntry {
    fn weight_bytes(&self) -> usize {
        match self {
            CachedClusterEntry::Graph(c) => c.bytes,
            CachedClusterEntry::Codes(c) => c.bytes.len(),
            CachedClusterEntry::NodeMeta(m) => m.bytes,
        }
    }
}

/// Weighed size of one unified cache entry, in KiB.
fn cache_entry_kib(value: &CachedClusterEntry) -> u32 {
    ((value.weight_bytes() / 1024) as u32).max(1)
}

/// Async singleflight gate deduplicating concurrent remote fetches per key.
///
/// When multiple concurrent queries miss the same shard footer or cluster blob,
/// only the first caller (`Claim::Leader`) issues the object-store request;
/// concurrent peers (`Claim::Follower`) receive a `watch::Receiver<bool>` and
/// await the leader's completion. Dropping [`SingleFlightGuard`] unconditionally
/// removes the entry and wakes all waiters, even on task cancellation or panic.
struct SingleFlight<K: Eq + std::hash::Hash + Clone> {
    inflight: Arc<parking_lot::Mutex<HashMap<K, tokio::sync::watch::Receiver<bool>>>>,
}

impl<K: Eq + std::hash::Hash + Clone> Default for SingleFlight<K> {
    fn default() -> Self {
        Self {
            inflight: Arc::new(parking_lot::Mutex::new(HashMap::new())),
        }
    }
}

enum SingleFlightClaim<K: Eq + std::hash::Hash + Clone> {
    Leader(SingleFlightGuard<K>),
    Follower(tokio::sync::watch::Receiver<bool>),
}

struct SingleFlightGuard<K: Eq + std::hash::Hash + Clone> {
    key: K,
    tx: Option<tokio::sync::watch::Sender<bool>>,
    inflight: Arc<parking_lot::Mutex<HashMap<K, tokio::sync::watch::Receiver<bool>>>>,
}

impl<K: Eq + std::hash::Hash + Clone> Drop for SingleFlightGuard<K> {
    fn drop(&mut self) {
        self.inflight.lock().remove(&self.key);
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(true);
        }
    }
}

impl<K: Eq + std::hash::Hash + Clone> SingleFlight<K> {
    fn try_claim(&self, key: K) -> SingleFlightClaim<K> {
        let mut map = self.inflight.lock();
        if let Some(rx) = map.get(&key) {
            return SingleFlightClaim::Follower(rx.clone());
        }
        let (tx, rx) = tokio::sync::watch::channel(false);
        map.insert(key.clone(), rx);
        SingleFlightClaim::Leader(SingleFlightGuard {
            key,
            tx: Some(tx),
            inflight: self.inflight.clone(),
        })
    }
}

/// Wait for a batch of singleflight peer receivers to signal completion.
async fn await_singleflight_peers(peers: Vec<tokio::sync::watch::Receiver<bool>>) {
    for mut rx in peers {
        let _ = rx.wait_for(|&done| done).await;
    }
}

/// Encode pre-quantized uniform-length byte codes into a self-describing flat
/// `cluster-codes` blob.
fn encode_codes_blob(encoded: &[Vec<u8>]) -> Result<Vec<u8>> {
    let num_vectors = u32::try_from(encoded.len())
        .map_err(|_| anyhow::anyhow!("cluster has too many vectors for codes blob"))?;
    let code_len = encoded.first().map(|c| c.len()).unwrap_or(0);
    let code_len_u32 = u32::try_from(code_len)
        .map_err(|_| anyhow::anyhow!("code length {code_len} overflows u32"))?;
    let mut out = Vec::with_capacity(CODES_HEADER_LEN + encoded.len().saturating_mul(code_len));
    out.extend_from_slice(&CODES_BLOB_MAGIC);
    out.extend_from_slice(&CODES_BLOB_VERSION.to_le_bytes());
    out.extend_from_slice(&num_vectors.to_le_bytes());
    out.extend_from_slice(&code_len_u32.to_le_bytes());
    for (idx, code) in encoded.iter().enumerate() {
        if code.len() != code_len {
            anyhow::bail!(
                "cluster codes blob requires uniform code length: expected {code_len}, got {} at index {idx}",
                code.len()
            );
        }
        out.extend_from_slice(code);
    }
    Ok(out)
}

/// Encode unquantized `f32` vectors into a self-describing flat `cluster-codes`
/// blob (little-endian `f32`, stride `dim * 4`).
fn encode_f32_codes_blob(vectors: &[Vec<f32>]) -> Result<Vec<u8>> {
    let num_vectors = u32::try_from(vectors.len())
        .map_err(|_| anyhow::anyhow!("cluster has too many vectors for f32 codes blob"))?;
    let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
    let code_len = dim.saturating_mul(4);
    let code_len_u32 = u32::try_from(code_len)
        .map_err(|_| anyhow::anyhow!("f32 code length {code_len} overflows u32"))?;
    let mut out = Vec::with_capacity(CODES_HEADER_LEN + vectors.len().saturating_mul(code_len));
    out.extend_from_slice(&CODES_BLOB_MAGIC);
    out.extend_from_slice(&CODES_BLOB_VERSION.to_le_bytes());
    out.extend_from_slice(&num_vectors.to_le_bytes());
    out.extend_from_slice(&code_len_u32.to_le_bytes());
    for (idx, vec) in vectors.iter().enumerate() {
        if vec.len() != dim {
            anyhow::bail!(
                "cluster f32 codes blob requires uniform dim: expected {dim}, got {} at index {idx}",
                vec.len()
            );
        }
        for &val in vec {
            out.extend_from_slice(&val.to_le_bytes());
        }
    }
    Ok(out)
}

/// Validate and wrap a `cluster-codes` blob zero-copy.
fn parse_codes_blob(bytes: Bytes) -> Result<CachedClusterCodes> {
    if bytes.len() < CODES_HEADER_LEN {
        anyhow::bail!(
            "cluster codes blob truncated: {} bytes < header {CODES_HEADER_LEN}",
            bytes.len()
        );
    }
    if bytes[0..4] != CODES_BLOB_MAGIC {
        anyhow::bail!("cluster codes blob has invalid magic");
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != CODES_BLOB_VERSION {
        anyhow::bail!("unsupported cluster codes blob version {version}");
    }
    let num_vectors = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let code_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    let expected = CODES_HEADER_LEN
        .checked_add(
            num_vectors
                .checked_mul(code_len)
                .ok_or_else(|| anyhow::anyhow!("cluster codes blob size overflow"))?,
        )
        .ok_or_else(|| anyhow::anyhow!("cluster codes blob size overflow"))?;
    if bytes.len() != expected {
        anyhow::bail!(
            "cluster codes blob length mismatch: expected {expected} ({num_vectors} x {code_len} + {CODES_HEADER_LEN}), got {}",
            bytes.len()
        );
    }
    Ok(CachedClusterCodes {
        bytes,
        num_vectors,
        code_len,
    })
}

/// Reorder global centroids by spatial locality so that centroids frequently
/// co-probed by a query receive adjacent cluster IDs and pack into the same
/// sequential Puffin shard.
///
/// Uses deterministic recursive furthest-pair projection bisection (a high-
/// dimensional space-filling ordering in $O(k \log k \cdot d)$) with Hilbert-
/// style seam flipping so consecutive subtrees stay close across split
/// boundaries. Because centroids are ordered *before* row assignment and shard
/// partitioning, shards retain strictly monotonic `[cluster_start..=cluster_end]`
/// ranges for [`ShardDescriptor`] routing.
fn order_centroids_by_locality(centroids: &mut [Vec<f32>]) {
    if centroids.len() <= 2 {
        return;
    }
    let dim = centroids.first().map(|c| c.len()).unwrap_or(0);
    if dim == 0 {
        return;
    }

    fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
        let mut s = 0.0f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let d = x - y;
            s += d * d;
        }
        s
    }

    fn recurse(slice: &mut [Vec<f32>], dim: usize) {
        let n = slice.len();
        if n <= 2 {
            return;
        }
        // Compute mean in f64 for numerical stability.
        let mut mean = vec![0.0f64; dim];
        for c in slice.iter() {
            for (m, &v) in mean.iter_mut().zip(c.iter()) {
                *m += v as f64;
            }
        }
        let inv_n = 1.0 / (n as f64);
        let mean_f32: Vec<f32> = mean.into_iter().map(|v| (v * inv_n) as f32).collect();

        // Pivot 1: centroid furthest from the mean.
        let mut p1_idx = 0usize;
        let mut max_d1 = -1.0f32;
        for (i, c) in slice.iter().enumerate() {
            let d = sq_dist(c, &mean_f32);
            if d > max_d1 {
                max_d1 = d;
                p1_idx = i;
            }
        }

        // Pivot 2: centroid furthest from Pivot 1 (principal spread axis).
        let p1 = slice[p1_idx].clone();
        let mut p2_idx = p1_idx;
        let mut max_d2 = -1.0f32;
        for (i, c) in slice.iter().enumerate() {
            let d = sq_dist(c, &p1);
            if d > max_d2 {
                max_d2 = d;
                p2_idx = i;
            }
        }
        if max_d2 <= 0.0 {
            return;
        }

        let dir: Vec<f32> = slice[p2_idx]
            .iter()
            .zip(p1.iter())
            .map(|(&b, &a)| b - a)
            .collect();

        let mut indexed: Vec<(f32, usize)> = slice
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let proj: f32 = c
                    .iter()
                    .zip(p1.iter())
                    .zip(dir.iter())
                    .map(|((&v, &a), &d)| (v - a) * d)
                    .sum();
                (proj, i)
            })
            .collect();
        indexed.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut reordered: Vec<Vec<f32>> = Vec::with_capacity(n);
        for (_, idx) in indexed {
            reordered.push(std::mem::take(&mut slice[idx]));
        }
        for (dst, src) in slice.iter_mut().zip(reordered) {
            *dst = src;
        }

        let mid = n / 2;
        let (left, right) = slice.split_at_mut(mid);
        recurse(left, dim);
        recurse(right, dim);

        // Hilbert-style continuity across the bisection seam: flip `right` if
        // its tail is closer to `left`'s tail than its head is.
        if let (Some(l_last), Some(r_first), Some(r_last)) =
            (left.last(), right.first(), right.last())
        {
            if sq_dist(l_last, r_last) < sq_dist(l_last, r_first) {
                right.reverse();
            }
        }
    }

    recurse(centroids, dim);
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
    /// Unified byte-weighed cache for cluster graphs, flat codes blobs, and
    /// per-cluster node metadata tables.
    cache: moka::sync::Cache<ClusterCacheKey, CachedClusterEntry>,
    /// Cluster slices resolved lazily for remote shards, keyed by shard index.
    ///
    /// A remote open reads shard 0 and the shard descriptor only, so shards 1..N
    /// have no slice map when the index is constructed. The first query to probe
    /// a shard reads that shard's footer (two ranged reads) and caches the map
    /// here; later queries reuse it. A `std::sync::Mutex` is enough because the
    /// critical sections are a lookup and an insert with no await between them.
    remote_slices: std::sync::Mutex<HashMap<usize, Arc<HashMap<usize, ClusterSlices>>>>,
    /// Singleflight gates deduplicating concurrent remote footer and blob GETs.
    inflight_footers: SingleFlight<usize>,
    inflight_codes: SingleFlight<usize>,
    inflight_meta: SingleFlight<usize>,
    inflight_clusters: SingleFlight<usize>,
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

        let mut centroids = global_centroids::train_global(&sample, k, None)?;
        order_centroids_by_locality(&mut centroids);
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
                     bytes, but only {} is available. Set BDB_CLUSTER_BUILD_TMPDIR to a \
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
        //
        // Each shard stages its blobs into three disk-backed streams (`codes`,
        // `meta`, `graph`) and seals them in codes-first order:
        //   [headers (shard 0)] [all codes] [all node_meta] [all hnsw.graph]
        // Staging on disk keeps peak build RAM independent of `target_bytes`,
        // while codes-first packing puts every cluster's codes contiguously at
        // the start of the shard for single-GET S3 coalescing.
        let target_bytes = target_shard_bytes_override.unwrap_or_else(target_shard_bytes);
        let mut shards_out: Vec<ClusterShard> = Vec::new();

        struct StagedShard {
            shard_idx: usize,
            cluster_start: usize,
            cluster_end: usize,
            path: std::path::PathBuf,
            codes_path: std::path::PathBuf,
            meta_path: std::path::PathBuf,
            graph_path: std::path::PathBuf,
            codes_writer: std::io::BufWriter<std::fs::File>,
            meta_writer: std::io::BufWriter<std::fs::File>,
            graph_writer: std::io::BufWriter<std::fs::File>,
            /// `(cluster_id, codes_len, meta_len, graph_blob_type, graph_len)`
            entries: Vec<(usize, usize, usize, &'static str, usize)>,
            staged_bytes: usize,
        }

        let open_staged_shard = |shard_idx: usize| -> Result<StagedShard> {
            let path = tmp_dir.join(format!("shard_{shard_idx}.puffin"));
            let codes_path = tmp_dir.join(format!("shard_{shard_idx}_codes.stage"));
            let meta_path = tmp_dir.join(format!("shard_{shard_idx}_meta.stage"));
            let graph_path = tmp_dir.join(format!("shard_{shard_idx}_graph.stage"));
            let codes_file = std::fs::File::create(&codes_path)
                .with_context(|| format!("creating codes staging file {}", codes_path.display()))?;
            let meta_file = std::fs::File::create(&meta_path)
                .with_context(|| format!("creating meta staging file {}", meta_path.display()))?;
            let graph_file = std::fs::File::create(&graph_path)
                .with_context(|| format!("creating graph staging file {}", graph_path.display()))?;
            Ok(StagedShard {
                shard_idx,
                cluster_start: usize::MAX,
                cluster_end: 0,
                path,
                codes_path,
                meta_path,
                graph_path,
                codes_writer: std::io::BufWriter::new(codes_file),
                meta_writer: std::io::BufWriter::new(meta_file),
                graph_writer: std::io::BufWriter::new(graph_file),
                entries: Vec::new(),
                staged_bytes: 0,
            })
        };

        let seal_shard = |mut shard: StagedShard| -> Result<ClusterShard> {
            shard.codes_writer.flush()?;
            shard.meta_writer.flush()?;
            shard.graph_writer.flush()?;
            drop(shard.codes_writer);
            drop(shard.meta_writer);
            drop(shard.graph_writer);

            let out_file = std::fs::File::create(&shard.path)
                .with_context(|| format!("creating shard file {}", shard.path.display()))?;
            let mut writer = PuffinWriter::new(std::io::BufWriter::new(out_file))?;
            if shard.shard_idx == 0 {
                write_header_blobs(
                    &mut writer,
                    &centroids,
                    metric,
                    quantizer.as_ref(),
                    &segment_table,
                )?;
            }

            // 1. All codes blobs contiguously at the front of the shard.
            {
                let mut codes_reader =
                    std::io::BufReader::new(std::fs::File::open(&shard.codes_path).with_context(
                        || format!("opening codes staging file {}", shard.codes_path.display()),
                    )?);
                for &(cluster_id, codes_len, _, _, _) in &shard.entries {
                    let props = HashMap::from([
                        ("cluster_id".to_string(), cluster_id.to_string()),
                        (
                            "filename".to_string(),
                            format!("cluster_{cluster_id}.codes"),
                        ),
                    ]);
                    writer.add_blob_from_reader(
                        PUFFIN_BLOB_VECTOR_CLUSTER_CODES.to_string(),
                        Vec::new(),
                        0,
                        1,
                        codes_reader.by_ref().take(codes_len as u64),
                        props,
                    )?;
                }
            }
            let _ = std::fs::remove_file(&shard.codes_path);

            // 2. All node_meta blobs contiguously after codes.
            {
                let mut meta_reader =
                    std::io::BufReader::new(std::fs::File::open(&shard.meta_path).with_context(
                        || format!("opening meta staging file {}", shard.meta_path.display()),
                    )?);
                for &(cluster_id, _, meta_len, _, _) in &shard.entries {
                    let props = HashMap::from([
                        ("cluster_id".to_string(), cluster_id.to_string()),
                        (
                            "filename".to_string(),
                            format!("cluster_{cluster_id}.node_meta.parquet"),
                        ),
                    ]);
                    writer.add_blob_from_reader(
                        PUFFIN_BLOB_VECTOR_CLUSTER_NODE_META.to_string(),
                        Vec::new(),
                        0,
                        1,
                        meta_reader.by_ref().take(meta_len as u64),
                        props,
                    )?;
                }
            }
            let _ = std::fs::remove_file(&shard.meta_path);

            // 3. All HNSW graph blobs after node_meta.
            {
                let mut graph_reader =
                    std::io::BufReader::new(std::fs::File::open(&shard.graph_path).with_context(
                        || format!("opening graph staging file {}", shard.graph_path.display()),
                    )?);
                for &(cluster_id, _, _, graph_blob_type, graph_len) in &shard.entries {
                    let props = HashMap::from([
                        ("cluster_id".to_string(), cluster_id.to_string()),
                        (
                            "filename".to_string(),
                            format!("cluster_{cluster_id}.hnsw.graph"),
                        ),
                    ]);
                    writer.add_blob_from_reader(
                        graph_blob_type.to_string(),
                        Vec::new(),
                        0,
                        1,
                        graph_reader.by_ref().take(graph_len as u64),
                        props,
                    )?;
                }
            }
            let _ = std::fs::remove_file(&shard.graph_path);

            let mut buf_writer = writer.finish()?;
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

        let mut current_shard = open_staged_shard(0)?;

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
                if !current_shard.entries.is_empty() && current_shard.staged_bytes >= target_bytes {
                    let next_idx = current_shard.shard_idx + 1;
                    let next_shard = match open_staged_shard(next_idx) {
                        Ok(s) => s,
                        Err(e) => {
                            build_result = Err(e);
                            break 'windows;
                        }
                    };
                    let old_shard = std::mem::replace(&mut current_shard, next_shard);
                    let sealed = match seal_shard(old_shard) {
                        Ok(s) => s,
                        Err(e) => {
                            build_result = Err(e);
                            break 'windows;
                        }
                    };
                    shards_out.push(sealed);
                }

                if let Err(e) = current_shard
                    .codes_writer
                    .write_all(&blobs.codes_bytes)
                    .and_then(|_| current_shard.meta_writer.write_all(&blobs.meta_bytes))
                    .and_then(|_| current_shard.graph_writer.write_all(&blobs.graph_bytes))
                {
                    build_result = Err(anyhow::Error::from(e).context("staging cluster blobs"));
                    break 'windows;
                }

                current_shard.staged_bytes +=
                    blobs.codes_bytes.len() + blobs.meta_bytes.len() + blobs.graph_bytes.len();
                current_shard.entries.push((
                    cluster_id,
                    blobs.codes_bytes.len(),
                    blobs.meta_bytes.len(),
                    blobs.graph_blob_type,
                    blobs.graph_bytes.len(),
                ));

                if current_shard.cluster_start == usize::MAX {
                    current_shard.cluster_start = cluster_id;
                }
                current_shard.cluster_end = cluster_id;
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
        let mut centroids = global_centroids::train_global(&vectors, k, None)?;
        order_centroids_by_locality(&mut centroids);
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
            pending: Vec<BuiltClusterBlobs>,
            staged_bytes: usize,
        }

        let seal_mem_shard = |shard: ActiveShard| -> Result<ClusterShard> {
            let mut writer = PuffinWriter::new(Cursor::new(Vec::new()))?;
            if shard.shard_idx == 0 {
                write_header_blobs(
                    &mut writer,
                    &centroids,
                    metric,
                    quantizer.as_ref(),
                    &segment_table,
                )?;
            }
            write_shard_blobs_codes_first(&mut writer, &shard.pending)?;
            let container = Bytes::from(writer.finish()?.into_inner());
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

        let mut current_shard = ActiveShard {
            shard_idx: 0,
            cluster_start: usize::MAX,
            cluster_end: 0,
            pending: Vec::new(),
            staged_bytes: 0,
        };

        for cluster_id in sorted_cluster_ids {
            let members = &grouped[&cluster_id];
            if members.is_empty() {
                continue;
            }

            if !current_shard.pending.is_empty() && current_shard.staged_bytes >= target_bytes {
                let next_idx = current_shard.shard_idx + 1;
                let old_shard = std::mem::replace(
                    &mut current_shard,
                    ActiveShard {
                        shard_idx: next_idx,
                        cluster_start: usize::MAX,
                        cluster_end: 0,
                        pending: Vec::new(),
                        staged_bytes: 0,
                    },
                );
                shards_out.push(seal_mem_shard(old_shard)?);
            }

            if let Some(blobs) = build_cluster_blobs(
                cluster_id,
                members,
                quantizer.as_ref(),
                metric,
                m,
                ef_construction,
            )? {
                current_shard.staged_bytes +=
                    blobs.codes_bytes.len() + blobs.meta_bytes.len() + blobs.graph_bytes.len();
                current_shard.pending.push(blobs);
                if current_shard.cluster_start == usize::MAX {
                    current_shard.cluster_start = cluster_id;
                }
                current_shard.cluster_end = cluster_id;
            }
        }

        shards_out.push(seal_mem_shard(current_shard)?);

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
                .weigher(|_key: &ClusterCacheKey, value: &CachedClusterEntry| {
                    cache_entry_kib(value)
                })
                .max_capacity(cluster_cache_budget_kib())
                .build(),
            remote_slices: std::sync::Mutex::new(resolved),
            inflight_footers: SingleFlight::default(),
            inflight_codes: SingleFlight::default(),
            inflight_meta: SingleFlight::default(),
            inflight_clusters: SingleFlight::default(),
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
}

/// Candidate tracked in the codes-only first-stage max-heap.
///
/// Ordered by ascending distance so [`BinaryHeap::peek`] exposes the worst
/// (largest) distance currently in the top-`k` set.
#[derive(Debug, Clone, Copy)]
struct TopKCandidate {
    dist: f32,
    cluster_id: usize,
    local_id: usize,
}

impl PartialEq for TopKCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist.to_bits() == other.dist.to_bits()
            && self.cluster_id == other.cluster_id
            && self.local_id == other.local_id
    }
}

impl Eq for TopKCandidate {}

impl PartialOrd for TopKCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TopKCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.cluster_id.cmp(&other.cluster_id))
            .then_with(|| self.local_id.cmp(&other.local_id))
    }
}

#[inline]
fn push_top_k(heap: &mut BinaryHeap<TopKCandidate>, k: usize, cand: TopKCandidate) -> f32 {
    if heap.len() < k {
        heap.push(cand);
    } else if let Some(worst) = heap.peek() {
        if cand.dist < worst.dist {
            heap.pop();
            heap.push(cand);
        }
    }
    if heap.len() >= k {
        heap.peek().map(|c| c.dist).unwrap_or(f32::INFINITY)
    } else {
        f32::INFINITY
    }
}

#[inline]
fn heap_threshold(heap: &BinaryHeap<TopKCandidate>, k: usize) -> f32 {
    if heap.len() >= k {
        heap.peek().map(|c| c.dist).unwrap_or(f32::INFINITY)
    } else {
        f32::INFINITY
    }
}

impl ClusterIndex {
    /// Search the table-level index synchronously.
    ///
    /// Uses the codes-only first stage with pipelined local prefetch whenever
    /// the artifact carries `cluster-codes` blobs (falling back transparently to
    /// HNSW graph traversal on legacy artifacts or when
    /// `BDB_CLUSTER_SEARCH_MODE=graph`).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        n_probe: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>> {
        self.search_with_mode(query, k, n_probe, ef_search, configured_search_mode())
    }

    /// Search synchronously with an explicit [`ClusterSearchMode`].
    pub fn search_with_mode(
        &self,
        query: &[f32],
        k: usize,
        n_probe: usize,
        ef_search: Option<usize>,
        mode: ClusterSearchMode,
    ) -> Result<Vec<ScoredResult>> {
        if k == 0 || self.centroids.is_empty() || query.len() != self.dim {
            return Ok(Vec::new());
        }

        let probe_ids = self.rank_probes(query, n_probe);
        let prepared_query = self
            .quantizer
            .as_ref()
            .map(|q| q.prepare_query(query, self.metric));
        let fallback_query = VectorValue::Float32(query.to_vec());
        let ef = ef_search.unwrap_or(40);
        let cluster_k = (k * 4).max(ef.min(k * 8));
        let prefer_codes = !matches!(mode, ClusterSearchMode::Graph);

        let mut codes_heap: BinaryHeap<TopKCandidate> = BinaryHeap::with_capacity(k + 1);
        let mut graph_candidates: Vec<ScoredResult> = Vec::new();
        let mut f32_scratch: Vec<f32> = Vec::new();

        // Pipelined local prefetch: advise batch `i + 1` before scanning batch `i`,
        // overlapping NVMe readahead with CPU scoring.
        let batches: Vec<&[usize]> = probe_ids.chunks(LOCAL_PREFETCH_BATCH).collect();
        if let Some(first_batch) = batches.first() {
            self.prefetch_batch(first_batch, prefer_codes);
        }

        for (batch_idx, batch) in batches.iter().enumerate() {
            if let Some(next_batch) = batches.get(batch_idx + 1) {
                self.prefetch_batch(next_batch, prefer_codes);
            }
            for &cluster_id in *batch {
                self.score_probed_cluster(
                    cluster_id,
                    query,
                    prepared_query.as_ref(),
                    &fallback_query,
                    k,
                    cluster_k,
                    ef,
                    prefer_codes,
                    &mut codes_heap,
                    &mut graph_candidates,
                    &mut f32_scratch,
                );
            }
        }

        // Phase 3 (selective metadata resolution): only load `node_meta` for the
        // small set of clusters that actually produced a top-k candidate.
        if !codes_heap.is_empty() {
            let winning_clusters = unique_winning_clusters(&codes_heap);
            self.prefetch_node_meta(&winning_clusters);
            self.resolve_codes_winners(codes_heap, &mut graph_candidates);
        }

        Ok(self.finalize_candidates(graph_candidates, k))
    }

    /// Search the table-level index asynchronously with full S3-compatible
    /// pipelining, singleflight request deduplication, and selective metadata
    /// hydration.
    ///
    /// On local stores (`local_root.is_some()`), delegates straight to the
    /// synchronous local pipelined path. On remote stores (`local_root.is_none()`),
    /// pipelines batches of codes-only ranged GETs against CPU scanning, then
    /// fetches `node_meta` only for the winning clusters that contributed to the
    /// top-`k` results.
    pub async fn search_async(
        self: &Arc<Self>,
        store: &std::sync::Arc<dyn object_store::ObjectStore>,
        query: &[f32],
        k: usize,
        n_probe: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>> {
        self.search_async_with_mode(
            store,
            query,
            k,
            n_probe,
            ef_search,
            configured_search_mode(),
        )
        .await
    }

    /// Async search with an explicit [`ClusterSearchMode`].
    pub async fn search_async_with_mode(
        self: &Arc<Self>,
        store: &std::sync::Arc<dyn object_store::ObjectStore>,
        query: &[f32],
        k: usize,
        n_probe: usize,
        ef_search: Option<usize>,
        mode: ClusterSearchMode,
    ) -> Result<Vec<ScoredResult>> {
        if k == 0 || self.centroids.is_empty() || query.len() != self.dim {
            return Ok(Vec::new());
        }

        if self.local_root.is_some() {
            return self.search_with_mode(query, k, n_probe, ef_search, mode);
        }

        let probe_ids = self.rank_probes(query, n_probe);
        let prepared_query = self
            .quantizer
            .as_ref()
            .map(|q| q.prepare_query(query, self.metric));
        let fallback_query = VectorValue::Float32(query.to_vec());
        let ef = ef_search.unwrap_or(40);
        let cluster_k = (k * 4).max(ef.min(k * 8));
        let prefer_codes = !matches!(mode, ClusterSearchMode::Graph);

        let mut codes_heap: BinaryHeap<TopKCandidate> = BinaryHeap::with_capacity(k + 1);
        let mut graph_candidates: Vec<ScoredResult> = Vec::new();
        let mut f32_scratch: Vec<f32> = Vec::new();

        // Phase 1 + 2: Pipelined remote hydration and CPU scoring.
        // Spawn the next batch's S3 ranged GETs on the Tokio runtime before
        // scanning the current batch on CPU so network RTT overlaps scoring.
        let batches: Vec<Vec<usize>> = probe_ids
            .chunks(REMOTE_PIPELINE_BATCH)
            .map(|c| c.to_vec())
            .collect();

        let mut next_prefetch: Option<tokio::task::JoinHandle<()>> =
            if let Some(first) = batches.first() {
                let idx = Arc::clone(self);
                let st = Arc::clone(store);
                let ids = first.clone();
                Some(tokio::spawn(async move {
                    idx.hydrate_probe_batch(&st, &ids, prefer_codes).await;
                }))
            } else {
                None
            };

        for (batch_idx, batch) in batches.iter().enumerate() {
            if let Some(handle) = next_prefetch.take() {
                let _ = handle.await;
            }
            if let Some(next_ids) = batches.get(batch_idx + 1) {
                let idx = Arc::clone(self);
                let st = Arc::clone(store);
                let ids = next_ids.clone();
                next_prefetch = Some(tokio::spawn(async move {
                    idx.hydrate_probe_batch(&st, &ids, prefer_codes).await;
                }));
            }

            for &cluster_id in batch {
                self.score_probed_cluster(
                    cluster_id,
                    query,
                    prepared_query.as_ref(),
                    &fallback_query,
                    k,
                    cluster_k,
                    ef,
                    prefer_codes,
                    &mut codes_heap,
                    &mut graph_candidates,
                    &mut f32_scratch,
                );
            }
        }

        // Phase 3: Selective `node_meta` hydration for winning clusters only.
        if !codes_heap.is_empty() {
            let winning_clusters = unique_winning_clusters(&codes_heap);
            self.hydrate_node_meta(store, &winning_clusters).await;
            self.resolve_codes_winners(codes_heap, &mut graph_candidates);
        }

        Ok(self.finalize_candidates(graph_candidates, k))
    }

    /// Score a single probed cluster via codes-only linear scan when available,
    /// falling back to HNSW graph traversal on legacy clusters or `Graph` mode.
    #[allow(clippy::too_many_arguments)]
    fn score_probed_cluster(
        &self,
        cluster_id: usize,
        query: &[f32],
        prepared_query: Option<&PreparedQuantizedQuery<'_>>,
        fallback_query: &VectorValue,
        k: usize,
        cluster_k: usize,
        ef: usize,
        prefer_codes: bool,
        codes_heap: &mut BinaryHeap<TopKCandidate>,
        graph_candidates: &mut Vec<ScoredResult>,
        f32_scratch: &mut Vec<f32>,
    ) {
        if prefer_codes {
            if let Some(codes) = self.get_codes(cluster_id) {
                self.scan_cluster_codes(
                    cluster_id,
                    &codes,
                    query,
                    prepared_query,
                    k,
                    codes_heap,
                    f32_scratch,
                );
                return;
            }
        }

        let Some(cluster) = self.get_cluster(cluster_id) else {
            return;
        };
        let graph = &cluster.graph;
        let node_meta = &cluster.node_meta;
        // The vectors this query could return from this cluster. Summed over
        // the probe set it is the coverage denominator, which is what makes
        // two clusterings comparable at equal work.
        stats::CLUSTER_CORPUS_ROWS.fetch_add(node_meta.len() as u64, AtomicOrdering::Relaxed);
        let cluster_ef = ef.min((node_meta.len() * 3) / 4).max(cluster_k);
        for neighbour in
            graph.search_with_quantizer(fallback_query, prepared_query, cluster_k, cluster_ef, None)
        {
            let local_id = neighbour.d_id;
            let Some(meta) = node_meta.get(local_id) else {
                continue;
            };
            let Some(segment_id) = self.segment_table.get(meta.segment_idx as usize) else {
                continue;
            };
            graph_candidates.push(ScoredResult {
                segment_id: segment_id.clone(),
                row_id: meta.row_id,
                score: neighbour.distance,
            });
        }
    }

    /// Linearly scan a cluster's flat `codes` blob into the global top-`k` heap.
    #[allow(clippy::too_many_arguments)]
    fn scan_cluster_codes(
        &self,
        cluster_id: usize,
        codes: &CachedClusterCodes,
        query: &[f32],
        prepared_query: Option<&PreparedQuantizedQuery<'_>>,
        k: usize,
        heap: &mut BinaryHeap<TopKCandidate>,
        f32_scratch: &mut Vec<f32>,
    ) {
        stats::CLUSTER_CORPUS_ROWS.fetch_add(codes.num_vectors as u64, AtomicOrdering::Relaxed);
        let code_len = codes.code_len;
        if codes.num_vectors == 0 || code_len == 0 {
            return;
        }
        let data = codes.data();

        match prepared_query {
            Some(pq) => {
                let initial_thresh = heap_threshold(heap, k);
                pq.scan_codes(data, code_len, initial_thresh, |local_id, dist| {
                    push_top_k(
                        heap,
                        k,
                        TopKCandidate {
                            dist,
                            cluster_id,
                            local_id,
                        },
                    )
                });
            }
            None => {
                let expected_bytes = self.dim.saturating_mul(4);
                if code_len != expected_bytes {
                    return;
                }
                let mut threshold = heap_threshold(heap, k);
                let is_l2 = self.metric == VectorMetric::L2;
                let to_sq_thresh = |t: f32| -> f32 {
                    if !t.is_finite() {
                        f32::INFINITY
                    } else {
                        t * t * (1.0 + 1e-4) + 1e-6
                    }
                };
                let mut sq_thresh = to_sq_thresh(threshold);
                #[cfg(target_endian = "little")]
                {
                    if let Ok(all_f32) = bytemuck::try_cast_slice::<u8, f32>(data) {
                        if is_l2 {
                            for (local_id, vec_slice) in all_f32.chunks_exact(self.dim).enumerate()
                            {
                                let dist_sq = crate::core::index::distance::l2_distance_squared(
                                    query, vec_slice,
                                );
                                if dist_sq <= sq_thresh {
                                    let dist = dist_sq.max(0.0).sqrt();
                                    if dist < threshold {
                                        threshold = push_top_k(
                                            heap,
                                            k,
                                            TopKCandidate {
                                                dist,
                                                cluster_id,
                                                local_id,
                                            },
                                        );
                                        sq_thresh = to_sq_thresh(threshold);
                                    }
                                }
                            }
                        } else {
                            for (local_id, vec_slice) in all_f32.chunks_exact(self.dim).enumerate()
                            {
                                let dist = metric_distance(self.metric, query, vec_slice);
                                if dist < threshold {
                                    threshold = push_top_k(
                                        heap,
                                        k,
                                        TopKCandidate {
                                            dist,
                                            cluster_id,
                                            local_id,
                                        },
                                    );
                                }
                            }
                        }
                        return;
                    }
                }
                f32_scratch.resize(self.dim, 0.0);
                for (local_id, raw) in data.chunks_exact(code_len).enumerate() {
                    let (chunks, _) = raw.as_chunks::<4>();
                    for (dst, &chunk) in f32_scratch.iter_mut().zip(chunks) {
                        *dst = f32::from_le_bytes(chunk);
                    }
                    if is_l2 {
                        let dist_sq =
                            crate::core::index::distance::l2_distance_squared(query, f32_scratch);
                        if dist_sq <= sq_thresh {
                            let dist = dist_sq.max(0.0).sqrt();
                            if dist < threshold {
                                threshold = push_top_k(
                                    heap,
                                    k,
                                    TopKCandidate {
                                        dist,
                                        cluster_id,
                                        local_id,
                                    },
                                );
                                sq_thresh = to_sq_thresh(threshold);
                            }
                        }
                    } else {
                        let dist = metric_distance(self.metric, query, f32_scratch);
                        if dist < threshold {
                            threshold = push_top_k(
                                heap,
                                k,
                                TopKCandidate {
                                    dist,
                                    cluster_id,
                                    local_id,
                                },
                            );
                        }
                    }
                }
            }
        }
    }

    /// Resolve top-`k` codes-only candidates to `(segment_id, row_id)` using
    /// per-cluster `node_meta` loaded only for the winning clusters.
    fn resolve_codes_winners(&self, heap: BinaryHeap<TopKCandidate>, out: &mut Vec<ScoredResult>) {
        let mut by_cluster: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for cand in heap {
            by_cluster
                .entry(cand.cluster_id)
                .or_default()
                .push((cand.local_id, cand.dist));
        }
        for (cluster_id, items) in by_cluster {
            let Some(node_meta) = self.get_node_meta(cluster_id) else {
                continue;
            };
            for (local_id, dist) in items {
                let Some(meta) = node_meta.entries.get(local_id) else {
                    continue;
                };
                let Some(segment_id) = self.segment_table.get(meta.segment_idx as usize) else {
                    continue;
                };
                out.push(ScoredResult {
                    segment_id: segment_id.clone(),
                    row_id: meta.row_id,
                    score: dist,
                });
            }
        }
    }

    fn finalize_candidates(
        &self,
        mut candidates: Vec<ScoredResult>,
        k: usize,
    ) -> Vec<ScoredResult> {
        candidates.sort_by(|a, b| a.score.total_cmp(&b.score));
        candidates.dedup_by(|a, b| a.segment_id == b.segment_id && a.row_id == b.row_id);
        candidates.truncate(k);

        // The table level tier builds no `RowSelection`: it resolves candidates
        // from the shard blobs and the caller materialises exactly those rows.
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
        candidates
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
        cluster_distances.sort_by(|a, b| a.1.total_cmp(&b.1));
        let probes = n_probe.clamp(1, cluster_distances.len());
        cluster_distances
            .into_iter()
            .take(probes)
            .map(|(cluster_id, _)| cluster_id)
            .collect()
    }

    /// Fetch this query's probe clusters from a remote store into the cache.
    ///
    /// For callers that invoke `hydrate_for_query` followed by synchronous
    /// `search`, this hydrates codes + metadata (skipping the HNSW graph blob)
    /// whenever the artifact carries `cluster-codes` blobs, and falls back to
    /// graph + metadata on legacy artifacts or `BDB_CLUSTER_SEARCH_MODE=graph`.
    /// Callers with an `Arc<ClusterIndex>` should prefer [`Self::search_async`],
    /// which pipelines codes hydration and skips `node_meta` for non-winning
    /// clusters altogether.
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
    /// Fetches touched shards concurrently with singleflight request
    /// deduplication across concurrent queries. When `cluster-codes` blobs are
    /// present and `BDB_CLUSTER_SEARCH_MODE != graph`, fetches codes and
    /// `node_meta` only (never the HNSW graph).
    pub async fn hydrate_clusters(
        &self,
        store: &std::sync::Arc<dyn object_store::ObjectStore>,
        cluster_ids: &[usize],
    ) {
        let prefer_codes = !matches!(configured_search_mode(), ClusterSearchMode::Graph);
        self.hydrate_probe_batch(store, cluster_ids, prefer_codes)
            .await;
        if prefer_codes {
            // Ensure `node_meta` is resident for any codes-only cluster so a
            // subsequent synchronous `search()` can resolve winners immediately.
            let codes_clusters: Vec<usize> = cluster_ids
                .iter()
                .copied()
                .filter(|cid| self.cache.contains_key(&ClusterCacheKey::Codes(*cid)))
                .collect();
            if !codes_clusters.is_empty() {
                self.hydrate_node_meta(store, &codes_clusters).await;
            }
        }
    }

    /// Hydrate one probe batch from a remote store: fetches ONLY `codes` blobs
    /// for clusters carrying split codes (when `prefer_codes` is true) and full
    /// `graph + node_meta` only for legacy clusters or `Graph` mode.
    async fn hydrate_probe_batch(
        &self,
        store: &std::sync::Arc<dyn object_store::ObjectStore>,
        cluster_ids: &[usize],
        prefer_codes: bool,
    ) {
        let mut per_shard: HashMap<usize, Vec<usize>> = HashMap::new();
        for &cluster_id in cluster_ids {
            let hit = if prefer_codes {
                self.cache.contains_key(&ClusterCacheKey::Codes(cluster_id))
                    || self.cache.contains_key(&ClusterCacheKey::Graph(cluster_id))
            } else {
                self.cache.contains_key(&ClusterCacheKey::Graph(cluster_id))
            };
            if hit {
                stats::CLUSTER_CACHE_HITS.fetch_add(1, AtomicOrdering::Relaxed);
                continue;
            }
            if let Some(&shard_idx) = self.cluster_to_shard.get(&cluster_id) {
                per_shard.entry(shard_idx).or_default().push(cluster_id);
            }
        }

        if per_shard.is_empty() {
            self.record_cache_gauges();
            return;
        }

        futures::stream::iter(per_shard)
            .for_each_concurrent(
                Some(REMOTE_SHARD_CONCURRENCY),
                |(shard_idx, ids)| async move {
                    self.hydrate_shard_probe_ids(store, shard_idx, ids, prefer_codes)
                        .await;
                },
            )
            .await;

        self.record_cache_gauges();
    }

    async fn hydrate_shard_probe_ids(
        &self,
        store: &std::sync::Arc<dyn object_store::ObjectStore>,
        shard_idx: usize,
        ids: Vec<usize>,
        prefer_codes: bool,
    ) {
        let Some(shard) = self.shards.get(shard_idx) else {
            return;
        };
        let Some(path) = shard.file_path.as_deref().map(str::to_string) else {
            return;
        };
        let Some(slices) = self.shard_slices(store, shard_idx).await else {
            return;
        };

        let mut codes_ids = Vec::new();
        let mut graph_ids = Vec::new();
        for cluster_id in ids {
            let Some(cs) = slices.get(&cluster_id) else {
                continue;
            };
            if prefer_codes && cs.codes.is_some() {
                codes_ids.push(cluster_id);
            } else {
                graph_ids.push(cluster_id);
            }
        }

        if !codes_ids.is_empty() {
            self.hydrate_shard_codes(store, shard_idx, &path, &slices, codes_ids)
                .await;
        }
        if !graph_ids.is_empty() {
            self.hydrate_shard_graphs(store, shard_idx, &path, &slices, graph_ids)
                .await;
        }
    }

    /// Fetch `codes` blobs for `ids` within one shard using coalesced ranged GETs
    /// and singleflight deduplication.
    async fn hydrate_shard_codes(
        &self,
        store: &std::sync::Arc<dyn object_store::ObjectStore>,
        shard_idx: usize,
        path: &str,
        slices: &HashMap<usize, ClusterSlices>,
        ids: Vec<usize>,
    ) {
        let mut claimed_ids = Vec::new();
        let mut guards = Vec::new();
        let mut peers = Vec::new();

        for cluster_id in ids {
            if self.cache.contains_key(&ClusterCacheKey::Codes(cluster_id)) {
                stats::CLUSTER_CACHE_HITS.fetch_add(1, AtomicOrdering::Relaxed);
                continue;
            }
            match self.inflight_codes.try_claim(cluster_id) {
                SingleFlightClaim::Leader(guard) => {
                    // Re-check cache after claiming to close any check-then-claim race.
                    if self.cache.contains_key(&ClusterCacheKey::Codes(cluster_id)) {
                        drop(guard);
                        stats::CLUSTER_CACHE_HITS.fetch_add(1, AtomicOrdering::Relaxed);
                    } else {
                        claimed_ids.push(cluster_id);
                        guards.push(guard);
                    }
                }
                SingleFlightClaim::Follower(rx) => peers.push(rx),
            }
        }

        if !claimed_ids.is_empty() {
            let mut spans = Vec::with_capacity(claimed_ids.len());
            for &cluster_id in &claimed_ids {
                if let Some(codes_range) = slices.get(&cluster_id).and_then(|s| s.codes) {
                    spans.push(codes_range);
                }
            }
            let object = object_store::path::Path::from(path);
            for (offset, len) in coalesce_spans(spans, PREFETCH_MERGE_GAP) {
                let range = offset as u64..(offset + len) as u64;
                let bytes = match store.get_range(&object, range).await {
                    Ok(bytes) => {
                        stats::CLUSTER_RANGE_GETS.fetch_add(1, AtomicOrdering::Relaxed);
                        stats::CLUSTER_RANGE_BYTES
                            .fetch_add(bytes.len() as u64, AtomicOrdering::Relaxed);
                        bytes
                    }
                    Err(e) => {
                        tracing::warn!(shard_idx, offset, len, error = %e, "cluster_index: codes ranged read failed");
                        continue;
                    }
                };
                for &cluster_id in &claimed_ids {
                    if self.cache.contains_key(&ClusterCacheKey::Codes(cluster_id)) {
                        continue;
                    }
                    let Some(codes_range) = slices.get(&cluster_id).and_then(|s| s.codes) else {
                        continue;
                    };
                    if !span_covers(offset, len, codes_range) {
                        continue;
                    }
                    let codes_bytes =
                        bytes.slice(codes_range.0 - offset..codes_range.0 - offset + codes_range.1);
                    self.cache_codes_bytes(cluster_id, codes_bytes);
                }
            }
        }

        // Release all owned guards BEFORE awaiting peers to prevent deadlocks.
        drop(guards);
        await_singleflight_peers(peers).await;
    }

    /// Fetch `node_meta` blobs only for the specified winning clusters across
    /// remote shards, concurrently and singleflight-deduplicated.
    async fn hydrate_node_meta(
        &self,
        store: &std::sync::Arc<dyn object_store::ObjectStore>,
        cluster_ids: &[usize],
    ) {
        let mut per_shard: HashMap<usize, Vec<usize>> = HashMap::new();
        for &cluster_id in cluster_ids {
            if self
                .cache
                .contains_key(&ClusterCacheKey::NodeMeta(cluster_id))
                || self.cache.contains_key(&ClusterCacheKey::Graph(cluster_id))
            {
                stats::CLUSTER_CACHE_HITS.fetch_add(1, AtomicOrdering::Relaxed);
                continue;
            }
            if let Some(&shard_idx) = self.cluster_to_shard.get(&cluster_id) {
                per_shard.entry(shard_idx).or_default().push(cluster_id);
            }
        }

        if per_shard.is_empty() {
            self.record_cache_gauges();
            return;
        }

        futures::stream::iter(per_shard)
            .for_each_concurrent(Some(REMOTE_SHARD_CONCURRENCY), |(shard_idx, ids)| async move {
                let Some(shard) = self.shards.get(shard_idx) else {
                    return;
                };
                let Some(path) = shard.file_path.as_deref().map(str::to_string) else {
                    return;
                };
                let Some(slices) = self.shard_slices(store, shard_idx).await else {
                    return;
                };

                let mut claimed_ids = Vec::new();
                let mut guards = Vec::new();
                let mut peers = Vec::new();

                for cluster_id in ids {
                    if self
                        .cache
                        .contains_key(&ClusterCacheKey::NodeMeta(cluster_id))
                    {
                        stats::CLUSTER_CACHE_HITS.fetch_add(1, AtomicOrdering::Relaxed);
                        continue;
                    }
                    match self.inflight_meta.try_claim(cluster_id) {
                        SingleFlightClaim::Leader(guard) => {
                            if self
                                .cache
                                .contains_key(&ClusterCacheKey::NodeMeta(cluster_id))
                            {
                                drop(guard);
                                stats::CLUSTER_CACHE_HITS.fetch_add(1, AtomicOrdering::Relaxed);
                            } else {
                                claimed_ids.push(cluster_id);
                                guards.push(guard);
                            }
                        }
                        SingleFlightClaim::Follower(rx) => peers.push(rx),
                    }
                }

                if !claimed_ids.is_empty() {
                    let mut spans = Vec::with_capacity(claimed_ids.len());
                    for &cluster_id in &claimed_ids {
                        if let Some(cs) = slices.get(&cluster_id) {
                            spans.push(cs.node_meta);
                        }
                    }
                    let object = object_store::path::Path::from(path.as_str());
                    for (offset, len) in coalesce_spans(spans, PREFETCH_MERGE_GAP) {
                        let range = offset as u64..(offset + len) as u64;
                        let bytes = match store.get_range(&object, range).await {
                            Ok(bytes) => {
                                stats::CLUSTER_RANGE_GETS.fetch_add(1, AtomicOrdering::Relaxed);
                                stats::CLUSTER_RANGE_BYTES
                                    .fetch_add(bytes.len() as u64, AtomicOrdering::Relaxed);
                                bytes
                            }
                            Err(e) => {
                                tracing::warn!(shard_idx, offset, len, error = %e, "cluster_index: node_meta ranged read failed");
                                continue;
                            }
                        };
                        for &cluster_id in &claimed_ids {
                            if self
                                .cache
                                .contains_key(&ClusterCacheKey::NodeMeta(cluster_id))
                            {
                                continue;
                            }
                            let Some(cs) = slices.get(&cluster_id) else {
                                continue;
                            };
                            if !span_covers(offset, len, cs.node_meta) {
                                continue;
                            }
                            let meta_bytes = bytes.slice(
                                cs.node_meta.0 - offset
                                    ..cs.node_meta.0 - offset + cs.node_meta.1,
                            );
                            self.cache_node_meta_bytes(cluster_id, meta_bytes);
                        }
                    }
                }

                drop(guards);
                await_singleflight_peers(peers).await;
            })
            .await;

        self.record_cache_gauges();
    }

    /// Fetch full graph + node_meta blobs for legacy clusters or `Graph` mode
    /// within one shard using coalesced ranged GETs and singleflight.
    async fn hydrate_shard_graphs(
        &self,
        store: &std::sync::Arc<dyn object_store::ObjectStore>,
        shard_idx: usize,
        path: &str,
        slices: &HashMap<usize, ClusterSlices>,
        ids: Vec<usize>,
    ) {
        let mut claimed_ids = Vec::new();
        let mut guards = Vec::new();
        let mut peers = Vec::new();

        for cluster_id in ids {
            if self.cache.contains_key(&ClusterCacheKey::Graph(cluster_id)) {
                stats::CLUSTER_CACHE_HITS.fetch_add(1, AtomicOrdering::Relaxed);
                continue;
            }
            match self.inflight_clusters.try_claim(cluster_id) {
                SingleFlightClaim::Leader(guard) => {
                    if self.cache.contains_key(&ClusterCacheKey::Graph(cluster_id)) {
                        drop(guard);
                        stats::CLUSTER_CACHE_HITS.fetch_add(1, AtomicOrdering::Relaxed);
                    } else {
                        claimed_ids.push(cluster_id);
                        guards.push(guard);
                    }
                }
                SingleFlightClaim::Follower(rx) => peers.push(rx),
            }
        }

        if !claimed_ids.is_empty() {
            let mut spans = Vec::with_capacity(claimed_ids.len() * 2);
            for &cluster_id in &claimed_ids {
                if let Some(cs) = slices.get(&cluster_id) {
                    spans.push(cs.graph);
                    spans.push(cs.node_meta);
                }
            }
            let object = object_store::path::Path::from(path);
            for (offset, len) in coalesce_spans(spans, PREFETCH_MERGE_GAP) {
                let range = offset as u64..(offset + len) as u64;
                let bytes = match store.get_range(&object, range).await {
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
                for &cluster_id in &claimed_ids {
                    if self.cache.contains_key(&ClusterCacheKey::Graph(cluster_id)) {
                        continue;
                    }
                    let Some(cs) = slices.get(&cluster_id) else {
                        continue;
                    };
                    if !span_covers(offset, len, cs.graph)
                        || !span_covers(offset, len, cs.node_meta)
                    {
                        continue;
                    }
                    let graph_bytes =
                        bytes.slice(cs.graph.0 - offset..cs.graph.0 - offset + cs.graph.1);
                    let meta_bytes = bytes
                        .slice(cs.node_meta.0 - offset..cs.node_meta.0 - offset + cs.node_meta.1);
                    self.cache_deserialized(cluster_id, graph_bytes, meta_bytes);
                }
            }

            // Fallback for clusters whose graph and metadata landed in separate spans
            // (which is the normal case in codes-first shards when running in Graph mode).
            for &cluster_id in &claimed_ids {
                if self.cache.contains_key(&ClusterCacheKey::Graph(cluster_id)) {
                    continue;
                }
                let Some(cs) = slices.get(&cluster_id) else {
                    continue;
                };
                let fetch = |range: (usize, usize)| {
                    store.get_range(&object, range.0 as u64..(range.0 + range.1) as u64)
                };
                let (graph_res, meta_res) =
                    futures::future::join(fetch(cs.graph), fetch(cs.node_meta)).await;
                let graph_bytes = match graph_res {
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
                let meta_bytes = match meta_res {
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
                self.cache_deserialized(cluster_id, graph_bytes, meta_bytes);
            }
        }

        drop(guards);
        await_singleflight_peers(peers).await;
    }

    fn record_cache_gauges(&self) {
        stats::CLUSTER_CACHE_ENTRIES.store(self.cache.entry_count(), AtomicOrdering::Relaxed);
        stats::CLUSTER_CACHE_WEIGHTED_KIB
            .store(self.cache.weighted_size(), AtomicOrdering::Relaxed);
    }

    /// Cluster slices for `shard_idx`, reading that shard's footer on first use
    /// with singleflight deduplication so concurrent queries issue one footer
    /// read instead of one per thread.
    async fn shard_slices(
        &self,
        store: &std::sync::Arc<dyn object_store::ObjectStore>,
        shard_idx: usize,
    ) -> Option<Arc<HashMap<usize, ClusterSlices>>> {
        let lookup = || {
            self.remote_slices
                .lock()
                .ok()
                .and_then(|map| map.get(&shard_idx).cloned())
        };
        if let Some(slices) = lookup() {
            return Some(slices);
        }

        match self.inflight_footers.try_claim(shard_idx) {
            SingleFlightClaim::Leader(_guard) => {
                if let Some(slices) = lookup() {
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
            SingleFlightClaim::Follower(mut rx) => {
                let _ = rx.wait_for(|&done| done).await;
                lookup()
            }
        }
    }

    /// Deserialize one cluster's full graph + metadata into the unified cache.
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
            ClusterCacheKey::Graph(cluster_id),
            CachedClusterEntry::Graph(Arc::new(CachedCluster {
                graph,
                node_meta,
                bytes,
            })),
        );
    }

    /// Validate and insert one cluster's zero-copy flat codes blob into the cache.
    fn cache_codes_bytes(&self, cluster_id: usize, codes_bytes: Bytes) {
        match parse_codes_blob(codes_bytes) {
            Ok(parsed) => {
                self.cache.insert(
                    ClusterCacheKey::Codes(cluster_id),
                    CachedClusterEntry::Codes(Arc::new(parsed)),
                );
            }
            Err(e) => {
                tracing::warn!(cluster_id, error = %e, "cluster_index: codes blob parse failed");
            }
        }
    }

    /// Parse and insert one cluster's node metadata table into the cache.
    fn cache_node_meta_bytes(&self, cluster_id: usize, meta_bytes: Bytes) {
        let raw_len = meta_bytes.len();
        match node_meta_from_parquet(meta_bytes) {
            Ok(entries) => {
                let weight = (entries
                    .len()
                    .saturating_mul(std::mem::size_of::<NodeMeta>()))
                .max(raw_len);
                self.cache.insert(
                    ClusterCacheKey::NodeMeta(cluster_id),
                    CachedClusterEntry::NodeMeta(Arc::new(CachedNodeMeta {
                        entries,
                        bytes: weight,
                    })),
                );
            }
            Err(e) => {
                tracing::warn!(cluster_id, error = %e, "cluster_index: node metadata parse failed");
            }
        }
    }

    /// Advise the page cache for a batch of probed clusters on a local store.
    ///
    /// When `prefer_codes` is true and a cluster has a `codes` blob, advises
    /// ONLY the codes span (skipping the 3–26x larger HNSW graph and skipping
    /// `node_meta` until winning clusters are known). Legacy clusters without a
    /// `codes` blob advise `graph + node_meta` as before.
    fn prefetch_batch(&self, cluster_ids: &[usize], prefer_codes: bool) {
        #[cfg(unix)]
        {
            if self.local_root.is_none() || cluster_ids.is_empty() {
                return;
            }
            let mut per_shard: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
            for &cluster_id in cluster_ids {
                let Some(&shard_idx) = self.cluster_to_shard.get(&cluster_id) else {
                    continue;
                };
                let Some(shard) = self.shards.get(shard_idx) else {
                    continue;
                };
                let Some(slices) = shard.cluster_slices.get(&cluster_id) else {
                    continue;
                };
                if prefer_codes {
                    if let Some(codes_range) = slices.codes {
                        if self.cache.contains_key(&ClusterCacheKey::Codes(cluster_id)) {
                            continue;
                        }
                        per_shard.entry(shard_idx).or_default().push(codes_range);
                        continue;
                    }
                }
                if self.cache.contains_key(&ClusterCacheKey::Graph(cluster_id)) {
                    continue;
                }
                let spans = per_shard.entry(shard_idx).or_default();
                spans.push(slices.graph);
                spans.push(slices.node_meta);
            }
            self.advise_shard_spans(per_shard);
        }
        #[cfg(not(unix))]
        let _ = (cluster_ids, prefer_codes);
    }

    /// Advise the page cache for winning clusters' `node_meta` blobs only.
    fn prefetch_node_meta(&self, cluster_ids: &[usize]) {
        #[cfg(unix)]
        {
            if self.local_root.is_none() || cluster_ids.is_empty() {
                return;
            }
            let mut per_shard: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
            for &cluster_id in cluster_ids {
                if self
                    .cache
                    .contains_key(&ClusterCacheKey::NodeMeta(cluster_id))
                    || self.cache.contains_key(&ClusterCacheKey::Graph(cluster_id))
                {
                    continue;
                }
                let Some(&shard_idx) = self.cluster_to_shard.get(&cluster_id) else {
                    continue;
                };
                let Some(shard) = self.shards.get(shard_idx) else {
                    continue;
                };
                let Some(slices) = shard.cluster_slices.get(&cluster_id) else {
                    continue;
                };
                per_shard
                    .entry(shard_idx)
                    .or_default()
                    .push(slices.node_meta);
            }
            self.advise_shard_spans(per_shard);
        }
        #[cfg(not(unix))]
        let _ = cluster_ids;
    }

    #[cfg(unix)]
    fn advise_shard_spans(&self, per_shard: HashMap<usize, Vec<(usize, usize)>>) {
        use std::os::unix::io::AsRawFd;
        let Some(root) = self.local_root.as_ref() else {
            return;
        };
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

    /// Load a cluster's flat codes blob from cache or its local mapped shard,
    /// coalescing concurrent local misses via `moka::sync::Cache::optionally_get_with`.
    fn get_codes(&self, cluster_id: usize) -> Option<Arc<CachedClusterCodes>> {
        let key = ClusterCacheKey::Codes(cluster_id);
        let entry = self.cache.optionally_get_with(key, || {
            let shard_idx = *self.cluster_to_shard.get(&cluster_id)?;
            let shard = self.shards.get(shard_idx)?;
            if shard.container.is_empty() {
                return None;
            }
            let slices = shard.cluster_slices.get(&cluster_id)?;
            let (off, len) = slices.codes?;
            let codes_bytes = shard.container.slice(off..off + len);
            match parse_codes_blob(codes_bytes) {
                Ok(parsed) => Some(CachedClusterEntry::Codes(Arc::new(parsed))),
                Err(e) => {
                    tracing::warn!(cluster_id, shard_idx, error = %e, "cluster_index: codes parse failed");
                    None
                }
            }
        })?;
        match entry {
            CachedClusterEntry::Codes(c) => Some(c),
            _ => None,
        }
    }

    /// Load a cluster's node metadata from cache or its local mapped shard,
    /// coalescing concurrent misses.
    fn get_node_meta(&self, cluster_id: usize) -> Option<Arc<CachedNodeMeta>> {
        let key = ClusterCacheKey::NodeMeta(cluster_id);
        let entry = self.cache.optionally_get_with(key, || {
            let shard_idx = *self.cluster_to_shard.get(&cluster_id)?;
            let shard = self.shards.get(shard_idx)?;
            if shard.container.is_empty() {
                return None;
            }
            let slices = shard.cluster_slices.get(&cluster_id)?;
            let meta_bytes = shard
                .container
                .slice(slices.node_meta.0..slices.node_meta.0 + slices.node_meta.1);
            let raw_len = meta_bytes.len();
            match node_meta_from_parquet(meta_bytes) {
                Ok(entries) => {
                    let weight = (entries
                        .len()
                        .saturating_mul(std::mem::size_of::<NodeMeta>()))
                    .max(raw_len);
                    Some(CachedClusterEntry::NodeMeta(Arc::new(CachedNodeMeta {
                        entries,
                        bytes: weight,
                    })))
                }
                Err(e) => {
                    tracing::warn!(cluster_id, shard_idx, error = %e, "cluster_index: node metadata parse failed");
                    None
                }
            }
        })?;
        match entry {
            CachedClusterEntry::NodeMeta(m) => Some(m),
            _ => None,
        }
    }

    /// Load a cluster's graph and node metadata from its containing shard,
    /// caching the deserialized form with per-key singleflight deduplication.
    fn get_cluster(&self, cluster_id: usize) -> Option<Arc<CachedCluster>> {
        let key = ClusterCacheKey::Graph(cluster_id);
        let entry = self.cache.optionally_get_with(key, || {
            let shard_idx = *self.cluster_to_shard.get(&cluster_id)?;
            let shard = self.shards.get(shard_idx)?;
            // Remote shards carry no container: `hydrate_*` owns fetching their
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

            Some(CachedClusterEntry::Graph(Arc::new(CachedCluster {
                graph,
                node_meta,
                bytes: slices.graph.1 + slices.node_meta.1,
            })))
        })?;
        match entry {
            CachedClusterEntry::Graph(c) => Some(c),
            _ => None,
        }
    }
}

/// Unique winning cluster IDs present in the top-`k` codes heap, sorted for
/// deterministic shard grouping.
fn unique_winning_clusters(heap: &BinaryHeap<TopKCandidate>) -> Vec<usize> {
    let mut set: HashSet<usize> = HashSet::with_capacity(heap.len());
    for cand in heap.iter() {
        set.insert(cand.cluster_id);
    }
    let mut ids: Vec<usize> = set.into_iter().collect();
    ids.sort_unstable();
    ids
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
    codes_bytes: Vec<u8>,
    graph_bytes: Vec<u8>,
    meta_bytes: Vec<u8>,
}

/// Construct and serialize one cluster's flat codes, HNSW graph, and node
/// metadata in memory.
///
/// Embarrassingly parallel: takes only immutable references and creates a
/// thread-unique temp dump file before returning the owned byte payloads. When
/// a quantizer is present, every vector is encoded once and shared between the
/// flat `cluster-codes` blob and the HNSW graph build.
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

    // Order cluster members by `(row_id, segment_idx)` so segments are
    // interleaved deterministically rather than clumped by Pass 2 scan order.
    // Without this, exact quantized-code ties across segments bias a linear
    // codes scan entirely toward whichever segment was streamed first.
    let mut order: Vec<usize> = (0..members.len()).collect();
    order.sort_unstable_by_key(|&i| (members[i].1, members[i].0));

    let vectors: Vec<Vec<f32>> = order.iter().map(|&i| members[i].2.clone()).collect();
    let (codes_bytes, graph) = if let Some(q) = quantizer {
        let encoded: Vec<Vec<u8>> = vectors.iter().map(|v| q.encode(v)).collect();
        let codes_bytes = encode_codes_blob(&encoded)?;
        let graph = build_graph_from_encoded(&encoded, q, m, ef_construction);
        (codes_bytes, graph)
    } else {
        let codes_bytes = encode_f32_codes_blob(&vectors)?;
        let graph = build_graph(&vectors, None, metric, m, ef_construction);
        (codes_bytes, graph)
    };

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

    let node_meta: Vec<NodeMeta> = order
        .iter()
        .map(|&i| NodeMeta {
            segment_idx: members[i].0,
            row_id: members[i].1,
        })
        .collect();
    let meta_bytes = node_meta_to_parquet(&node_meta)?;

    Ok(Some(BuiltClusterBlobs {
        cluster_id,
        graph_blob_type: blob_type_for(quantizer),
        codes_bytes,
        graph_bytes,
        meta_bytes,
    }))
}

/// Write a shard's pre-built clusters in codes-first layout:
/// `[all codes blobs] [all node_meta blobs] [all HNSW graph blobs]`.
fn write_shard_blobs_codes_first<W: std::io::Write + std::io::Seek>(
    writer: &mut PuffinWriter<W>,
    clusters: &[BuiltClusterBlobs],
) -> Result<()> {
    for blobs in clusters {
        let codes_props = HashMap::from([
            ("cluster_id".to_string(), blobs.cluster_id.to_string()),
            (
                "filename".to_string(),
                format!("cluster_{}.codes", blobs.cluster_id),
            ),
        ]);
        writer.add_blob(
            PUFFIN_BLOB_VECTOR_CLUSTER_CODES.to_string(),
            Vec::new(),
            0,
            1,
            &blobs.codes_bytes,
            codes_props,
        )?;
    }
    for blobs in clusters {
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
    }
    for blobs in clusters {
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
    }
    Ok(())
}

/// Build an HNSW graph from already-quantized byte codes (avoids encoding twice).
fn build_graph_from_encoded(
    encoded: &[Vec<u8>],
    quantizer: &QuantizerImpl,
    m: usize,
    ef_construction: usize,
) -> HnswGraph {
    let max_layers = 16;
    let mut hnsw = match quantizer {
        QuantizerImpl::BenoQuant(tq) => {
            if tq.bits() == 4 {
                HnswGraph::TurboQuant4(Hnsw::new(
                    m,
                    encoded.len(),
                    max_layers,
                    ef_construction,
                    crate::core::index::distance::DistL2u4,
                ))
            } else {
                HnswGraph::TurboQuant8(Hnsw::new(
                    m,
                    encoded.len(),
                    max_layers,
                    ef_construction,
                    crate::core::index::distance::DistL2u8,
                ))
            }
        }
        QuantizerImpl::Sq(q) => HnswGraph::Sq8(Hnsw::new(
            m,
            encoded.len(),
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
                encoded.len(),
                max_layers,
                ef_construction,
                dist,
            ))
        }
    };

    hnsw.set_extend_candidates(true);
    hnsw.set_keeping_pruned(true);
    for (local_id, code) in encoded.iter().enumerate() {
        hnsw.insert((VectorValue::Binary(code.clone()), local_id));
    }
    hnsw
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
/// Normalize a manifest `file_path` (either store-relative or a full `file://` /
/// `s3://bucket/...` URI from an external Iceberg manifest) to an
/// [`object_store::path::Path`] relative to the target `ObjectStore` root.
fn object_store_path_for(path: &str) -> object_store::path::Path {
    if path.contains("://") {
        if let Ok(url) = url::Url::parse(path) {
            return object_store::path::Path::from(url.path().trim_start_matches('/'));
        }
    }
    object_store::path::Path::from(path)
}

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
    let object_path = object_store_path_for(path);
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
    let mut codes: HashMap<usize, (usize, usize)> = HashMap::new();

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
        } else if blob.r#type == PUFFIN_BLOB_VECTOR_CLUSTER_CODES {
            codes.insert(cluster_id, (offset, length));
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
                    codes: codes.get(&cluster_id).copied(),
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
        assert_eq!(cache_entry_kib(&CachedClusterEntry::Graph(entry)), 900);
        let codes = Arc::new(CachedClusterCodes {
            bytes: Bytes::from(vec![0u8; 128 * 1024]),
            num_vectors: 1000,
            code_len: 128,
        });
        assert_eq!(cache_entry_kib(&CachedClusterEntry::Codes(codes)), 128);
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
                codes: None,
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

    #[test]
    fn order_centroids_by_locality_groups_spatial_clusters() {
        // Two well-separated spatial groups interleaved in the input.
        let mut centroids = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![100.0, 100.0, 100.0, 100.0],
            vec![0.5, 0.2, 0.1, 0.0],
            vec![100.4, 99.8, 100.1, 100.2],
            vec![1.0, 0.5, 0.3, 0.2],
            vec![99.5, 100.3, 99.9, 100.0],
            vec![0.2, 0.8, 0.1, 0.4],
            vec![100.1, 100.5, 99.7, 100.3],
        ];
        order_centroids_by_locality(&mut centroids);

        assert_eq!(centroids.len(), 8);
        // First 4 must all belong to the same spatial group, and last 4 to the other.
        let first_half_high = centroids[..4].iter().filter(|c| c[0] > 50.0).count();
        assert!(
            first_half_high == 0 || first_half_high == 4,
            "locality order must not split spatial groups across halves: {centroids:?}"
        );
    }

    #[test]
    fn codes_blob_round_trip_and_validation() {
        let encoded = vec![vec![1u8, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]];
        let blob = encode_codes_blob(&encoded).unwrap();
        let parsed = parse_codes_blob(Bytes::from(blob)).unwrap();
        assert_eq!(parsed.num_vectors, 3);
        assert_eq!(parsed.code_len, 4);
        assert_eq!(parsed.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

        // Truncated or bad magic must be rejected cleanly.
        assert!(parse_codes_blob(Bytes::from_static(b"short")).is_err());
        assert!(parse_codes_blob(Bytes::from_static(b"XXXX000000000000")).is_err());

        // Raw f32 round-trip.
        let vecs = vec![vec![1.0f32, 2.5], vec![-3.0, 4.25]];
        let f32_blob = encode_f32_codes_blob(&vecs).unwrap();
        let f32_parsed = parse_codes_blob(Bytes::from(f32_blob)).unwrap();
        assert_eq!(f32_parsed.num_vectors, 2);
        assert_eq!(f32_parsed.code_len, 8);
    }

    #[test]
    fn codes_first_shard_layout_packs_codes_before_meta_and_graphs() {
        let records = blob(240, 16, 0.0);
        let index = ClusterIndex::build_from_records(
            records,
            segments(),
            "embedding",
            &IndexAlgorithm::hnsw_tq8(),
            8,
            16,
            64,
        )
        .unwrap();

        assert_eq!(index.shards.len(), 1);
        let container = &index.shards[0].container;
        let reader = PuffinReader::new(Cursor::new(container.as_ref())).unwrap();

        let mut max_codes_end = 0i64;
        let mut min_meta_start = i64::MAX;
        let mut max_meta_end = 0i64;
        let mut min_graph_start = i64::MAX;
        let mut codes_count = 0usize;

        for b in &reader.footer().blobs {
            if b.r#type == PUFFIN_BLOB_VECTOR_CLUSTER_CODES {
                codes_count += 1;
                max_codes_end = max_codes_end.max(b.offset + b.length);
            } else if b.r#type == PUFFIN_BLOB_VECTOR_CLUSTER_NODE_META {
                min_meta_start = min_meta_start.min(b.offset);
                max_meta_end = max_meta_end.max(b.offset + b.length);
            } else if b.r#type == PUFFIN_BLOB_VECTOR_HNSW_GRAPH {
                min_graph_start = min_graph_start.min(b.offset);
            }
        }

        assert!(codes_count >= 2, "expected multiple codes blobs");
        assert!(
            max_codes_end <= min_meta_start,
            "all codes blobs ({max_codes_end}) must precede all node_meta blobs ({min_meta_start})"
        );
        assert!(
            max_meta_end <= min_graph_start,
            "all node_meta blobs ({max_meta_end}) must precede all graph blobs ({min_graph_start})"
        );
    }

    #[test]
    fn codes_only_search_matches_or_exceeds_graph_search_across_quantizers() {
        let records: Vec<(u32, u32, Vec<f32>)> = (0..160usize)
            .map(|i| {
                let seg = (i % 3) as u32;
                let row = i as u32;
                let v: Vec<f32> = (0..16)
                    .map(|d| (i as f32) * 0.5 + ((i * 7 + d * 13) % 11) as f32 * 0.05)
                    .collect();
                (seg, row, v)
            })
            .collect();

        for algo in [
            IndexAlgorithm::hnsw_tq8(),
            IndexAlgorithm::hnsw_tq4(),
            IndexAlgorithm::hnsw_sq8(),
            IndexAlgorithm::hnsw().with_complexity(16).with_quality(100),
        ] {
            let index = ClusterIndex::build_from_records(
                records.clone(),
                segments(),
                "embedding",
                &algo,
                6,
                16,
                100,
            )
            .unwrap();

            let query = records[155].2.clone();
            let codes_res = index
                .search_with_mode(&query, 5, 6, Some(64), ClusterSearchMode::Codes)
                .unwrap();
            let graph_res = index
                .search_with_mode(&query, 5, 6, Some(64), ClusterSearchMode::Graph)
                .unwrap();

            assert!(!codes_res.is_empty(), "codes search empty for {algo:?}");
            assert!(!graph_res.is_empty(), "graph search empty for {algo:?}");
            assert!(
                codes_res.iter().any(|r| r.row_id == 155),
                "codes search missed exact target row 155 for {algo:?}: {codes_res:?}"
            );
            // Codes-only exact scan over probed clusters must find a top-1
            // distance at least as good as (<=) approximate HNSW traversal.
            assert!(
                codes_res[0].score <= graph_res[0].score + 1e-3,
                "codes top-1 ({}) worse than graph top-1 ({}) for {algo:?}",
                codes_res[0].score,
                graph_res[0].score
            );
        }
    }

    #[test]
    fn legacy_artifact_without_codes_blob_falls_back_to_graph() {
        let records = blob(120, 8, 0.0);
        let mut index = ClusterIndex::build_from_records(
            records,
            segments(),
            "embedding",
            &IndexAlgorithm::hnsw_tq8(),
            4,
            16,
            64,
        )
        .unwrap();

        // Simulate a legacy artifact by removing every `codes` slice from the shard map.
        for shard in &mut index.shards {
            for slices in shard.cluster_slices.values_mut() {
                slices.codes = None;
            }
        }

        let query = vec![0.0; 8];
        let results = index
            .search_with_mode(&query, 5, 4, Some(64), ClusterSearchMode::Auto)
            .unwrap();
        assert!(!results.is_empty(), "legacy fallback returned no results");
        assert!(
            index
                .cache
                .contains_key(&ClusterCacheKey::Graph(index.rank_probes(&query, 1)[0])),
            "legacy artifact must populate Graph cache entry"
        );
    }

    #[tokio::test]
    async fn remote_s3_search_async_fetches_only_codes_and_winning_meta() {
        // Build a multi-shard TQ8 index with 8 well-separated clusters so only 1
        // cluster holds the true nearest neighbors while the query probes all 8.
        let dim = 16usize;
        let mut records = Vec::new();
        for cluster_i in 0..8usize {
            let center = (cluster_i as f32) * 50.0;
            for j in 0..25usize {
                let row = (cluster_i * 25 + j) as u32;
                let seg = row % 3;
                let v: Vec<f32> = (0..dim)
                    .map(|d| center + (j as f32) * 0.05 + (d as f32) * 0.001)
                    .collect();
                records.push((seg, row, v));
            }
        }

        let built = ClusterIndex::build_from_records_with_target(
            records,
            segments(),
            "embedding",
            &IndexAlgorithm::hnsw_tq8(),
            8,
            16,
            64,
            8_000, // Force multiple shards
        )
        .unwrap();
        assert!(
            built.shards.len() >= 2,
            "test requires multi-shard layout, got {} shards",
            built.shards.len()
        );

        let store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let saved_paths = built
            .save(store.clone(), "idx/cluster.puffin")
            .await
            .unwrap();

        // Open through `load_sharded`: `InMemory` has no local root, so this
        // exercises the exact S3 remote path (`load_sharded_remote`).
        let remote = Arc::new(
            ClusterIndex::load_sharded(store.clone(), &saved_paths, "embedding", 1)
                .await
                .unwrap(),
        );
        assert!(remote.local_root.is_none());

        let query = vec![0.1f32; dim];
        let hits = remote
            .search_async(&store, &query, 5, 8, Some(64))
            .await
            .unwrap();
        assert_eq!(hits.len(), 5);
        for h in &hits {
            assert!(
                h.row_id < 25,
                "expected cluster 0 hits (row_id < 25), got row_id={}",
                h.row_id
            );
        }

        // Verify architectural invariants:
        // 1. ZERO Graph entries were deserialized or downloaded!
        for cid in 0..8usize {
            assert!(
                !remote.cache.contains_key(&ClusterCacheKey::Graph(cid)),
                "codes-only remote search must not hydrate graph for cluster {cid}"
            );
        }
        // 2. Codes entries were hydrated for all 8 probed clusters.
        let codes_cached = (0..8usize)
            .filter(|&cid| remote.cache.contains_key(&ClusterCacheKey::Codes(cid)))
            .count();
        assert_eq!(codes_cached, 8, "all 8 probed clusters should cache codes");

        // 3. ONLY the winning cluster(s) hydrated `node_meta` (selective Phase 3)!
        let meta_cached = (0..8usize)
            .filter(|&cid| remote.cache.contains_key(&ClusterCacheKey::NodeMeta(cid)))
            .count();
        assert!(
            meta_cached <= 2,
            "selective Phase 3 should hydrate node_meta for only winning clusters, got {meta_cached}/8"
        );
    }

    #[tokio::test]
    async fn remote_singleflight_coalesces_concurrent_queries() {
        let dim = 16usize;
        let records = blob(160, dim, 0.0);
        let built = ClusterIndex::build_from_records_with_target(
            records,
            segments(),
            "embedding",
            &IndexAlgorithm::hnsw_tq8(),
            6,
            16,
            64,
            6_000,
        )
        .unwrap();

        let store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let saved_paths = built
            .save(store.clone(), "idx/singleflight.puffin")
            .await
            .unwrap();

        let barrier = Arc::new(tokio::sync::Barrier::new(8));
        let query = vec![0.5f32; dim];

        // Wrap `InMemory` to count exact object-store `get_range` requests and bytes.
        #[derive(Debug)]
        struct CountingStore {
            inner: Arc<dyn ObjectStore>,
            range_gets: Arc<std::sync::atomic::AtomicU64>,
            range_bytes: Arc<std::sync::atomic::AtomicU64>,
        }
        impl std::fmt::Display for CountingStore {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "CountingStore")
            }
        }
        #[async_trait::async_trait]
        impl ObjectStore for CountingStore {
            async fn put_opts(
                &self,
                location: &object_store::path::Path,
                payload: object_store::PutPayload,
                opts: object_store::PutOptions,
            ) -> object_store::Result<object_store::PutResult> {
                self.inner.put_opts(location, payload, opts).await
            }
            async fn put_multipart_opts(
                &self,
                location: &object_store::path::Path,
                opts: object_store::PutMultipartOptions,
            ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
                self.inner.put_multipart_opts(location, opts).await
            }
            async fn get_opts(
                &self,
                location: &object_store::path::Path,
                options: object_store::GetOptions,
            ) -> object_store::Result<object_store::GetResult> {
                self.inner.get_opts(location, options).await
            }
            async fn get_range(
                &self,
                location: &object_store::path::Path,
                range: std::ops::Range<u64>,
            ) -> object_store::Result<Bytes> {
                self.range_gets.fetch_add(1, AtomicOrdering::Relaxed);
                let out = self.inner.get_range(location, range).await?;
                self.range_bytes
                    .fetch_add(out.len() as u64, AtomicOrdering::Relaxed);
                Ok(out)
            }
            async fn delete(
                &self,
                location: &object_store::path::Path,
            ) -> object_store::Result<()> {
                self.inner.delete(location).await
            }
            fn list(
                &self,
                prefix: Option<&object_store::path::Path>,
            ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
            {
                self.inner.list(prefix)
            }
            async fn list_with_delimiter(
                &self,
                prefix: Option<&object_store::path::Path>,
            ) -> object_store::Result<object_store::ListResult> {
                self.inner.list_with_delimiter(prefix).await
            }
            async fn copy(
                &self,
                from: &object_store::path::Path,
                to: &object_store::path::Path,
            ) -> object_store::Result<()> {
                self.inner.copy(from, to).await
            }
            async fn copy_if_not_exists(
                &self,
                from: &object_store::path::Path,
                to: &object_store::path::Path,
            ) -> object_store::Result<()> {
                self.inner.copy_if_not_exists(from, to).await
            }
        }

        let range_gets = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let range_bytes = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let counted: Arc<dyn ObjectStore> = Arc::new(CountingStore {
            inner: store.clone(),
            range_gets: range_gets.clone(),
            range_bytes: range_bytes.clone(),
        });

        // Open a fresh remote index so all footers and cluster blobs start cold.
        let remote = Arc::new(
            ClusterIndex::load_sharded(counted.clone(), &saved_paths, "embedding", 1)
                .await
                .unwrap(),
        );
        let open_gets = range_gets.swap(0, AtomicOrdering::Relaxed);
        let _ = open_gets;
        let _ = range_bytes.swap(0, AtomicOrdering::Relaxed);

        let mut tasks = Vec::with_capacity(8);
        for _ in 0..8 {
            let idx = Arc::clone(&remote);
            let st = Arc::clone(&counted);
            let b = Arc::clone(&barrier);
            let q = query.clone();
            tasks.push(tokio::spawn(async move {
                b.wait().await;
                idx.search_async_with_mode(&st, &q, 5, 6, Some(64), ClusterSearchMode::Codes)
                    .await
                    .unwrap()
            }));
        }

        let mut all_results = Vec::new();
        for t in tasks {
            all_results.push(t.await.unwrap());
        }
        let concurrent_codes_gets = range_gets.load(AtomicOrdering::Relaxed);
        let concurrent_codes_bytes = range_bytes.load(AtomicOrdering::Relaxed);

        for res in &all_results {
            assert_eq!(res.len(), 5);
            assert_eq!(
                res.iter().map(|r| r.row_id).collect::<Vec<_>>(),
                all_results[0].iter().map(|r| r.row_id).collect::<Vec<_>>(),
                "all concurrent queries must observe identical top-k hits"
            );
        }

        // Compare against Graph mode on a fresh cold open of the exact same artifact.
        let remote_graph = Arc::new(
            ClusterIndex::load_sharded(counted.clone(), &saved_paths, "embedding", 1)
                .await
                .unwrap(),
        );
        range_gets.store(0, AtomicOrdering::Relaxed);
        range_bytes.store(0, AtomicOrdering::Relaxed);
        let graph_res = remote_graph
            .search_async_with_mode(&counted, &query, 5, 6, Some(64), ClusterSearchMode::Graph)
            .await
            .unwrap();
        let single_graph_bytes = range_bytes.load(AtomicOrdering::Relaxed);
        assert_eq!(graph_res.len(), 5);

        // 8 concurrent codes-only queries must transfer strictly fewer bytes than
        // even a SINGLE graph-mode query, and singleflight must keep GETs well
        // below 8x fan-out.
        assert!(
            concurrent_codes_bytes < single_graph_bytes,
            "8 concurrent codes-only queries ({concurrent_codes_bytes} B) must transfer fewer bytes than 1 graph query ({single_graph_bytes} B)"
        );
        // Each shard needs at most 2 footer GETs + 1 coalesced codes GET + 1 meta GET = 4 GETs.
        // Without singleflight, 8 concurrent threads would issue up to 8x that.
        let max_expected_singleflight_gets = (remote.shards.len() as u64) * 4 + 2;
        assert!(
            concurrent_codes_gets <= max_expected_singleflight_gets,
            "singleflight must coalesce 8 concurrent queries: got {concurrent_codes_gets} GETs (max expected {max_expected_singleflight_gets})"
        );
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
