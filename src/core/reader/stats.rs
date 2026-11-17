// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Counters for the candidate rerank read path.
//!
//! A claim that sparse `RowSelection` skips work rather than filtering after
//! the read cannot be checked from wall clock time alone: a filter and a skip
//! can produce the same latency on a warm cache. These counters record what the
//! reader was asked for (`ROWS_SELECTED`), what it actually emitted
//! (`ROWS_DECODED`), and how many row groups it opened. If the decoded count
//! tracks the selected count, the reader materialised the candidates and
//! nothing else. If it tracks the file row count, the selection is decorative
//! and the I/O claim is false.
//!
//! The counters are process wide atomics, reset by the benchmark harness before
//! each pass through `bsdb.reset_reader_stats()`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Segment reads that reached the candidate read path.
pub static READS: AtomicU64 = AtomicU64::new(0);
/// Rows in the file, summed over the reads.
pub static ROWS_REQUESTED: AtomicU64 = AtomicU64::new(0);
/// Rows the `RowSelection` asked for.
pub static ROWS_SELECTED: AtomicU64 = AtomicU64::new(0);
/// Rows the reader actually emitted, before equality deletes are applied.
pub static ROWS_DECODED: AtomicU64 = AtomicU64::new(0);
/// Row groups opened, summed over the reads.
pub static ROW_GROUPS_READ: AtomicU64 = AtomicU64::new(0);
/// Row groups available in the files that were read.
pub static ROW_GROUPS_TOTAL: AtomicU64 = AtomicU64::new(0);

// Cluster-file counters for the object-store path. On a remote store the table
// tier's `search` is synchronous, so a query's probe clusters are fetched ahead
// of it (`hydrate_for_query`) with one coalesced ranged GET per shard span. What
// that costs is requests and bytes, not latency: an object store bills per
// request and per byte, and a cache miss is the only thing that produces either.
// Without these counters a request-count claim is an inference from egress.
/// Range GETs issued to fetch cluster bytes.
pub static CLUSTER_RANGE_GETS: AtomicU64 = AtomicU64::new(0);
/// Bytes those range GETs returned.
pub static CLUSTER_RANGE_BYTES: AtomicU64 = AtomicU64::new(0);
/// Probe clusters served from the cache with no request at all.
pub static CLUSTER_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
/// Vectors in the clusters a query actually scanned, summed over queries.
///
/// This is the coverage denominator that a probe count cannot supply. Two
/// clusterings of the same corpus can have very different cluster sizes, so
/// `nprobe = 16` examines a different number of vectors under each; recall at
/// equal `nprobe` therefore compares two different amounts of work. Recall at
/// equal `CLUSTER_CORPUS_ROWS` compares the same amount of work.
pub static CLUSTER_CORPUS_ROWS: AtomicU64 = AtomicU64::new(0);
/// Clusters resident in the cache after the most recent hydration (a gauge).
pub static CLUSTER_CACHE_ENTRIES: AtomicU64 = AtomicU64::new(0);
/// Weighed KiB the cache holds after the most recent hydration (a gauge).
///
/// Compared against [`CLUSTER_CACHE_ENTRIES`] this gives the cache's own view of
/// an entry's size, which is what decides how many clusters the budget can
/// retain. Compared against [`CLUSTER_RANGE_BYTES`] it says whether that view
/// agrees with the bytes the store actually returned.
pub static CLUSTER_CACHE_WEIGHTED_KIB: AtomicU64 = AtomicU64::new(0);

/// A consistent read of every counter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReaderCounters {
    pub reads: u64,
    pub rows_requested: u64,
    pub rows_selected: u64,
    pub rows_decoded: u64,
    pub row_groups_read: u64,
    pub row_groups_total: u64,
    pub cluster_range_gets: u64,
    pub cluster_range_bytes: u64,
    pub cluster_cache_hits: u64,
    pub cluster_corpus_rows: u64,
    pub cluster_cache_entries: u64,
    pub cluster_cache_weighted_kib: u64,
}

impl ReaderCounters {
    /// Load every counter with relaxed ordering.
    ///
    /// Relaxed is sufficient because the harness reads these after the pass has
    /// completed and the threads have joined; there is no concurrent consumer
    /// that needs a happens before edge on an individual counter.
    pub fn load() -> Self {
        Self {
            reads: READS.load(Ordering::Relaxed),
            rows_requested: ROWS_REQUESTED.load(Ordering::Relaxed),
            rows_selected: ROWS_SELECTED.load(Ordering::Relaxed),
            rows_decoded: ROWS_DECODED.load(Ordering::Relaxed),
            row_groups_read: ROW_GROUPS_READ.load(Ordering::Relaxed),
            row_groups_total: ROW_GROUPS_TOTAL.load(Ordering::Relaxed),
            cluster_range_gets: CLUSTER_RANGE_GETS.load(Ordering::Relaxed),
            cluster_range_bytes: CLUSTER_RANGE_BYTES.load(Ordering::Relaxed),
            cluster_cache_hits: CLUSTER_CACHE_HITS.load(Ordering::Relaxed),
            cluster_corpus_rows: CLUSTER_CORPUS_ROWS.load(Ordering::Relaxed),
            cluster_cache_entries: CLUSTER_CACHE_ENTRIES.load(Ordering::Relaxed),
            cluster_cache_weighted_kib: CLUSTER_CACHE_WEIGHTED_KIB.load(Ordering::Relaxed),
        }
    }

    /// Zero every counter.
    pub fn reset() {
        for counter in [
            &READS,
            &ROWS_REQUESTED,
            &ROWS_SELECTED,
            &ROWS_DECODED,
            &ROW_GROUPS_READ,
            &ROW_GROUPS_TOTAL,
            &CLUSTER_RANGE_GETS,
            &CLUSTER_RANGE_BYTES,
            &CLUSTER_CACHE_HITS,
            &CLUSTER_CORPUS_ROWS,
            &CLUSTER_CACHE_ENTRIES,
            &CLUSTER_CACHE_WEIGHTED_KIB,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }

    /// Fraction of the file the reader emitted, when the read had any rows.
    pub fn decoded_fraction(&self) -> Option<f64> {
        if self.rows_requested == 0 {
            return None;
        }
        Some(self.rows_decoded as f64 / self.rows_requested as f64)
    }
}

/// Whether the reader is allowed to build a page skipping `RowSelection`.
///
/// `BSDB_READER_SKIP=0` forces a full selection, which is the control arm for
/// the rerank I/O ablation: same quantizer, same projection, same candidate
/// count, but every row group read to the end. Read once per process so a run
/// cannot drift between arms mid pass.
pub fn skip_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("BSDB_READER_SKIP")
            .map(|value| value.trim() != "0")
            .unwrap_or(true)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_reset_roundtrip() {
        ReaderCounters::reset();
        READS.fetch_add(2, Ordering::Relaxed);
        ROWS_REQUESTED.fetch_add(1_000, Ordering::Relaxed);
        ROWS_SELECTED.fetch_add(40, Ordering::Relaxed);
        ROWS_DECODED.fetch_add(40, Ordering::Relaxed);
        ROW_GROUPS_READ.fetch_add(1, Ordering::Relaxed);
        ROW_GROUPS_TOTAL.fetch_add(5, Ordering::Relaxed);

        let snapshot = ReaderCounters::load();
        assert_eq!(snapshot.reads, 2);
        assert_eq!(snapshot.rows_selected, 40);
        assert_eq!(snapshot.rows_decoded, 40);
        assert_eq!(snapshot.rows_decoded, snapshot.rows_selected);
        assert_eq!(snapshot.decoded_fraction(), Some(0.04));

        ReaderCounters::reset();
        assert_eq!(ReaderCounters::load(), ReaderCounters::default());
        assert_eq!(ReaderCounters::default().decoded_fraction(), None);
    }
}
