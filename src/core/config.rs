// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Process configuration resolution for the hot path environment keys.
//!
//! Environment variables remain the operator facing contract: the names,
//! defaults, and precedence below match the point of use reads they replace,
//! variable by variable. Two things change, both internal to the crate:
//!
//! 1. Parsing lives in pure functions taking `Option<&str>`, so unit tests
//!    assert the fallback ladder without mutating process global state.
//! 2. Each value resolves once per process through `OnceLock` and is reused.
//!    A running process no longer picks up mid flight environment changes for
//!    these keys; set them before process start, as with every other server
//!    database. The per call reads these replace were re parsing (and, for a
//!    malformed `BDB_DISK_CACHE_GB`, re warning) on every cache charge,
//!    session build, or search.
//!
//! Deliberately not here yet: the remaining `BDB_*` keys that are read at
//! startup or at low frequency. They migrate to this module as their call
//! sites need the same treatment.

use std::sync::OnceLock;

const GIB: u64 = 1024 * 1024 * 1024;

/// Default on-disk budget for the secondary disk cache when `BDB_DISK_CACHE_GB`
/// is unset.
///
/// The disk cache exists to keep *remote* objects (S3/Azure/GCS) off the
/// network. It is not scratch space: with no budget it grows without bound, and
/// a single 10M-vector benchmark run wrote 409 GB of copies of indexes that were
/// already sitting on a local disk.
pub const DEFAULT_DISK_CACHE_BUDGET_GB: u64 = 16;

// ---------------------------------------------------------------------------
// BDB_DISK_CACHE_GB / BDB_DISK_CACHE  (consumed by core::cache)
// ---------------------------------------------------------------------------

/// Pure parser for `BDB_DISK_CACHE_GB`. `Ok(bytes)` on a valid value
/// (saturating at `u64::MAX`), `Err(Some(raw))` when set but not a number so
/// the caller can warn, `Err(None)` when unset.
fn parse_disk_cache_budget_bytes(raw: Option<&str>) -> Result<u64, Option<String>> {
    match raw {
        None => Ok(DEFAULT_DISK_CACHE_BUDGET_GB * GIB),
        Some(v) => v
            .trim()
            .parse::<u64>()
            .map(|gb| gb.saturating_mul(GIB))
            .map_err(|_| Some(v.to_string())),
    }
}

/// On-disk budget for the secondary [`DiskCache`], from `BDB_DISK_CACHE_GB`
/// (default 16 GiB). Resolved once per process.
///
/// [`DiskCache`]: crate::core::cache::DiskCache
pub(crate) fn disk_cache_budget_bytes() -> u64 {
    static CACHED: OnceLock<u64> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let raw = std::env::var("BDB_DISK_CACHE_GB").ok();
        match parse_disk_cache_budget_bytes(raw.as_deref()) {
            Ok(bytes) => bytes,
            Err(Some(bad)) => {
                tracing::warn!(
                    value = %bad,
                    "BDB_DISK_CACHE_GB is not a number; using the default budget"
                );
                DEFAULT_DISK_CACHE_BUDGET_GB * GIB
            }
            Err(None) => DEFAULT_DISK_CACHE_BUDGET_GB * GIB,
        }
    })
}

/// Pure parser for the `BDB_DISK_CACHE` off switch: `off|0|false|no` (case
/// insensitive, trimmed) disables the secondary disk cache.
fn parse_disk_cache_disabled(raw: Option<&str>) -> bool {
    raw.map(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        )
    })
    .unwrap_or(false)
}

/// Whether the secondary disk cache is enabled at all: the off switch must be
/// unset and the budget must be positive. Resolved once per process.
pub(crate) fn disk_cache_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        !parse_disk_cache_disabled(std::env::var("BDB_DISK_CACHE").ok().as_deref())
            && disk_cache_budget_bytes() > 0
    })
}

// ---------------------------------------------------------------------------
// BDB_TARGET_PARTITIONS  (consumed by core::sql)
// ---------------------------------------------------------------------------

/// Pure parser for `BDB_TARGET_PARTITIONS`. `None` when unset, non numeric, or
/// zero, so the caller falls back to detected parallelism.
fn parse_target_partitions(raw: Option<&str>) -> Option<usize> {
    raw.and_then(|n| n.parse::<usize>().ok()).filter(|n| *n > 0)
}

/// Effective DataFusion `target_partitions`: the `BDB_TARGET_PARTITIONS`
/// override when valid, otherwise the CPU budget visible to the process
/// (respects cgroup limits via `std::thread::available_parallelism`).
/// Resolved once per process.
pub(crate) fn target_partitions() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_target_partitions(std::env::var("BDB_TARGET_PARTITIONS").ok().as_deref())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(4)
            })
    })
}

// ---------------------------------------------------------------------------
// BDB_VACUUM_MIN_FILE_AGE_SECS  (consumed by the manifest vacuum)
// ---------------------------------------------------------------------------

/// Pure parser for `BDB_VACUUM_MIN_FILE_AGE_SECS`. A missing, non numeric, or
/// negative value falls back to the 60 second default; `0` is meaningful (it
/// disables the grace period) and is preserved.
fn parse_vacuum_min_file_age_secs(raw: Option<&str>) -> i64 {
    raw.and_then(|v| v.parse::<i64>().ok())
        .filter(|v| *v >= 0)
        .unwrap_or(60)
}

/// Minimum age (seconds) before vacuum will delete an unreferenced file.
///
/// This is the safety margin that closes the GC-vs-writer race: a file uploaded
/// by a concurrent writer is young, so it is never reaped before the writer's
/// commit becomes visible. Override with `BDB_VACUUM_MIN_FILE_AGE_SECS`
/// (default 60s; `0` disables the grace period). Resolved once per process.
pub(crate) fn vacuum_min_file_age_secs() -> i64 {
    static CACHED: OnceLock<i64> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_vacuum_min_file_age_secs(
            std::env::var("BDB_VACUUM_MIN_FILE_AGE_SECS")
                .ok()
                .as_deref(),
        )
    })
}

// ---------------------------------------------------------------------------
// BDB_MANIFEST_MAX_FILES_BEFORE_CONSOLIDATION  (consumed by the manifest commit)
// ---------------------------------------------------------------------------

/// Pure parser for `BDB_MANIFEST_MAX_FILES_BEFORE_CONSOLIDATION`. A missing,
/// non numeric, or zero value falls back to the default of 32.
fn parse_manifest_max_files(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(32)
}

/// Maximum number of manifest files the append-only fast path will accumulate
/// before it consolidates with a full rewrite.
///
/// The append-only path writes only the new entries and references the previous
/// manifest files, so each commit adds one manifest file to the list. Reads walk
/// every referenced manifest file, so an unbounded list would make reads
/// O(number of commits). Once the list reaches this many files the next commit
/// falls back to the full rewrite, which re-encodes the live entries into a
/// single (chunked) manifest file and resets the list. This bounds read cost
/// while still amortizing the common burst of small appends. Resolved once per
/// process.
pub(crate) fn manifest_max_files_before_consolidation() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_manifest_max_files(
            std::env::var("BDB_MANIFEST_MAX_FILES_BEFORE_CONSOLIDATION")
                .ok()
                .as_deref(),
        )
    })
}

// ---------------------------------------------------------------------------
// BDB_INGEST_BACKPRESSURE_GRACE_SECS  (consumed by the table write path)
// ---------------------------------------------------------------------------

/// Parse the fail-open grace (seconds). A missing, non-positive, or unparseable
/// value falls back to the 60-second default so a typo cannot disable the
/// liveness safety net. Pure so it can be asserted without process-global env.
pub(crate) fn backpressure_grace_secs(raw: Option<&str>) -> f64 {
    raw.and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(60.0)
}

/// The resolved ingest back-pressure fail-open grace. Resolved once per
/// process.
pub(crate) fn ingest_backpressure_grace_secs() -> f64 {
    static CACHED: OnceLock<f64> = OnceLock::new();
    *CACHED.get_or_init(|| {
        backpressure_grace_secs(
            std::env::var("BDB_INGEST_BACKPRESSURE_GRACE_SECS")
                .ok()
                .as_deref(),
        )
    })
}

// ---------------------------------------------------------------------------
// BDB_INGEST_MEMORY_BUDGET_GB  (consumed by core::memory)
// ---------------------------------------------------------------------------

/// Pure parser for the `BDB_INGEST_MEMORY_BUDGET_GB` override. `None` when
/// unset, non numeric, or non positive.
fn parse_ingest_memory_budget_gb(raw: Option<&str>) -> Option<f64> {
    raw.and_then(|s| s.parse::<f64>().ok()).filter(|g| *g > 0.0)
}

/// The `BDB_INGEST_MEMORY_BUDGET_GB` override in GiB, or `None` when unset or
/// invalid, in which case callers apply their own default. Resolved once per
/// process.
pub(crate) fn ingest_memory_budget_gb_override() -> Option<f64> {
    static CACHED: OnceLock<Option<f64>> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_ingest_memory_budget_gb(std::env::var("BDB_INGEST_MEMORY_BUDGET_GB").ok().as_deref())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_cache_budget_falls_back_to_default_when_unset() {
        assert_eq!(parse_disk_cache_budget_bytes(None), Ok(16 * GIB));
    }

    #[test]
    fn disk_cache_budget_parses_trims_and_saturates() {
        assert_eq!(parse_disk_cache_budget_bytes(Some(" 32 ")), Ok(32 * GIB));
        assert_eq!(parse_disk_cache_budget_bytes(Some("0")), Ok(0));
        // 2^34 GiB * 2^30 = 2^64 exactly, so saturating_mul pins at u64::MAX.
        assert_eq!(
            parse_disk_cache_budget_bytes(Some("17179869184")),
            Ok(u64::MAX)
        );
        assert_eq!(
            parse_disk_cache_budget_bytes(Some("notanumber")),
            Err(Some("notanumber".to_string()))
        );
        assert_eq!(
            parse_disk_cache_budget_bytes(Some("99999999999999999999")),
            Err(Some("99999999999999999999".to_string()))
        );
    }

    #[test]
    fn disk_cache_off_values_disable() {
        for v in ["0", "off", "false", "no", "OFF", "False", "No", " off "] {
            assert!(parse_disk_cache_disabled(Some(v)), "failed for {v:?}");
        }
        assert!(!parse_disk_cache_disabled(None));
        assert!(!parse_disk_cache_disabled(Some("on")));
        assert!(!parse_disk_cache_disabled(Some("")));
    }

    #[test]
    fn target_partitions_ignores_zero_and_garbage() {
        assert_eq!(parse_target_partitions(Some("8")), Some(8));
        assert_eq!(parse_target_partitions(Some("0")), None);
        assert_eq!(parse_target_partitions(Some("-4")), None);
        assert_eq!(parse_target_partitions(Some("x")), None);
        assert_eq!(parse_target_partitions(None), None);
    }

    #[test]
    fn vacuum_age_accepts_zero_and_rejects_negative() {
        assert_eq!(parse_vacuum_min_file_age_secs(Some("0")), 0);
        assert_eq!(parse_vacuum_min_file_age_secs(Some("120")), 120);
        assert_eq!(parse_vacuum_min_file_age_secs(Some("-1")), 60);
        assert_eq!(parse_vacuum_min_file_age_secs(Some("bad")), 60);
        assert_eq!(parse_vacuum_min_file_age_secs(None), 60);
    }

    #[test]
    fn manifest_consolidation_requires_positive() {
        assert_eq!(parse_manifest_max_files(Some("1")), 1);
        assert_eq!(parse_manifest_max_files(Some("0")), 32);
        assert_eq!(parse_manifest_max_files(Some("bad")), 32);
        assert_eq!(parse_manifest_max_files(None), 32);
    }

    #[test]
    fn backpressure_grace_ladder() {
        assert_eq!(backpressure_grace_secs(None), 60.0);
        assert_eq!(backpressure_grace_secs(Some("15")), 15.0);
        assert_eq!(backpressure_grace_secs(Some("2.5")), 2.5);
        assert_eq!(backpressure_grace_secs(Some("0")), 60.0);
        assert_eq!(backpressure_grace_secs(Some("-3")), 60.0);
        assert_eq!(backpressure_grace_secs(Some("nope")), 60.0);
    }

    #[test]
    fn ingest_budget_override_requires_positive() {
        assert_eq!(parse_ingest_memory_budget_gb(Some("4.5")), Some(4.5));
        assert_eq!(parse_ingest_memory_budget_gb(Some("0")), None);
        assert_eq!(parse_ingest_memory_budget_gb(Some("-1")), None);
        assert_eq!(parse_ingest_memory_budget_gb(Some("bad")), None);
        assert_eq!(parse_ingest_memory_budget_gb(None), None);
    }
}
