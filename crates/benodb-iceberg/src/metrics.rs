// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Metrics owned by the format layer.
//!
//! Metric registration uses static, compile-time-constant names and help
//! strings. The `prometheus` `register_*` macros are only fallible on an
//! invalid name or a duplicate registration — a programming error caught by
//! the tests, never a runtime condition — and the crate offers no infallible
//! constructor. This is a documented residual invariant in
//! NO_PANIC_POLICY.md; failure happens once at first access, not on a request
//! or ingest path.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use lazy_static::lazy_static;
use prometheus::register_int_counter_vec;
use prometheus::IntCounterVec;

lazy_static! {
    /// Outcome of the per-file parsed-delete cache lookup. Label `result`
    /// is `hit` or `miss`.
    pub static ref DELETE_FILE_CACHE_TOTAL: IntCounterVec = register_int_counter_vec!(
        "bdb_delete_file_cache_total",
        "Per-file parsed-delete cache lookups by outcome",
        &["result"]
    )
    .unwrap();
}
