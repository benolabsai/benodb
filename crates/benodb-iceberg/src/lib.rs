// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Apache Iceberg format layer for BenoDB.
//!
//! Implements the Iceberg table format plumbing: table metadata JSON (V2/V3),
//! Avro manifests and manifest lists, schema JSON to Arrow conversion,
//! partition transforms, value conversions, position/equality delete files,
//! and `deletion-vector-v1` Puffin blobs.
//!
//! Scope boundary: this is the format layer, not a complete Iceberg engine.
//! There is no catalog client (the engine's `core::catalog` owns REST, Glue,
//! Hive, Unity, Nessie, and JDBC catalogs) and no scan planning (the engine's
//! reader owns data file reads). The crate is consumed by the BenoDB engine
//! through facade re-exports in `core::iceberg`, `core::manifest::types`,
//! `core::puffin`, and `core::cache`, so engine paths are unchanged. The
//! engine's own error type lives in the engine; this crate's `error` module
//! carries only format-level failures.
#![deny(warnings)]
// No-panic policy for production paths (see docs/NO_PANIC_POLICY.md).
//
// Enforced by `scripts/no_panic_check.sh`, which ratchets this crate's `--lib`
// target against the shared baseline and runs the hard gate below. The
// `#[cfg(test)]` code inside the crate is exempt: tests are allowed to unwrap
// freely.
#![cfg_attr(
    all(not(test), feature = "no-panic"),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

pub mod catalog;
pub mod delete;
pub mod error;
pub mod expr;
pub mod manifest;
pub mod manifest_types;
pub mod metrics;
pub mod puffin;
pub mod scan;
pub mod schema;
pub mod transaction;
pub mod transform;
pub mod types;
pub mod value;
pub mod writer;

// Re-export `iceberg_delete` for backward compatibility with existing callers
// that reference `crate::core::iceberg::iceberg_delete::IcebergDeleteWriter`.
// The actual implementation lives in `delete` module.
pub mod iceberg_delete {
    pub use crate::delete::IcebergDeleteWriter;
}

// ── Types ──────────────────────────────────────────────────────────────────
pub use types::{
    DeleteContent, IcebergDataFile, IcebergManifestEntry, IcebergManifestListEntry,
    IcebergSnapshot, IcebergTableMetadata,
};

// ── Manifest reading & conversion ─────────────────────────────────────────
pub use manifest::{read_manifest, read_manifest_list};

// ── Schema ─────────────────────────────────────────────────────────────────
pub use schema::{iceberg_json_to_arrow_schema, iceberg_partition_spec_to_benodb};

// ── Delete readers ────────────────────────────────────────────────────────
pub use delete::{EqualityDeleteReader, PositionDeleteReader};

// ── Writers ────────────────────────────────────────────────────────────────
pub use writer::{GpuPuffinWriter, IcebergWriter, MANIFEST_LIST_SCHEMA_V2};

// ── Transforms ─────────────────────────────────────────────────────────────
pub use transform::{murmur3_32_x86, IcebergTransform};

// ── Value conversion ──────────────────────────────────────────────────────
pub use value::{
    avro_to_json, decode_iceberg_value, json_to_avro_value, parse_avro_value_bytes,
    parse_avro_value_bytes_with_type,
};

use moka::future::Cache;
use once_cell::sync::Lazy;
use std::sync::Arc;
use std::time::Duration;

/// Global cache for fully merged position-delete files, keyed by the delete
/// file path. The value maps data file path to the set of deleted row ids,
/// because the same physical delete file may contain positions for multiple
/// data files and consumers filter to the target during parsing.
pub static FULL_DELETE_FILE_CACHE: Lazy<
    Cache<String, Arc<std::collections::HashMap<String, std::collections::HashSet<i64>>>>,
> = Lazy::new(|| {
    Cache::builder()
        .max_capacity(50_000)
        .time_to_idle(Duration::from_secs(60 * 10)) // 10 mins idle
        .build()
});

/// Normalize single-slash local file URIs emitted by Java/Hadoop (`file:/path`)
/// into standard `file:///path` form while leaving other schemes unchanged.
pub fn normalize_file_uri(uri: &str) -> String {
    if uri.starts_with("file:/") && !uri.starts_with("file://") {
        format!("file://{}", &uri["file:".len()..])
    } else {
        uri.to_string()
    }
}

/// Strip a `file://` or single-slash `file:` prefix, returning `&str` untouched
/// for bare paths or non-`file` URIs.
pub fn strip_file_scheme(uri: &str) -> &str {
    uri.strip_prefix("file://")
        .or_else(|| {
            if uri.starts_with("file:/") {
                uri.strip_prefix("file:")
            } else {
                None
            }
        })
        .unwrap_or(uri)
}

/// Qualify a manifest-relative data-file path to the absolute URI that Iceberg
/// manifests record.
///
/// Position-delete rows must reference exactly the data-file path declared by
/// the data manifests (absolute). The engine keeps paths store-relative
/// internally, so deletes need this conversion at write time; otherwise readers
/// cannot match a delete to its file and silently return the deleted row.
pub fn qualify_data_file_path(root_uri: &str, path: &str) -> String {
    if path.contains("://") || path.starts_with("file:/") || path.starts_with('/') {
        normalize_file_uri(path)
    } else {
        format!(
            "{}/{}",
            normalize_file_uri(root_uri).trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualify_and_normalize_handles_hadoop_single_slash_file_uris() {
        assert_eq!(
            normalize_file_uri("file:/tmp/warehouse/db/events"),
            "file:///tmp/warehouse/db/events"
        );
        assert_eq!(
            qualify_data_file_path(
                "file:/tmp/warehouse/db/events",
                "file:/tmp/warehouse/db/events/data/00000.parquet"
            ),
            "file:///tmp/warehouse/db/events/data/00000.parquet"
        );
        assert_eq!(
            qualify_data_file_path("file:/tmp/warehouse/db/events", "data/00000.parquet"),
            "file:///tmp/warehouse/db/events/data/00000.parquet"
        );
    }
}
