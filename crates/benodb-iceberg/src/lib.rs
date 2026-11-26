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

pub mod delete;
pub mod error;
pub mod manifest;
pub mod manifest_types;
pub mod metrics;
pub mod puffin;
pub mod schema;
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
    IcebergManifestObject, IcebergSnapshot, IcebergTableMetadata,
};

// ── Manifest reading & conversion ─────────────────────────────────────────
pub use manifest::{convert_iceberg_to_object, read_manifest, read_manifest_list};

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

/// Qualify a manifest-relative data-file path to the absolute URI that Iceberg
/// manifests record.
///
/// Position-delete rows must reference exactly the data-file path declared by
/// the data manifests (absolute). The engine keeps paths store-relative
/// internally, so deletes need this conversion at write time; otherwise readers
/// cannot match a delete to its file and silently return the deleted row.
pub fn qualify_data_file_path(root_uri: &str, path: &str) -> String {
    if path.contains("://") || path.starts_with('/') {
        path.to_string()
    } else {
        format!(
            "{}/{}",
            root_uri.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }
}
