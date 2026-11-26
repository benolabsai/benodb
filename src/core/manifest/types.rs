// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! The engine's manifest vocabulary: the snapshot manifest (`v{N}.json`)
//! types and the index inventory. The Iceberg spec types (`Schema`,
//! `PartitionSpec`, `DeleteFile`, ...) live in the `benodb-iceberg` crate and
//! are re-exported here, so every `core::manifest::types::` path keeps
//! resolving.

use crate::core::iceberg::value::decode_iceberg_value;
use anyhow::Result;
pub use benodb_iceberg::manifest_types::{
    encode_composite_key, ColumnStats, DeleteContent, DeleteFile, IndexAlgorithm,
    ManifestListEntry, ManifestValue, PartitionField, PartitionSpec, Schema, SchemaField,
    VectorStats,
};
use benodb_iceberg::types::IcebergManifestEntry;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

pub type SegmentId = String;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct IndexFile {
    /// Format version of the serialized index structure (e.g., 1 for legacy, 2 for tightened/Puffin)
    #[serde(default = "default_index_format_version")]
    pub format_version: u32,

    /// Unique identifier for this index instance
    pub index_id: String,

    /// High-level index category: "vector", "lexical", "scalar", "composite", "bloom"
    pub index_category: String,

    /// Concrete algorithm implementation: "hnsw_turboquant8", "hnsw_f32", "bm25_v1", "roaring_bitmap_v1"
    pub algorithm: String,

    /// Target column name within the Parquet schema
    pub column_name: String,

    // -----------------------------------------------------------------------
    // Strict Lineage Bindings (Data Integrity Guards)
    // -----------------------------------------------------------------------
    /// Exact Iceberg Snapshot ID active when this index was constructed
    pub source_snapshot_id: i64,

    /// Cryptographic SHA-256 hash of the target Parquet data file
    pub source_data_checksum: String,

    /// Exact record count of the target Parquet data file
    /// Enforces a strict 1-to-1 bijection between index point IDs and Parquet row offsets
    pub source_record_count: i64,

    /// Cryptographic SHA-256 hash of the index artifact itself (detects storage corruption / bitrot)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_checksum: Option<String>,

    // -----------------------------------------------------------------------
    // Physical Storage Location & Puffin Bundle Offsets
    // -----------------------------------------------------------------------
    /// Physical URI or relative storage path of the index artifact (or Puffin bundle)
    pub file_path: String,

    /// Puffin blob type if packed into an Iceberg Puffin container
    /// e.g., "org.apache.iceberg.vector.hnsw-graph", "org.apache.iceberg.lexical.bm25"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_type: Option<String>,

    /// Byte offset within the container file where the index payload begins
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_offset: Option<i64>,

    /// Length in bytes of the index payload
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_length: Option<i64>,

    /// Compression algorithm applied to the index blob ("zstd", "lz4", "none")
    #[serde(default = "default_index_compression")]
    pub compression: String,
}

fn default_index_format_version() -> u32 {
    2
}

fn default_index_compression() -> String {
    "zstd".to_string()
}

/// A **table-level** index artifact: one Puffin file that is not owned by any
/// single data file.
///
/// [`IndexFile`] hangs off a [`ManifestEntry`], so it is per-data-file and its
/// lineage bindings (`source_snapshot_id`, `source_data_checksum`,
/// `source_record_count`) describe exactly one Parquet file. A cluster-keyed
/// vector index spans every data file in the table, so it has no single file to
/// bind to.
///
/// The lineage contract for this type supports a two-tier LSM architecture:
///
/// * `source_manifest_version` binds the artifact to the manifest version it was
///   built from.
/// * `source_segments` records the exact set of base data files it covers.
/// * Commits that append data files retain `Manifest::table_indexes` so that the
///   compacted base tier can be queried alongside uncompacted delta segments.
/// * Any commit that removes or rewrites data files prunes `TableIndexFile`
///   entries whose `source_segments` were deleted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TableIndexFile {
    /// Format version of the serialized artifact.
    #[serde(default = "default_index_format_version")]
    pub format_version: u32,

    /// Unique identifier for this index instance.
    pub index_id: String,

    /// High-level index category. Always `"vector"` today.
    pub index_category: String,

    /// Concrete algorithm implementation, e.g. `"cluster_hnsw_tq8"`.
    pub algorithm: String,

    /// Target column name within the Parquet schema.
    pub column_name: String,

    // -----------------------------------------------------------------------
    // Table-level lineage bindings
    // -----------------------------------------------------------------------
    /// Manifest version this artifact was built from.
    pub source_manifest_version: u64,

    /// Data files (segment ids) covered by this artifact. A query whose manifest
    /// contains a data file outside this set must not use the artifact.
    #[serde(default)]
    pub source_segments: Vec<String>,

    /// Total number of indexed rows across all segments.
    #[serde(default)]
    pub source_record_count: i64,

    /// Cryptographic SHA-256 hash of the artifact itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_checksum: Option<String>,

    // -----------------------------------------------------------------------
    // Physical storage location
    // -----------------------------------------------------------------------
    /// Physical URI or relative storage path of the Puffin container.
    pub file_path: String,

    /// Individual sequential shard files when the table index is partitioned.
    #[serde(default)]
    pub shard_files: Vec<String>,

    /// Puffin blob type of the container's primary payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_type: Option<String>,

    /// Number of global clusters in the artifact.
    #[serde(default)]
    pub cluster_count: usize,

    /// Vector dimensionality.
    #[serde(default)]
    pub dim: usize,

    /// Distance metric name (`l2`, `cosine`, `ip`, ...).
    #[serde(default)]
    pub metric: String,

    /// Compression algorithm applied to the artifact ("zstd", "lz4", "none").
    #[serde(default = "default_index_compression")]
    pub compression: String,
}

impl TableIndexFile {
    /// Whether this artifact covers exactly the given set of segment ids.
    ///
    /// Order-insensitive. Used by the read path to reject an artifact that was
    /// built before a data file was added or removed.
    pub fn covers_segments(&self, segments: &[String]) -> bool {
        if self.source_segments.len() != segments.len() {
            return false;
        }
        let mut a = self.source_segments.clone();
        let mut b = segments.to_vec();
        a.sort_unstable();
        b.sort_unstable();
        a == b
    }

    /// Whether all source segments covered by this artifact are present in the
    /// given segment set (and this artifact covers at least one segment).
    ///
    /// In a two-tier LSM model, this artifact covers the historical base tier
    /// (S_base). Uncompacted delta segments (S_delta = segments \ S_base) are
    /// queried independently and merged at retrieval time.
    pub fn is_valid_base_for(&self, segments: &[String]) -> bool {
        if self.source_segments.is_empty() {
            return false;
        }
        let set: std::collections::HashSet<&String> = segments.iter().collect();
        self.source_segments.iter().all(|s| set.contains(s))
    }
}

impl IndexFile {
    /// Whether this index belongs to the lexical (full-text) family.
    ///
    /// The canonical category is `"lexical"`. Legacy manifests written before
    /// the rename used `"inverted"`, and some code
    /// paths historically used `"bm25"`. All three denote the same family
    /// (inverted postings / BM25) and must be treated equivalently so that
    /// legacy tables keep working and new tables are recognized by every
    /// consumer.
    pub fn is_lexical(&self) -> bool {
        matches!(
            self.index_category.as_str(),
            "lexical" | "inverted" | "bm25"
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SortDirection {
    Asc,
    Desc,
}

/// Iceberg's `null-order` for a sort field.
///
/// The spec spells these `nulls-first` / `nulls-last`. `rename_all =
/// "lowercase"` produced `nullsfirst` / `nullslast`, which the Java client
/// rejects: `SortOrderParser` upper-cases the value and replaces `-` with `_`
/// before `NullOrder.valueOf`, so `nullslast` becomes `NULLSLAST` and does not
/// exist. Spark and Trino therefore refused to open any BenoStreamDB table that
/// had a sort order, with `IllegalArgumentException: Unexpected null order:
/// nullslast`.
///
/// The old concatenated spelling stays as an `alias` so metadata written by
/// earlier releases still parses.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NullOrder {
    #[serde(rename = "nulls-first", alias = "nullsfirst")]
    NullsFirst,
    #[serde(rename = "nulls-last", alias = "nullslast")]
    NullsLast,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct SortField {
    pub source_id: i32,
    pub transform: String, // e.g. "identity"
    pub direction: SortDirection,
    pub null_order: NullOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct SortOrder {
    pub order_id: i32,
    pub fields: Vec<SortField>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ManifestEntry {
    pub file_path: String,
    pub file_size_bytes: i64,
    pub record_count: i64,
    /// BenoStream Extension: File checksum (e.g. SHA256) for data integrity verification
    #[serde(default)]
    pub file_checksum: Option<String>,
    /// BenoStream Extension: Sidecar Index Files
    pub index_files: Vec<IndexFile>,
    /// BenoStream Extension: Merge-on-Read Delete Files (Iceberg v2 compliant)
    #[serde(default)]
    pub delete_files: Vec<DeleteFile>,
    /// Column Statistics for Pruning (Min/Max/Nulls)
    #[serde(default)]
    pub column_stats: HashMap<String, ColumnStats>,
    /// Partition values for this file
    #[serde(default)]
    pub partition_values: HashMap<String, Value>,
    /// BenoStream Extension: Clustering metadata for advanced pruning
    #[serde(default)]
    pub clustering_strategy: Option<String>,
    #[serde(default)]
    pub clustering_columns: Option<Vec<String>>,
    #[serde(default)]
    pub min_clustering_score: Option<u64>,
    #[serde(default)]
    pub max_clustering_score: Option<u64>,
    #[serde(default)]
    pub normalization_mins: Option<Vec<Value>>,
    #[serde(default)]
    pub normalization_maxs: Option<Vec<Value>>,
    /// Iceberg V3 Row Lineage: the first `_row_id` assigned to the first row in
    /// this data file. A row's `_row_id` is `first_row_id + row_position`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_row_id: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ManifestList {
    pub manifest_files: Vec<ManifestListEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Manifest {
    pub version: u64,
    /// Iceberg table format version (1, 2, or 3)
    #[serde(default = "default_format_version")]
    pub format_version: i32,
    pub timestamp_ms: i64,
    /// Path to the ManifestList file (Iceberg-style scalability)
    #[serde(default)]
    pub manifest_list_path: Option<String>,
    /// List of active entries (Directly in manifest for small tables, otherwise in ManifestList)
    pub entries: Vec<ManifestEntry>,
    /// Authoritative, partition-scoped list of merge-on-read delete files.
    ///
    /// Delete files are stored **once** here (not attached to every data
    /// entry). `ManifestEntry::delete_files` is a read-time *derived view*:
    /// the reader resolves it by matching `partition_values`, and the commit
    /// never persists it. This avoids the read-attach -> commit-persist cycle
    /// that previously made the per-entry list grow without bound.
    #[serde(default)]
    pub delete_files: Vec<DeleteFile>,
    /// Pointer to previous version (for history/rollback)
    pub prev_version: Option<u64>,
    /// Explicit Schema Tracking (Iceberg-style)
    #[serde(default)]
    pub schemas: Vec<Schema>,
    #[serde(default)]
    pub current_schema_id: i32,
    /// Partition Specification
    #[serde(default)]
    pub partition_spec: PartitionSpec,
    #[serde(default)]
    pub partition_specs: Vec<PartitionSpec>,
    #[serde(default)]
    pub default_spec_id: i32,
    /// Sort Orders (Iceberg spec)
    #[serde(default)]
    pub sort_orders: Vec<SortOrder>,
    #[serde(default)]
    pub default_sort_order_id: i32,
    #[serde(default)]
    pub properties: HashMap<String, String>,
    #[serde(default)]
    pub last_column_id: i32,
    /// Table-level index artifacts (not owned by any single data file).
    ///
    /// `#[serde(default)]` keeps every pre-existing manifest readable: an old
    /// table loads with an empty list and the read path falls back to the
    /// per-segment index. The field is intentionally **not** carried forward by
    /// [`Manifest::new`] and friends, so any data commit invalidates the
    /// artifact — see [`TableIndexFile`] for the lineage contract.
    #[serde(default)]
    pub table_indexes: Vec<TableIndexFile>,
}

fn default_format_version() -> i32 {
    2 // Default to V2 for backward compatibility
}

impl Manifest {
    pub fn new(version: u64, entries: Vec<ManifestEntry>, prev_version: Option<u64>) -> Self {
        Self {
            version,
            format_version: 2,
            timestamp_ms: Utc::now().timestamp_millis(),
            manifest_list_path: None,
            entries,
            delete_files: Vec::new(),
            prev_version,
            schemas: Vec::new(),
            current_schema_id: 0,
            partition_spec: PartitionSpec::default(),
            partition_specs: Vec::new(),
            default_spec_id: 0,
            sort_orders: Vec::new(),
            default_sort_order_id: 0,
            properties: HashMap::new(),
            last_column_id: 0,
            table_indexes: Vec::new(),
        }
    }

    pub fn new_with_schema(
        version: u64,
        entries: Vec<ManifestEntry>,
        prev_version: Option<u64>,
        schemas: Vec<Schema>,
        current_schema_id: i32,
    ) -> Self {
        let last_id = schemas
            .iter()
            .flat_map(|s| s.fields.iter().map(|f| f.id))
            .max()
            .unwrap_or(0);
        Self {
            version,
            format_version: 2,
            timestamp_ms: Utc::now().timestamp_millis(),
            manifest_list_path: None,
            entries,
            delete_files: Vec::new(),
            prev_version,
            schemas,
            current_schema_id,
            partition_spec: PartitionSpec::default(),
            partition_specs: Vec::new(),
            default_spec_id: 0,
            sort_orders: Vec::new(),
            default_sort_order_id: 0,
            properties: HashMap::new(),
            last_column_id: last_id,
            table_indexes: Vec::new(),
        }
    }

    pub fn new_with_spec(
        version: u64,
        entries: Vec<ManifestEntry>,
        prev_version: Option<u64>,
        schema_list: Vec<Schema>,
        current_schema_id: i32,
        partition_spec: PartitionSpec,
    ) -> Self {
        let last_id = schema_list
            .iter()
            .flat_map(|s| s.fields.iter().map(|f| f.id))
            .max()
            .unwrap_or(0);
        let spec_id = partition_spec.spec_id;
        Self {
            version,
            format_version: 2,
            timestamp_ms: Utc::now().timestamp_millis(),
            manifest_list_path: None,
            entries,
            delete_files: Vec::new(),
            prev_version,
            schemas: schema_list,
            current_schema_id,
            partition_spec: partition_spec.clone(),
            partition_specs: vec![partition_spec], // Track spec history
            default_spec_id: spec_id,
            sort_orders: Vec::new(),
            default_sort_order_id: 0,
            properties: HashMap::new(),
            last_column_id: last_id,
            table_indexes: Vec::new(),
        }
    }
}
// Iceberg Target Manifest Size = 8MB

pub const MANIFEST_TARGET_SIZE_BYTES: usize = 8 * 1024 * 1024;

/// Enum to distinguish between Data and Delete entries in the manifest
pub enum IcebergManifestObject {
    Data(Box<crate::core::manifest::types::ManifestEntry>),
    Delete(crate::core::manifest::types::DeleteFile),
}

/// Convert Iceberg Manifest Entry to BenoStream Manifest Entry or Delete File
pub fn convert_iceberg_to_object(
    iceberg_entry: &IcebergManifestEntry,
    schema: &crate::core::manifest::types::Schema,
    partition_spec: &crate::core::manifest::types::PartitionSpec,
) -> Result<IcebergManifestObject> {
    use crate::core::manifest::types::{ColumnStats, DeleteContent, DeleteFile, ManifestEntry};
    use std::collections::HashMap;

    let df = &iceberg_entry.data_file;

    // Process partition values
    let mut partition_values = HashMap::new();
    if !df.partition.is_empty() && partition_spec.fields.len() == df.partition.len() {
        for (i, p_field) in partition_spec.fields.iter().enumerate() {
            partition_values.insert(p_field.name.clone(), df.partition[i].clone());
        }
    }

    if df.content == 0 {
        // Data File
        let mut column_stats = HashMap::new();
        let field_map: HashMap<i32, &crate::core::manifest::types::SchemaField> =
            schema.fields.iter().map(|f| (f.id, f)).collect();

        if let (Some(lowers), Some(uppers), Some(nulls)) =
            (&df.lower_bounds, &df.upper_bounds, &df.null_value_counts)
        {
            for (id, lower_bytes) in lowers {
                if let Some(field) = field_map.get(id) {
                    let min_val =
                        decode_iceberg_value(&serde_json::json!(field.type_str), lower_bytes);
                    let max_val = uppers.get(id).map(|upper_bytes| {
                        decode_iceberg_value(&serde_json::json!(field.type_str), upper_bytes)
                    });
                    let null_count = *nulls.get(id).unwrap_or(&0);
                    column_stats.insert(
                        field.name.clone(),
                        ColumnStats {
                            min: Some(min_val),
                            max: max_val,
                            null_count,
                            ..Default::default()
                        },
                    );
                }
            }
        }

        let index_files = if let Some(idx_json) = &df.index_files {
            serde_json::from_str(idx_json).unwrap_or_default()
        } else {
            Vec::new()
        };

        Ok(IcebergManifestObject::Data(Box::new(ManifestEntry {
            file_path: df.file_path.clone(),
            file_size_bytes: df.file_size_in_bytes,
            record_count: df.record_count,
            column_stats,
            partition_values,
            index_files,
            file_checksum: df.file_checksum.clone(),
            first_row_id: df.first_row_id,
            ..Default::default()
        })))
    } else {
        // Delete File.
        //
        // Two encodings are accepted for V3 Deletion Vectors:
        //  - spec shape: `content = 1` (POSITION_DELETES), `file_format = "PUFFIN"`,
        //    non-null `referenced_data_file` + `content_offset` + `content_size_in_bytes`;
        //  - legacy BenoStreamDB shape: the custom `content = 3` code, which earlier
        //    builds wrote before the V3 spec layout was implemented.
        let dv_fields = (df.content_offset, df.content_size_in_bytes);
        let is_puffin = df.file_format.eq_ignore_ascii_case("PUFFIN");
        let content = if df.content == 1 && is_puffin {
            match dv_fields {
                (Some(offset), Some(size)) => DeleteContent::DeletionVector {
                    puffin_file_path: df.file_path.clone(),
                    referenced_data_file: df.referenced_data_file.clone(),
                    content_offset: offset,
                    content_size_in_bytes: size,
                },
                // A PUFFIN-format position-delete entry without blob coordinates
                // cannot be fetched; failing closed here would silently drop the
                // deletes, so surface it as a position delete and let the reader
                // warn on the unreadable path.
                _ => DeleteContent::Position,
            }
        } else if df.content == 1 {
            DeleteContent::Position
        } else if df.content == 2 {
            DeleteContent::Equality {
                equality_ids: df.equality_ids.clone().unwrap_or_default(),
            }
        } else if df.content == 3 {
            // Legacy DV encoding written before the V3 spec layout landed.
            if let (Some(offset), Some(size)) = dv_fields {
                DeleteContent::DeletionVector {
                    puffin_file_path: df.file_path.clone(),
                    referenced_data_file: df.referenced_data_file.clone(),
                    content_offset: offset,
                    content_size_in_bytes: size,
                }
            } else {
                DeleteContent::Position
            }
        } else {
            // Unknown content type, default to position
            DeleteContent::Position
        };

        Ok(IcebergManifestObject::Delete(DeleteFile {
            file_path: df.file_path.clone(),
            content,
            file_size_bytes: df.file_size_in_bytes,
            record_count: df.record_count,
            partition_values,
        }))
    }
}

#[cfg(test)]
mod sort_order_spec_spelling_tests {
    use super::*;

    /// Spark and Trino parse `null-order` with
    /// `NullOrder.valueOf(value.to_uppercase().replace('-', '_'))`, so the value
    /// must be `nulls-first` / `nulls-last`. Writing `nullsfirst` made every
    /// foreign reader reject the table with
    /// `IllegalArgumentException: Unexpected null order: nullslast`.
    #[test]
    fn null_order_is_written_with_the_spec_hyphen() {
        let field = SortField {
            source_id: 1,
            transform: "identity".to_string(),
            direction: SortDirection::Desc,
            null_order: NullOrder::NullsLast,
        };
        let json = serde_json::to_string(&field).expect("serialize sort field");
        assert!(json.contains(r#""null-order":"nulls-last""#), "{json}");
        assert!(json.contains(r#""direction":"desc""#), "{json}");

        let json = serde_json::to_string(&NullOrder::NullsFirst).expect("serialize null order");
        assert_eq!(json, r#""nulls-first""#);
    }

    /// Metadata written by releases that emitted the concatenated spelling must
    /// still load, or the fix would strand existing tables.
    #[test]
    fn legacy_concatenated_null_order_still_deserializes() {
        let legacy =
            r#"{"source-id":1,"transform":"identity","direction":"desc","null-order":"nullslast"}"#;
        let field: SortField = serde_json::from_str(legacy).expect("parse legacy sort field");
        assert_eq!(field.null_order, NullOrder::NullsLast);

        let legacy_first: NullOrder =
            serde_json::from_str(r#""nullsfirst""#).expect("parse legacy null order");
        assert_eq!(legacy_first, NullOrder::NullsFirst);
    }

    #[test]
    fn test_encode_composite_key_disambiguation() {
        let single = encode_composite_key(["bob-1"]);
        assert_eq!(single, "bob-1");

        let tuple1 = encode_composite_key(["a\u{1f}b", "c"]);
        let tuple2 = encode_composite_key(["a", "b\u{1f}c"]);
        assert_ne!(
            tuple1, tuple2,
            "Disambiguated length prefixing must prevent delimiter collisions"
        );
        assert_eq!(tuple1, "3:a\u{1f}b\u{1f}1:c");
        assert_eq!(tuple2, "1:a\u{1f}3:b\u{1f}c");
    }
}
