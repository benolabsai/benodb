// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct IcebergTableMetadata {
    pub format_version: i32,
    pub table_uuid: String,
    pub location: String,
    pub last_sequence_number: i64,
    pub last_updated_ms: i64,
    pub current_snapshot_id: Option<i64>,
    pub snapshots: Vec<IcebergSnapshot>,
    pub schemas: Vec<serde_json::Value>,
    pub current_schema_id: i32,
    pub partition_specs: Vec<serde_json::Value>,
    pub default_spec_id: i32,
    /// The highest field id ever assigned (the spec's
    /// `last-assigned-field-id`). `None` when the metadata source does not
    /// carry it; requirement validation fails closed on `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_assigned_field_id: Option<i32>,
    /// The table's snapshot references (branches and tags).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub refs: HashMap<String, SnapshotRef>,
    /// The table's properties (the spec's free-form settings map).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub properties: HashMap<String, String>,
}

/// A named snapshot reference (a branch or a tag).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct SnapshotRef {
    /// `branch` or `tag`.
    #[serde(rename = "type")]
    pub ref_type: String,
    #[serde(rename = "snapshot-id")]
    pub snapshot_id: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_snapshots_to_keep: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_snapshot_age_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct IcebergSnapshot {
    pub snapshot_id: i64,
    pub timestamp_ms: i64,
    pub manifest_list: String,
    pub summary: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct IcebergManifestListEntry {
    pub manifest_path: String,
    pub manifest_length: i64,
    pub partition_spec_id: i32,
    pub added_snapshot_id: i64,
    pub content: i32,
    pub sequence_number: i64,
    pub min_sequence_number: i64,
    pub added_files_count: i32,
    pub existing_files_count: i32,
    pub deleted_files_count: i32,
    pub added_rows_count: i64,
    pub existing_rows_count: i64,
    pub deleted_rows_count: i64,
}

#[derive(Debug, Clone)]
pub struct IcebergManifestEntry {
    pub status: i32,
    pub snapshot_id: Option<i64>,
    pub data_file: IcebergDataFile,
    /// BenoDB extension: the delete files associated with this data entry,
    /// written as spec-compliant delete records in the same manifest.
    pub delete_files: Vec<crate::manifest_types::DeleteFile>,
}

#[derive(Debug, Clone)]
pub struct IcebergDataFile {
    pub content: i32,
    pub file_path: String,
    pub file_format: String,
    pub partition: Vec<serde_json::Value>,
    pub record_count: i64,
    pub file_size_in_bytes: i64,
    // Column stats (Field ID -> Value)
    pub column_sizes: Option<std::collections::HashMap<i32, i64>>,
    pub value_counts: Option<std::collections::HashMap<i32, i64>>,
    pub null_value_counts: Option<std::collections::HashMap<i32, i64>>,
    pub nan_value_counts: Option<std::collections::HashMap<i32, i64>>,
    pub lower_bounds: Option<std::collections::HashMap<i32, Vec<u8>>>,
    pub upper_bounds: Option<std::collections::HashMap<i32, Vec<u8>>>,
    pub equality_ids: Option<Vec<i32>>,
    // V3 Deletion Vector fields
    pub referenced_data_file: Option<String>,
    pub content_offset: Option<i64>,
    pub content_size_in_bytes: Option<i64>,
    /// BenoStream Extension: Serialized Index Files (JSON)
    pub index_files: Option<String>,
    /// BenoStream Extension: File Checksum for integrity validation
    pub file_checksum: Option<String>,
    /// Iceberg V3 Row Lineage: first `_row_id` assigned in this data file.
    pub first_row_id: Option<i64>,
}

/// Content type for delete files
#[derive(Debug, Clone)]
pub enum DeleteContent {
    Position,
    Equality {
        equality_ids: Vec<i32>,
    },
    DeletionVector {
        puffin_file_path: String,
        content_offset: i64,
        content_size_in_bytes: i64,
    },
}
