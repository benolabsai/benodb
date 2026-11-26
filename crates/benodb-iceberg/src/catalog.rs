// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! The catalog abstraction: the Iceberg REST spec's table requirements and
//! updates as typed enums, and the `Catalog` trait third parties implement
//! or consume.
//!
//! The wire shapes follow the Iceberg REST spec's internally tagged enums:
//! a requirement is `{"requirement": "assert-table-uuid", "uuid": ...}`, an
//! update is `{"action": "add-snapshot", ...}`. The engine's catalog
//! backends speak the same JSON, so the typed enums serialize to and parse
//! from the payloads the backends already handle.

use crate::manifest_types::Schema;
use crate::types::IcebergSnapshot;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The Iceberg REST spec's table requirements: assertions evaluated against
/// the catalog's current metadata before any update in the same commit
/// applies. A failed assertion rejects the whole commit.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "requirement", rename_all = "kebab-case")]
pub enum TableRequirement {
    /// The table's UUID must match — the table has not been dropped and
    /// recreated concurrently.
    AssertTableUuid { uuid: String },
    /// The current schema id must match.
    AssertCurrentSchemaId {
        #[serde(rename = "current-schema-id")]
        current_schema_id: i32,
    },
    /// The last assigned column (field) id must match — no concurrent schema
    /// evolution.
    AssertLastAssignedFieldId {
        #[serde(rename = "last-assigned-field-id")]
        last_assigned_field_id: i32,
    },
    /// The last sequence number must match — no concurrent commit changed
    /// the table's data.
    AssertLastSequenceNumber {
        #[serde(rename = "last-sequence-number")]
        last_sequence_number: i64,
    },
    /// A named reference (branch or tag) must point at the given snapshot
    /// (`None` asserts the ref does not exist).
    AssertRefSnapshotId {
        #[serde(rename = "ref")]
        r#ref: String,
        #[serde(rename = "snapshot-id")]
        snapshot_id: Option<i64>,
    },
}

/// The Iceberg REST spec's table updates: the changes a commit applies to
/// the table metadata after its requirements pass.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "action", rename_all = "kebab-case")]
pub enum TableUpdate {
    /// Add a schema to the table's schema list.
    AddSchema { schema: Schema },
    /// Set the current schema id (`-1` resolves to the newest added schema).
    SetCurrentSchema {
        #[serde(rename = "schema-id")]
        schema_id: i32,
    },
    /// Add a snapshot to the table's snapshot list.
    AddSnapshot { snapshot: IcebergSnapshot },
    /// Set or create a snapshot reference (branch or tag).
    SetSnapshotRef {
        #[serde(rename = "ref-name")]
        ref_name: String,
        #[serde(rename = "type")]
        ref_type: String,
        #[serde(rename = "snapshot-id")]
        snapshot_id: i64,
    },
    /// Set table properties.
    SetProperties { properties: HashMap<String, String> },
    /// Remove table properties.
    RemoveProperties {
        #[serde(rename = "properties")]
        properties: Vec<String>,
    },
    /// Set the table's location.
    SetLocation { location: String },
    /// Upgrade the table's format version (never downgrades).
    UpgradeFormatVersion {
        #[serde(rename = "format-version")]
        format_version: i32,
    },
}

/// A catalog: the Iceberg namespace/table registry.
///
/// Implementations back the trait with a catalog service (REST, Glue, Hive,
/// Nessie, JDBC, ...) or in-process storage. All methods are async; the
/// trait is object safe (`async_trait`), so a catalog composes into
/// runtimes that erase the concrete backend.
#[async_trait::async_trait]
pub trait Catalog: Send + Sync {
    /// Create a namespace (a database or schema in service-specific terms).
    async fn create_namespace(&self, namespace: &str) -> anyhow::Result<()>;

    /// Whether the namespace exists.
    async fn namespace_exists(&self, namespace: &str) -> anyhow::Result<bool>;

    /// Drop an empty namespace.
    async fn drop_namespace(&self, namespace: &str) -> anyhow::Result<()>;

    /// Create a table with the given spec schema at `location` (or the
    /// catalog's default location for the namespace).
    async fn create_table(
        &self,
        namespace: &str,
        table: &str,
        schema: &Schema,
        location: Option<&str>,
    ) -> anyhow::Result<IcebergTableMetadata>;

    /// Drop a table.
    async fn drop_table(&self, namespace: &str, table: &str) -> anyhow::Result<()>;

    /// List the tables in a namespace.
    async fn list_tables(&self, namespace: &str) -> anyhow::Result<Vec<String>>;

    /// Load a table's current metadata.
    async fn load_table(&self, namespace: &str, table: &str) -> anyhow::Result<IcebergTableMetadata>;

    /// Whether the table exists.
    async fn table_exists(&self, namespace: &str, table: &str) -> anyhow::Result<bool>;

    /// Commit updates atomically: every requirement is evaluated against the
    /// catalog's current metadata, and the commit applies only if all pass.
    /// Returns the committed (post-update) metadata.
    async fn commit_table(
        &self,
        namespace: &str,
        table: &str,
        requirements: &[TableRequirement],
        updates: &[TableUpdate],
    ) -> anyhow::Result<IcebergTableMetadata>;
}

use crate::types::IcebergTableMetadata;

#[cfg(test)]
mod tests {
    use super::*;

    /// The REST spec's requirement wire shape: internally tagged with
    /// `requirement`, kebab-case fields.
    #[test]
    fn requirement_serializes_to_the_rest_shape() {
        let r = TableRequirement::AssertTableUuid { uuid: "u-1".into() };
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"requirement": "assert-table-uuid", "uuid": "u-1"})
        );
        let back: TableRequirement = serde_json::from_value(json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn ref_snapshot_requirement_allows_null_snapshot() {
        let json = serde_json::json!({
            "requirement": "assert-ref-snapshot-id",
            "ref": "main",
            "snapshot-id": null
        });
        let r: TableRequirement = serde_json::from_value(json).unwrap();
        assert_eq!(
            r,
            TableRequirement::AssertRefSnapshotId { r#ref: "main".into(), snapshot_id: None }
        );
    }

    /// The REST spec's update wire shape: internally tagged with `action`.
    #[test]
    fn update_serializes_to_the_rest_shape() {
        let u = TableUpdate::SetProperties {
            properties: [("write.format".to_string(), "parquet".to_string())]
                .into_iter()
                .collect(),
        };
        let json = serde_json::to_value(&u).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "action": "set-properties",
                "properties": {"write.format": "parquet"}
            })
        );
        let back: TableUpdate = serde_json::from_value(json).unwrap();
        assert_eq!(back, u);
    }

    #[test]
    fn add_snapshot_update_round_trips() {
        let u = TableUpdate::AddSnapshot {
            snapshot: IcebergSnapshot {
                snapshot_id: 7,
                timestamp_ms: 1_700,
                manifest_list: "s3://t/mList.avro".to_string(),
                summary: [("operation".to_string(), "append".to_string())]
                    .into_iter()
                    .collect(),
            },
        };
        let json = serde_json::to_value(&u).unwrap();
        assert_eq!(json["action"], "add-snapshot");
        assert_eq!(json["snapshot"]["snapshot-id"], 7);
        let back: TableUpdate = serde_json::from_value(json).unwrap();
        assert_eq!(back, u);
    }
}
