// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! The spec catalog adapter: wraps the engine's catalog trait object and
//! implements the `benodb-iceberg` crate's `Catalog` trait over it.
//!
//! This is the bridge that lets the engine's eight backends (REST, Glue,
//! Hive, Unity, Nessie, JDBC, S3 Tables, Memory) satisfy the crate's
//! spec-pure trait without forking the trait definitions: the engine's trait
//! keeps its public shape (Arrow schemas in, the engine's metadata out), and
//! this wrapper converts at the boundary — spec schema to Arrow on create,
//! engine metadata to the crate's simplified spec shape on load and commit.

use crate::core::catalog::Catalog as EngineCatalog;
use crate::core::metadata::TableMetadata as EngineTableMetadata;
use benodb_iceberg::catalog::{Catalog as SpecCatalogTrait, TableRequirement, TableUpdate};
use benodb_iceberg::types::{IcebergSnapshot, IcebergTableMetadata};
use std::collections::HashMap;
use std::sync::Arc;

/// Convert the engine's full kebab-case metadata into the crate's
/// simplified spec shape. Fields the crate's shape does not carry (sort
/// orders, identifier fields, snapshot/metadata logs, V3 row lineage
/// bookkeeping) are dropped here; they remain authoritative in the engine's
/// metadata.
pub fn to_iceberg_metadata(m: &EngineTableMetadata) -> IcebergTableMetadata {
    IcebergTableMetadata {
        format_version: m.format_version,
        table_uuid: m.table_uuid.clone(),
        location: m.location.clone(),
        last_sequence_number: m.last_sequence_number,
        last_updated_ms: m.last_updated_ms,
        current_snapshot_id: m.current_snapshot_id,
        snapshots: m
            .snapshots
            .iter()
            .map(|s| IcebergSnapshot {
                snapshot_id: s.snapshot_id,
                timestamp_ms: s.timestamp_ms,
                manifest_list: s.manifest_list.clone(),
                summary: s.summary.clone(),
            })
            .collect(),
        schemas: m
            .schemas
            .iter()
            .map(|s| serde_json::to_value(s).unwrap_or_default())
            .collect(),
        current_schema_id: m.current_schema_id,
        partition_specs: m
            .partition_specs
            .iter()
            .map(|s| serde_json::to_value(s).unwrap_or_default())
            .collect(),
        default_spec_id: m.default_spec_id,
        last_assigned_field_id: Some(m.last_column_id),
        refs: HashMap::new(),
        properties: m.properties.clone(),
    }
}

/// A catalog adapter over the engine's trait object.
///
/// Construct with [`SpecCatalog::new`] and use it anywhere the crate's
/// `Catalog` trait is the contract — third-party scan planning, external
/// commit tooling, or the crate's transaction API.
pub struct SpecCatalog {
    inner: Arc<dyn EngineCatalog>,
}

impl SpecCatalog {
    pub fn new(inner: Arc<dyn EngineCatalog>) -> Self {
        Self { inner }
    }
}

#[async_trait::async_trait]
impl SpecCatalogTrait for SpecCatalog {
    async fn create_namespace(&self, namespace: &str) -> anyhow::Result<()> {
        self.inner.create_namespace(namespace).await
    }

    async fn namespace_exists(&self, namespace: &str) -> anyhow::Result<bool> {
        self.inner.namespace_exists(namespace).await
    }

    async fn drop_namespace(&self, namespace: &str) -> anyhow::Result<()> {
        self.inner.drop_namespace(namespace).await
    }

    async fn create_table(
        &self,
        namespace: &str,
        table: &str,
        schema: &benodb_iceberg::manifest_types::Schema,
        location: Option<&str>,
    ) -> anyhow::Result<IcebergTableMetadata> {
        // Spec schema → JSON → Arrow: the engine's create path thinks in
        // Arrow, and the conversion is the same one the manifest reader uses.
        let schema_json = serde_json::to_value(schema)?;
        let arrow_schema = crate::core::iceberg::iceberg_json_to_arrow_schema(&schema_json)?;
        self.inner
            .create_table(namespace, table, arrow_schema, location)
            .await?;
        self.load_table(namespace, table).await
    }

    async fn drop_table(&self, namespace: &str, table: &str) -> anyhow::Result<()> {
        self.inner.drop_table(namespace, table).await
    }

    async fn list_tables(&self, namespace: &str) -> anyhow::Result<Vec<String>> {
        self.inner.list_tables(namespace).await
    }

    async fn load_table(&self, namespace: &str, table: &str) -> anyhow::Result<IcebergTableMetadata> {
        let m = self.inner.load_table(namespace, table).await?;
        Ok(to_iceberg_metadata(&m))
    }

    async fn table_exists(&self, namespace: &str, table: &str) -> anyhow::Result<bool> {
        self.inner.table_exists(namespace, table).await
    }

    async fn commit_table(
        &self,
        namespace: &str,
        table: &str,
        requirements: &[TableRequirement],
        updates: &[TableUpdate],
    ) -> anyhow::Result<IcebergTableMetadata> {
        // The typed enums serialize to exactly the REST-spec JSON objects the
        // engine's backends already parse (internally tagged, kebab-case).
        let updates_json: Vec<serde_json::Value> = updates
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<_, _>>()?;
        let requirements_json: Vec<serde_json::Value> = requirements
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<_, _>>()?;

        self.inner
            .commit_table_with_requirements(namespace, table, updates_json, requirements_json)
            .await?;

        self.load_table(namespace, table).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The metadata conversion drops engine-only fields and maps the shared
    /// ones. The engine's shape is authoritative; the crate's shape is the
    /// spec subset third parties see.
    #[test]
    fn metadata_conversion_maps_the_shared_fields() {
        let engine = EngineTableMetadata {
            format_version: 2,
            table_uuid: "u-1".to_string(),
            location: "s3://bucket/t".to_string(),
            last_sequence_number: 9,
            last_updated_ms: 1_700,
            current_snapshot_id: Some(7),
            snapshots: vec![crate::core::metadata::Snapshot {
                snapshot_id: 7,
                parent_snapshot_id: None,
                timestamp_ms: 1_700,
                sequence_number: Some(9),
                summary: [("operation".to_string(), "append".to_string())]
                    .into_iter()
                    .collect(),
                manifest_list: "s3://bucket/t/mList.avro".to_string(),
                schema_id: None,
                first_row_id: None,
                added_rows: None,
            }],
            schemas: vec![crate::core::manifest::Schema::default()],
            current_schema_id: 0,
            partition_specs: vec![crate::core::manifest::PartitionSpec::default()],
            default_spec_id: 0,
            properties: HashMap::new(),
            ..Default::default()
        };

        let spec = to_iceberg_metadata(&engine);
        assert_eq!(spec.format_version, 2);
        assert_eq!(spec.table_uuid, "u-1");
        assert_eq!(spec.last_sequence_number, 9);
        assert_eq!(spec.current_snapshot_id, Some(7));
        assert_eq!(spec.snapshots.len(), 1);
        assert_eq!(spec.snapshots[0].snapshot_id, 7);
        assert_eq!(spec.snapshots[0].summary["operation"], "append");
        assert_eq!(spec.schemas.len(), 1);
        assert_eq!(spec.default_spec_id, 0);
        // The engine-only fields do not exist on the crate's shape at all.
    }

    /// The typed requirements serialize to the JSON the engine's backends
    /// already parse — the adapter's commit path is a serialization, not a
    /// translation.
    #[test]
    fn commit_payload_serialization_matches_the_rest_shapes() {
        let requirements = vec![TableRequirement::AssertTableUuid { uuid: "u-1".into() }];
        let updates = vec![TableUpdate::SetLocation {
            location: "s3://bucket/t2".to_string(),
        }];
        let requirements_json: Vec<serde_json::Value> = requirements
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<_, _>>()
            .unwrap();
        let updates_json: Vec<serde_json::Value> = updates
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(requirements_json[0]["requirement"], "assert-table-uuid");
        assert_eq!(updates_json[0]["action"], "set-location");
        assert_eq!(updates_json[0]["location"], "s3://bucket/t2");
    }
}
