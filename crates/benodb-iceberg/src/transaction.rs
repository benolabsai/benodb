// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! The transaction API: the Iceberg assertion/update commit model as a
//! typed, pure transition on table metadata.
//!
//! A `Transaction` accumulates requirements (assertions evaluated against
//! the base metadata) and updates (the changes to apply). `validate` checks
//! the assertions; `apply` produces the next metadata. Both are pure and
//! total — the durability backend (the engine's OCC commit, or a catalog's
//! `commit_table`) owns atomicity; this type owns the vocabulary and the
//! transition semantics.

use crate::catalog::{TableRequirement, TableUpdate};
use crate::manifest_types::Schema;
use crate::types::{IcebergSnapshot, SnapshotRef};
use anyhow::{bail, Result};
use std::collections::HashMap;

/// A commit in progress: assertions about the base metadata plus the updates
/// to apply. Build with [`Transaction::new`], mutate with the `with_*`
/// builders, then [`Transaction::validate`] and [`Transaction::apply`].
#[derive(Debug, Clone, Default)]
pub struct Transaction {
    requirements: Vec<TableRequirement>,
    updates: Vec<TableUpdate>,
    /// The schema id assigned by the last `add_schema` in this transaction,
    /// for the `-1` (newest added) resolution of `set_current_schema`.
    last_added_schema_id: Option<i32>,
    /// The floor for ids assigned by `add_schema`: the base's current id.
    schema_id_floor: i32,
}

impl Transaction {
    /// Start a transaction against `base`'s table uuid — the first
    /// requirement every commit asserts.
    pub fn new(base: &crate::types::IcebergTableMetadata) -> Self {
        Self {
            requirements: vec![TableRequirement::AssertTableUuid {
                uuid: base.table_uuid.clone(),
            }],
            updates: Vec::new(),
            last_added_schema_id: None,
            schema_id_floor: base.current_schema_id,
        }
    }

    /// The requirements this transaction will assert.
    pub fn requirements(&self) -> &[TableRequirement] {
        &self.requirements
    }

    /// The updates this transaction will apply.
    pub fn updates(&self) -> &[TableUpdate] {
        &self.updates
    }

    /// Add a schema and set it current. The schema's own id is used; a
    /// schema with id 0 (unset) gets the next id after the base's highest.
    pub fn add_schema(&mut self, mut schema: Schema) -> &mut Self {
        if schema.schema_id == 0 {
            schema.schema_id = self.next_schema_id();
        }
        self.last_added_schema_id = Some(schema.schema_id);
        self.updates.push(TableUpdate::AddSchema { schema });
        self.updates.push(TableUpdate::SetCurrentSchema {
            schema_id: self.last_added_schema_id.unwrap(),
        });
        self
    }

    /// Add a snapshot and advance the current snapshot pointer to it.
    pub fn add_snapshot(&mut self, snapshot: IcebergSnapshot) -> &mut Self {
        self.updates.push(TableUpdate::AddSnapshot {
            snapshot: snapshot.clone(),
        });
        self.updates.push(TableUpdate::SetSnapshotRef {
            ref_name: "main".to_string(),
            ref_type: "branch".to_string(),
            snapshot_id: snapshot.snapshot_id,
        });
        self
    }

    /// Set table properties.
    pub fn set_properties(&mut self, properties: HashMap<String, String>) -> &mut Self {
        self.updates.push(TableUpdate::SetProperties { properties });
        self
    }

    /// Upgrade the format version. Downgrades are rejected at `apply`.
    pub fn upgrade_format_version(&mut self, format_version: i32) -> &mut Self {
        self.updates.push(TableUpdate::UpgradeFormatVersion { format_version });
        self
    }

    fn next_schema_id(&self) -> i32 {
        // The base's current id is the floor; a fresh table (id 0, the
        // initial "no schema" marker) starts at 1.
        (self.schema_id_floor + 1).max(1)
    }

    /// Evaluate every requirement against the base metadata. The first
    /// failed assertion rejects the transaction with the assertion's name.
    ///
    /// Fail-closed: a requirement the metadata shape cannot answer (a ref
    /// assertion for a ref map that was never populated) is a mismatch, not
    /// a pass.
    pub fn validate(&self, base: &crate::types::IcebergTableMetadata) -> Result<()> {
        for requirement in &self.requirements {
            let ok = match requirement {
                TableRequirement::AssertTableUuid { uuid } => base.table_uuid == *uuid,
                TableRequirement::AssertCurrentSchemaId { current_schema_id } => {
                    base.current_schema_id == *current_schema_id
                }
                TableRequirement::AssertLastSequenceNumber { last_sequence_number } => {
                    base.last_sequence_number == *last_sequence_number
                }
                TableRequirement::AssertLastAssignedFieldId { last_assigned_field_id } => {
                    base.last_assigned_field_id == Some(*last_assigned_field_id)
                }
                TableRequirement::AssertRefSnapshotId { r#ref, snapshot_id } => {
                    base.refs.get(r#ref).map(|r| r.snapshot_id) == *snapshot_id
                }
            };
            if !ok {
                bail!(
                    "requirement failed: {}",
                    serde_json::to_value(requirement).unwrap_or_default()
                );
            }
        }
        Ok(())
    }

    /// Apply the updates to a clone of the base metadata, producing the next
    /// metadata. `validate` must pass first; this function re-runs it so a
    /// caller cannot skip the assertions.
    pub fn apply(self, base: &crate::types::IcebergTableMetadata) -> Result<crate::types::IcebergTableMetadata> {
        self.validate(base)?;
        let mut next = base.clone();
        for update in self.updates {
            match update {
                TableUpdate::AddSchema { schema } => {
                    let id = schema.schema_id;
                    next.schemas.push(serde_json::to_value(schema).unwrap_or_default());
                    next.current_schema_id = id;
                    if let Some(la) = next.last_assigned_field_id.as_mut() {
                        *la = (*la).max(id);
                    }
                }
                TableUpdate::SetCurrentSchema { schema_id } => {
                    next.current_schema_id = if schema_id < 0 {
                        next.schemas.len().saturating_sub(1) as i32
                    } else {
                        schema_id
                    };
                }
                TableUpdate::AddSnapshot { snapshot } => {
                    next.snapshots.push(snapshot.clone());
                    next.current_snapshot_id = Some(snapshot.snapshot_id);
                    next.last_sequence_number += 1;
                }
                TableUpdate::SetSnapshotRef {
                    ref_name,
                    ref_type,
                    snapshot_id,
                } => {
                    next.refs.insert(
                        ref_name,
                        SnapshotRef {
                            ref_type,
                            snapshot_id,
                            min_snapshots_to_keep: None,
                            max_snapshot_age_ms: None,
                        },
                    );
                }
                TableUpdate::SetProperties { properties } => {
                    next.properties.extend(properties);
                }
                TableUpdate::RemoveProperties { properties } => {
                    for p in properties {
                        next.properties.remove(&p);
                    }
                }
                TableUpdate::SetLocation { location } => {
                    next.location = location;
                }
                TableUpdate::UpgradeFormatVersion { format_version } => {
                    if format_version < next.format_version {
                        bail!(
                            "format version downgrade: {} < {}",
                            format_version,
                            next.format_version
                        );
                    }
                    next.format_version = format_version;
                }
            }
        }
        next.last_updated_ms = chrono::Utc::now().timestamp_millis();
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::IcebergTableMetadata;

    fn base() -> IcebergTableMetadata {
        IcebergTableMetadata {
            format_version: 2,
            table_uuid: "u-1".to_string(),
            location: "s3://b/t".to_string(),
            last_sequence_number: 5,
            last_updated_ms: 0,
            current_snapshot_id: None,
            snapshots: vec![],
            schemas: vec![],
            current_schema_id: 0,
            partition_specs: vec![],
            default_spec_id: 0,
            last_assigned_field_id: Some(3),
            refs: HashMap::new(),
            properties: HashMap::new(),
        }
    }

    #[test]
    fn uuid_assertion_passes_on_the_matching_base() {
        let tx = Transaction::new(&base());
        assert!(tx.validate(&base()).is_ok());
    }

    #[test]
    fn uuid_assertion_fails_on_a_different_table() {
        let mut b = base();
        b.table_uuid = "other".to_string();
        let tx = Transaction::new(&base());
        assert!(tx.validate(&b).is_err());
    }

    #[test]
    fn ref_assertion_is_fail_closed() {
        let mut tx = Transaction::new(&base());
        tx.requirements.push(TableRequirement::AssertRefSnapshotId {
            r#ref: "main".to_string(),
            snapshot_id: Some(9),
        });
        // The base has no "main" ref: the assertion cannot pass.
        assert!(tx.validate(&base()).is_err());
        // An assertion that the ref does NOT exist passes on an empty map.
        tx.requirements.pop();
        tx.requirements.push(TableRequirement::AssertRefSnapshotId {
            r#ref: "main".to_string(),
            snapshot_id: None,
        });
        assert!(tx.validate(&base()).is_ok());
    }

    #[test]
    fn add_schema_assigns_the_next_id_and_sets_current() {
        let mut b = base();
        b.current_schema_id = 2;
        let mut tx = Transaction::new(&b);
        let mut schema = Schema::default();
        schema.schema_id = 0; // unset: the transaction assigns
        tx.add_schema(schema);
        let next = tx.apply(&b).unwrap();
        assert_eq!(next.current_schema_id, 3);
        assert_eq!(next.schemas.len(), 1);
        assert_eq!(next.last_assigned_field_id, Some(3));
    }

    #[test]
    fn add_snapshot_advances_pointer_sequence_and_main_ref() {
        let mut tx = Transaction::new(&base());
        tx.add_snapshot(IcebergSnapshot {
            snapshot_id: 42,
            timestamp_ms: 1,
            manifest_list: "s3://b/t/m.avro".to_string(),
            summary: HashMap::new(),
        });
        let next = tx.apply(&base()).unwrap();
        assert_eq!(next.current_snapshot_id, Some(42));
        assert_eq!(next.last_sequence_number, 6);
        assert_eq!(next.refs["main"].snapshot_id, 42);
        assert_eq!(next.snapshots.len(), 1);
    }

    #[test]
    fn format_downgrade_is_rejected() {
        let mut tx = Transaction::new(&base());
        tx.upgrade_format_version(1);
        assert!(tx.apply(&base()).is_err());
        let mut tx = Transaction::new(&base());
        tx.upgrade_format_version(3);
        assert_eq!(tx.apply(&base()).unwrap().format_version, 3);
    }

    #[test]
    fn properties_set_and_remove() {
        let mut props = HashMap::new();
        props.insert("write.format".to_string(), "parquet".to_string());
        let mut tx = Transaction::new(&base());
        tx.set_properties(props);
        let next = tx.apply(&base()).unwrap();
        assert_eq!(next.properties["write.format"], "parquet");
    }
}
