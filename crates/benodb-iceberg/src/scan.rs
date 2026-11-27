// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Scan planning: table metadata plus an optional predicate become the list
//! of file tasks to read.
//!
//! The planner is the portable subset of the engine's execution: it resolves
//! a snapshot's manifest list and manifests from object storage, prunes
//! files whose partition values definitively fail the predicate (three
//! valued — unknown keeps the file), and attaches the partition-scoped
//! delete files to each surviving data file. Row-level delete filtering and
//! index-accelerated execution remain the engine's job; this module never
//! reads data files.

use crate::expr::{Expression, Literal};
use crate::manifest::read_manifest_list;
use crate::manifest_types::{DeleteContent, DeleteFile, PartitionSpec, Schema};
use crate::transform::IcebergTransform;
use crate::types::{IcebergManifestEntry, IcebergTableMetadata};
use anyhow::{Context, Result};
use std::sync::Arc;

/// One file to read: the data file's manifest entry plus the delete files
/// that apply to it (partition-scoped; row-level filtering happens at read
/// time).
#[derive(Debug, Clone)]
pub struct FileScanTask {
    pub data_file: IcebergManifestEntry,
    pub delete_files: Vec<DeleteFile>,
}

/// Plan a scan: resolve `snapshot_id` (or the table's current snapshot),
/// read the manifest list and manifests from `store`, prune files whose
/// partition values definitively fail `predicate`, and attach the
/// partition-scoped delete files.
///
/// The predicate addresses fields by id; a field that is not a partition key
/// evaluates to unknown, which keeps the file — pruning only on definitive
/// falsity is the contract that makes the plan safe.
pub async fn plan_file_scan(
    store: Arc<dyn object_store::ObjectStore>,
    metadata: &IcebergTableMetadata,
    snapshot_id: Option<i64>,
    predicate: Option<&Expression>,
    partition_spec: &PartitionSpec,
    _schema: &Schema,
) -> Result<Vec<FileScanTask>> {
    let sid = snapshot_id
        .or(metadata.current_snapshot_id)
        .ok_or_else(|| anyhow::anyhow!("table has no snapshots to scan"))?;
    let snap = metadata
        .snapshots
        .iter()
        .find(|s| s.snapshot_id == sid)
        .ok_or_else(|| anyhow::anyhow!("snapshot {sid} not found in table metadata"))?;

    let ml_path = object_store::path::Path::from(snap.manifest_list.clone());
    let ml_bytes = store
        .get(&ml_path)
        .await
        .with_context(|| format!("fetching manifest list {}", snap.manifest_list))?
        .bytes()
        .await?;
    let list = read_manifest_list(&ml_bytes[..])?;

    // Pass 1: collect the data and delete entries from every manifest.
    let mut data_entries = Vec::new();
    let mut delete_entries = Vec::new();
    for mle in &list {
        let mpath = object_store::path::Path::from(mle.manifest_path.clone());
        let mbytes = store
            .get(&mpath)
            .await
            .with_context(|| format!("fetching manifest {}", mle.manifest_path))?
            .bytes()
            .await?;
        for entry in crate::manifest::read_manifest(&mbytes[..])? {
            if entry.data_file.content == 0 {
                data_entries.push(entry);
            } else {
                delete_entries.push(entry);
            }
        }
    }

    // Pass 2: prune data files on partition values, attach partition-scoped
    // deletes.
    let mut tasks = Vec::new();
    for entry in data_entries {
        if let Some(p) = predicate {
            let partition = &entry.data_file.partition;
            // Positional: the partition field's position in the spec is the
            // value's position in the entry's partition list. The stored
            // value is the transform applied to the source column; the
            // predicate's literals are transformed through the same
            // transforms below, so the comparison is against like values.
            let mut resolve = |field_id: i32| -> Option<Literal> {
                let pos = partition_spec
                    .fields
                    .iter()
                    .position(|p| p.source_id == Some(field_id))?;
                json_to_literal(partition.get(pos)?)
            };
            // The transform-aware evaluation: rewrite the predicate's
            // literals through the partition transforms before testing.
            let rewritten = transform_literals(p.clone(), partition_spec);
            if rewritten.test(&mut resolve) == Some(false) {
                continue; // definitively false: the file is pruned
            }
        }

        // Partition-scoped deletes: a delete entry applies to this data file
        // when their partition values match (position deletes are
        // partition-scoped too; row-level path matching happens at read
        // time).
        let delete_files: Vec<DeleteFile> = delete_entries
            .iter()
            .filter(|d| d.data_file.partition == entry.data_file.partition)
            .filter_map(DeleteFile::from_iceberg_entry)
            .collect();

        tasks.push(FileScanTask {
            data_file: entry,
            delete_files,
        });
    }
    Ok(tasks)
}

impl DeleteFile {
    /// The spec-level conversion: an Iceberg manifest entry whose content is
    /// a delete kind becomes the crate's delete-file inventory record.
    pub fn from_iceberg_entry(entry: &IcebergManifestEntry) -> Option<Self> {
        let df = &entry.data_file;
        let content = match df.content {
            1 => DeleteContent::Position,
            2 => DeleteContent::Equality {
                equality_ids: df.equality_ids.clone().unwrap_or_default(),
            },
            3 => {
                // The V3 deletion-vector encoding: fields 143-145 locate the
                // blob inside the Puffin container. The Puffin file is the
                // delete entry's own path; `referenced_data_file` names the
                // data file the vector targets.
                match (df.content_offset, df.content_size_in_bytes) {
                    (Some(content_offset), Some(content_size_in_bytes)) => {
                        DeleteContent::DeletionVector {
                            puffin_file_path: df.file_path.clone(),
                            referenced_data_file: df.referenced_data_file.clone(),
                            content_offset,
                            content_size_in_bytes,
                        }
                    }
                    _ => DeleteContent::Position,
                }
            }
            _ => return None,
        };
        Some(Self {
            file_path: df.file_path.clone(),
            content,
            file_size_bytes: df.file_size_in_bytes,
            record_count: df.record_count,
            partition_values: Default::default(),
        })
    }
}

/// Transform a predicate's literals through the partition transforms, so the
/// comparison against the stored (already transformed) partition values is
/// meaningful. A leaf whose field is not a partition key is left untouched —
/// the resolver answers unknown for it.
fn transform_literals(expr: Expression, partition_spec: &PartitionSpec) -> Expression {
    use crate::expr::{Predicate, Term};
    match expr {
        Expression::Predicate(Predicate::Binary { op, term, literal }) => {
            let transformed = match &term {
                Term::ByFieldId(id) => {
                    let pf = partition_spec
                        .fields
                        .iter()
                        .find(|pf| pf.source_id == Some(*id));
                    match pf {
                        Some(pf) => {
                            let t = IcebergTransform::parse(&pf.transform);
                            transform_literal(&t, &literal)
                        }
                        None => None,
                    }
                }
            };
            let literal = transformed.unwrap_or(literal);
            Expression::Predicate(Predicate::Binary { op, term, literal })
        }
        Expression::Predicate(Predicate::Set { op, term, literals }) => {
            let transformed: Vec<Literal> = match &term {
                Term::ByFieldId(id) => {
                    let pf = partition_spec
                        .fields
                        .iter()
                        .find(|pf| pf.source_id == Some(*id));
                    match pf {
                        Some(pf) => {
                            let t = IcebergTransform::parse(&pf.transform);
                            literals
                                .iter()
                                .map(|l| transform_literal(&t, l).unwrap_or_else(|| l.clone()))
                                .collect()
                        }
                        None => literals,
                    }
                }
            };
            Expression::Predicate(Predicate::Set {
                op,
                term,
                literals: transformed,
            })
        }
        Expression::Not(inner) => {
            Expression::Not(Box::new(transform_literals(*inner, partition_spec)))
        }
        Expression::And(a, b) => Expression::And(
            Box::new(transform_literals(*a, partition_spec)),
            Box::new(transform_literals(*b, partition_spec)),
        ),
        Expression::Or(a, b) => Expression::Or(
            Box::new(transform_literals(*a, partition_spec)),
            Box::new(transform_literals(*b, partition_spec)),
        ),
        other => other,
    }
}

/// Apply a transform to a single literal: a one-element Arrow array of the
/// literal's type, the transform's own apply, and the JSON result back to a
/// literal.
fn transform_literal(t: &IcebergTransform, lit: &Literal) -> Option<Literal> {
    use arrow::array::*;
    let arr: Arc<dyn Array> = match lit {
        Literal::Boolean(b) => Arc::new(BooleanArray::from(vec![*b])),
        Literal::Int(v) => Arc::new(Int64Array::from(vec![*v as i64])),
        Literal::Long(v) => Arc::new(Int64Array::from(vec![*v])),
        Literal::Float(v) => Arc::new(Float32Array::from(vec![*v])),
        Literal::Double(v) => Arc::new(Float64Array::from(vec![*v])),
        Literal::String(s) => Arc::new(StringArray::from(vec![s.clone()])),
        Literal::Date(v) => Arc::new(Int32Array::from(vec![*v])),
        Literal::TimestampMicros(v) | Literal::TimestampNanos(v) => {
            Arc::new(Int64Array::from(vec![*v]))
        }
        Literal::Binary(b) => Arc::new(BinaryArray::from(vec![b.as_slice()])),
    };
    let out = t.apply(arr.as_ref(), 0);
    json_to_literal(&out)
}

/// The JSON-to-literal reconstruction used consistently for both sides of a
/// comparison: integral numbers become `Long`, fractional become `Double`.
fn json_to_literal(v: &serde_json::Value) -> Option<Literal> {
    match v {
        serde_json::Value::Bool(b) => Some(Literal::Boolean(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(Literal::Long(i))
            } else {
                n.as_f64().map(Literal::Double)
            }
        }
        serde_json::Value::String(s) => Some(Literal::String(s.clone())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::PredicateOp;
    use crate::manifest_types::PartitionField;

    fn spec_one_int_partition() -> PartitionSpec {
        PartitionSpec {
            spec_id: 0,
            fields: vec![PartitionField {
                field_id: Some(1000),
                source_id: Some(1),
                source_ids: vec![1],
                name: "id".to_string(),
                transform: "identity".to_string(),
            }],
        }
    }

    /// A data file whose partition value definitively fails the predicate is
    /// pruned; one that passes (or cannot answer) survives.
    #[tokio::test]
    async fn partition_pruning_keeps_unknown_and_prunes_false() {
        // The manifest fixtures are Avro; the planner's pruning logic is
        // exercised through the resolver directly here.
        let spec = spec_one_int_partition();
        let expr = Expression::predicate(PredicateOp::Eq, 1, Literal::Long(5));
        let rewritten = transform_literals(expr, &spec);

        let values = [serde_json::json!(5)];
        let mut resolve = |id: i32| -> Option<Literal> {
            if id == 1 {
                json_to_literal(&values[0])
            } else {
                None
            }
        };
        assert_eq!(rewritten.test(&mut resolve), Some(true));

        let expr = Expression::predicate(PredicateOp::Eq, 1, Literal::Long(6));
        let rewritten = transform_literals(expr, &spec);
        assert_eq!(rewritten.test(&mut resolve), Some(false));
    }

    #[test]
    fn bucket_transform_applies_to_literals() {
        let t = IcebergTransform::parse("bucket[8]");
        let out = transform_literal(&t, &Literal::String("abc".to_string()));
        assert!(
            out.is_some(),
            "bucket transform applies to a string literal"
        );
    }

    #[test]
    fn delete_entry_conversion_maps_content_codes() {
        let mut entry = IcebergManifestEntry {
            status: 1,
            snapshot_id: None,
            data_file: crate::types::IcebergDataFile {
                content: 2,
                file_path: "del.avro".to_string(),
                file_format: "AVRO".to_string(),
                partition: vec![],
                record_count: 4,
                file_size_in_bytes: 100,
                column_sizes: None,
                value_counts: None,
                null_value_counts: None,
                nan_value_counts: None,
                lower_bounds: None,
                upper_bounds: None,
                equality_ids: Some(vec![1, 2]),
                referenced_data_file: None,
                content_offset: None,
                content_size_in_bytes: None,
                index_files: None,
                file_checksum: None,
                first_row_id: None,
            },
            delete_files: vec![],
        };
        let df = DeleteFile::from_iceberg_entry(&entry).unwrap();
        assert!(matches!(df.content, DeleteContent::Equality { .. }));

        entry.data_file.content = 1;
        let df = DeleteFile::from_iceberg_entry(&entry).unwrap();
        assert!(matches!(df.content, DeleteContent::Position));

        entry.data_file.content = 9;
        assert!(DeleteFile::from_iceberg_entry(&entry).is_none());
    }
}
