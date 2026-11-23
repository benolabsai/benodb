// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benodb::core::table::Table;
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test]
async fn test_primary_key_uniqueness_enforcement() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    // 1. Setup Table with Primary Key
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

    let table = Table::create_async(uri.clone(), schema.clone()).await?;

    // Set the PK explicitly for this test
    table.set_primary_key(vec!["id".to_string()]);

    // 2. Write first batch (valid)
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )?;
    table.write_async(vec![batch1]).await?;

    // IMPORTANT: Commit and wait for background indexing to finish
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    // 3. Write second batch with DUPLICATE (id=3 already exists on disk)
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![3, 4, 5]))],
    )?;

    let res = table.write_async(vec![batch2]).await;

    assert!(res.is_err(), "Should have failed due to duplicate PK '3'");
    let err_msg = res.unwrap_err().to_string();
    println!("Caught expected error: {}", err_msg);
    assert!(err_msg.contains("Duplicate primary key error"));
    assert!(err_msg.contains("id = 3"));

    Ok(())
}

#[tokio::test]
async fn test_compound_primary_key_uniqueness() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let schema = Arc::new(Schema::new(vec![
        Field::new("pk1", DataType::Int32, false),
        Field::new("pk2", DataType::Int32, false),
        Field::new("val", DataType::Int32, true),
    ]));

    let table = Table::create_async(uri.clone(), schema.clone()).await?;
    table.set_primary_key(vec!["pk1".to_string(), "pk2".to_string()]);

    // Write (1, 100)
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![100])),
            Arc::new(Int32Array::from(vec![10])),
        ],
    )?;
    table.write_async(vec![batch1]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    // Write (1, 200) - Should SUCCEED (partial overlap)
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![200])),
            Arc::new(Int32Array::from(vec![20])),
        ],
    )?;
    table.write_async(vec![batch2]).await?;

    // Write (1, 100) - Should FAIL (exact overlap)
    let batch3 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![100])),
            Arc::new(Int32Array::from(vec![30])),
        ],
    )?;

    let res = table.write_async(vec![batch3]).await;
    assert!(
        res.is_err(),
        "Should have failed due to duplicate compound PK (1, 100)"
    );
    assert!(res
        .unwrap_err()
        .to_string()
        .contains("Duplicate primary key error"));

    Ok(())
}

#[tokio::test]
async fn test_primary_key_nullability_enforcement() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    // 1. Setup Table with Nullable Column
    let schema = Arc::new(Schema::new(vec![
        Field::new("pk", DataType::Int32, true), // ALLOW NULLS (for now)
    ]));

    let table = Table::create_async(uri.clone(), schema.clone()).await?;

    // 2. Set PK - should evolve schema to NOT NULL
    table.set_primary_key_async(vec!["pk".to_string()]).await?;

    // Verify Iceberg schema in manifest is now NOT NULL (required: true)
    let manifest_manager =
        benodb::core::manifest::ManifestManager::new(table.store.clone(), "", &table.uri);
    let (latest, _, _) = manifest_manager.load_latest_full().await?;
    let field = latest
        .schemas
        .last()
        .unwrap()
        .fields
        .iter()
        .find(|f| f.name == "pk")
        .unwrap();
    assert!(
        field.required,
        "PK column should have evolved to NOT NULL (expected required: true)"
    );

    // 3. Try to write NULL value - should be caught by pre-flight validation
    let batch_with_null = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]))],
    )?;

    let res = table.write_async(vec![batch_with_null]).await;
    assert!(
        res.is_err(),
        "Should have failed to write batch with NULL in PK column"
    );
    let err_msg = res.unwrap_err().to_string();
    assert!(
        err_msg.contains("Null constraint violation"),
        "Error message should mention null constraint violation"
    );
    assert!(
        err_msg.contains("'pk'"),
        "Error message should mention column 'pk'"
    );

    Ok(())
}

#[tokio::test]
async fn test_compound_primary_key_staged_write_uniqueness() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let schema = Arc::new(Schema::new(vec![
        Field::new("pk1", DataType::Int32, false),
        Field::new("pk2", DataType::Int32, false),
        Field::new("val", DataType::Int32, true),
    ]));

    let table = Table::create_async(uri.clone(), schema.clone()).await?;
    table.set_primary_key(vec!["pk1".to_string(), "pk2".to_string()]);

    // 1. Write and commit (10, 100)
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![10])),
            Arc::new(Int32Array::from(vec![100])),
            Arc::new(Int32Array::from(vec![1])),
        ],
    )?;
    table.write_async(vec![batch1]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    // 2. Stage (10, 101) - should SUCCEED (partial overlap, distinct composite key)
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![10])),
            Arc::new(Int32Array::from(vec![101])),
            Arc::new(Int32Array::from(vec![2])),
        ],
    )?;
    let staged_entries = table.stage_data_files_async(vec![batch2]).await?;
    assert!(!staged_entries.is_empty());
    table.commit_staged_entries_async(staged_entries).await?;
    table.wait_for_background_tasks_async().await?;

    // 3. Stage (10, 100) - should FAIL (exists in committed storage)
    let batch3 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![10])),
            Arc::new(Int32Array::from(vec![100])),
            Arc::new(Int32Array::from(vec![3])),
        ],
    )?;
    let res = table.stage_data_files_async(vec![batch3]).await;
    assert!(
        res.is_err(),
        "Should have failed due to duplicate compound PK in storage"
    );
    let err_msg = res.unwrap_err().to_string();
    assert!(err_msg.contains("Duplicate primary key error"));

    // 4. Stage duplicate within same staged batch - should FAIL
    let batch4 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![20, 20])),
            Arc::new(Int32Array::from(vec![200, 200])),
            Arc::new(Int32Array::from(vec![4, 5])),
        ],
    )?;
    let res4 = table.stage_data_files_async(vec![batch4]).await;
    assert!(
        res4.is_err(),
        "Should have failed due to duplicate compound PK within batch"
    );
    assert!(res4
        .unwrap_err()
        .to_string()
        .contains("Duplicate primary key error"));

    Ok(())
}

#[tokio::test]
async fn test_concurrent_primary_key_writes_serialized() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, true),
    ]));

    let table = Arc::new(Table::create_async(uri.clone(), schema.clone()).await?);
    table.set_primary_key(vec!["id".to_string()]);

    let batch_a = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![42])),
            Arc::new(Int32Array::from(vec![1])),
        ],
    )?;
    let batch_b = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![42])),
            Arc::new(Int32Array::from(vec![2])),
        ],
    )?;

    let t1 = table.clone();
    let t2 = table.clone();

    let h1 = tokio::spawn(async move { t1.write_async(vec![batch_a]).await });
    let h2 = tokio::spawn(async move { t2.write_async(vec![batch_b]).await });

    let (res1, res2) = tokio::join!(h1, h2);
    let r1 = res1?;
    let r2 = res2?;

    let (successes, failures) = match (r1, r2) {
        (Ok(_), Err(e)) => (1, vec![e]),
        (Err(e), Ok(_)) => (1, vec![e]),
        (Ok(_), Ok(_)) => (2, vec![]),
        (Err(e1), Err(e2)) => (0, vec![e1, e2]),
    };

    assert_eq!(successes, 1, "Exactly one concurrent write must succeed");
    assert_eq!(failures.len(), 1);
    assert!(failures[0]
        .to_string()
        .contains("Duplicate primary key error"));

    Ok(())
}

#[tokio::test]
async fn test_composite_pk_multi_row_single_batch() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let schema = Arc::new(Schema::new(vec![
        Field::new("tenant_id", DataType::Int32, false),
        Field::new("doc_id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, true),
    ]));

    let table = Table::create_async(uri.clone(), schema.clone()).await?;
    table.set_primary_key(vec!["tenant_id".to_string(), "doc_id".to_string()]);

    // Batch with repeated tenant_id (10), but distinct tuples (10, 100) and (10, 101)
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![10, 10])),
            Arc::new(Int32Array::from(vec![100, 101])),
            Arc::new(StringArray::from(vec!["doc1", "doc2"])),
        ],
    )?;

    // Must SUCCEED: distinct composite keys in the same batch
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    // Batch with duplicate tuple (10, 100)
    let dup_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![10])),
            Arc::new(Int32Array::from(vec![100])),
            Arc::new(StringArray::from(vec!["doc1_dup"])),
        ],
    )?;

    let res = table.write_async(vec![dup_batch]).await;
    assert!(res.is_err(), "Duplicate composite key must be rejected");
    assert!(res
        .unwrap_err()
        .to_string()
        .contains("Duplicate primary key error"));

    Ok(())
}

#[tokio::test]
async fn test_composite_pk_disambiguated_delimiter_values() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let schema = Arc::new(Schema::new(vec![
        Field::new("part_a", DataType::Utf8, false),
        Field::new("part_b", DataType::Utf8, false),
        Field::new("val", DataType::Int32, true),
    ]));

    let table = Table::create_async(uri.clone(), schema.clone()).await?;
    table.set_primary_key(vec!["part_a".to_string(), "part_b".to_string()]);

    // Distinct tuples that would collide under naive U+001F joining:
    // ("x\u{1f}y", "z") vs ("x", "y\u{1f}z")
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["x\u{1f}y"])),
            Arc::new(StringArray::from(vec!["z"])),
            Arc::new(Int32Array::from(vec![1])),
        ],
    )?;

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["x"])),
            Arc::new(StringArray::from(vec!["y\u{1f}z"])),
            Arc::new(Int32Array::from(vec![2])),
        ],
    )?;

    table.write_async(vec![batch1]).await?;
    table.commit_async().await?;

    // Second batch must SUCCEED because length-prefixed encoding distinguishes them
    table.write_async(vec![batch2]).await?;
    table.commit_async().await?;

    Ok(())
}

#[tokio::test]
async fn test_staged_writes_enforce_pk_uniqueness_at_commit_boundary() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, true),
    ]));

    let table = Table::create_async(uri.clone(), schema.clone()).await?;
    table.set_primary_key(vec!["id".to_string()]);

    let batch_task1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![99])),
            Arc::new(StringArray::from(vec!["from_task1"])),
        ],
    )?;

    let batch_task2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![99])),
            Arc::new(StringArray::from(vec!["from_task2"])),
        ],
    )?;

    // Both tasks stage concurrently against the same initial empty snapshot
    let staged1 = table.stage_data_files_async(vec![batch_task1]).await?;
    let staged2 = table.stage_data_files_async(vec![batch_task2]).await?;

    assert!(!staged1.is_empty());
    assert!(!staged2.is_empty());

    // Task 1 commits first
    table.commit_staged_entries_async(staged1).await?;

    // Task 2 attempts to commit the colliding key: MUST be rejected at commit boundary
    let res = table.commit_staged_entries_async(staged2).await;
    assert!(
        res.is_err(),
        "Task 2 must be rejected at commit boundary due to duplicate PK committed by Task 1"
    );
    assert!(res
        .unwrap_err()
        .to_string()
        .contains("Duplicate primary key error"));

    Ok(())
}
