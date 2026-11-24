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

#[tokio::test]
async fn test_cross_process_composite_pk_race_independent_table_instances() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let schema = Arc::new(Schema::new(vec![
        Field::new("tenant_id", DataType::Int32, false),
        Field::new("doc_id", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, true),
    ]));

    let table_init = Table::create_async(uri.clone(), schema.clone()).await?;
    table_init
        .set_primary_key_async(vec!["tenant_id".to_string(), "doc_id".to_string()])
        .await?;

    // Open two independent Table instances simulating separate processes/drivers
    let table_a = Table::new_async(uri.clone()).await?;
    let table_b = Table::new_async(uri.clone()).await?;

    let batch_a = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![10])),
            Arc::new(Int32Array::from(vec![100])),
            Arc::new(StringArray::from(vec!["writer_a"])),
        ],
    )?;

    let batch_b = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![10])),
            Arc::new(Int32Array::from(vec![100])),
            Arc::new(StringArray::from(vec!["writer_b"])),
        ],
    )?;

    // Both independent tables stage files concurrently
    let staged_a = table_a.stage_data_files_async(vec![batch_a]).await?;
    let staged_b = table_b.stage_data_files_async(vec![batch_b]).await?;

    // Attempt concurrent staged commit across independent table instances
    let (res_a, res_b) = tokio::join!(
        table_a.commit_staged_entries_async(staged_a),
        table_b.commit_staged_entries_async(staged_b)
    );

    let successes = [res_a.is_ok(), res_b.is_ok()]
        .iter()
        .filter(|&&x| x)
        .count();
    let failures = [res_a.is_err(), res_b.is_err()]
        .iter()
        .filter(|&&x| x)
        .count();

    assert_eq!(
        successes, 1,
        "Exactly one cross-process staged commit must succeed"
    );
    assert_eq!(
        failures, 1,
        "The competing cross-process staged commit must be rejected"
    );

    let err_msg = if let Err(e) = res_a {
        e.to_string()
    } else {
        res_b.unwrap_err().to_string()
    };
    assert!(
        err_msg.contains("Duplicate primary key error"),
        "Expected PrimaryKeyViolation, got {err_msg}"
    );

    Ok(())
}

#[tokio::test]
async fn test_mixed_write_path_pk_race_buffered_and_staged() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, true),
    ]));

    let table = Table::create_async(uri.clone(), schema.clone()).await?;
    table.set_primary_key(vec!["id".to_string()]);

    let batch_buffered = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![77])),
            Arc::new(StringArray::from(vec!["buffered"])),
        ],
    )?;

    let batch_staged = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![77])),
            Arc::new(StringArray::from(vec!["staged"])),
        ],
    )?;

    // Stage the file first (not committed yet)
    let staged = table.stage_data_files_async(vec![batch_staged]).await?;

    // Perform buffered write into memory
    table.write_async(vec![batch_buffered]).await?;

    // Now attempt to commit staged entries with colliding key: must be detected in pending writes!
    let res = table.commit_staged_entries_async(staged).await;
    assert!(
        res.is_err(),
        "Staged commit must reject key present in active write buffer"
    );
    assert!(res
        .unwrap_err()
        .to_string()
        .contains("Duplicate primary key error"));

    Ok(())
}

#[tokio::test]
async fn test_staged_validation_rejects_missing_schema_or_batch_pk_fields() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, true),
    ]));

    let table = Table::create_async(uri.clone(), schema.clone()).await?;
    // Configure a PK column that doesn't exist in the schema
    table.set_primary_key(vec!["non_existent_col".to_string()]);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(StringArray::from(vec!["test"])),
        ],
    )?;

    let staged = table.stage_data_files_async(vec![batch]).await?;
    let res = table.commit_staged_entries_async(staged).await;
    assert!(
        res.is_err(),
        "Must fail when primary key column is not in schema"
    );
    let err_str = res.unwrap_err().to_string();
    assert!(
        err_str.contains("ColumnNotFound") || err_str.contains("non_existent_col"),
        "Expected ColumnNotFound error, got: {err_str}"
    );

    Ok(())
}

#[tokio::test]
async fn test_pk_lock_timeout_fails_closed() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, true),
    ]));

    let table = Table::create_async(uri.clone(), schema.clone()).await?;
    table.set_primary_key_async(vec!["id".to_string()]).await?;

    // Pre-acquire the table's pk_commit lock externally to simulate another process holding it
    let lock_path = table.pk_lock_path();
    let external_lock =
        benodb::core::lock::FileBasedLock::new(table.store.clone(), lock_path.clone(), 300);
    let _held_guard = external_lock
        .try_acquire()
        .await?
        .expect("External lock must be acquired");

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(StringArray::from(vec!["test"])),
        ],
    )?;

    let staged = table.stage_data_files_async(vec![batch]).await?;

    // Staged commit must fail closed with LockAcquireFailed, rather than proceeding without the lock
    let res = table.commit_staged_entries_async(staged).await;
    assert!(
        res.is_err(),
        "Commit must fail closed when PK lock is held by another process"
    );
    let err_str = res.unwrap_err().to_string();
    assert!(
        err_str.contains("LockAcquireFailed")
            || err_str.contains("Failed to acquire distributed lock")
            || err_str.to_lowercase().contains("failed to acquire"),
        "Expected lock acquisition failure, got: {err_str}"
    );

    Ok(())
}

#[tokio::test]
async fn test_cross_instance_buffered_vs_staged_pk_collision() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, true),
    ]));

    let table_init = Table::create_async(uri.clone(), schema.clone()).await?;
    table_init
        .set_primary_key_async(vec!["id".to_string()])
        .await?;

    // Open two independent Table instances pointing to the same storage
    let table_a = Table::new_async(uri.clone()).await?;
    let table_b = Table::new_async(uri.clone()).await?;

    let batch_a = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![999])),
            Arc::new(StringArray::from(vec!["staged_by_a"])),
        ],
    )?;

    let batch_b = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![999])),
            Arc::new(StringArray::from(vec!["buffered_by_b"])),
        ],
    )?;

    // Table B writes key 999 into its memory buffer before Table A commits to storage
    table_b.write_async(vec![batch_b]).await?;

    // Table A stages and commits key 999 to durable storage concurrently
    let staged_a = table_a.stage_data_files_async(vec![batch_a]).await?;
    table_a.commit_staged_entries_async(staged_a).await?;

    // Table B flushes its buffer: flush must fail during storage validation under pk_commit lock
    let flush_res = table_b.flush_async().await;
    assert!(
        flush_res.is_err(),
        "Table B flush must be rejected due to collision with committed key 999"
    );
    let err_str = flush_res.unwrap_err().to_string();
    assert!(
        err_str.contains("Duplicate primary key error") || err_str.contains("PrimaryKeyViolation"),
        "Expected primary key violation on flush, got: {err_str}"
    );

    Ok(())
}

#[tokio::test]
async fn test_pk_lock_path_full_uri_isolation() -> anyhow::Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

    let dir1 = tempdir()?;
    let dir2 = tempdir()?;

    let uri_prod = format!("file://{}/prod/customer_a", dir1.path().display());
    let uri_archive = format!("file://{}/archive/customer_a", dir2.path().display());

    let table_prod = Table::create_async(uri_prod.clone(), schema.clone()).await?;
    let table_archive = Table::create_async(uri_archive, schema.clone()).await?;

    let lock_prod = table_prod.pk_lock_path();
    let lock_archive = table_archive.pk_lock_path();

    assert_ne!(
        lock_prod, lock_archive,
        "Tables with matching slug but distinct URI paths must not share pk_commit locks"
    );

    // Verify canonicalization equivalence for trailing slashes and relative segments
    let uri_prod_slash = format!("{}/", uri_prod);
    let uri_prod_dot = format!("{}/./", uri_prod);
    let canon1 = Table::canonicalize_table_uri(&uri_prod);
    let canon2 = Table::canonicalize_table_uri(&uri_prod_slash);
    let canon3 = Table::canonicalize_table_uri(&uri_prod_dot);
    assert_eq!(
        canon1, canon2,
        "Trailing slash must produce identical canonical URI"
    );
    assert_eq!(
        canon1, canon3,
        "Relative segment must produce identical canonical URI"
    );

    Ok(())
}

#[tokio::test]
async fn test_pk_lock_path_cloud_uri_scheme_aliases() -> anyhow::Result<()> {
    // 1. S3 and S3A scheme aliases
    let s3_uri = "s3://production-lakehouse/finance/transactions";
    let s3a_uri = "s3a://production-lakehouse/finance/transactions";
    let s3_case_uri = "S3A://PRODUCTION-LAKEHOUSE/finance/transactions/";
    assert_eq!(
        Table::canonicalize_table_uri(s3_uri),
        Table::canonicalize_table_uri(s3a_uri),
        "s3:// and s3a:// must canonicalize to identical storage URI"
    );
    assert_eq!(
        Table::canonicalize_table_uri(s3_uri),
        Table::canonicalize_table_uri(s3_case_uri),
        "Mixed case and trailing slash must canonicalize to identical storage URI"
    );
    assert_eq!(
        Table::pk_lock_path_for_uri(s3_uri),
        Table::pk_lock_path_for_uri(s3a_uri),
        "s3:// and s3a:// must derive identical pk_commit lock paths"
    );
    assert_eq!(
        Table::pk_lock_path_for_uri(s3_uri),
        Table::pk_lock_path_for_uri(s3_case_uri),
        "Case-variant S3 URIs must derive identical pk_commit lock paths"
    );

    // 2. GS and GCS scheme aliases
    let gs_uri = "gs://analytics-bucket/warehouse/events";
    let gcs_uri = "gcs://analytics-bucket/warehouse/events";
    let gcs_slash_uri = "gcs://ANALYTICS-BUCKET/warehouse/events/";
    assert_eq!(
        Table::canonicalize_table_uri(gs_uri),
        Table::canonicalize_table_uri(gcs_uri),
        "gs:// and gcs:// must canonicalize to identical storage URI"
    );
    assert_eq!(
        Table::canonicalize_table_uri(gs_uri),
        Table::canonicalize_table_uri(gcs_slash_uri),
        "Case-variant GCS URIs must canonicalize to identical storage URI"
    );
    assert_eq!(
        Table::pk_lock_path_for_uri(gs_uri),
        Table::pk_lock_path_for_uri(gcs_uri),
        "gs:// and gcs:// must derive identical pk_commit lock paths"
    );
    assert_eq!(
        Table::pk_lock_path_for_uri(gs_uri),
        Table::pk_lock_path_for_uri(gcs_slash_uri),
        "Case-variant GCS URIs must derive identical pk_commit lock paths"
    );

    // 3. Azure: AZ, ABFS, and ABFSS scheme aliases
    let az_uri = "az://blob-container/core/customers";
    let abfs_uri = "abfs://blob-container/core/customers";
    let abfss_uri = "abfss://BLOB-CONTAINER/core/customers/";
    assert_eq!(
        Table::canonicalize_table_uri(az_uri),
        Table::canonicalize_table_uri(abfs_uri),
        "az:// and abfs:// must canonicalize to identical storage URI"
    );
    assert_eq!(
        Table::canonicalize_table_uri(az_uri),
        Table::canonicalize_table_uri(abfss_uri),
        "az:// and abfss:// must canonicalize to identical storage URI"
    );
    assert_eq!(
        Table::pk_lock_path_for_uri(az_uri),
        Table::pk_lock_path_for_uri(abfs_uri),
        "az:// and abfs:// must derive identical pk_commit lock paths"
    );
    assert_eq!(
        Table::pk_lock_path_for_uri(az_uri),
        Table::pk_lock_path_for_uri(abfss_uri),
        "az:// and abfss:// must derive identical pk_commit lock paths"
    );

    // 4. Distinct tables and buckets must produce isolated lock paths
    let s3_other_table = "s3://production-lakehouse/finance/accounts";
    let s3_other_bucket = "s3://staging-lakehouse/finance/transactions";
    assert_ne!(
        Table::pk_lock_path_for_uri(s3_uri),
        Table::pk_lock_path_for_uri(s3_other_table),
        "Distinct table paths within the same bucket must have isolated lock paths"
    );
    assert_ne!(
        Table::pk_lock_path_for_uri(s3_uri),
        Table::pk_lock_path_for_uri(s3_other_bucket),
        "Matching table paths in different buckets must have isolated lock paths"
    );

    Ok(())
}

#[tokio::test]
async fn test_simultaneous_buffered_and_staged_pk_collision() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, true),
    ]));

    let table_init = Table::create_async(uri.clone(), schema.clone()).await?;
    table_init
        .set_primary_key_async(vec!["id".to_string()])
        .await?;

    // Open two independent Table instances pointing to the same storage
    let table_a = Table::new_async(uri.clone()).await?;
    let table_b = Table::new_async(uri.clone()).await?;

    let batch_a = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![555])),
            Arc::new(StringArray::from(vec!["staged_555"])),
        ],
    )?;

    let batch_b = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![555])),
            Arc::new(StringArray::from(vec!["buffered_555"])),
        ],
    )?;

    // Table A stages data files (not yet committed)
    let staged_a = table_a.stage_data_files_async(vec![batch_a]).await?;

    // Table B writes to write buffer (not yet flushed)
    table_b.write_async(vec![batch_b]).await?;

    // Barrier synchronization to release both commit attempts simultaneously
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let b1 = barrier.clone();
    let b2 = barrier.clone();

    let handle_a = tokio::spawn(async move {
        b1.wait().await;
        table_a.commit_staged_entries_async(staged_a).await
    });

    let handle_b = tokio::spawn(async move {
        b2.wait().await;
        table_b.flush_async().await
    });

    let (res_a, res_b) = tokio::join!(handle_a, handle_b);
    let res_a = res_a?;
    let res_b = res_b?;

    let successes = [res_a.is_ok(), res_b.is_ok()]
        .iter()
        .filter(|&&x| x)
        .count();
    let failures = [res_a.is_err(), res_b.is_err()]
        .iter()
        .filter(|&&x| x)
        .count();

    assert_eq!(
        successes, 1,
        "Exactly one concurrent write path must commit key 555"
    );
    assert_eq!(
        failures, 1,
        "Competing write path must be rejected with duplicate primary key error"
    );

    let err_str = if let Err(e) = res_a {
        e.to_string()
    } else {
        res_b.unwrap_err().to_string()
    };
    assert!(
        err_str.contains("Duplicate primary key error") || err_str.contains("PrimaryKeyViolation"),
        "Expected primary key error, got: {err_str}"
    );

    // Verify durable storage contains exactly one row
    let table_verify = Table::new_async(uri.clone()).await?;
    let manager =
        benodb::core::manifest::ManifestManager::new(table_verify.store.clone(), "", &uri);
    let (manifest, _) = manager.load_latest_direct().await?;
    let all_entries = manager.load_all_entries(&manifest).await?;
    let total_rows: i64 = all_entries.iter().map(|e| e.record_count).sum();
    assert_eq!(
        total_rows, 1,
        "Table must contain exactly one committed row"
    );

    Ok(())
}

#[tokio::test]
async fn test_pk_publication_fenced_when_lease_lost() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, true),
    ]));

    let table = Table::create_async(uri.clone(), schema.clone()).await?;
    table.set_primary_key_async(vec!["id".to_string()]).await?;

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![42])),
            Arc::new(StringArray::from(vec!["fenced"])),
        ],
    )?;

    let staged = table.stage_data_files_async(vec![batch]).await?;

    // Acquire lock and obtain fencing verifier
    let lock =
        benodb::core::lock::FileBasedLock::new(table.store.clone(), table.pk_lock_path(), 30);
    let guard = lock.try_acquire().await?.expect("Must acquire lock");
    let verifier = guard.fencing_verifier();

    // Release/steal lock externally before manifest commit
    guard.release().await;

    // Verify fencing verifier closure fails closed immediately on released/lost lease
    assert!(
        (verifier)().await.is_err(),
        "verifier().await must fail closed when lock lease is released or lost"
    );

    // Check version before commit attempt
    let manifest_manager =
        benodb::core::manifest::ManifestManager::new(table.store.clone(), "", &uri);
    let (_, initial_ver) = manifest_manager.load_latest_direct().await?;

    // Construct commit metadata carrying the now-fenced verifier
    let commit_metadata = benodb::core::manifest::CommitMetadata {
        publication_fence: Some(verifier),
        ..Default::default()
    };

    // Attempt manifest publication directly through ManifestManager: must be rejected
    let commit_res = manifest_manager.commit(&staged, &[], commit_metadata).await;
    assert!(
        commit_res.is_err(),
        "ManifestManager::commit must refuse publication when publication fence rejects"
    );
    let err_str = commit_res.unwrap_err().to_string();
    assert!(
        err_str.contains("Lease lost")
            || err_str.contains("Lease fencing violation")
            || err_str.contains("already been released"),
        "Expected lease fencing error, got: {err_str}"
    );

    // Verify durable storage was NOT updated
    let (_, final_ver) = manifest_manager.load_latest_direct().await?;
    assert_eq!(
        final_ver, initial_ver,
        "No new manifest version must be published when publication fencing fails"
    );

    Ok(())
}
