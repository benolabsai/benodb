// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Proves a vector query reaches the index instead of decoding the table.
//!
//! Results alone cannot show this: a flat scan returns the same nearest
//! neighbours as an index, so a soak that only checks the rows passes when the
//! index is missing, ignored, or silently dropped. The reader counters are the
//! mechanism signal, and this test measures both sides of it. The indexed table
//! must decode a small fraction of its rows, and an unindexed table with the
//! same data must decode all of them, which is what makes the first number
//! meaningful rather than a counter that reports nothing.

use arrow::array::{FixedSizeListArray, Int64Array};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use benodb::core::index::VectorValue;
use benodb::core::manifest::IndexAlgorithm;
use benodb::core::reader::stats::ReaderCounters;
use benodb::core::table::VectorSearchParams;
use benodb::Table;
use std::sync::Arc;
use tempfile::tempdir;

const DIM: usize = 64;
const ROWS: usize = 2_000;
const K: usize = 5;

fn batch(start: i64) -> RecordBatch {
    let ids: Vec<i64> = (start..start + ROWS as i64).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
    ]));

    let vectors: Vec<Option<Vec<Option<f32>>>> = ids
        .iter()
        .map(|id| {
            Some(
                (0..DIM)
                    .map(|d| Some(((id % 97) as f32) * 0.01 + (d as f32) * 0.001))
                    .collect(),
            )
        })
        .collect();

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vectors, DIM as i32),
            ),
        ],
    )
    .expect("build batch")
}

/// Run one k nearest neighbour query and report what the reader did.
async fn query(table: &Table) -> (usize, ReaderCounters) {
    ReaderCounters::reset();
    let params = VectorSearchParams::new("embedding", VectorValue::Float32(vec![0.5; DIM]), K);
    let batches = table
        .read_async(None, Some(vec![params]), None)
        .await
        .expect("vector query");
    let rows = batches.iter().map(|b| b.num_rows()).sum();
    (rows, ReaderCounters::load())
}

#[tokio::test]
async fn indexed_vector_query_decodes_a_fraction_of_the_table() -> anyhow::Result<()> {
    let indexed_dir = tempdir()?;
    let indexed_uri = format!("file://{}", indexed_dir.path().to_str().expect("utf8"));
    let indexed = Table::new_async(indexed_uri).await?;
    // Two segments: a single segment is already one compact unit, and
    // `compact_indexes` correctly declines to build an artifact for it.
    indexed.write_async(vec![batch(0)]).await?;
    indexed.commit_async().await?;
    indexed.write_async(vec![batch(ROWS as i64)]).await?;
    indexed.commit_async().await?;
    indexed
        .add_index("embedding".to_string(), IndexAlgorithm::hnsw_tq8())
        .await?;
    indexed.wait_for_background_tasks_async().await?;

    let (rows, counters) = query(&indexed).await;
    assert_eq!(rows, K, "the indexed query must return k rows");
    assert!(
        counters.rows_decoded > 0,
        "the counters did not observe the indexed read path"
    );
    assert!(
        counters.rows_decoded * 4 < (2 * ROWS) as u64,
        "the indexed query decoded {} of {} rows; an index resolves candidates, \
         a scan decodes the table",
        counters.rows_decoded,
        2 * ROWS
    );

    // The table level tier must report through the same counters. Compaction
    // publishes one artifact covering the single segment, so the query resolves
    // through the cluster artifact and materialises k rows, not the table.
    let compaction = indexed
        .compact_indexes_async("embedding", None)
        .await
        .expect("compact the table level artifact");
    assert!(
        compaction.output_shards >= 1,
        "compaction published no shard"
    );
    let (cluster_rows, cluster_counters) = query(&indexed).await;
    assert_eq!(cluster_rows, K, "the cluster tier must return k rows");
    assert!(
        cluster_counters.rows_selected > 0,
        "the cluster tier reported no counters"
    );
    assert!(
        cluster_counters.rows_decoded * 4 < (2 * ROWS) as u64,
        "the cluster query decoded {} of {} rows",
        cluster_counters.rows_decoded,
        2 * ROWS
    );

    // Control: identical data, no index. Without this number the check above
    // could be satisfied by a counter that simply never counts.
    let plain_dir = tempdir()?;
    let plain_uri = format!("file://{}", plain_dir.path().to_str().expect("utf8"));
    let plain = Table::new_async(plain_uri).await?;
    plain.write_async(vec![batch(0)]).await?;
    plain.commit_async().await?;

    let (plain_rows, plain_counters) = query(&plain).await;
    assert_eq!(plain_rows, K, "the unindexed query must return k rows");
    assert!(
        plain_counters.rows_decoded >= ROWS as u64,
        "the unindexed query decoded {} of {} rows; a flat scan decodes every row",
        plain_counters.rows_decoded,
        ROWS
    );

    Ok(())
}
