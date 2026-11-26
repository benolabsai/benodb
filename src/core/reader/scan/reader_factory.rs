// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Parquet reader construction: the small file bytes cache, the
//! in-memory `BytesReader`, and the metadata cache loader.

use super::*;

/// Maximum parquet file size (bytes) cached whole in memory.
///
/// Small segments are read repeatedly (a scan touches every segment, and a
/// workload issues several scans per step). Reading them through the object
/// store each time pays a per-range `spawn_blocking` + open. Caching the whole
/// file and serving ranges from memory removes that. Larger files are left to
/// `ParquetObjectReader`, which fetches only the needed column chunks.
const PARQUET_BYTES_CACHE_MAX_FILE: u64 = 4 * 1024 * 1024;

/// An `AsyncFileReader` that serves byte ranges from an in-memory buffer.
///
/// Used with `ParquetRecordBatchStreamBuilder::new_with_metadata`, so only
/// `get_bytes`/`get_byte_ranges` are exercised; `get_metadata` is never called
/// (the metadata is supplied by the caller from `PARQUET_META_CACHE`).
struct BytesReader {
    bytes: Bytes,
}

impl parquet::arrow::async_reader::AsyncFileReader for BytesReader {
    fn get_bytes(
        &mut self,
        range: std::ops::Range<u64>,
    ) -> futures::future::BoxFuture<'_, parquet::errors::Result<Bytes>> {
        let start = (range.start as usize).min(self.bytes.len());
        let end = (range.end as usize).min(self.bytes.len());
        let slice = if start <= end {
            self.bytes.slice(start..end)
        } else {
            Bytes::new()
        };
        Box::pin(async move { Ok(slice) })
    }

    fn get_metadata<'a>(
        &'a mut self,
        _options: Option<&'a ArrowReaderOptions>,
    ) -> futures::future::BoxFuture<
        'a,
        parquet::errors::Result<Arc<parquet::file::metadata::ParquetMetaData>>,
    > {
        Box::pin(async move {
            Err(parquet::errors::ParquetError::General(
                "BytesReader does not load metadata; it is supplied by the caller".to_string(),
            ))
        })
    }
}

impl HybridReader {
    /// Build a parquet reader for `path`. Small files are served from an
    /// in-memory cache (see `PARQUET_BYTES_CACHE`); larger files use the object
    /// store directly so only the needed column chunks are fetched.
    pub(super) async fn parquet_reader(
        &self,
        store: &Arc<dyn ObjectStore>,
        path: &Path,
        size: u64,
    ) -> Box<dyn parquet::arrow::async_reader::AsyncFileReader> {
        if size <= PARQUET_BYTES_CACHE_MAX_FILE {
            let key = format!("{}/{}", self.cache_root(), path);
            if let Some(bytes) = crate::core::cache::PARQUET_BYTES_CACHE.get(&key).await {
                return Box::new(BytesReader { bytes });
            }
            // Retry briefly on EMFILE ("Too many open files"). Under heavy
            // concurrency (background compaction rewriting segments while reads
            // are saturated) the process can transiently exceed its
            // file-descriptor limit; a short backoff lets other readers release
            // their handles instead of failing the query with `os error 24`.
            let mut attempt = 0u32;
            let mut delay = std::time::Duration::from_millis(5);
            loop {
                match store.get(path).await {
                    Ok(res) => {
                        if let Ok(bytes) = res.bytes().await {
                            crate::core::cache::PARQUET_BYTES_CACHE
                                .insert(key, bytes.clone())
                                .await;
                            return Box::new(BytesReader { bytes });
                        }
                        break;
                    }
                    Err(e) => {
                        if crate::core::storage::is_emfile(&e) && attempt < 6 {
                            attempt += 1;
                            tokio::time::sleep(delay).await;
                            delay = (delay * 2).min(std::time::Duration::from_millis(200));
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        Box::new(ParquetObjectReader::new(store.clone(), path.clone()))
    }

    pub(super) async fn load_cached_parquet_meta(
        &self,
        pq_path: &Path,
    ) -> Result<(Arc<ArrowReaderMetadata>, usize)> {
        let pq_path_str = pq_path.to_string();
        let cache_key = format!("{}/{}", self.cache_root(), pq_path_str);
        if let Some((meta, size)) = crate::core::cache::PARQUET_META_CACHE
            .get_with_metrics(&cache_key, "parquet_meta")
            .await
        {
            Ok((meta, size))
        } else {
            let store = self.data_store();
            let object_meta = store
                .head(pq_path)
                .await
                .context("Failed to get segment metadata")?;
            let size = object_meta.size as usize;
            let mut reader = ParquetObjectReader::new(store.clone(), object_meta.location);
            let options = ArrowReaderOptions::new().with_page_index(true);
            let arrow_meta = ArrowReaderMetadata::load_async(&mut reader, options).await?;
            let meta = Arc::new(arrow_meta);
            crate::core::cache::PARQUET_META_CACHE
                .insert(cache_key, (meta.clone(), size))
                .await;
            Ok((meta, size))
        }
    }
}
