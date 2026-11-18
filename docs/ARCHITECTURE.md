# Architecture: The Indexed Lakehouse

BenoDB implements an indexed, compute-disaggregated lakehouse storage architecture that pairs authoritative open table storage with advisory, persistent secondary indexes and a unified retrieval layer.

```text
               Iceberg Table
                     │
       ┌─────────────┴──────────────┐
       │                            │
Authoritative Storage        Advisory Index Overlay
       │                            │
  Parquet Files              Bitmap / Bloom / BM25 / HNSW / TQ
```

## Core Philosophy

**"Your Data is Standard. Your Index is Custom."**

BenoDB attaches persistent, reconstructible sidecar index files *alongside* standard Parquet files:
*   **100% Format Compatibility**: Standard data engines (Spark, Trino, DuckDB, Pandas) read the underlying Parquet and Iceberg tables directly at native speed.
*   **O(log N) Accelerated Retrieval**: BenoDB-aware query engines and REST search gateways leverage inverted bitmap and vector indexes for low-latency queries directly on object storage.
*   **The Overlay Invariant**: Indexes are derived state. If an index file is absent, corrupted, or stale, queries safely degrade to Parquet scanning without failing or returning incorrect results. See [ADR-001: Tightened Overlay Index Artifact Contract & Compound Puffin Bundles](https://benolabsai.com/docs/guides/adr_001.html).

---

## Storage Layout: Hybrid Segments

Data is stored in immutable **Segments** pairing open columnar storage with advisory index sidecars:

1.  **Authoritative Raw Data**:
    *   `data/segment_id.parquet`: Standard Apache Iceberg columnar data files written with ZSTD compression and dictionary encoding. 100% readable by external engines (DuckDB, Spark, Trino, Polars).
2.  **Advisory Secondary Indexes (Apache Iceberg Puffin Compound Bundles)**:
    *   `metadata/segment_id.puffin`: As specified in [ADR-001](https://benolabsai.com/docs/guides/adr_001.html), production deployments consolidate secondary index artifacts into standard Apache Iceberg Puffin (`.puffin`) compound bundles. A single `.puffin` file encapsulates Roaring Bitmap inverted indexes, BM25 term frequencies, and Voronoi-clustered HNSW/TurboQuant vector graphs queried via HTTP byte-range offsets, preventing cloud object count explosion.
    *   *(Uncompacted / Local Debug Mode)*: Prior to Voronoi cluster compaction (`rebuild_cluster_index`), transient per-column sidecars (`.inv.parquet`, `.bm25.parquet`, `.centroids.parquet`, `.hnsw.graph`) may be emitted before `compact()` or `vacuum()` merges them into compound Puffin bundles.
3.  **Iceberg Table Metadata**:
    *   `_manifest/v{N}.avro`: Standard Iceberg V2/V3 snapshot manifests and manifest lists.
    *   `_metadata/v{N}.metadata.json`: Iceberg table schema, sort orders, partition specs, and snapshot lineage.

---

## Ingestion Pipeline: Non-Blocking Indexing & Async WAL

BenoDB achieves high ingestion throughput while maintaining real-time queryability:

1.  **Memtable & Async WAL**: Incoming records write to an in-memory batch buffer with adaptive or immediate WAL fsync durability.
2.  **Parquet Flushes**: Records flush to compressed Parquet files for immediate durability.
3.  **Background Indexing**: High-dimensional vector graphs (HNSW-IVF) and full-text indexes build asynchronously across a Rayon thread pool without blocking writes.
4.  **Atomic Manifest Snapshot**: Once background index construction completes, the manifest is atomically swapped via Optimistic Concurrency Control (OCC) or distributed CAS locks (`FileBasedLock`).

---

## The Read Path & Hot Row Cache

BenoDB combines scalar pre-filtering, vector search, and in-memory caching for ultra-low latencies:

1.  **Scalar Pruning**: Query planner evaluates inverted index sidecars, producing a candidate row bitmap.
2.  **Vector / Keyword Scoring**:
    *   **Vector Search**: Traverses HNSW graph sidecars, evaluating candidates with dynamic SIMD metrics.
    *   **Full-Text Search**: Scores terms using BM25 Okapi sidecars.
    *   **Hybrid RRF**: Combines ranks from vector and keyword search using Reciprocal Rank Fusion ($1 / (k + \text{rank} + 1)$).
3.  **Hot Row Cache (`BLOCK_CACHE`)**:
    *   Decoded Parquet `RecordBatch` chunks are held in an in-memory LRU block cache.
    *   Scattered kNN candidate row fetches resolve directly from memory in sub-millisecond time, completely bypassing disk I/O and eliminating tail latency spikes.

---

## Table-Level Cluster Indexing & Sequential 512 MiB Puffin Sharding

For multi-million and 100M+ vector tables, per-segment vector sidecars create query fan-out overhead across hundreds of small HNSW graphs. BenoDB provides **Table-Level Cluster Indexing** (`ClusterIndex`) via `table.rebuild_cluster_index()`:

```text
                        Table-Level Cluster Index (v42)
                        (Locality-Ordered Centroids 0..K-1)
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       │                              │                              │
Shard 0000 (≤512 MiB)          Shard 0001 (≤512 MiB)          Shard 0002 (≤512 MiB)
Clusters [0 .. K0]             Clusters [K0+1 .. K1]          Clusters [K1+1 .. K2]
  1. cluster-codes (BSCD v1)     1. cluster-codes (BSCD v1)     1. cluster-codes (BSCD v1)
  2. (seg_id, row_id) Parquet    2. (seg_id, row_id) Parquet    2. (seg_id, row_id) Parquet
  3. HNSW Graphs (IPC)           3. HNSW Graphs (IPC)           3. HNSW Graphs (IPC)
```

### Architectural Guarantees:
1.  **Decoupled Overlay Invariant**: Cluster nodes carry `(segment_idx, row_id)` tuples stored in Arrow Parquet metadata tables (`org.apache.iceberg.vector.cluster-node-meta`). The underlying Parquet data files are read once and **never rewritten or compacted**, preserving standard Apache Iceberg read compatibility.
2.  **Codes-First Split Blob Layout & Codes-Only Query Path**: Each shard stages and seals blobs in contiguous **codes-first** order (`org.apache.iceberg.vector.cluster-codes` → `cluster-node-meta` → `hnsw-graph`). At query time, `ClusterIndex` scans the compact 16-byte-aligned flat codes blob directly (skipping HNSW graph deserialization and I/O entirely) and hydrates `node_meta` **only** for the small set of winning clusters that contribute to top-$k$. Legacy shards written without `cluster-codes` blobs fall back transparently to HNSW graph traversal; set `BSDB_CLUSTER_SEARCH_MODE=graph` to force graph traversal for A/B benchmarking.
3.  **Locality-Ordered Centroids & Sequential 512 MiB Rolling Shards**: Global centroids are ordered before assignment by deterministic recursive furthest-pair projection bisection with Hilbert-style seam orientation, then packed into rolling $\le 512\text{ MiB}$ Puffin shards (`cluster_{col}_{version}_shard_{i}.puffin`) with strictly monotonic disjoint cluster ranges (`min_cluster..=max_cluster`). Co-probed centroids land in adjacent offsets within the same shard, enabling single-GET S3 coalescing and respecting cloud single-PUT limits.
4.  **S3 Pipelining, Singleflight Coalescing & Unified Cache Budget**: On local storage, shards are memory-mapped zero-copy (`memmap2`) with chunked `posix_fadvise(WILLNEED)` pipelining over codes spans only. On remote object storage (`s3://`, `gs://`, `az://`), queries resolve footers and fetch touched shards concurrently (`get_range` coalesced across adjacent codes blobs), pipeline batches on Tokio while scanning on CPU, and deduplicate concurrent misses via per-key `SingleFlight` gates under a single unified byte-weighed `moka` cache (`BSDB_CLUSTER_CACHE_KIB` or `BSDB_CACHE_GB / 4`).
5.  **Cross-Architecture SIMD Parity & Multi-Threaded Rayon Compaction**: Centroid assignment (Pass 2) and intra-bucket cluster construction (Pass 3) run across CPU cores via Rayon with single-pass quantization and disk-staged shard sealing to bound build memory inside 16 GiB containers. Batch code scanning dispatches to hoisted **AVX2+FMA** kernels on `x86_64` (Intel/AMD) and native **ARM64 NEON** kernels on `aarch64` (Apple Silicon M-series / AWS Graviton / Azure Cobalt) across TQ8, TQ4 (16-nibble unpacking), SQ8, Cosine/IP, L1, and F32, plus 4-way unrolled L1-LUT accumulation for PQ.

---

## Multi-Protocol Search Gateway (`bsdb-search`)

BenoDB exposes its columnar storage and index overlays via standard REST protocols:
*   **OpenSearch / Elasticsearch 7.10 API (Port 9200)**: Drop-in compatibility for `_search`, `_bulk`, `_mapping`, and `_cat/indices`.
*   **Qdrant Vector API (Port 6333)**: Qdrant v1.x REST emulation — collections, point upsert/retrieve, payload & vector edits, vector search, aliases, and the universal query API. See [QDRANT_COMPATIBILITY.md](https://github.com/benolabsai/benodb/blob/main/contrib/benodb-search/docs/QDRANT_COMPATIBILITY.md).
*   **Arrow Flight SQL (Port 50051)**: Low-latency zero-copy gRPC queries for analytical tools.
