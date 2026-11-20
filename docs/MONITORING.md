# BenoDB Monitoring & Operations

BenoDB is designed to be highly observable and easy to operate in serverless, ephemeral environments as well as traditional long-running daemon setups.

## Observability by deployment mode

The engine records metrics in one place (the core registry) and every host
exposes them, so a scrape sees the engine's ingest/query/compaction/index
metrics regardless of how the engine is deployed.

| Deployment mode | Metrics endpoint | Health |
|---|---|---|
| **Embedded library** (Python / Rust) | `http://<BDB_METRICS_BIND>:<BDB_METRICS_PORT>/metrics` (default `127.0.0.1:9090`), enabled by the `observability` feature | `/health`, `/readyz` |
| **Flight SQL server** (`benodb-flight`) | `http://<BDB_METRICS_BIND>:<BDB_METRICS_PORT>/metrics` (default `127.0.0.1:9090`) | `/health`, `/readyz` |
| **Search gateway** (`bdb-search`) | `http://<BDB_SEARCH_BIND>:<BDB_SEARCH_PORT>/metrics` (default `127.0.0.1:9200`) | `/_health`, `/healthz`, `/livez`, `/readyz` |
| **Gateway / Iceberg REST bins** | `GET /metrics` (gateway), `GET /v1/metrics` (iceberg_rest) | `GET /health`, `GET /v1/health` |
| **Spark / Trino connectors** | `BenoDBJNIBridge.gatherMetrics()` / `BenoDBJNIBridge.gatherMetrics()` — bridge into the host's metrics system | host-provided |

The search gateway's `/metrics` serves **both** its own `bdb_search_*` registry
and the core engine's `bdb_*` metrics, so one scrape covers the whole
process.

### Metrics configuration

| Variable | Description | Default |
|:---|:---|:---|
| `BDB_METRICS_BIND` | Bind address for the embedded/Flight metrics HTTP listener. | `127.0.0.1` |
| `BDB_METRICS_PORT` | Port for the embedded/Flight metrics HTTP listener. | `9090` |

> The listener defaults to **localhost**. Set `BDB_METRICS_BIND=0.0.0.0` only
> when a scraper runs on another host, and front it with auth/TLS.
>
> **⚠️ Security:** none of these endpoints authenticate. Run every BenoDB
> server on a **trusted internal network**, behind a gateway/reverse proxy that
> terminates TLS and enforces authentication. Do **not** expose them to the
> public internet. See [SECURITY.md](https://github.com/benolabsai/benodb/blob/main/SECURITY.md).

## Metric catalog

### Engine metrics (`bdb_*`)

Recorded with the `prometheus` crate in `src/telemetry/metrics.rs`:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `bdb_ingest_rows_total` | counter | — | Rows ingested. |
| `bdb_query_latency_seconds` | histogram | — | Query latency. |
| `bdb_search_latency_seconds` | histogram | — | Vector/keyword search latency. |
| `bdb_commit_duration_seconds` | histogram | — | Manifest commit duration. |
| `bdb_compaction_duration_seconds` | histogram | — | Compaction duration. |
| `bdb_index_build_duration_seconds` | histogram | — | HNSW-IVF build duration. |
| `bdb_active_files` | gauge | — | Active Parquet files. |
| `bdb_active_segments` | gauge | — | Active segments. |
| `bdb_cache_hits_total` / `_misses_total` | counter | `cache_name` | Cache hit/miss by cache. |
| `bdb_io_bytes_read_total` / `_written_total` | counter | — | Object-store I/O bytes. |
| `bdb_manifest_conflicts_total` | counter | — | OCC commit conflicts. |
| `bdb_ingest_rss_bytes` | gauge | — | Process RSS during ingest. |
| `bdb_ingest_backpressure_pauses_total` | counter | — | Ingest pauses on the RAM high-water mark. |
| `bdb_ingest_backpressure_pause_seconds` | histogram | — | Duration of each pause. |
| `bdb_index_build_gate_wait_seconds` | histogram | — | Wait for an index-build permit. |
| `bdb_free_disk_bytes` | gauge | — | Free disk for local tables. |
| `bdb_merged_deletes_phase_seconds` | histogram | `phase` | Merge-on-read delete phases. |
| `bdb_merged_deletes_cache_total` | counter | `result` | Merged-deletes cache hit/miss. |
| `bdb_delete_file_cache_total` | counter | `result` | Parsed-delete cache hit/miss. |
| `bdb_merged_deletes_files_total` | counter | `kind` | Delete files merged by kind. |
| `bdb_merged_deletes_calls_total` | counter | — | `load_merged_deletes` calls. |
| `bdb_read_phase_seconds` | histogram | `phase` | Parquet read phases. |
| `bdb_parquet_meta_cache_total` | counter | `result` | Parquet-metadata cache hit/miss. |

Recorded with the `metrics` facade (query planner, compaction, manifest commit):

| Metric | Type | Meaning |
|---|---|---|
| `bdb_manifest_commit_skipped_removals_total` | counter | Commits that skipped removals. |
| `bdb_manifest_commit_retries_total` | counter | OCC commit retries. |
| `bdb_manifest_commit_rebases_total` | counter | OCC commit rebases. |
| `bdb.query.planning_duration` | histogram | Query planning duration. |
| `bdb.query.segment_pruning_duration` | histogram | Segment pruning duration. |
| `bdb.query.segment_search_duration` | histogram | Per-segment search duration. |
| `bdb.pruned.vector_bbox` | counter | Segments pruned by vector bounding box. |
| `bdb_data_files_compacted` | counter | Data files compacted. |
| `bdb_compaction_bytes_written` | counter | Bytes written by compaction. |

### Search gateway metrics (`bdb_search_*`)

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `bdb_search_active_requests` | gauge | — | In-flight HTTP requests. |
| `bdb_search_http_requests_total` | counter | `method`, `route` | Requests by route class. |
| `bdb_search_http_request_duration_seconds` | histogram | `method`, `route` | Request latency. |
| `bdb_search_query_seconds` | histogram | `op` | Search latency by op (`match`/`knn`/`hybrid`/`filter`). |
| `bdb_search_docs_indexed_total` | counter | `result` | Documents indexed. |
| `bdb_search_bulk_items_total` | counter | `status` | Bulk items by outcome. |
| `bdb_search_refresh_seconds` | histogram | — | `_refresh` duration. |
| `bdb_search_index_cache_hits_total` / `_misses_total` | counter | — | Index table-cache hit/miss. |
| `bdb_search_index_fetch_bytes_total` | counter | `kind` | Index bytes fetched from object storage. |

## Telemetry & Tracing

BenoDB leverages a push-based OpenTelemetry (OTLP) pipeline to export distributed traces. Because it often runs in serverless functions (like AWS Lambda) where instances may be frozen between invocations, the `Table` API explicitly flushes traces on teardown (via the `Drop` trait) to ensure no data is lost.

### Configuration

Tracing is disabled by default. To enable tracing, set the `JAEGER_ENABLED` environment variable. When enabled, traces are automatically pushed to an OTLP-compatible endpoint.

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| `JAEGER_ENABLED` | Set to `true` to enable OpenTelemetry exporting. | `false` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | The destination URL for OTLP traces. | `http://localhost:4317` (OTLP/gRPC default) |
| `RUST_LOG` | The log level filter (e.g., `info`, `benodb=debug`). | `info` |

### Core Instrumentation
We instrument key paths to provide visibility into latency and bottlenecks:
- **Write Path:** `write_async`, `commit_async`
- **Read Path:** `read_async`, `stream_all`, `vector_search_index`, `vector_search_flat`
- **Manifest Orchestration:** Optimistic concurrency loops, schema updates, and conflict resolution in the `ManifestManager`.

## Stateless Operational CLI (`bdb`)

The `bdb` binary is a standalone CLI tool that performs administrative actions directly against the object storage tier without requiring a long-running database server to be active.

### Usage

```bash
# Start an interactive SQL REPL
bdb repl

# Execute a single SQL query
bdb query --query "SELECT * FROM my_table LIMIT 10"

# Register a table in the session
bdb register --name my_table --uri s3://my-bucket/my-table
```

### Table Management

You can perform routine maintenance using the `bdb table` subcommand:

```bash
# Inspect table metadata and statistics
bdb table inspect --uri s3://my-bucket/my-table

# Compact small data files to optimize read performance
bdb table compact --uri s3://my-bucket/my-table

# Vacuum (delete) data files that are no longer referenced and older than N days
bdb table vacuum --uri s3://my-bucket/my-table --older-than-days 7
```

*Note: The CLI is stateless; it interacts directly with the storage URI.*
