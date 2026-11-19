# BenoStreamDB Search (`benodb-search`)

OpenSearch / Elasticsearch 7.10 and Qdrant REST compatibility gateway for BenoStreamDB.

> **⚠️ Security:** This gateway ships **stateless authentication** (API key and/or JWT — see [Authentication](#authentication)) but **no TLS**. Bind to `127.0.0.1` or an internal interface and terminate TLS at a reverse proxy. Do **not** expose it directly to the public internet. See [SECURITY.md](../../SECURITY.md).

## Overview

`benodb-search` enables drop-in integration with applications, vector libraries, and ingest pipelines that speak:
1. **OpenSearch / Elasticsearch 7.10 REST API** (`/_bulk`, `/{index}/_doc`, `/{index}/_search`, aggregations, index templates).
2. **Qdrant REST API** (`/collections`, `/collections/{name}/points`, search, scroll).

It runs as a standalone daemon (`bsdb-search`) backed directly by BenoStreamDB tables on local or cloud object storage.

## Installation

### Via Cargo
```bash
cargo install benodb-search
```

### Via Python / PyPI
```bash
pip install benodb-search
```

## Running the Server

```bash
# Start server listening on port 9200 (OpenSearch) and 6333 (Qdrant)
bsdb-search --bind 0.0.0.0 --port 9200 --storage-uri file:///data/benodb
```

## Configuration

All configuration is via environment variables (or the equivalent CLI flags).
These variables are specific to this gateway; the core engine's variables are
documented in [`docs/CONFIGURATION.md`](../../docs/CONFIGURATION.md).

### Server & storage

| Variable | Description | Default |
|:---|:---|:---|
| `BDB_SEARCH_PORT` | Port for the OpenSearch/Elasticsearch 7.10 compatible HTTP REST API. | `9200` |
| `BDB_SEARCH_BIND` | Bind IP address for the OpenSearch REST server. | `0.0.0.0` |
| `BDB_SEARCH_DEVICE` | Compute device target (`"auto"`, `"cpu"`, `"cuda"`, `"rocm"`, `"metal"`). | `"auto"` |
| `BDB_SEARCH_STORAGE_URI` | Storage URI for search indices (e.g. `s3://bucket/search` or `file:///data`). | `./data` |
| `BDB_SEARCH_INDEX_CACHE_GB` | Size cap in GB for on-demand sidecar index file LRU cache. | `1` |
| `BDB_SEARCH_PRELOAD` | Preload index sidecars into memory on open (`0`/`false`/`no`/`off` disables). | `true` |
| `BDB_SEARCH_PRELOAD_GB` | In-memory budget in GB for preloaded index sidecars; overflow spills to the mmap disk cache. | `4` |
| `BDB_SEARCH_AUTO_REFRESH_SECS` | Background timer interval in seconds to flush memtables to Iceberg snapshots. | `1.0` |
| `BDB_SEARCH_WAL_DURABILITY` | Gateway ingestion durability (`"async"` for batched WAL, `"sync"` for per-doc fsync). | `"async"` |
| `BDB_SEARCH_RRF_K` | Smoothing constant ($k$) for Reciprocal Rank Fusion hybrid search scoring. | `60` |
| `BDB_QDRANT_BIND` | Bind IP address for the Qdrant-compatible Vector REST API listener. | Matches `BDB_SEARCH_BIND` |
| `BDB_QDRANT_PORT` | Port for the Qdrant-compatible Vector REST API listener. | `6333` |

### Authentication

The gateway supports **stateless** authentication — no user database to maintain.
Identity is delegated to a shared secret or an external IdP's signing key. The
same credentials protect **both** the OpenSearch/ES API and the Qdrant API.

| Variable | Description | Default |
|:---|:---|:---|
| `BDB_API_KEY` | Shared secret. Accepted via `Authorization: Bearer <key>`, `Authorization: ApiKey <key>`, or `api-key: <key>` (Qdrant). | unset |
| `BDB_JWT_SECRET` | HS256 shared secret for JWT verification. | unset |
| `BDB_JWT_PUBLIC_KEY` | RS256 public key (PEM) for JWT verification — the "delegate to your IdP" path. | unset |
| `BDB_JWT_AUDIENCE` | Required `aud` claim (if set). | unset |
| `BDB_JWT_ISSUER` | Required `iss` claim (if set). | unset |
| `BDB_AUTH_REQUIRED` | Fail closed (`401`) when no credential is configured. | `false` |
| `BDB_METRICS_PUBLIC` | Exempt `/metrics` from auth (for an internal Prometheus scraper). | `false` |

Health endpoints (`/_health`, `/_cluster/health`, `/_cat/health`, `/healthz`,
`/readyz`, `/livez`) and `GET /` are always public.

```bash
# API key
export BDB_API_KEY="$(openssl rand -hex 32)"
curl -H "api-key: $BDB_API_KEY" http://127.0.0.1:9200/_cat/indices

# JWT (HS256) — verify a token minted by your IdP
export BDB_JWT_SECRET="shared-secret"
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:9200/_search
```

**Where to keep the secret:** never in the repo or a world-readable env file.
Use a secret manager (Vault, AWS/GCP/Azure Secrets Manager) or a `0600` file
mounted at `/run/secrets/` (Docker/K8s) or `/etc/benodb/secrets/`, loaded
via systemd `LoadCredential=`/`EnvironmentFile=` (mode `0600`).

### Catalog storage policy (access control)

The gateway is an **untrusted, network-facing connector**: reads and writes are
governed by the authoritative **catalog storage policy** — a
`.benostream-policy.json` object in the warehouse/bucket root (a local path or an
object store such as `s3://`, resolved through the core object store).

- **Schemas come from the backend — the gateway never creates one.** A schema
  exists because an operator provisioned it and named it in the policy. Writing
  an index whose schema is not granted returns `403`.
- **Writes** are allowed only to schemas listed in `readwrite_schemas`. There is
  **no default `scratch` namespace** (unlike the local MCP server): a server
  connector is **read-only until an operator publishes a policy**.
- **Reads** have a read-only floor — with **no policy** every schema is readable
  (read-only) — and, when a policy is present, its `readwrite_schemas` /
  `readonly_schemas` / immutable lists are an **allowlist** for reads too. A read
  of a schema the policy does not name is denied before the index's existence is
  even checked.
- Immutable system schemas (`default`, `public`, `information_schema`, `sys`,
  `system`) are never writable.
- Policy updates are read-modify-write with optimistic concurrency (CAS on
  object stores that support conditional PUTs).

Enforcement is identical across the OpenSearch API (`_search`, `_count`,
`_doc/{id}`, `_mapping`, `_bulk`, aliases, index create/delete) and the Qdrant
API (search / query / scroll / retrieve / count / recommend / discover plus every
mutating route).

Example `.benostream-policy.json` at the warehouse root:

```json
{
  "policy_version": "1.0",
  "readwrite_schemas": ["analytics", "reporting"],
  "readonly_schemas": ["archive"],
  "no_access_schemas": ["pii"],
  "default_write_schema": "analytics"
}
```

With this policy the gateway may write `analytics.*` and `reporting.*` indexes,
may only read `analytics`, `reporting`, and `archive`, and denies everything
else.

> **Operational note:** a deployment that publishes **no** policy is read-only —
> the gateway logs a `WARN` at startup (`"search gateway is READ-ONLY: no writable
> schema"`). Provision the schema(s) and publish a policy before expecting writes.
> Set `BDB_SEARCH_REQUIRE_POLICY=1` to **refuse to start** rather than run
> read-only. See [`docs/CATALOG_USAGE.md`](../../docs/CATALOG_USAGE.md) and
> [`docs/SUPPORT_MATRIX.md`](../../docs/SUPPORT_MATRIX.md) for the full model.

### Iceberg catalog integration

`benodb-search` can automatically synchronize search indices with an
external Apache Iceberg catalog (Snowflake / Apache Polaris, Project Nessie, AWS
Glue, Hive Metastore, or Unity Catalog) so ingested documents immediately advance
the catalog's current snapshot.

| Variable | Description | Default |
|:---|:---|:---|
| `BDB_SEARCH_CATALOG_TYPE` | Catalog provider: `"rest"` (Snowflake Polaris / Lakekeeper), `"nessie"`, `"glue"`, `"hive"`, `"unity"`. | None (path-based Iceberg) |
| `BDB_SEARCH_CATALOG_URL` | Catalog REST/Service endpoint (e.g. `https://polaris.example.com/api/catalog/v1`). | None |
| `BDB_SEARCH_CATALOG_URI` | Alias for `BDB_SEARCH_CATALOG_URL` (also accepts `BDB_CATALOG_URI`). | None |
| `BDB_SEARCH_CATALOG_NAMESPACE` | Catalog namespace where search index tables are created and committed. | `default` |
| `BDB_SEARCH_CATALOG_CREDENTIAL` | OAuth2 Client Credentials (`<client_id>:<client_secret>`) for Polaris / REST. | None |
| `BDB_SEARCH_CATALOG_TOKEN` | Bearer token for Iceberg REST authentication. | None |
| `BDB_SEARCH_CATALOG_PREFIX` | Warehouse or catalog prefix (e.g. Polaris warehouse name). | None |
| `BDB_SEARCH_CATALOG_ID` | AWS Glue Account / Catalog ID (when using Glue). | None |

> **Note:** The gateway also accepts the core-engine aliases
> `BDB_CATALOG_TYPE`, `BDB_CATALOG_URL` / `BDB_CATALOG_URI`,
> `BDB_CATALOG_TOKEN`, `BDB_CATALOG_CREDENTIAL`, and
> `BDB_CATALOG_NAMESPACE` as fallbacks when the corresponding
> `BDB_SEARCH_CATALOG_*` variable is unset.

> **Snowflake Polaris setup:** Set `BDB_SEARCH_CATALOG_TYPE=rest`,
> `BDB_SEARCH_CATALOG_URL=https://<account>.snowflakecomputing.com/polaris/api/catalog/v1`,
> and `BDB_SEARCH_CATALOG_CREDENTIAL=<client_id>:<client_secret>`. Every bulk
> ingest via `/_bulk` or `_doc` automatically executes an Iceberg atomic swap
> against Snowflake Polaris.

## Table-Level Cluster Indexing & Cache Warming

For vector tables, the gateway provides endpoints to compact multi-segment indexes into global Voronoi clusters and prime server RAM:

* **`POST /{index}/_rebuild_cluster`**: Rebuilds global Voronoi cluster index into rolling $\le 512\text{ MiB}$ Puffin shards sealed into the Iceberg manifest.
  ```json
  POST /articles/_rebuild_cluster
  {
    "column": "vector",
    "k": 1024,
    "m": 16,
    "ef_construction": 200,
    "target_shard_mb": 512
  }
  ```

* **`POST /{index}/_warmup`**: Probes the Voronoi centroid router to prefault index shard pages into the OS page cache before servicing client queries.
  ```json
  POST /articles/_warmup
  {
    "n_probes": 500,
    "max_memory_gb": 8.0
  }
  ```
