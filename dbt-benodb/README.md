# dbt Adapter for BenoDB (`dbt-benodb`)

`dbt-benodb` is the official dbt adapter enabling analytics engineering, transformation models, and data lineage directly against BenoDB.

## Features

- **Table & View Materializations**: Build standard and optimized Iceberg-format tables.
- **Incremental Materializations**: Efficient append and merge updates leveraging BenoDB's primary key upserts and delete vector acceleration.
- **Vector & Graph Extensions**: Built-in macros for vector distance scoring, hybrid search, and graph relationship traversals (`benodb.vector_search(...)`, `benodb.graph_neighbors(...)`).

## Installation

```bash
pip install dbt-benodb
```

## Configuration (`profiles.yml`)

```yaml
benodb_profile:
  target: dev
  outputs:
    dev:
      type: benodb
      database: default
      schema: public
      threads: 4
      uri: "file:///path/to/benodb/warehouse"
```

## Remote / containerised runs: use `dbt-trino`

This adapter is an **embedded** adapter: it opens the warehouse through the
engine's JNI/`benodb` Python module, so the warehouse path has to be
mounted in the process running dbt.

For Airflow, Cloud Run, CI, or any topology where dbt should not mount the
warehouse, point `dbt-trino` at a Trino coordinator that has the
`trino-benodb` plugin installed. This is the supported remote path and
needs no extra transport in this adapter:

```yaml
# profiles.yml — remote access through the BenoDB Trino connector
benodb_remote:
  target: prod
  outputs:
    prod:
      type: trino
      method: none            # or ldap / jwt, per your coordinator
      host: trino.internal
      port: 8080
      catalog: benodb     # the catalog backed by trino-benodb
      schema: default
      threads: 8
```

Notes:

- The catalog name must match the entry in the coordinator's
  `etc/catalog/benodb.properties` (`connector.name=benodb`).
- Table functions work remotely too: `system.vector_search`,
  `system.hybrid_search`, `system.list_indexes`, `system.regional_drift_search`.
  `hybrid_search` now honours its `filter` argument, so a filtered hybrid query
  is pushed to the engine instead of being rejected.
- `vector_search` accepts a sixth `metric` argument (`cosine` by default, plus
  `l2`, `dot`, `l1`, `hamming`, `jaccard`).
- Keep the embedded adapter for local development and for features that need the
  Python API directly (index building, graph traversal from Python).

## Cluster Indexing & Cache Warming Macros

Use post-hooks in your model files to automatically rebuild table-level Voronoi cluster indexes and prime memory caches following data ingestion:

```sql+jinja
{{ config(
    materialized='table',
    post_hook=[
        rebuild_cluster_index(this, 'embedding', k=1024, m=16, ef_construction=200, target_shard_mb=512),
        warmup(this, n_probes=500, max_memory_gb=8.0)
    ]
) }}
```
