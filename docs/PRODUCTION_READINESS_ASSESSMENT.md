# BenoDB — Release Readiness Assessment
- **Status:** Living document — **Developer Preview Ready** (Core storage, concurrency, and bounded Iceberg V2/V3 integrity gates passed; distributed clustering, native TLS, and enterprise HA deferred).
- **Initial reviews:** 2026-09-30 (concurrency & workspace boundary review); 2026-10-03 (v0.13.0 core storage WS1–WS5 audit)
- **Last verified:** 2026-11-25 (updated for `v0.17.3`: `ADR-008` `PendingCommit`, `ADR-009` catalog CAS parity, `ADR-010` Iceberg V3 DVs & `iceberg-rust`/Spark/Trino conformance, `0.17.0` `benodb` rename & `bsql`/`ClusterIndex`, and `0.17.3` Flight SQL RBAC, HNSW IPC bounds, WAL/segment sync, and doc alignment)
- **Scope:** Architectural audit of core storage invariants, differential oracles, crash recovery, cross-engine interop, and security against the **Developer Preview Release Bar**.
- **Current Version:** 0.17.3

> **Release Scope Notice (v0.17.3):**
> Releases from `v0.15.2` through `v0.17.3` remain at the **Developer Preview** bar (single-table object-store coordination rather than distributed multi-node consensus). Since `v0.15.1`, the codebase has landed:
> - **`M2` / `ADR-008` (`PendingCommit` state machine):** Explicit `CommitPhase::Staging` -> `ManifestCommitted` -> `IcebergPublished`, linearizable `buffer_epoch` read double-checking, WAL segment rotation/pruning with torn-EOF recovery, and idempotent `benostream.committed_wal_tx` convergence ([ADR-008](https://benolabsai.com/docs/guides/adr_008.html)).
> - **`ADR-009` (Catalog parity & CAS hardening):** Equivalent compare-and-swap / HTTP `409` conflict semantics across `rest`, `s3tables`, `memory`, `jdbc`, `glue`, `hive`, `unity`, and `nessie`, backed by a shared real-service conformance suite ([ADR-009](https://github.com/benolabsai/benodb/blob/main/docs/architecture/ADR_009_CATALOG_PARITY_AND_CAS_HARDENING.md)).
> - **`ADR-010` (Bounded Iceberg V3 conformance):** Spec-exact `deletion-vector-v1` Puffin blobs, manifest fields `143–145`, V3 row lineage (`_row_id`, `_last_updated_sequence_number`), and independent reader validation via `apache/iceberg-rust` per-PR and Apache Spark 3.5/4.0 + Trino 482/484 in `.github/workflows/iceberg-interop.yml` ([ADR-010](https://github.com/benolabsai/benodb/blob/main/docs/architecture/ADR_010_ICEBERG_V3_DELETION_VECTORS_AND_CROSS_ENGINE_CONFORMANCE.md)).
> - **`v0.17.0`–`v0.17.3`:** Canonical `benodb` rename, `bsql`/`bdb` CLIs, codes-first `ClusterIndex`, manifest-version caching, search-gateway segment-aware write auth, Flight SQL `admin`-role write RBAC + SQL identifier validation, SPKI RS256 verification, HNSW IPC bounds checks, WAL/segment directory fsyncs, conditional lease release, batch coercion propagation, serialized PK validation, CSR alignment validation, and SIMD dimension guards.
>
> Connector-specific evidence and the boundary between the supported core and experimental/deferred surfaces live in [`docs/SUPPORT_MATRIX.md`](SUPPORT_MATRIX.md).

This assessment summarizes the verification of core invariants (WS1-WS5) and the end-to-end benchmark suite.

---

## 1. Overall Verdict

| Area | Status |
|---|---|
| Core storage architecture | **Proven** |
| Index architecture (overlay) | **Proven** (differential oracles pass on evaluated workloads) |
| Iceberg integration | **V2 and bounded V3 proven** (V2 core compliant via PyIceberg; implemented V3 subset — row lineage, `deletion-vector-v1` Puffin blobs, manifest `format-version` and fields `143–145` — verified by `apache/iceberg-rust` in CI and by Spark 3.5/4.0 & Trino 482/484 in `iceberg-interop.yml`; `variant` and deferred V3 types are out of scope per [`SUPPORT_MATRIX.md`](SUPPORT_MATRIX.md)) |
| Query engine | **Proven** (matches DuckDB/DataFusion row counts on tested suites) |
| Rust implementation | **Proven** (library `--features no-panic` gate enforced in CI; see [`NO_PANIC_POLICY.md`](NO_PANIC_POLICY.md) for residual `hnswio`/binary scope) |
| Testing infrastructure | **Hardened** (WS1-WS5 differential harnesses, fuzzing, real-service catalog suite, and release soak gate active) |
| Crash/recovery confidence | **High** (survives randomized WS2 crash-injection sweep; remote object-store host-loss RPO=0 requires persistent `BDB_WAL_DIR`) |
| Multi-writer concurrency | **Single-table OCC Ready** (survives WS3/M2 concurrency harness with object-store/catalog faults; no distributed cross-table or multi-node consensus) |
| Security | **Developer Preview** (API key + HS256/RS256 JWT on gateway and Flight SQL; OAuth2/SigV4 for catalogs; catalog storage policy; requires trusted network or TLS-terminating reverse proxy) |
| Operational tooling | Progressing (metrics, tracing, and `bdb-admin` shipped; unified index-freshness and WAL-backlog observability still maturing) |
| **Overall** | **Developer Preview Ready** |

### ✅ Resolved Blocker (Resolved 2026-10-10 / Verified 2026-10-28): `lastfm-64-dot` Vector Recall Anomaly
During initial post-audit execution of the end-to-end benchmark suite, an anomaly was discovered where vector recall dropped to **~6%** specifically on the `lastfm-64-dot` dataset for `hnsw_tq8`. 
- **Root Cause:** The MIPS dataset `lastfm-64-dot` contains padded training vectors with a uniform small norm (~0.7256), whereas the query vectors possessed wildly varying, unpadded natural norms (up to ~31.23). The query vectors severely overflowed the `TurboQuant` encoder's dynamic range during SDC, clamping their values to zero or max, thus destroying vector directionality.
- **Resolution:** A pre-quantization dynamic scaling step was introduced for `InnerProduct` and `Cosine` metrics in `HnswGraph` (with exact dot-product/angular rescore validated across full benchmark sweeps by `2026-10-28`, `86391664`). Because query magnitude does not alter metric rankings, the query is scaled into the quantizer's `offset` and `scale` window, avoiding overflow and restoring 8-bit precision.

---

## 2. Core Invariants Verified

The following core invariants have been tested and proven under adversarial conditions:
1. **Overlay Invariant** — indexes are derived, reconstructible state.
2. **Publication Invariant** — a manifest references only durable artifacts.
3. **Durability Invariant** — WAL truncation only after a committed snapshot (`WalDurability::Sync` guarantees RPO=0 across process crashes; surviving host/container loss on remote object stores requires pointing `BDB_WAL_DIR` / `with_wal_dir()` at a persistent volume).
4. **Maintenance Invariant** — delete only if not referenced / not in-flight.
5. **Resource Invariant** — memory-scaled build concurrency; library no-panic gate (`--features no-panic` on `--lib`, with documented `hnswio` / `metrics.rs` / binary exemptions in [`NO_PANIC_POLICY.md`](NO_PANIC_POLICY.md)).

---

## 3. Workstream Execution Summary

All five workstreams testing the storage/transaction/recovery boundaries have been successfully executed:

### WS1: Differential indexed-vs-full-scan oracle (Complete)
**Goal:** Prove `indexed(query) == full_scan(query)` on the same snapshot.
- **Findings & Fixes:**
  - `explain()` misreported the vector access path for TurboQuant indexes by failing to match the file pattern; resolved by reading from the manifest instead of globbing.
  - A BM25 index silently dropped equality matches by inappropriately applying tokenized indexes to exact-match filters; resolved by restricting exact matches to the `identity` analyzer.

### WS2: Crash-injection matrix at every boundary (Complete)
**Goal:** Kill the process at boundaries (WAL/manifest/index) and prove deterministic recovery.
- **Findings & Fixes:**
  - WAL replay was not idempotent. A crash between manifest commit and WAL truncation caused duplicate rows on recovery. Fixed by recording committed WAL transaction IDs in the manifest properties (`benostream.committed_wal_tx`), allowing recovery to skip already-committed records.

### WS3: Multi-writer / object-store concurrency (Complete)
**Goal:** Prove OCC + distributed-locking design is safe with N concurrent writers against shared object storage.
- **Findings & Fixes:**
  - Vacuum deleted all data files due to empty `Manifest.entries` for tiered manifests; fixed by resolving entries correctly via `load_all_entries`.
  - GC-vs-writer race caused new files to be deleted; fixed by re-validating the live set before deletion and adding a grace period.
  - Partition-spec updates dropped data by dropping `manifest_list_path`; fixed to preserve it.
  - Delete files were incorrectly written to the local FS for S3 stores; fixed to write to the `ObjectStore` relative to the table root.

### WS4: Iceberg interoperability / conformance (V2 & Bounded V3 Complete)
**Goal:** Validate V2 and implemented V3 conformance via independent implementations (`PyIceberg` for V2, `apache/iceberg-rust` per-PR and Apache Spark 3.5/4.0 + Trino 482/484 weekly/on-tag in `.github/workflows/iceberg-interop.yml` for V2/V3).
- **Findings & Fixes:**
  - PyIceberg validation uncovered issues with relative paths in `manifest-list`, `manifest_path`, and `file_path`. Fixed to use absolute URIs on write and relativize on read.
  - Avro schemas were missing spec-required metadata (`field-id`, map `logicalType`, list `element-id`). Added to schemas.
  - **Closed in `v0.16.0`:** V3 `deletion-vector-v1` framing and write/read wiring, Avro `format-version = 3` metadata with fields `143–145`, V3 snapshot row ranges, and the `apache/iceberg-rust` reader oracle (see [`ADR-010`](https://github.com/benolabsai/benodb/blob/main/docs/architecture/ADR_010_ICEBERG_V3_DELETION_VECTORS_AND_CROSS_ENGINE_CONFORMANCE.md)); cross-catalog CAS parity landed as [`ADR-009`](https://github.com/benolabsai/benodb/blob/main/docs/architecture/ADR_009_CATALOG_PARITY_AND_CAS_HARDENING.md). (Deferred V3 features such as `variant` stay listed in [`SUPPORT_MATRIX.md`](SUPPORT_MATRIX.md).)

### WS5: Long-duration GC / compaction / delete / recovery (Complete)
**Goal:** Prove the Maintenance Invariant under sustained churn.
- **Findings & Fixes:**
  - Compaction resurrected deleted rows because it didn't pass delete files into the reader; fixed to apply the delete mask during compaction.
  - Compaction misclassified index sidecars as data files; fixed by strictly matching the data parquet file name.
  - Read-after-delete hung due to unbounded per-entry delete file lists; fixed by storing delete files once partition-scoped in the manifest.
  - Vector recall miss due to querying during a background index build; fixed test-side by awaiting background tasks.
  - Append-only commits rewrote the entire manifest list; optimized to only write new entries and reference unchanged prior manifests, falling back to consolidation after a threshold.

---

## 4. Security Posture

Authentication is implemented as a shared, stateless module (`src/core/auth.rs`)
used by both the search gateway and the Flight SQL server:

- **API key** — constant-time comparison against `BDB_API_KEY`.
- **JWT** — HS256 (`BDB_JWT_SECRET`) or RS256 (`BDB_JWT_PUBLIC_KEY`); claims
  extraction for subject, roles, audience, issuer, and expiry.
- **RBAC** — roles extracted from JWT claims (`roles` array); available to
  authorization logic.
- **Search gateway** (`contrib/benodb-search/src/auth.rs`) — Axum
  middleware accepting `Authorization: Bearer <jwt>`, `Authorization: ApiKey
  <key>`, and `api-key: <key>` (Qdrant convention). Public-path exemption for
  health checks; fail-closed when `required()` is set.
- **Flight SQL** (`server/flight_sql/src/main.rs`, `server/flight_sql/src/server.rs`) — Tonic interceptor using the
  same `AuthConfig` (`authorization` metadata header); verifies caller identity,
  enforces `"admin"` role on DDL/DML/ingest/transaction/maintenance mutations when
  authenticated, and supports server-level `BDB_FLIGHT_READ_ONLY`.
- **Fail-closed on network exposure** — both servers refuse to bind to a
  non-loopback address without credentials (`process::exit(1)`). Neither server terminates TLS natively; production or shared-network deployments must sit behind a TLS-terminating reverse proxy or trusted internal network.
- **Catalog authentication** — OAuth2 client-credentials grant with token
  caching and expiry (`src/core/catalog/rest.rs`); Bearer token and AWS SigV4 support; SSRF
  guard on catalog URLs (`src/core/catalog/mod.rs`).
- **SQL session hardening** — `ALLOWED_SESSION_SETTINGS` prevents SQL `SET`
  from rewriting environment variables (credentials, endpoints).

Remaining gaps: no built-in user database (identity is delegated to an external
IdP or a shared secret); no per-table or per-column access control; no audit logging of authentication events; no native TLS termination on network servers.

## 5. Final Exit Criteria

With the completion of WS1-WS5 and the resolution of the quantization boundary anomaly, the core storage engine has survived millions of operations (insert/update/delete/predicate/vector/snapshot/commit/compaction/index-rebuild) with injected failures without divergence between indexed and full-scan execution.

The system meets the exit criteria for a **Developer Preview** (pending real-world usage and operational tooling maturity).

---

## 6. Review & Resolution History

| Initial Review | Scope / Finding | Status | Resolution Date & Evidence |
|---|---|---|---|
| **2026-09-30** | Pre-OSS concurrency & boundary review (`H1`: atomic `PendingWrite` vs dual-lock extraction; `H2`: `maintenance_lock` barrier on `truncate_async`; `H3`: propagate manifest-load errors on `truncate`; `contrib/` three-tier repo split) | **Resolved** | **2026-09-30** (`bba0dc46` / `203b8bc8`): `pending_writes` and `maintenance_lock` implemented in `src/core/table/mod.rs` and `src/core/table/write.rs`; `tests/test_h1_h2_concurrency_regression.rs` added; non-core gateways isolated under `contrib/`. |
| **2026-09-30** | Extended audit `M1`: `remove_orphan_files_async()` and `vacuum()` fencing against concurrent readers, `_staging/`, and `_wal/` | **Resolved** | **2026-10-03** (WS3/WS5, `203b8bc8`): Grace-period + live-set re-validation shipped; verified by `tests/test_crash_injection.rs` and `crash_during_vacuum_preserves_live_rows` in `tests/test_crash_injection_workload.rs`. |
| **2026-09-30** | Extended audit `M2`: Explicit `PendingCommit` transaction state machine evaluation | **Resolved** | **2026-11-10** (shipped in `v0.15.2` / `v0.16.0`): Explicit `CommitPhase` (`Staging` -> `ManifestCommitted` -> `IcebergPublished`) and `PendingCommit` state machine with `buffer_epoch` linearizable read barrier and `restore_staging_pending_commit_to_buffer` (`src/core/table/mod.rs`, `write.rs`, `read.rs`, `stage.rs`, documented in `docs/architecture/ADR_008_PENDING_COMMIT_STATE_MACHINE.md`); WAL rotation/watermark pruning and torn-EOF/header tail tolerance (`rotate_async`, `prune_committed_segments`, `open_wal_segment`, `SafeIpcSegmentIter`, `fuzz_replay_wal_bytes` in `src/core/wal.rs`); atomic `benostream.committed_wal_tx` CSV merge, V3 row-lineage OCC retry cleanup, monotonic `LATEST_VERSION_CACHE`, CAS `commit_update_atomic` / `advance_version_hint`, REST catalog CAS retry, and nanosecond `ManifestValue` decode (`src/core/manifest/manager.rs`, `commit.rs`, `types.rs`, `src/core/metadata.rs`, `src/core/catalog/rest.rs`); plus calibrated asymmetric quantized ADC scoring, out-of-core tail sampling, and Euclidean flat-vs-HNSW L2 alignment (`src/core/index/{benoquant,sq,pq,mod,hnsw_ivf,cluster_index}.rs`, `src/core/reader/scan.rs`). Certified by `tests/test_m2_commit_state_machine.rs` (Pillars 1–9; 14/14 including subprocess `SIGABRT`/`SIGKILL`, `ScriptedStore`, Axum REST 409, V3 row lineage, and linearizability), `tests/test_multi_writer_differential.rs` (4/4), `tests/test_randomized_differential_workload.rs` (`BDB_WORKLOAD_STEPS=400`, 3/3), `fuzz/fuzz_targets/wal_segment.rs` (`5,000` runs, 0 crashes), `tests/test_iceberg_conformance.rs` (6/6), `tests/test_multi_writer_concurrency.rs`, and `scripts/pre_release_soak.sh`. |
| **2026-10-03** | Core storage invariant audit (`WS1`–`WS5`) + security posture for `v0.13.0` | **Resolved** | **2026-10-03** (`5ef7b25b`): Differential oracles, crash-injection matrix, PyIceberg V2/V3 conformance, and stateless auth verified. |
| **2026-10-10** | `lastfm-64-dot` `InnerProduct` / `Cosine` recall collapse under `hnsw_tq8` | **Resolved** | **2026-10-10** / verified **2026-10-28** (`da2ea446`, `86391664`, `e3f933a2`): Pre-quantization dynamic query scaling + exact rescore shipped. |
| **2026-11-09** | `v0.15.1` scope carry-forward & Arrow C FFI string offset conformance | **Verified** | **2026-11-09** (`a51b245b`, `5ac9838a`, `b88cfcbe`): Fixed string/vector search offsets across Arrow C FFI; `cargo deny check` and fuzz smoke tests enforced on pre-push gate. |
| **2026-11-13** | `v0.16.0` catalog CAS parity (`ADR-009`), bounded Iceberg V3 DVs + `iceberg-rust` oracle (`ADR-010`), real-service catalog suite | **Resolved** | **2026-11-13**: Shipped `MemoryCatalog`, `S3Tables` + AWS SigV4, CAS across `jdbc`/`glue`/`hive`/`unity`/`nessie`, `deletion-vector-v1` Puffin blobs + fields `143–145`, and `tests/iceberg_rust_oracle`. |
| **2026-11-25** | `v0.17.0`–`v0.17.3` release hardening & documentation credibility alignment | **Resolved** | **2026-11-23 / 2026-11-25**: Shipped `benodb` rename, `bsql`/`bdb` CLIs, codes-first `ClusterIndex`, manifest-version cache, gateway segment-aware write auth, Flight SQL `admin` write RBAC + `DoGet` DML/DDL guard, HNSW IPC bounds checks, WAL/segment sync hardening, SPKI RS256 verification, conditional lease release, batch coercion propagation, serialized PK validation, CSR alignment & SIMD dimension guards, Spark 3.5/4.0 + Trino 482/484 CI interop green, and `v0.17.3` doc alignment. |

