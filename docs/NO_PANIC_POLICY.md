# No-Panic Policy

BenoDB is a long-running database process. A single `unwrap()` on an
unexpected `None`/`Err` in a request path, an ingest loop, or a decode path
crashes the whole daemon: in-flight queries are dropped, the process restarts,
and the WAL has to recover. Panics are therefore treated as defects in
production code, not as an acceptable error strategy.

## Scope

| Code | Rule |
| --- | --- |
| Production library code (`src/**` compiled without `cfg(test)`) | No `unwrap()` / `expect()` / `panic!` except documented invariants |
| Production binaries (`src/bin/{bdb,gateway,iceberg_rest,bdb-admin}.rs`, `benodb-search/src/main.rs`) | Same rule |
| `#[cfg(test)] mod tests` blocks | Exempt — tests may unwrap freely |
| `tests/**` integration crates | Exempt |
| `build.rs` | Build scripts are exempt in spirit, but `build.rs` was converted to return `Result` anyway |

## The rule

Prefer, in order:

1. **Propagate**: `?` with the native error type (`HyperstreamError`,
   `DataFusionError`, `anyhow::Error`) via `ok_or_else` / `context`.
2. **Make the function total**: where Arrow's `data_type()` (or an equivalent
   guard) already proves the downcast cannot fail, use
   `.map(|a| ...).unwrap_or(<neutral>)` or `if let ... else { <neutral> }`
   instead of `unwrap()`. A violated invariant then degrades to `null` / `false`
   / `skip` rather than panicking.
3. **Documented invariant**: only where the API genuinely has no infallible
   constructor and failure is a programming error caught at startup, add a
   narrowly scoped `#[allow(clippy::unwrap_used, clippy::expect_used)]` with a
   justification comment. Every such site must also be listed below.

## Enforcement

CI runs [`scripts/no_panic_check.sh`](https://github.com/benolabsai/benodb/blob/main/scripts/no_panic_check.sh) as the `no-panic` job in `.github/workflows/ci.yml`. It combines two checks:

### 1. Ratchet (`scripts/no_panic_check.sh`)

```bash
bash scripts/no_panic_check.sh           # fail if production panic sites grew, then run hard gate
bash scripts/no_panic_check.sh --write   # lower the baseline after a fix
```

The script compiles with `-D clippy::unwrap_used -D clippy::expect_used -D clippy::panic` across `benodb` (`--lib`), `benodb-search` (`--lib`), `benodb-iceberg` (`--lib`), `benodb-flight` (`--bin benodb-flight`), and `benodb-mcp` (`--bin benodb-mcp`). Using explicit non-test targets leaves `#[cfg(test)]` tests out while comparing the total un-exempted site count against `scripts/no_panic_baseline.txt`.

### 2. Library hard gate (`--features no-panic`)

`src/lib.rs`, `benodb-search/src/lib.rs`, and `crates/benodb-iceberg/src/lib.rs` carry:

```rust
#![cfg_attr(
    all(not(test), feature = "no-panic"),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
```

After the ratchet passes, `scripts/no_panic_check.sh` runs `cargo clippy --no-deps --lib --features no-panic` so any new un-exempted `unwrap()`, `expect()`, or `panic!` in the core library fails CI directly. The `benodb-iceberg` crate carries its own `no-panic` feature (the root package's flag does not reach it), and the script gates it separately with `cargo clippy --no-deps -p benodb-iceberg --lib --features no-panic`.

## Baseline and inventory

Initial production baseline (measured before remediation): **289** sites
(247 core library + 42 search library). **Current baseline: 0** — the `no-panic` job enforces both the `0`-site ratchet and the `--lib --features no-panic` hard gate in CI.

Two module-scoped exemptions remain, both justified in source:

| Exemption | Sites | Why |
| --- | ---: | --- |
| `src/core/index/hnsw_rs/**` | 69 | Vendored `hnsw_rs` (dual MIT/Apache-2.0). `unwrap`/`expect`/`panic` here assert graph invariants in the builder and inner search loop, plus `hnswio` file dump/load paths. `src/core/index/hnsw_rs/mod.rs` carries `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]`. |
| `src/telemetry/metrics.rs` | 13 | `prometheus` metric constructors for static definitions; the crate has no infallible API. See the allowlist below. |

Everything else in the core and search library crates — graph UDFs, the storage/read path, the ingest and index-build path, merge/compaction/catalog, the algorithms, the GPU backends, the Python/FFI bindings, and `src/lib.rs` — is clean under the `--lib --features no-panic` gate (with the narrowly scoped startup invariants in the allowlist below).

### Remaining gaps (not panic-free end to end)

- **`hnswio` file I/O.** A truncated or corrupt index file can panic in
  `read_exact`/`write_all` inside the vendored dump/load path, which the
  `hnsw_rs` module-level exemption still covers. Converting those entry points to
  `Result` changes upstream signatures; it remains a **user-reachable panic path**
  when loading corrupt HNSW sidecars.
- **Root binary targets.** `scripts/no_panic_check.sh` ratchets `benodb-flight`
  and `benodb-mcp` binary targets, while root-crate binaries (`src/bin/*`) and
  `benodb-search/src/main.rs` are not yet compiled with `--features no-panic`
  by the CI hard gate (which invokes `--lib` only). Consequently, the library
  gate is enforced, not whole-application panic freedom across every binary entry
  point.

Both gaps are Phase 2, targeted for the **0.18.x** line: convert the `hnswio`
entry points to `Result` (closing the last user-reachable panic path), and
compile the root binaries and `benodb-search/src/main.rs` under the hard gate.
Until then the library gate — the surface every embedding consumer links — is
fully enforced.

## Residual-invariant allowlist

These sites are permitted to keep a documented `#[allow]` because the API has no
infallible form and failure is a startup-time programming error, not a runtime
condition:

- **`src/telemetry/metrics.rs`** — `prometheus` metric constructors
  (`IntGauge::new`, `IntCounterVec::new`, …) return `Result` and are only
  fallible on an invalid name or duplicate registration. All names/help strings
  are static constants. Failure happens once at startup before the server binds
  and is unreachable from a request. Guarded by
  `#[allow(clippy::unwrap_used, clippy::expect_used)]` on `Metrics::new`.

- **Compile-time regex literals** (e.g. `Regex::new(r"...")`) — the pattern is a
  literal, so construction cannot fail. These may use `expect` with a message.

- **`chrono::NaiveDate::from_ymd_opt(1970, 1, 1)`** — replaced by
  `NaiveDate::default()` where possible (it *is* 1970-01-01), which is
  infallible.

- **`src/lib.rs` jemalloc ctl** — `tikv_jemalloc_ctl::epoch::mib()` is a
  process-lifetime singleton; failure indicates a broken allocator build.

- **`src/core/table/state.rs` `Table::runtime()`** — `rt` is set by every public
  constructor; the `expect` asserts a structural invariant. Returning
  `Option`/`Result` would ripple through every synchronous caller just to
  express "this cannot happen". Guarded by a function-level
  `#[allow(clippy::expect_used)]`.

- **`src/python/helpers.rs` `TOKIO_RUNTIME`** — `Runtime::new()` has no
  infallible form and fails only if the OS cannot give the process a reactor and
  worker threads. If that happens the binding layer is unusable, so failing once
  at first use with a clear message is correct. Guarded by
  `#[allow(clippy::expect_used)]`.

- **`src/core/segment.rs` `indexing_pool()` and
  `src/core/table/maintenance.rs` `maintenance_pool()`** — `rayon`'s
  `ThreadPoolBuilder::build()` has no infallible form and fails only on OS
  resource exhaustion (thread creation). The pools are process-lifetime
  singletons built on first use; a failure is unrecoverable at startup. Guarded
  by a function-level `#[allow(clippy::expect_used)]`.

Anything not in this list should be converted to `?`, made total, or the
allowlist entry justified in review.
