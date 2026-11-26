# ADR-011: The Iceberg Format Layer Is a Standalone Crate, and Its Public API Speaks Only Spec Concepts

> **Status: accepted (post-`v0.17.3`).** Implemented across [`crates/benodb-iceberg/`](https://github.com/benolabsai/benodb/blob/main/crates/benodb-iceberg/src/lib.rs), the facade re-exports in [`core/iceberg/mod.rs`](https://github.com/benolabsai/benodb/blob/main/src/core/iceberg/mod.rs), [`core/manifest/mod.rs`](https://github.com/benolabsai/benodb/blob/main/src/core/manifest/mod.rs), [`core/puffin.rs`](https://github.com/benolabsai/benodb/blob/main/src/core/puffin.rs), and [`core/error.rs`](https://github.com/benolabsai/benodb/blob/main/src/core/error.rs), the engine adapter in [`core/manifest/manager/commit.rs`](https://github.com/benolabsai/benodb/blob/main/src/core/manifest/manager/commit.rs), and the catalog/expression/transaction additions (`catalog.rs`, `expr.rs`, `transaction.rs`).
> Related: [`ADR-009`](https://github.com/benolabsai/benodb/blob/main/docs/architecture/ADR_009_CATALOG_PARITY_AND_CAS_HARDENING.md) (catalog CAS parity), [`ADR-010`](https://github.com/benolabsai/benodb/blob/main/docs/architecture/ADR_010_ICEBERG_V3_DELETION_VECTORS_AND_CROSS_ENGINE_CONFORMANCE.md) (V3 conformance), and the parity specification (`docs/GOAL_C_PARITY_SPEC.md`, untracked).

---

## 1. Context

Two forces made a decision necessary.

**Force 1 — the engine is one 129,000-line compilation unit.** The `core` module declares roughly thirty top-level modules in a single crate. The dependency direction (`sql → table → index → manifest → iceberg → storage`) was verified by text scan, not enforced by anything; any future `use crate::core::table` inside `core/index` compiles without complaint. Compile time scales with the whole unit, and the de facto public API is the entire tree, because `lib.rs` declares `pub mod core;` — every `pub` item is reachable as `benodb::core::...` whether anyone intends it or not.

**Force 2 — the Iceberg layer has standalone value.** The format machinery (table metadata V2/V3, Avro manifests, schemas, transforms, values, delete files, deletion vectors, Puffin containers) is useful to consumers who want an Iceberg implementation without an embedded database. Inside the monolith, that consumer does not exist.

A first extraction attempt failed on line-range surgery: `manifest_types.rs` interleaves `#[derive]` attributes with items without blank separators and embeds raw-string JSON with braces inside struct bodies, so text-level cuts corrupt both destination files. The attempt was reverted; the corrected seam map and the tooling requirement are recorded in the review document's appendices.

## 2. Decision

**D1 — The format layer is a workspace crate (`benodb-iceberg`), extracted along the verified dependency direction.** The crate carries the Iceberg spec modules (`types`, `manifest`, `schema`, `transform`, `value`, `writer`, `delete`), the Puffin container, and the manifest type vocabulary. The engine consumes it through facade re-exports (`core::iceberg`, `core::manifest::types`, `core::puffin`, `core::error`, `core::cache`), so every `crate::core::` path and the entire public API keep resolving.

*Rejected:* splitting `core/table` first. It is the highest-coupling node; the review's recommendation table ranks it behind the leaf extractions for that reason.

*Rejected:* a big-bang crate split. One crate per release, facade re-exports from the start, and the migration never blocks a shipping deadline.

**D2 — The crate's public API speaks only spec concepts; engine vocabulary lives engine-side.** The engine's `ManifestEntry`/`Manifest`/`IndexFile`/`IndexAlgorithm` composition chain and the engine's `BenoStreamError` belong to the engine; the crate's API carries the spec types (`Schema`, `PartitionSpec`, `DeleteFile`, `IcebergManifestEntry`) and its own `IcebergError`. Where the two meet — the manifest writer — the writer's API takes the spec entry type, and the engine adapts at two call sites (`to_iceberg_manifest_entry`: partition ordering by spec field, column stats to the field-id-keyed maps via the same `encode_iceberg_value` the read side uses).

*Rejected:* keeping the engine types in the crate for convenience. The crate would carry BenoDB's internal bookkeeping (`IndexFile`, clustering metadata) in its public API, and a third party asking for "an Iceberg implementation" would find engine bookkeeping in the contract.

*Rejected:* changing `benodb`'s public API to make the split easier. Every historical path resolves; the one signature change (the two writer methods' entries parameter is now the spec type) is disclosed and independently revertible.

**D3 — The container owns framing and hashing; the caller owns what a blob means.** The Puffin split is the model for the crate's seams: the crate's `PuffinWriter` exposes blob-level primitives (`add_blob`, `add_blob_range_from_store` — framing, the blob registry, SHA-256 while streaming), and the engine's `PuffinIndexWriter` (moved engine-side) builds the `IndexFile` inventory entries on top. The deletion-vector codec stays in the crate with its own typed error (`IcebergError::DeletionVectorDeserialize`); the engine's error enum moved back to the engine byte-identical.

*Rejected:* leaving the index-inventory writer in the crate. It would drag `IndexFile` and the engine's clustering vocabulary into the crate's public API.

**D4 — The catalog, expression, and transaction layers are crate API (Goal C).** The `Catalog` trait (spec vocabulary, `commit_table` taking the REST spec's typed `TableRequirement`/`TableUpdate` — internally tagged, kebab-case, the exact wire shapes the engine's eight backends already parse), the spec predicate tree (`Expression`/`Predicate`/`Literal`, field-id addressing, three-valued evaluation where `None` keeps the file), and the `Transaction` (assertions fail closed; `apply` is a pure metadata transition) are crate modules. The engine's `SpecCatalog` adapter wraps the engine's trait object; the engine's OCC commit remains the durability backend.

*Rejected:* implementing the crate's trait via a blanket `impl<T: EngineCatalog> CrateCatalog for T`. The orphan rule forbids it (E0210); the wrapper type is the honest shape.

## 3. Consequences

**Positive.** The dependency direction is compiler-enforced at the crate boundary: `core/index` cannot reach `core/table` without an explicit cross-crate dependency that review would see. The format crate builds without `datafusion` (dropped when the error enum went home), without the GPU stack, and without the network servers — it builds in constrained environments the full engine will not. The crate's test suite (22+ items) runs independently. Third parties get an Iceberg format implementation with a spec-pure API; the engine's V4 maintenance lands in one crate.

**Negative and accepted.**
- **The facade indirection is permanent** for as long as `benodb`'s public API is frozen: `core::iceberg`, `core::manifest::types`, `core::puffin`, and `core::error` are re-export shims, and every engine file's `crate::core::X` path costs one hop of indirection when reading.
- **The manifest type vocabulary is split across two crates by composition, not location**: the engine's `ManifestEntry` (in the engine) references the crate's `DeleteFile`; the crate's `ManifestListEntry` references the crate's `IndexAlgorithm`. The composition chain `Manifest → ManifestEntry → IndexFile` is engine-side; the spec types are crate-side. A reader must hold both maps.
- **The read-side conversion still lives in the crate** (`convert_iceberg_to_object` reconstructs the engine's `ManifestEntry`), relocated engine-side in a follow-up; until then the crate's read API produces one engine type.
- **The no-panic ratchet does not cover the new crate** — `scripts/no_panic_check.sh` enumerates four targets; adding the crate to the ratchet is follow-up work (the crate's code is currently clean: 0 sites).
- **One public signature changed** (the two writer methods' entries parameter is the spec type). It is disclosed here as the single deliberate deviation from the frozen-API rule, and it is independently revertible.
