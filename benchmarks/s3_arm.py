#!/usr/bin/env python3
# Copyright (c) 2026 Richard Albright and BenoDB Contributors.
"""Exercise the S3 (object store) read path and report what it transfers.

The 100M local sweep measures the mmap path. This measures the other one: the
table level cluster artifact lives in an S3 bucket, `load_sharded_remote` opens it
from footers alone, and each query hydrates its probe clusters with coalesced
range reads. The number that matters is not wall clock -- it is bytes and
requests, which is what an object store bills for.

**Provenance: every number this produces is from a containerized RustFS instance
on the loopback interface**, not from AWS S3, GCS or Azure. That makes the byte
counts meaningful (the same ranges, the same coalescing) and the latency numbers
optimistic: there is no WAN round trip, no TLS, no request throttling and no
egress charge. Any write-up must say so, and must not present these as public
cloud figures.

Byte accounting comes from the S3 server's own interface counters (`docker stats`
on the RustFS container), so it measures what left the store rather than what the
client thinks it asked for.

Usage:
    AWS_ENDPOINT=http://localhost:9000 ./s3_arm.py build
    AWS_ENDPOINT=http://localhost:9000 ./s3_arm.py measure
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa

N = 200_000
DIM = 64
CLUSTERS = 512
# One flush for the whole set, so the ingest does not exercise the multipart
# path at all and this script measures the read arm only. (The multipart floor is
# handled in `core::storage::multipart_part_bytes`; an earlier version of this
# script hit `EntityTooSmall` because both writers emitted one 2 MiB part per
# `File::read`, which was our defect, not a server deviation.)
SEGMENT_ROWS = 200_000
SHARD_BYTES = 24 * 1024 * 1024


def netio(container: str) -> tuple[int, int]:
    """Cumulative (rx, tx) bytes for a container, from the store's counters."""
    raw = subprocess.check_output(
        ["docker", "stats", "--no-stream", "--format", "{{.NetIO}}", container],
        text=True,
    ).strip()
    rx_s, tx_s = [part.strip() for part in raw.split("/")]
    return _parse_bytes(rx_s), _parse_bytes(tx_s)


def _parse_bytes(text: str) -> int:
    text = text.strip()
    for unit, scale in (
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
        ("GB", 1000**3),
        ("MB", 1000**2),
        ("kB", 1000),
        ("B", 1),
    ):
        if text.endswith(unit):
            return int(float(text[: -len(unit)]) * scale)
    return int(float(text))


def _table(bsdb, uri: str):
    return bsdb.Table(uri)


def _existing_objects(uri: str) -> int:
    """Objects already under `uri`.

    `write` appends, so building twice into one prefix silently doubles the
    table. An earlier version of this script did exactly that while the
    multipart defect was being diagnosed: six attempts appended to the same
    prefix, and the resulting table was 6.3x the 200k rows the module claims
    (338 MB of Parquet where 200k x 64 float32 is 51 MB). Every absolute figure
    from those runs described a million-row table. Refusing a non-empty prefix
    is cheaper than noticing that after the fact.
    """
    if not uri.startswith("s3://"):
        return 0
    import boto3

    bucket, _, prefix = uri[len("s3://") :].partition("/")
    s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT"))
    total = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        total += len(page.get("Contents", []))
    return total


def build(
    bsdb,
    uri: str,
    container: str,
    algorithm: str | None = None,
    m: int = 16,
    rows: int = N,
    dim: int = DIM,
) -> None:
    existing = _existing_objects(uri)
    if existing:
        raise SystemExit(
            f"{uri} already holds {existing} objects; write appends, so use a fresh prefix"
        )
    rx0, tx0 = netio(container)
    t0 = time.time()
    table = _table(bsdb, uri)
    rng = np.random.default_rng(0)
    for start in range(0, rows, SEGMENT_ROWS):
        end = min(start + SEGMENT_ROWS, rows)
        vectors = rng.random((end - start, dim), dtype=np.float32)
        fsl = pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), dim)
        batch = pa.Table.from_arrays(
            [
                pa.array(np.arange(start, end, dtype=np.int64)),
                fsl,
            ],
            names=["id", "embedding"],
        )
        table.write(batch)
        # One segment per batch keeps each flushed object under the multipart
        # threshold (see SEGMENT_ROWS) and gives the cluster index several shards
        # to spread clusters over, which is what the coalescing is meant to face.
        table.flush()
    ingest_s = time.time() - t0

    # `rebuild_cluster_index` builds with the column's configured algorithm, and
    # the default is `hnsw_tq8`, so a plain build is already quantized. Naming the
    # algorithm here is what makes the payload comparison an experiment instead of
    # an assumption.
    if algorithm:
        table.add_index("embedding", algorithm=algorithm)

    t0 = time.time()
    table.rebuild_cluster_index(
        "embedding",
        k=CLUSTERS,
        m=m,
        ef_construction=200,
        target_shard_bytes=SHARD_BYTES,
    )
    index_s = time.time() - t0

    rx1, tx1 = netio(container)
    print(
        f"build: {rows:,}x{dim} in {ingest_s:.1f}s, index {CLUSTERS} clusters "
        f"(algorithm {algorithm or 'default'}, m={m}) in {index_s:.1f}s | "
        f"store rx {(rx1 - rx0) / 1e6:.1f} MB, tx {(tx1 - tx0) / 1e6:.1f} MB"
    )
    del table


def exact_topk(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Exact top-k by squared L2, one query at a time.

    200k x 64 float32 is 51 MB, so the squared norms are computed once and each
    query costs a single matvec. This is the oracle the recall column is measured
    against; without it a payload reduction is only a smaller number.
    """
    norms = (vectors * vectors).sum(axis=1)
    out = np.empty((len(query), k), dtype=np.int64)
    for i, q in enumerate(query):
        d = norms - 2.0 * (vectors @ q) + float(q @ q)
        part = np.argpartition(d, k)[:k]
        out[i] = part[np.argsort(d[part])]
    return out


def recall_at_k(returned: list[list[int]], truth: np.ndarray, k: int) -> float:
    hits = 0
    for got, want in zip(returned, truth):
        hits += len(set(got[:k]) & set(int(w) for w in want[:k]))
    return hits / (len(returned) * k)


def measure(
    bsdb,
    uri: str,
    container: str,
    probes: list[int],
    queries: int,
    k: int = 10,
    rows: int = N,
    dim: int = DIM,
) -> list[dict]:
    rng = np.random.default_rng(1)
    query_vectors = rng.random((queries, dim), dtype=np.float32)
    # The build regenerates the corpus from the same seed and shape, so the oracle
    # needs no reads of its own.
    corpus = np.random.default_rng(0).random((rows, dim), dtype=np.float32)
    truth = exact_topk(corpus, query_vectors, k)

    # Cold open plus one query: this is what triggers `load_sharded_remote`, so
    # the delta is the whole initialisation cost. The request count is the point
    # of the shard descriptor: an open that reads every footer pays three
    # requests per shard, one that reads the descriptor pays five in total.
    bsdb.reset_reader_stats()
    rx0, tx0 = netio(container)
    t0 = time.time()
    table = _table(bsdb, uri)
    first = _query(table, query_vectors[0], k, 64, probes[0])
    init_s = time.time() - t0
    init_stats = bsdb.reader_stats()
    rx1, tx1 = netio(container)
    egress_mb = max(tx1 - tx0, 0) / 1e6
    print(
        f"init: open + first query in {init_s:.2f}s | store egress {egress_mb:.2f} MB "
        f"| slice GETs {init_stats['cluster_range_gets']} "
        f"({init_stats['cluster_range_bytes'] / 1024:.0f} KiB) "
        f"| returned {len(first)} ids"
    )

    records: list[dict] = []
    for n_probe in probes:
        rx0, tx0 = netio(container)
        bsdb.reset_reader_stats()
        returned: list[list[int]] = []
        t0 = time.time()
        for i in range(queries):
            returned.append(_ids(_query(table, query_vectors[i], k, 64, n_probe)))
        elapsed = time.time() - t0
        rx1, tx1 = netio(container)
        st = bsdb.reader_stats()
        egress = max(tx1 - tx0, 0) / 1e6
        # Coverage is the axis that makes two clusterings comparable: the vectors
        # the probe set actually contains, over the corpus. `nprobe` cannot serve,
        # because equal probe counts examine different amounts of a corpus under
        # two different clusterings.
        coverage = (
            st["cluster_corpus_rows"] / (queries * rows) if queries and rows else 0.0
        )
        gets = st["cluster_range_gets"]
        cached_kb = st["cluster_range_bytes"] / 1024
        hits = st["cluster_cache_hits"]
        # A probe is either a cache hit or it needs at least one request, so the
        # hit ratio is hits over hits plus probed clusters that missed. Misses are
        # not observable directly, but a lower bound is `hits + gets`, which is
        # exactly right whenever each miss costs one request per cluster.
        touched = hits + gets
        ratio = f"{100.0 * hits / touched:5.1f}%" if touched else "   n/a"
        entries = st["cluster_cache_entries"]
        weighted = st["cluster_cache_weighted_kib"]
        per_entry = f"{weighted / entries:.0f} KiB" if entries else "n/a"
        print(
            f"probes={n_probe:<3} {queries} queries in {elapsed:6.2f}s "
            f"-> {queries / elapsed:6.1f} QPS | store egress {egress:8.2f} MB "
            f"({egress * 1e6 / queries / 1024:8.1f} KiB/query)\n"
            f"            cluster range GETs {gets} ({gets / queries:.2f}/query, "
            f"{cached_kb / queries:.1f} KiB/query) | cache hits {hits} "
            f"({hits / queries:.2f}/query, {ratio})\n"
            f"            cache resident {entries} clusters, weighed {weighted} KiB "
            f"({per_entry}/cluster) | coverage {100.0 * coverage:6.2f}% of corpus "
            f"| recall@{k} {recall_at_k(returned, truth, k):.4f}"
        )
        records.append(
            {
                "probes": n_probe,
                "coverage": coverage,
                "qps": queries / elapsed,
                "gets_per_query": gets / queries,
                "kib_per_query": cached_kb / queries,
                "hits": hits,
                "hit_ratio": (hits / touched) if touched else 0.0,
                "per_cluster_kib": (weighted / entries) if entries else 0.0,
                "recall": recall_at_k(returned, truth, k),
            }
        )
    return records



def compare(left_path: str, right_path: str, k: int, points: int = 9) -> None:
    """Recall at matched coverage for two measured curves.

    Interpolating on coverage rather than on `nprobe` is the whole point: two
    centroid trainers produce differently sized clusters, so the same probe count
    scans different fractions of the corpus and a recall-at-equal-`nprobe`
    comparison answers the wrong question.
    """
    import json

    def load(path: str) -> tuple[str, list[dict]]:
        with open(path, encoding="utf-8") as fh:
            recs = json.load(fh)
        recs = [r for r in recs if r.get("coverage")]
        recs.sort(key=lambda r: r["coverage"])
        return path, recs

    def at(recs: list[dict], cov: float) -> float | None:
        """Linear interpolation of recall at `cov`, None outside the measured range."""
        if not recs or cov < recs[0]["coverage"] or cov > recs[-1]["coverage"]:
            return None
        for a, b in zip(recs, recs[1:]):
            if a["coverage"] <= cov <= b["coverage"]:
                span = b["coverage"] - a["coverage"]
                t = 0.0 if span == 0 else (cov - a["coverage"]) / span
                return a["recall"] + t * (b["recall"] - a["recall"])
        return recs[-1]["recall"]

    lname, left = load(left_path)
    rname, right = load(right_path)
    lo = max(left[0]["coverage"], right[0]["coverage"])
    hi = min(left[-1]["coverage"], right[-1]["coverage"])
    if hi <= lo:
        print(f"no overlapping coverage: {lname} [{left[0]['coverage']:.3f}, "
              f"{left[-1]['coverage']:.3f}] vs {rname} [{right[0]['coverage']:.3f}, "
              f"{right[-1]['coverage']:.3f}]")
        return
    print(f"| coverage | {lname} recall@{k} | {rname} recall@{k} | delta |")
    print("|---|---|---|---|")
    for i in range(points):
        cov = lo + (hi - lo) * i / (points - 1)
        a = at(left, cov)
        b = at(right, cov)
        if a is None or b is None:
            continue
        print(f"| {100.0 * cov:5.1f}% | {a:.4f} | {b:.4f} | {a - b:+.4f} |")


def rows_for_dim(dim: int, cap: int, budget_mb: float = 120.0) -> int:
    """Rows for one arm, sized so every arm ingests about the same bytes.

    Holding the row count fixed would push 410 MB of vectors through the store for
    the 2048-dim arm alone. The comparison that needs matching is *within* a
    dimension, so each (dim, quantizer) pair shares a row count and the row count
    varies between dimensions to stay inside the byte budget.

    There is a floor as well as a ceiling: below roughly 50 rows per cluster the
    per-cluster graph and the cluster assignment both degenerate, and recall then
    measures the row count rather than the quantizer. `--rows-budget-mb` raises
    the budget when a high-dimension cell needs its clusters populated.
    """
    by_budget = int(budget_mb * 1_000_000) // (4 * dim)
    return max(2_000, min(cap, by_budget))


def sweep(
    bsdb,
    base_uri: str,
    container: str,
    dims: list[int],
    algorithms: list[str],
    rows: int,
    queries: int,
    k: int,
    probes: int,
    m: int,
    rows_budget_mb: float = 120.0,
) -> list[dict]:
    """Build and measure every (dimension, quantizer) cell, then tabulate.

    The whole point is the padding boundary. `BenoQuantEncoder` rotates and then
    pads to the next power of two, so TQ8 stores `next_pow2(dim)` code bytes per
    point while SQ8 stores `dim`. A dimension that is not a power of two is where
    that costs bytes; a power of two is where it costs nothing, which is why a
    single dimension cannot decide between the two quantizers.
    """
    rows_out: list[dict] = []
    for dim in dims:
        rows = rows_for_dim(dim, rows, rows_budget_mb)
        for algo in algorithms:
            short = algo.replace("hnsw_", "").replace("hnsw", "f32")
            uri = f"{base_uri.rstrip('/')}/q{dim}_{short}_{rows}"
            print(f"\n=== dim {dim} ({short}), {rows:,} rows ===", flush=True)
            build(bsdb, uri, container, algo, m, rows, dim)
            records = measure(
                bsdb, uri, container, [probes], queries, k, rows, dim
            )
            rec = records[0]
            rec.update(
                {
                    "dim": dim,
                    "algorithm": short,
                    "padded_dim": 1 << (dim - 1).bit_length(),
                    "rows": rows,
                }
            )
            rows_out.append(rec)
            print(
                f"    -> {rec['per_cluster_kib']:.0f} KiB/cluster, "
                f"recall@{k} {rec['recall']:.4f}, {rec['qps']:.1f} QPS",
                flush=True,
            )
    return rows_out


def render_sweep(records: list[dict], k: int) -> str:
    out = [
        "| dim | padded | quantizer | KiB/cluster | recall@%d | QPS |" % k,
        "|---|---|---|---|---|---|",
    ]
    for r in records:
        pad = r["padded_dim"]
        pad_cell = f"{pad} (+{100.0 * (pad - r['dim']) / r['dim']:.1f}%)" if pad != r["dim"] else f"{pad} (none)"
        out.append(
            f"| {r['dim']} | {pad_cell} | {r['algorithm']} | "
            f"{r['per_cluster_kib']:.0f} | {r['recall']:.4f} | {r['qps']:.1f} |"
        )
    return "\n".join(out)


def _query(table, vector, k: int, ef_search: int, probes: int):
    return table.to_arrow(
        vector_filter={
            "column": "embedding",
            "query": vector.tolist(),
            "k": k,
            # Pinned so the recall oracle knows which ranking to reproduce.
            "metric": "l2",
            "ef_search": ef_search,
            "probes": probes,
        },
        columns=["id"],
    )


def _ids(table) -> list[int]:
    """The `id` values of a result, in the order the engine returned them."""
    return table.column("id").to_pylist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["build", "measure", "sweep", "compare"])
    parser.add_argument("--left", default=None, help="`compare`: JSON from one arm")
    parser.add_argument("--right", default=None, help="`compare`: JSON from the other")
    parser.add_argument("--uri", default="s3://bsdb-bench/bench")
    parser.add_argument("--container", default="rustfs")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--probes", default="8,16,32")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--json-out",
        default=None,
        help="write the per-pass records here, so `compare` can read two curves",
    )
    parser.add_argument(
        "--algorithm",
        default=None,
        help="index algorithm for the vector column, e.g. hnsw, hnsw_tq8, hnsw_tq4",
    )
    parser.add_argument("--m", type=int, default=16, help="cluster graph degree")
    parser.add_argument("--rows", type=int, default=N)
    parser.add_argument(
        "--rows-budget-mb",
        type=float,
        default=120.0,
        help="vector bytes per sweep arm, which sets its row count; raise it for high dimensions",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DIM,
        help="vector dimensions; a power of two is the case where TQ8 does not pad",
    )
    parser.add_argument(
        "--dims",
        default="64,100,300,384,512,768,1024,1536,2048,3072",
        help=(
            "dimensions for `sweep`, comma separated: 64/512/1024/2048 are powers "
            "of two (TQ8 does not pad), 100 is GloVe-100d, 300 is GloVe/fastText, "
            "384 is all-MiniLM-L6-v2, 768 is BERT-base/mpnet-base/Cohere v2, "
            "1536 is OpenAI text-embedding-3-small and ada-002, 3072 is "
            "text-embedding-3-large (pads to 4096)"
        ),
    )
    parser.add_argument(
        "--algorithms",
        default="hnsw_sq8,hnsw_tq8",
        help="index algorithms for `sweep`, comma separated",
    )
    args = parser.parse_args()

    import benodb as bsdb

    probes = [int(p) for p in args.probes.split(",") if p.strip()]
    if args.phase == "build":
        build(bsdb, args.uri, args.container, args.algorithm, args.m, args.rows, args.dim)
    elif args.phase == "sweep":
        dims = [int(d) for d in args.dims.split(",") if d.strip()]
        algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
        # One probe value for the whole grid: every cluster, so the probe stage
        # cannot mask the quantizer's effect on ranking.
        records = sweep(
            bsdb,
            args.uri,
            args.container,
            dims,
            algos,
            args.rows,
            args.queries,
            args.k,
            probes[0],
            args.m,
            args.rows_budget_mb,
        )
        print()
        print(render_sweep(records, args.k))
        out = os.environ.get("SWEEP_OUT")
        if out:
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(render_sweep(records, args.k) + "\n")
            print(f"wrote {out}")
    elif args.phase == "compare":
        compare(args.left, args.right, args.k)
    else:
        records = measure(
            bsdb,
            args.uri,
            args.container,
            probes,
            args.queries,
            args.k,
            args.rows,
            args.dim,
        )
        if args.json_out:
            import json

            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2)
            print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
