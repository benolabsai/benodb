#!/usr/bin/env python3
# Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
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


def build(bsdb, uri: str, container: str) -> None:
    existing = _existing_objects(uri)
    if existing:
        raise SystemExit(
            f"{uri} already holds {existing} objects; write appends, so use a fresh prefix"
        )
    rx0, tx0 = netio(container)
    t0 = time.time()
    table = _table(bsdb, uri)
    rng = np.random.default_rng(0)
    for start in range(0, N, SEGMENT_ROWS):
        end = min(start + SEGMENT_ROWS, N)
        vectors = rng.random((end - start, DIM), dtype=np.float32)
        fsl = pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), DIM)
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

    t0 = time.time()
    table.rebuild_cluster_index(
        "embedding",
        k=CLUSTERS,
        m=16,
        ef_construction=200,
        target_shard_bytes=SHARD_BYTES,
    )
    index_s = time.time() - t0

    rx1, tx1 = netio(container)
    print(
        f"build: {N:,}x{DIM} in {ingest_s:.1f}s, index {CLUSTERS} clusters "
        f"in {index_s:.1f}s | store rx {(rx1 - rx0) / 1e6:.1f} MB, "
        f"tx {(tx1 - tx0) / 1e6:.1f} MB"
    )
    del table


def measure(bsdb, uri: str, container: str, probes: list[int], queries: int) -> None:
    rng = np.random.default_rng(1)
    query_vectors = rng.random((queries, DIM), dtype=np.float32)

    # Cold open plus one query: this is what triggers `load_sharded_remote`, so
    # the delta is the whole initialisation cost. The request count is the point
    # of the shard descriptor: an open that reads every footer pays three
    # requests per shard, one that reads the descriptor pays five in total.
    bsdb.reset_reader_stats()
    rx0, tx0 = netio(container)
    t0 = time.time()
    table = _table(bsdb, uri)
    first = _query(table, query_vectors[0], 10, 64, probes[0])
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

    for n_probe in probes:
        rx0, tx0 = netio(container)
        bsdb.reset_reader_stats()
        t0 = time.time()
        for i in range(queries):
            _query(table, query_vectors[i], 10, 64, n_probe)
        elapsed = time.time() - t0
        rx1, tx1 = netio(container)
        st = bsdb.reader_stats()
        egress = max(tx1 - tx0, 0) / 1e6
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
            f"({per_entry}/cluster)"
        )


def _query(table, vector, k: int, ef_search: int, probes: int):
    return table.to_arrow(
        vector_filter={
            "column": "embedding",
            "query": vector.tolist(),
            "k": k,
            "ef_search": ef_search,
            "probes": probes,
        },
        columns=["id"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["build", "measure"])
    parser.add_argument("--uri", default="s3://bsdb-bench/bench")
    parser.add_argument("--container", default="rustfs")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--probes", default="8,16,32")
    args = parser.parse_args()

    import benostreamdb as bsdb

    probes = [int(p) for p in args.probes.split(",") if p.strip()]
    if args.phase == "build":
        build(bsdb, args.uri, args.container)
    else:
        measure(bsdb, args.uri, args.container, probes, args.queries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
