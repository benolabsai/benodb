#!/usr/bin/env python3
# Copyright (c) 2026 Richard Albright and BenoDB Contributors.
"""Build one quantizer's index inside a detached overlay directory of its own.

Run inside the envelope by `run_quantizer_index_dirs.sh build <mode>`, as

    docker run --entrypoint python3 ... bsdb-neurips:latest \\
        /opt/bench/current/quantizer_index_build.py

Environment:
    QIX_UPSTREAM      upstream read-only Iceberg table directory (mounted via
                      `Table.mount_external_iceberg` when set)
    QIX_DIR           overlay (or standalone table) directory
    QIX_MODE          tq8 | sq8 | tq4 | pq | f32
    QIX_COLUMN        vector column
    QIX_CLUSTERS      table-level cluster count
    QIX_M             cluster graph degree
    QIX_EFC           construction beam
    QIX_METRIC        distance metric (default: l2)
    QIX_SKIP_REINDEX  "1" to reuse the segment tier already present
"""

from __future__ import annotations

import json
import os
import sys
import time


def main() -> int:
    import benodb as bsdb

    upstream = os.environ.get("QIX_UPSTREAM", "").strip()
    directory = os.environ["QIX_DIR"]
    mode = os.environ["QIX_MODE"]
    column = os.environ.get("QIX_COLUMN", "embedding")
    clusters = int(os.environ.get("QIX_CLUSTERS", "512"))
    m = int(os.environ.get("QIX_M", "16"))
    efc = int(os.environ.get("QIX_EFC", "200"))
    metric = os.environ.get("QIX_METRIC", "l2")
    skip_segment = os.environ.get("QIX_SKIP_REINDEX") == "1"

    idx_type = "hnsw" if mode == "f32" else mode

    print(f"   upstream: {upstream or '(standalone)'}", flush=True)
    print(f"   overlay:  {directory}", flush=True)
    print(
        f"   mode:     {mode}   segment reindex: {'skipped' if skip_segment else 'yes'}",
        flush=True,
    )

    os.makedirs(directory, exist_ok=True)
    if upstream:
        table = bsdb.Table.mount_external_iceberg(upstream, directory)
    else:
        table = bsdb.Table(directory)

    # Always register the target quantizer so `rebuild_cluster_index` selects it
    # even when `skip_segment` is set. On a fresh overlay, `add_index` triggers
    # an automatic background backfill of unindexed segments; waiting on it
    # builds the segment tier once without a redundant second pass.
    t0 = time.time()
    table.add_index(
        column,
        {
            "type": idx_type,
            "complexity": m,
            "quality": efc,
            "metric": metric,
        },
    )
    table.wait_for_background_tasks()
    if not skip_segment and not upstream:
        table.rebuild_index(column)
        table.wait_for_background_tasks()
    print(f"   segment tier ready in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    table.rebuild_cluster_index(column, k=clusters, m=m, ef_construction=efc)
    print(f"   table artifact rebuilt in {time.time() - t0:.1f}s", flush=True)

    marker = {
        "mode": mode,
        "algorithm": f"hnsw_{mode}" if mode != "f32" else "hnsw",
        "m": m,
        "ef_construction": efc,
        "clusters": clusters,
        "upstream": upstream or None,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(directory, ".index_mode.json"), "w") as f:
        json.dump(marker, f, indent=2, sort_keys=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
