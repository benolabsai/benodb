#!/usr/bin/env python3
"""Official NeurIPS BigANN benchmark runner for BenoStreamDB.

The run is split into two stages so that a large dataset is ingested **once**
and then re-indexed per quantizer, instead of being re-ingested for every mode:

  ingest  Create the table (no index configured) and stream the raw vectors in.
          This is a pure columnar write: no HNSW, no quantizer, no index files.
  query   For each mode: configure the algorithm, rebuild the index *from the
          stored Parquet* (``Table.rebuild_index``), preload, then query.

``--modes`` runs the ingest stage once (unless the DB already exists) and
then one query pass per mode, all inside a single process, so the table is
never re-read from the raw ``.u8bin`` and the index is never rebuilt by
re-ingesting.

Intra-engine ablation:
1. Native SQ8
2. TurboQuant TQ8 (HNSW-TQ8: H * D)
3. TurboQuant TQ4 (HNSW-TQ4: H * D + 2B scale)
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import psutil
import pyarrow as pa

DEFAULT_DATA_DIR = "/home/ralbright/data/datasets/bigann"
DEFAULT_DIM = 128

# NeurIPS BigANN/Billion-Scale & Dense Retrieval tracks: file tag -> (tag, base_count, default_dim, default_metric).
DATASETS = {
    "10m": ("10M", 10_000_000, 128, "l2"),
    "100m": ("100M", 100_000_000, 128, "l2"),
    "cohere-10m": ("10M", 10_000_000, 768, "inner_product"),
    "cohere": ("10M", 10_000_000, 768, "inner_product"),
}
QUERY_SETS = {
    "public": ("query.public.10K.u8bin", 10_000),
    "private": ("query.private.10K.u8bin", 10_000),
    "learn": ("query.learn.10K.u8bin", 10_000),
}


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def probe_bin_file(path: str):
    """Probe binary file header [count: int32, dim: int32] and detect dtype."""
    if not path or not os.path.exists(path):
        return None, None, np.float32
    try:
        with open(path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=2)
        if header.size < 2:
            return None, None, np.float32
        count, dim = int(header[0]), int(header[1])
        is_fbin = path.endswith(".fbin") or "fbin" in path.lower() or "float" in path.lower()
        dtype = np.float32 if is_fbin else np.uint8
        return count, dim, dtype
    except Exception:
        return None, None, np.float32


def resolve_dataset(args):
    """Resolve (base, queries, gt, n, dim, dtype, metric) from dataset args.

    Auto-probes vector count, dimensionality and data type (uint8 / float32)
    from binary file headers.
    """
    if args.dataset in DATASETS:
        tag, count, def_dim, def_metric = DATASETS[args.dataset]
    else:
        tag, count, def_dim, def_metric = (args.dataset, 10_000_000, DEFAULT_DIM, "l2")

    qfile, qcount = QUERY_SETS.get(args.query_set, (f"query.{args.query_set}.10K.u8bin", 10_000))

    # Candidate paths for base vectors
    base = args.base
    if base is None:
        for cand in (
            f"base.{tag}.fbin",
            f"base.{tag}.u8bin",
            f"base.{tag}.bin",
            "base.fbin",
            "base.u8bin",
            f"base.{tag}",
        ):
            p = os.path.join(args.data_dir, cand)
            if os.path.exists(p):
                base = p
                break
        else:
            base = os.path.join(args.data_dir, f"base.{tag}.u8bin")

    # Probe file properties from base
    file_count, file_dim, file_dtype = probe_bin_file(base)
    dim = args.dim or file_dim or def_dim
    dtype = file_dtype
    metric = args.metric or def_metric
    if metric in ("ip", "inner_product"):
        metric = "inner_product"

    # Candidate paths for query vectors
    query = getattr(args, "query", None)
    if query is None:
        for cand in (
            "queries.fbin",
            "query.fbin",
            f"query.{tag}.fbin",
            qfile,
            f"query.{tag}.u8bin",
            "queries.u8bin",
        ):
            p = os.path.join(args.data_dir, cand)
            if os.path.exists(p):
                query = p
                break
        else:
            query = os.path.join(args.data_dir, qfile)

    # Candidate paths for ground truth
    gt = args.gt
    if gt is None:
        for cand in (
            f"GT.{tag}.{args.query_set}",
            f"GT.{tag}",
            f"gt.{tag}",
            f"GT.{tag}.10K",
            "gt_100.fbin",
            "gt.ibin",
            "groundtruth.ibin",
            f"groundtruth.{tag}.ibin",
        ):
            p = os.path.join(args.data_dir, cand)
            if os.path.exists(p):
                gt = p
                break
        else:
            gt = os.path.join(args.data_dir, f"GT.{tag}")

    n = args.limit if args.limit and args.limit > 0 else (file_count or count)
    return base, query, gt, n, tag, (file_count or count), qcount, dim, dtype, metric


def load_queries(q_path: str, queries_count: int, dim: int, dtype: np.dtype):
    q_count, q_dim, q_dtype = probe_bin_file(q_path)
    if q_dim is None:
        q_dim = dim
    if q_dtype is None:
        q_dtype = dtype
    if q_count is None:
        q_count = 10_000
    queries = np.memmap(q_path, dtype=q_dtype, mode="r", offset=8, shape=(q_count, q_dim))
    q = min(queries_count, len(queries))
    return queries[:q].astype(np.float32), q


def exact_ground_truth(base_path: str, queries, n: int, k: int, dim: int, dtype: np.dtype, metric: str = "l2", chunk: int = 1_000_000):
    """Exact top-k for a subset small enough to brute force (<= ~2M vectors)."""
    base = np.memmap(base_path, dtype=dtype, mode="r", offset=8, shape=(n, dim))
    out = np.empty((len(queries), k), dtype=np.int64)
    for i, qv in enumerate(queries):
        if metric in ("inner_product", "ip"):
            dists = np.concatenate([
                -np.dot(base[s: s + chunk].astype(np.float32), qv)
                for s in range(0, n, chunk)
            ])
        else:
            dists = np.concatenate([
                np.sum((base[s: s + chunk].astype(np.float32) - qv) ** 2, axis=1)
                for s in range(0, n, chunk)
            ])
        out[i] = np.argsort(dists)[:k]
    return out


def load_ground_truth(gt_path: str, n: int, dataset_count: int, queries_count: int, k: int):
    """Official precomputed GT, but only for the corpus it actually describes.

    `GT.10M` holds neighbour ids into the *full* 10M base, so using it for a
    `--limit 1000000` subset compares against vectors that were never ingested
    and reports ~0 recall. A subset must use exact GT (computed by the caller
    when it is small enough) or report recall as null.

    Format: 8-byte header (n_queries: int32, k: int32), followed by
    n_queries * k uint32 neighbour IDs, followed by n_queries * k float32 distances.
    Only read the ID slice — reading the whole file without count doubles the
    stride (200 ints per row instead of 100) and corrupts every query past #0.
    """
    if n != dataset_count or not os.path.exists(gt_path):
        return None
    with open(gt_path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
    if header.size < 2:
        return None
    n_q, file_k = int(header[0]), int(header[1])
    if file_k < k:
        return None
    gt = np.fromfile(gt_path, dtype=np.uint32, offset=8, count=n_q * file_k)
    if gt.size != n_q * file_k:
        return None
    return gt.reshape((n_q, file_k))[:queries_count, :k]


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def db_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024.0 * 1024.0)


def db_breakdown(path: str) -> dict:
    """Split the table directory into data bytes and index bytes.

    `db_size_mb` returns the whole directory, which is what the ingest stage
    wants but not what the paper means by "index size". A table directory holds
    `seg_*.parquet` (the columnar data) and `seg_*.puffin` (the advisory index
    sidecar); reporting the sum as the index overstates it by the size of the
    data. Measured at 10M: 5.0 GB parquet + 4.1 GB puffin, reported as 9.17 GB.
    """
    data = index = other = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            if os.path.islink(fp):
                continue
            size = os.path.getsize(fp)
            if fn.endswith(".puffin"):
                index += size
            elif fn.endswith(".parquet"):
                data += size
            else:
                other += size
    mb = 1024.0 * 1024.0
    return {
        "data_size_mb": round(data / mb, 2),
        "index_size_mb": round(index / mb, 2),
        "other_size_mb": round(other / mb, 2),
        "db_size_mb": round((data + index + other) / mb, 2),
    }


def ingest_stage(bsdb, args, base_path: str, dim: int = DEFAULT_DIM, dtype: np.dtype = np.uint8) -> dict:
    """Create the table and stream the raw vectors in, without any index."""
    db_path = args.db_dir
    if args.rebuild_db and os.path.exists(db_path):
        print(f"    -> Removing existing DB at {db_path}", flush=True)
        shutil.rmtree(db_path)

    n = args.n
    bytes_per_vector = dim * np.dtype(dtype).itemsize

    # Auto-scale flush_every for high-dimensional vectors (e.g. 768-dim float32 is 3,072 B/vec)
    # to guarantee write buffer stays under ~1 GB memory regardless of dimension.
    batch_size = args.batch_size
    flush_every = args.flush_every
    if flush_every == 5_000_000 and dim >= 256:
        flush_every = max(250_000, 1_000_000_000 // max(bytes_per_vector, 1))

    # Use standard sequential binary streaming with fadvise DONTNEED instead of
    # a static 30+ GB memory map, preventing kernel page-cache saturation and
    # thrashing under 16 GiB cgroups.
    raw_file = open(base_path, "rb")
    raw_fd = raw_file.fileno()
    try:
        os.posix_fadvise(raw_fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
    except Exception:
        pass
    raw_file.seek(8)  # skip 8-byte header (n_vectors: int32, dim: int32)

    process = psutil.Process(os.getpid())

    t0 = time.time()
    if getattr(args, "auto_warmup", False):
        inner = bsdb._RustTable(db_path, auto_warmup=True)
        table = bsdb.Table(db_path, inner_table=inner)
    else:
        table = bsdb.Table(db_path)

    since_flush = 0
    t_ingest = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        cur_batch_count = end - start

        batch_raw = np.fromfile(raw_file, dtype=dtype, count=cur_batch_count * dim)
        if dtype == np.float32:
            batch_floats = batch_raw
        else:
            batch_floats = batch_raw.astype(np.float32, copy=False)

        fsl = pa.FixedSizeListArray.from_arrays(pa.array(batch_floats.ravel()), dim)
        ids = pa.array(np.arange(start, end, dtype=np.int64))
        tbl = pa.Table.from_arrays([ids, fsl], names=["id", "embedding"])
        table.write(tbl)
        since_flush += cur_batch_count

        # Evict processed file pages from page cache
        try:
            offset_bytes = 8 + start * bytes_per_vector
            length_bytes = cur_batch_count * bytes_per_vector
            os.posix_fadvise(raw_fd, offset_bytes, length_bytes, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass

        # Release Python & PyArrow temporary objects immediately
        del tbl, fsl, ids, batch_floats, batch_raw

        # Flush write buffer to Parquet when threshold is reached
        if flush_every and since_flush >= flush_every:
            table.flush()
            since_flush = 0
            gc.collect()
            try:
                pa.default_memory_pool().release_unused()
            except Exception:
                pass
            print(
                f"    -> Flushed write buffer at {end:,} vectors "
                f"(RSS: {process.memory_info().rss / (1024**3):.2f} GB)",
                flush=True,
            )

        if end % 1_000_000 == 0 or end == n:
            elapsed = time.time() - t_ingest
            rate = end / max(elapsed, 0.1)
            rss = process.memory_info().rss / (1024**3)
            print(
                f"    -> Ingested {end:,} / {n:,} vectors ({end/n*100:.1f}%) "
                f"[{rate:,.0f} vec/s] | RSS: {rss:.2f} GB",
                flush=True,
            )

    raw_file.close()

    print("    -> Committing ...", flush=True)
    table.commit()
    table.wait_for_background_tasks()
    build_time = time.time() - t0

    stats = {
        "stage": "ingest",
        "n_train": n,
        "db_dir": db_path,
        "ingest_time_s": round(build_time, 2),
        "ingest_rate_vps": round(n / build_time, 0) if build_time > 0 else 0,
        "data_size_mb": round(db_size_mb(db_path), 2),
        "peak_rss_mb": round(process.memory_info().rss / (1024 * 1024), 1),
    }
    del table
    return stats


def evict_file_caches(paths: list):
    """Evict clean pages from OS page cache using posix_fadvise (unprivileged)."""
    evicted_files = 0
    evicted_bytes = 0
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        if os.path.isfile(p):
            targets = [p]
        else:
            targets = []
            for root, _, files in os.walk(p):
                for f in files:
                    targets.append(os.path.join(root, f))
        for fp in targets:
            try:
                sz = os.path.getsize(fp)
                fd = os.open(fp, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
                evicted_files += 1
                evicted_bytes += sz
            except Exception:
                pass
    return evicted_files, evicted_bytes


def clear_evaluation_caches(table=None, paths=None, label: str = ""):
    """Flush engine memory caches, sync, and evict OS page caches.

    1. Invalidate engine resident in-memory caches (HNSW, IVF, CSR graph, inverted, vector routers).
    2. Try kernel drop_caches (/proc/sys/vm/drop_caches) if running privileged/root.
    3. Evict target file pages via posix_fadvise(POSIX_FADV_DONTNEED).
    4. Force Python garbage collection.
    """
    if table is not None and hasattr(table, "clear_cache"):
        try:
            table.clear_cache(all=True)
        except Exception:
            pass
    try:
        import benostreamdb as _bsdb
        if hasattr(_bsdb, "clear_cache"):
            _bsdb.clear_cache(all=True)
    except Exception:
        pass

    try:
        os.sync()
    except Exception:
        pass

    proc_ok = False
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        proc_ok = True
    except Exception:
        pass

    evicted_files, evicted_bytes = evict_file_caches(paths or [])
    tag = f" [{label}]" if label else ""
    evicted_gb = evicted_bytes / (1024**3)
    if proc_ok:
        print(f"    -> Cleared caches{tag}: engine caches wiped, /proc/sys/vm/drop_caches=3, "
              f"posix_fadvise ({evicted_files} files, {evicted_gb:.2f} GB evicted)", flush=True)
    else:
        print(f"    -> Cleared caches{tag}: engine caches wiped, "
              f"posix_fadvise ({evicted_files} files, {evicted_gb:.2f} GB evicted from page cache)", flush=True)

    import gc
    gc.collect()


def index_mode_marker(db_path: str) -> str:
    """Path of the harness-authored record of which quantizer built the index.

    Each quantizer is a first-class `IndexAlgorithm` (`hnsw`, `hnsw_sq8`,
    `hnsw_tq8`, `hnsw_tq4`, `hnsw_pq`), so the mode is recoverable from the
    manifest. The marker remains as a build-parameter record so `--skip-reindex`
    can still detect a mismatched index on disk.
    """
    return os.path.join(db_path, ".index_mode.json")


def read_index_mode(db_path: str) -> dict:
    """The recorded build parameters for the index currently on disk, if any."""
    try:
        with open(index_mode_marker(db_path)) as f:
            return json.load(f)
    except Exception:
        return {}


def write_index_mode(db_path: str, mode: str, idx_type: str, args) -> None:
    """Record the provenance of the index now on disk.

    This file is the index's manifest: which quantizer built it, with which graph
    parameters, at which Parquet page size, over how many vectors, and from which
    commit. A query sweep may reuse an index only when the manifest matches what
    the sweep asked for, which is what stops a run from reporting one quantizer
    while silently querying another.
    """
    record = {
        "mode": mode,
        "algorithm": idx_type,
        "complexity": args.m,
        "quality": args.ef_construction,
        "metric": getattr(args, "metric", None),
        "dim": getattr(args, "dim", None),
        "n_vectors": getattr(args, "n", None),
        "dataset": getattr(args, "dataset", None),
        "flush_every": getattr(args, "flush_every", None),
        "parquet_page_bytes": parquet_page_bytes(),
        "cache_gb": getattr(args, "cache_gb", None),
        "index_compaction": bool(getattr(args, "index_compaction", False)),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": git_state(),
    }
    with open(index_mode_marker(db_path), "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)


def reindex_and_preload(bsdb, args, mode: str):
    """Rebuild or reuse the index for `mode` in the table and warm it.

    This is paid once per mode — the query grid then runs against the same warm
    index. Switching modes overwrites the previous mode's Puffin sidecars, which
    is why the mode is recorded in `.index_mode.json`: a later `--skip-reindex`
    run can then tell whether the index on disk is the quantizer it asked for.
    """
    base_db = getattr(args, "base_db_dir", None) or args.db_dir or os.path.join(args.scratch_dir, "bigann_db")

    # One table, rebuilt in place per mode.
    #
    # Per-mode directories CANNOT work for this. An Iceberg table's identity is
    # the `location` in its metadata JSON, and the manifests store absolute
    # `file:///.../bigann_db/seg_*.parquet` URIs. Hard-linking a data directory
    # therefore yields a copy that still points at the original: `rebuild_index`
    # reads the base manifests, derives each Puffin path from the absolute base
    # Parquet path, and writes every index back into the base table, leaving the
    # copy with no index at all — `has_vector_indices: False`, preloaded 0 of 0,
    # and every "query" silently measuring a full scan. The Puffin filename is
    # also fixed at `{segment_id}.puffin` and overwritten on each rebuild, so a
    # copied directory could not hold two modes' blobs at the same time anyway.
    #
    # Rebuilding in place costs exactly what a shadow rebuild costs — this loop
    # already pays one build per mode — and every query then runs against a real
    # table. Which mode the index belongs to is recorded out of band.
    if not os.path.exists(base_db):
        raise RuntimeError(
            f"DB '{base_db}' does not exist; run the ingest command first "
            "(or drop the query subcommand)"
        )
    db_path = base_db

    if getattr(args, "auto_warmup", False):
        inner = bsdb._RustTable(db_path, auto_warmup=True)
        table = bsdb.Table(db_path, inner_table=inner)
    else:
        table = bsdb.Table(db_path)

    # Which algorithm this mode builds. Each quantizer is a distinct, first-class
    # `IndexAlgorithm` (`hnsw`, `hnsw_sq8`, `hnsw_tq8`, `hnsw_tq4`, `hnsw_pq`),
    # so the mode is recorded here and also inferable from the manifest.
    if mode == "f32":
        idx_type = "hnsw"
    elif mode == "tq4":
        idx_type = "tq4"
    elif mode == "sq8":
        idx_type = "sq8"
    elif mode == "pq":
        idx_type = "pq"
    else:
        idx_type = "tq8"

    force = getattr(args, "force_reindex", False)
    skip = getattr(args, "skip_reindex", False)

    # An index is reusable only if it is the one this mode asked for. Counting
    # `*.puffin` against `*.parquet` proves something was built; it does not
    # prove which quantizer built it, and reusing the wrong one reports a
    # quantizer that never ran.
    marker = read_index_mode(db_path)
    has_existing_index = False
    try:
        parquets = [f for f in os.listdir(db_path) if f.endswith(".parquet")]
        puffins = [f for f in os.listdir(db_path) if f.endswith(".puffin")]
        if len(parquets) > 0 and len(puffins) >= len(parquets):
            if marker.get("mode") == mode:
                has_existing_index = True
            elif marker:
                print(
                    f"    [!] on-disk index was built for {marker.get('mode')} "
                    f"({marker.get('algorithm')}), not {mode}; rebuilding",
                    flush=True,
                )
            else:
                print(
                    f"    [!] on-disk index carries no mode record; rebuilding for {mode}",
                    flush=True,
                )
    except Exception:
        pass

    if skip and not has_existing_index:
        print(
            f"    [!] --skip-reindex ignored: the index on disk is not {mode}",
            flush=True,
        )
        skip = False

    if skip or (has_existing_index and not force):
        print(f"    -> Reusing existing index on disk ({mode.upper()}) at {db_path}", flush=True)
        reindex_s = 0.0
    else:
        print(f"    -> Reindexing 'embedding' from stored data ({mode.upper()}) into {db_path} ...", flush=True)
        t_idx = time.time()
        table.add_index(
            "embedding",
            {
                "type": idx_type,
                "complexity": args.m,
                "quality": args.ef_construction,
                "metric": getattr(args, "metric", "l2"),
            },
        )
        table.rebuild_index("embedding")
        table.wait_for_background_tasks()
        if hasattr(table, "compact_indexes"):
            print("    -> Compacting index sidecars into Voronoi shards ...", flush=True)
            try:
                cstats = table.compact_indexes("embedding")
                print(f"    -> Compaction complete: {cstats}", flush=True)
            except Exception as e:
                print(f"    [!] compaction note: {e}", flush=True)
        reindex_s = time.time() - t_idx
        write_index_mode(db_path, mode, idx_type, args)

    # Index-only compaction: build the table-level, cluster-keyed artifact.
    #
    # This is the "global clustering" artifact from
    # docs/architecture/ADR_002_GLOBAL_VECTOR_CLUSTERING.md: one Puffin file, not owned by any
    # data file, keyed by global cluster, whose nodes carry (segment_id, row_id).
    # The Parquet files are read once and never rewritten. The query path
    # (read_expr_stream_async) prefers this artifact when present, so the sweep
    # below measures the compacted index instead of the per-segment fan-out.
    compaction_s = 0.0
    compaction_stats = None
    if getattr(args, "index_compaction", False):
        print(
            f"    -> Index compaction: building table-level cluster artifact "
            f"(k={args.compaction_k}, m={args.compaction_m}, "
            f"ef_c={args.compaction_ef_construction}) ...",
            flush=True,
        )
        t_comp = time.time()
        table.rebuild_cluster_index(
            "embedding",
            k=args.compaction_k,
            m=args.compaction_m,
            ef_construction=args.compaction_ef_construction,
            target_shard_bytes=getattr(args, "compaction_target_shard_bytes", None),
        )
        compaction_s = time.time() - t_comp
        compaction_stats = {
            "k": args.compaction_k,
            "m": args.compaction_m,
            "ef_construction": args.compaction_ef_construction,
            "build_s": round(compaction_s, 2),
        }
        print(
            f"    -> Index compaction complete in {compaction_s:.1f}s "
            f"({args.compaction_k} global clusters)",
            flush=True,
        )

    sizes = db_breakdown(db_path)
    if not (skip or (has_existing_index and not force)):
        print(
            f"    -> Reindexed in {reindex_s:.1f}s "
            f"(index {sizes['index_size_mb'] / 1024:.2f} GB, "
            f"data {sizes['data_size_mb'] / 1024:.2f} GB, "
            f"table {sizes['db_size_mb'] / 1024:.2f} GB)",
            flush=True,
        )
    else:
        print(
            f"    -> Found existing index "
            f"(index {sizes['index_size_mb'] / 1024:.2f} GB, "
            f"data {sizes['data_size_mb'] / 1024:.2f} GB, "
            f"table {sizes['db_size_mb'] / 1024:.2f} GB)",
            flush=True,
        )

    # Invalidate engine internal caches and evict unneeded Parquet build data from OS page cache
    clear_evaluation_caches(table, [db_path, getattr(args, "scratch_dir", None)], label=f"pre-warm-{mode.upper()}")

    print("    -> Preloading index into RAM cache ...", flush=True)
    warm = table.preload_indexes(
        max_memory_bytes=int(args.preload_gb * 1024**3), spill_to_disk=False
    )
    print(
        f"    -> Preloaded {warm.get('indexes_warmed', 0)} of {warm.get('indexes_seen', 0)} "
        f"index entries ({warm.get('bytes_in_memory', 0) / (1024**3):.2f} GB in RAM) "
        f"in {warm.get('elapsed_ms', 0):.0f}ms",
        flush=True,
    )
    if warm.get("last_error"):
        print(f"    [!] preload error: {warm['last_error']}", flush=True)
    if not warm.get("indexes_warmed"):
        print(
            "    [!] nothing warmed: queries will scan instead of using the index "
            f"({json.dumps(warm)})",
            flush=True,
        )

    return table, {
        "db_path": db_path,
        "reindex_s": round(reindex_s, 2),
        "compaction_s": round(compaction_s, 2),
        "compaction": compaction_stats,
        "preload_ms": warm.get("elapsed_ms", 0),
        **sizes,
    }


def search_pass(table, args, queries, gt_indices, ef_search: int, n_probe: int, mode: str = "") -> dict:
    """One query pass at a given (ef_search, n_probe) operating point."""
    q = len(queries)
    query_vectors = [queries[i].tolist() for i in range(q)]
    process = psutil.Process(os.getpid())

    def search_worker(idx):
        vf = {
            "column": "embedding",
            "query": query_vectors[idx],
            "k": args.k,
            "ef_search": ef_search,
            "probes": n_probe,
        }
        if getattr(args, "refine_factor", None) is not None and mode.lower() != "f32":
            vf["refine_factor"] = args.refine_factor
        t1 = time.perf_counter()
        res = table.to_arrow(vector_filter=vf, columns=["id"])
        lat = time.perf_counter() - t1
        ids = res.column("id").to_pylist() if "id" in res.column_names else []

        hit = None
        if gt_indices is not None:
            truth = set(int(x) for x in gt_indices[idx][: args.k])
            hit = len(set(ids) & truth) / args.k
        return lat, hit

    wall_start = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        results = list(executor.map(search_worker, range(q)))
    wall = time.time() - wall_start

    latencies = [r[0] for r in results]
    hits = [r[1] for r in results if r[1] is not None]
    recall = float(np.mean(hits)) if hits else None

    return {
        "queries": q,
        "k": args.k,
        "threads": args.threads,
        "ef_search": ef_search,
        "n_probe": n_probe,
        # Recorded so a refined point is distinguishable from an unrefined one in the
        # artifact: the engine treats a missing/1 factor as "off", and until now no
        # record carried the field at all, so `grep refine results/**/*.json` found
        # nothing even when `--refine-factor` had been passed.
        "refine_factor": getattr(args, "refine_factor", None),
        "recall_at_k": round(recall, 4) if recall is not None else None,
        "pure_qps": round(q / max(wall, 0.001), 1),
        "p50_ms": round(float(np.percentile(latencies, 50)) * 1000, 2),
        "p99_ms": round(float(np.percentile(latencies, 99)) * 1000, 2),
        "peak_rss_mb": round(process.memory_info().rss / (1024 * 1024), 1),
    }


# --------------------------------------------------------------------------- #
# Result persistence
# --------------------------------------------------------------------------- #


def _git(args: list) -> str:
    try:
        return subprocess.check_output(
            ["git"] + args, cwd=os.path.dirname(os.path.abspath(__file__)), text=True
        ).strip()
    except Exception:
        return "unknown"


def git_state() -> dict:
    """Revision and dirty state of the tree that produced a result.

    Every artifact before this change recorded `git_rev: unknown` and
    `git_dirty: true`, which made the numbers unverifiable: a result produced
    from a modified tree cannot be reproduced from any commit. The harness now
    resolves the revision, counts the modified paths, and refuses to write a
    result from a dirty tree unless `--allow-dirty` marks the run as throwaway.
    """
    porcelain = _git(["status", "--porcelain"])
    dirty_paths = [line for line in porcelain.splitlines() if line.strip()]
    return {
        "git_rev": _git(["rev-parse", "HEAD"]),
        "git_rev_short": _git(["rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(dirty_paths),
        "git_dirty_paths": 0 if porcelain == "unknown" else len(dirty_paths),
    }


def parquet_page_bytes() -> int:
    """The Parquet data page size the writer will use for this run.

    The default of 32 KB is what the rerank path is tuned for. The environment
    override exists so the page size ablation can write the same corpus at
    several page sizes and compare the physical bytes read per query, which is
    the measurement the paper's central claim currently lacks.
    """
    raw = os.environ.get("BSDB_PARQUET_PAGE_BYTES", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return 32 * 1024
    return value if value >= 1024 else 32 * 1024


def aggregate_repeats(runs: list) -> dict:
    """Median with interquartile range across repeated measurements of one point.

    A single run cannot separate a real difference from scheduling noise, and the
    paper reports differences of a few percent. Repeats are recorded rather than
    silently averaged: the report shows the median, and the interquartile range
    stays in the JSON so a reader can see the spread.
    """
    import statistics

    base = dict(runs[0])
    base["repeat"] = len(runs)
    if len(runs) == 1:
        return base

    def values(key):
        return sorted(r[key] for r in runs if r.get(key) is not None)

    for key in ("pure_qps", "p50_ms", "p99_ms", "recall_at_k"):
        vals = values(key)
        if not vals:
            continue
        base[key] = statistics.median(vals)
        if len(vals) >= 4:
            quartiles = statistics.quantiles(vals, n=4)
            base[f"{key}_iqr"] = quartiles[2] - quartiles[0]
        else:
            base[f"{key}_iqr"] = max(vals) - min(vals)
    return base


def render_report(args, records, meta) -> str:
    """Markdown report for the paper: provenance, table, and the quotable points."""
    labels = {
        "f32": "Uncompressed HNSW (f32)",
        "tq8": "TurboQuant TQ8 ($H \\cdot D$)",
        "tq4": "TurboQuant TQ4 ($H \\cdot D$ + 2B scale)",
        "sq8": "Native SQ8 (no rotation)",
        "pq": "Product Quantization (PQ)",
    }
    env = meta["envelope"]
    ds = meta["dataset"]
    lines = [
        f"# BigANN {ds['name']} — BenoStreamDB quantizer ablation",
        "",
        f"Generated: `{meta['generated_utc']}`  ·  engine `{meta['engine']['version']}` "
        f"(`{meta['engine']['git_rev']}`, dirty={meta['engine']['git_dirty']})",
        "",
        "## Provenance",
        "",
        f"- Base vectors: `{ds['base']}` ({ds['n_vectors']:,} of {ds['dataset_vectors']:,} vectors, d={ds['dim']})",
        f"- Queries: `{ds['queries']}`",
        f"- Ground truth: "
        + (
            f"computed — {ds['gt_source']}"
            if str(ds["gt_source"]).startswith("exact")
            else f"`{ds['gt']}` ({ds['gt_source']})"
        ),
        f"- Envelope: hardware=`{env['hardware']}`, threads={env['threads']}, "
        f"BSDB_CACHE_GB={env['cache_gb']}, preload={env['preload_gb']:.1f} GiB",
        f"- Storage: `{env.get('storage', {}).get('device', 'unknown')}` "
        f"{env.get('storage', {}).get('model') or ''}, "
        f"PCIe {env.get('storage', {}).get('pcie_generation', '?')} "
        f"x{env.get('storage', {}).get('link_width', '?')} "
        f"({env.get('storage', {}).get('link_speed') or 'unknown'}), "
        f"rotational={env.get('storage', {}).get('rotational')}, "
        f"{env.get('storage', {}).get('fstype') or 'fs?'} on "
        f"`{env.get('storage', {}).get('source') or '?'}`"
        + (f", declared `{env['storage_note']}`" if env.get("storage_note") else ""),
        f"- Index build: M={meta['harness']['m']}, ef_construction={meta['harness']['ef_construction']}",
        f"- Reindex cost: {records[0].get('reindex_s', 'n/a')} s, "
        f"index {records[0].get('index_size_mb', 0) / 1024:.2f} GB, "
        f"data {records[0].get('data_size_mb', 0) / 1024:.2f} GB, "
        f"table {records[0].get('db_size_mb', 0) / 1024:.2f} GB",
        f"- Parquet data page size: {meta['harness'].get('parquet_page_bytes')} bytes"
        + (" (index reused, not rebuilt)" if meta["harness"].get("reused_index") else ""),
        f"- Repeats per operating point: {records[0].get('repeat', 1)}"
        + (" (median shown, interquartile range in the JSON)"
           if records[0].get("repeat", 1) > 1 else ""),
        "",
        "## Pareto curve",
        "",
        f"| Quantizer | ef_search | n_probe | Recall@{args.k} | QPS | p50 (ms) | p99 (ms) |",
        "| :--- | :---: | :---: | :---: | ---: | ---: | ---: |",
    ]
    for r in sorted(records, key=lambda x: (x["mode"], -x["pure_qps"])):
        recall = "n/a" if r["recall_at_k"] is None else f"**{r['recall_at_k']:.4f}**"
        lines.append(
            f"| {labels.get(r['mode'], r['mode'])} | {r['ef_search']} | {r['n_probe']} | "
            f"{recall} | {r['pure_qps']:,.1f} | {r['p50_ms']:.2f} | {r['p99_ms']:.2f} |"
        )

    lines += ["", "## Fastest point clearing each recall bar", ""]
    for bar in (0.90, 0.95, 0.98):
        rows = []
        for mode in sorted({r["mode"] for r in records}):
            pts = [
                r
                for r in records
                if r["mode"] == mode and r["recall_at_k"] is not None and r["recall_at_k"] >= bar
            ]
            if pts:
                best = max(pts, key=lambda r: r["pure_qps"])
                rows.append(
                    f"  - {labels.get(mode, mode)}: {best['pure_qps']:,.1f} QPS at "
                    f"ef_s={best['ef_search']}, n_probe={best['n_probe']} "
                    f"(recall {best['recall_at_k']:.4f})"
                )
        lines.append(f"**recall ≥ {bar:.2f}**")
        lines.extend(rows or ["  - no operating point reached this recall"])
        lines.append("")
    return "\n".join(lines) + "\n"


def save_results(args, records, meta) -> None:
    """Persist JSON (machine), Markdown (paper), and an append-only ledger."""
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    # Include the vector count: `--dataset 10m --limit 1000000` is a 1M subset of
    # the 10M corpus, and a file named "bigann_10m" would mislabel it.
    tag = f"bigann_{args.dataset}_{meta['dataset']['n_vectors'] // 1000}k_{stamp}"

    with open(os.path.join(out_dir, f"{tag}.json"), "w") as fh:
        json.dump({"meta": meta, "records": records}, fh, indent=2)
    with open(os.path.join(out_dir, f"{tag}.md"), "w") as fh:
        fh.write(render_report(args, records, meta))
    with open(os.path.join(out_dir, "bigann_ledger.jsonl"), "a") as fh:
        fh.write(json.dumps({"tag": tag, "meta": meta, "records": records}) + "\n")

    print(f"    -> wrote {tag}.json / {tag}.md and appended bigann_ledger.jsonl "
          f"to {out_dir}", flush=True)


# --------------------------------------------------------------------------- #
# Hardware envelope
# --------------------------------------------------------------------------- #

GIB = 1024**3

# Reference machines for leaderboard-comparable runs:
# Azure Standard_D8lds_v5 = 8 vCPU, 16 GiB RAM, local NVMe.
# Azure Standard_D8s_v5   = 8 vCPU, 32 GiB RAM, local NVMe.
# Azure Standard_E8ds_v5  = 8 vCPU, 64 GiB RAM, local NVMe.
HARDWARE_PROFILES = {
    "d8lds": {"label": "Azure Standard_D8lds_v5 (8 vCPU / 16 GiB)", "cpus": 8, "mem_gb": 16},
    "d8s":   {"label": "Azure Standard_D8s_v5 (8 vCPU / 32 GiB)",   "cpus": 8, "mem_gb": 32},
    "e8ds":  {"label": "Azure Standard_E8ds_v5 (8 vCPU / 64 GiB)",  "cpus": 8, "mem_gb": 64},
}


def _read_sys(path: str) -> Optional[str]:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def storage_envelope(path: str) -> dict:
    """Describe the block device that holds `path`, including its PCIe generation.

    Storage generation belongs in the measurement envelope rather than in a
    footnote. A Gen3 x4 link caps sequential reads near 3.5 GB/s, so an out of
    core result measured on one is not comparable to the same run on a Gen4
    device. Recording the measured link speed, width, and rotational flag makes
    that difference visible in the artifact instead of inferred from the host
    name.
    """
    info: dict = {"path": os.path.abspath(path)}
    try:
        dev = os.stat(path).st_dev
    except OSError:
        return info

    node = f"/sys/dev/block/{os.major(dev)}:{os.minor(dev)}"
    if not os.path.exists(node):
        return info

    # A partition resolves to .../block/<disk>/<partition>, so the parent
    # directory name is the whole device. Using the last component would leave
    # every sysfs lookup below pointing at a partition node that has no model,
    # no PCIe endpoint, and no rotational flag.
    link = os.path.realpath(node)
    disk = os.path.basename(link)
    # /sys/block only lists whole devices, so testing /sys/block/<partition>
    # always fails and leaves `disk` holding a partition name. The partition
    # attribute lives in the device class directory, which does list partitions.
    if os.path.exists(f"/sys/class/block/{disk}/partition"):
        disk = os.path.basename(os.path.dirname(link))
    info["device"] = disk

    info["model"] = _read_sys(f"/sys/block/{disk}/device/model")
    info["firmware"] = _read_sys(f"/sys/block/{disk}/device/firmware_rev")
    rotational = _read_sys(f"/sys/block/{disk}/queue/rotational")
    info["rotational"] = None if rotational is None else rotational == "1"

    # The PCIe endpoint sits at different depths by transport: an NVMe
    # namespace points at the controller, whose `device` link reaches the PCI
    # function, while SCSI and virtio devices expose it one level up. Probe the
    # candidates in order and take the first that reports a link speed.
    speed = None
    width = None
    for ctrl in (
        f"/sys/block/{disk}/device/device",
        f"/sys/block/{disk}/device",
    ):
        speed = _read_sys(os.path.join(ctrl, "current_link_speed"))
        if speed:
            width = _read_sys(os.path.join(ctrl, "current_link_width"))
            break
    info["link_speed"] = speed
    info["link_width"] = width
    if speed:
        rate = speed.split()[0].replace("GT/s", "")
        try:
            rate = f"{float(rate):.1f}"
        except ValueError:
            rate = ""
        info["pcie_generation"] = {
            "2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6,
        }.get(rate)

    best = ""
    try:
        with open("/proc/mounts") as fh:
            for line in fh:
                # source, mount point, fstype, options
                parts = line.split()
                if len(parts) >= 3 and info["path"].startswith(parts[1]) and len(parts[1]) > len(best):
                    best = parts[1]
                    info["source"] = parts[0]
                    info["fstype"] = parts[2]
                    info["mount"] = parts[1]
    except OSError:
        pass
    return info


def cgroup_cpu_limit() -> Optional[int]:
    """CPU limit from cgroup v2 (`cpu.max`) or v1 (`cpu.cfs_*`), when set."""
    raw = _read_sys("/sys/fs/cgroup/cpu.max")
    if raw:
        parts = raw.split()
        if len(parts) == 2 and parts[0] != "max":
            quota, period = int(parts[0]), int(parts[1])
            if quota > 0 and period > 0:
                return max(1, round(quota / period))
    quota = _read_sys("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_sys("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period and quota != "-1" and int(quota) > 0 and int(period) > 0:
        return max(1, round(int(quota) / int(period)))
    return None


def cgroup_memory_limit_bytes() -> Optional[int]:
    """Container memory limit, when the process is actually constrained."""
    raw = _read_sys("/sys/fs/cgroup/memory.max")
    if raw and raw != "max":
        return int(raw)
    raw = _read_sys("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if raw:
        value = int(raw)
        if value < (1 << 60):  # cgroup v1 reports a huge sentinel when unlimited
            return value
    return None


def apply_hardware(args) -> None:
    """Constrain threads, cache and preload budget to the evaluation envelope.

    The point is comparability: a 48 GiB cache budget on a 16 GiB machine is not
    a faster engine, it is an OOM kill (or a swapped run). Half the envelope goes
    to the index cache; the rest is left for decode buffers, Arrow batches and
    the Python runtime.
    """
    detected_cpus = cgroup_cpu_limit()
    detected_mem = cgroup_memory_limit_bytes()
    profile = HARDWARE_PROFILES.get(args.hardware)

    if args.hardware == "local":
        cpus = args.threads or (os.cpu_count() or 1)
        mem = psutil.virtual_memory().total
        source = "this host (no envelope)"
    elif profile:
        cpus, mem = profile["cpus"], profile["mem_gb"] * GIB
        source = profile["label"]
        if args.strict_hardware:
            problems = []
            if detected_cpus is None:
                problems.append("no cgroup CPU limit is set")
            elif detected_cpus > cpus:
                problems.append(f"cgroup CPU limit {detected_cpus} > profile {cpus}")
            if detected_mem is None:
                problems.append("no cgroup memory limit is set")
            elif detected_mem > mem * 1.01:
                problems.append(
                    f"cgroup memory limit {detected_mem / GIB:.1f} GiB > "
                    f"profile {mem / GIB:.0f} GiB"
                )
            if problems:
                raise SystemExit(
                    "[ERROR] --strict-hardware: " + "; ".join(problems)
                    + "\n        run under `docker compose -f benchmarks/docker-compose.neurips.yml`"
                )
        # Never exceed what the container actually allows.
        if detected_cpus:
            cpus = min(cpus, detected_cpus)
        if detected_mem:
            mem = min(mem, detected_mem)
    else:  # auto
        cpus = detected_cpus or (os.cpu_count() or 1)
        mem = detected_mem or psutil.virtual_memory().total
        source = "cgroup limits" if (detected_cpus or detected_mem) else "host (unconstrained)"

    # `--cache-gb` overrides the half-memory default. A smaller cache leaves
    # more of the envelope for the index-build working set, which is what
    # OOM-kills a 100M build at 16 GiB. Never let the cache claim the whole
    # envelope: the build, Arrow batches and the runtime need the rest.
    args.cache_gb = args.cache_gb or max(1, int(mem * 0.5 / GIB))
    args.cache_gb = min(args.cache_gb, max(1, int(mem / GIB) - 1))
    args.threads = max(1, min(args.threads or cpus, cpus))
    if not args.preload_gb:
        args.preload_gb = float(args.cache_gb)
    args.preload_gb = min(args.preload_gb, float(args.cache_gb))

    print("=" * 90)
    print(f"HARDWARE ENVELOPE  (profile hardware={args.hardware} -> {source})")
    if detected_cpus or detected_mem:
        mem_txt = "unlimit" if detected_mem is None else f"{detected_mem / GIB:.1f} GiB"
        print(f"  detected cgroup limits: cpus={detected_cpus} mem={mem_txt}")
    print(f"  cpus: {cpus}   memory: {mem / GIB:.1f} GiB")
    print(
        f"  query threads: {args.threads}   BSDB_CACHE_GB: {args.cache_gb}   "
        f"preload budget: {args.preload_gb:.1f} GiB"
    )
    if args.hardware == "auto" and (detected_cpus is None or detected_mem is None):
        print(
            "  [!] NOT COMPARABLE: no cgroup limits detected, so this run uses the\n"
            "      whole host. Leaderboard numbers need --profile d8lds, or the\n"
            "      container in benchmarks/docker-compose.neurips.yml."
        )
    print("=" * 90 + "\n", flush=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Profiles and operating point grids
# --------------------------------------------------------------------------- #

PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")

# The machine settings a profile may supply, with the value used when neither the
# profile nor the command line sets one.
PROFILE_SETTINGS = {
    "hardware": "auto",
    "strict_hardware": False,
    "threads": 0,
    "cache_gb": 0,
    "preload_gb": 0.0,
    "batch_size": 100_000,
    "flush_every": 5_000_000,
    "warmup_queries": 500,
    "scratch_dir": None,
    "tmp_dir": None,
    # Free text for the profile to declare the storage class it expects, for
    # example "PCIe 3.0 x4 NVMe". Measured values are recorded separately.
    "storage_note": None,
}


def load_profile(name: str) -> dict:
    """Read an envelope profile from benchmarks/profiles/<name>.toml.

    The profile holds the machine description: hardware label, thread count,
    cache budget, preload, flush cadence, warmup, and scratch location. Keeping
    it in a file is what lets the command line stay small, and it removes the
    failure mode where a run silently used the wrong cache budget because one of
    ten machine flags was mistyped.
    """
    path = name if os.path.sep in name or name.endswith(".toml") else os.path.join(PROFILES_DIR, f"{name}.toml")
    if not os.path.exists(path):
        available = sorted(f[:-5] for f in os.listdir(PROFILES_DIR) if f.endswith(".toml"))
        raise SystemExit(f"[ERROR] no profile at {path}. Available: {', '.join(available)}")
    import tomllib

    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    unknown = sorted(set(data) - set(PROFILE_SETTINGS) - {"label"})
    if unknown:
        raise SystemExit(f"[ERROR] {path}: unknown keys {', '.join(unknown)}")
    return data


def apply_profile(args) -> None:
    """Fill every unset machine setting from the profile, then from the defaults.

    A flag passed on the command line always wins, so a profile is a default
    rather than a lock.
    """
    profile = load_profile(args.profile) if getattr(args, "profile", None) else {}
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for dest, default in PROFILE_SETTINGS.items():
        if getattr(args, dest, None) is not None:
            continue
        # Inside a container the scratch volume is a mount path that no profile
        # can know, so the environment wins over the profile for these two.
        env_override = {
            "scratch_dir": "BSDB_BENCH_SCRATCH",
            "tmp_dir": "BSDB_BENCH_TMP",
        }.get(dest)
        if env_override and os.environ.get(env_override):
            setattr(args, dest, os.environ[env_override])
            continue
        if dest == "scratch_dir":
            value = profile.get(dest) or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "scratch_10m"
            )
        else:
            value = profile.get(dest, default)
        if dest in ("scratch_dir", "tmp_dir") and isinstance(value, str) and not os.path.isabs(value):
            value = os.path.join(repo_root, value)
        setattr(args, dest, value)
    args.profile_label = profile.get("label")


def parse_grid(spec: str) -> list:
    """Parse ``ef/np,ef/np,...`` into operating points."""
    points = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "/" not in chunk:
            raise SystemExit(f"[ERROR] --grid entry {chunk!r} is not ef/np")
        ef, np_ = chunk.split("/", 1)
        points.append((int(ef), int(np_)))
    if not points:
        raise SystemExit("[ERROR] --grid is empty")
    return points


def main():
    parser = argparse.ArgumentParser(
        description="NeurIPS BigANN benchmark for BenoStreamDB",
        epilog="Examples:  run_neurips_bigann.py ingest --dataset 10m --profile d8lds | "
               "run_neurips_bigann.py query --dataset 10m --profile d8lds --modes tq8,sq8 "
               "--grid 256/32,64/4 --repeat 3",
    )
    parser.add_argument("command", nargs="?", choices=["ingest", "query", "compact"], default=None,
                        help="ingest writes the table, query measures it, compact builds the "
                             "table level Puffin artifact. Omit to run ingest and query in one process.")
    parser.add_argument("--profile", type=str, default=None,
                        help="Envelope profile from benchmarks/profiles (d8lds, d8s, e8ds, local). "
                             "Supplies hardware, threads, cache, preload, flush cadence, warmup, "
                             "and scratch location.")
    parser.add_argument("--grid", type=str, default=None,
                        help="Operating points as ef/np pairs, e.g. 256/32,64/4.")
    parser.add_argument("--reindex", action="store_true",
                        help="Rebuild the index for each mode before measuring, instead of "
                             "reusing the index already on disk.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="10m",
                        help="NeurIPS BigANN dataset to run (selects base + queries + GT)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Vectors to index; 0 (default) = the whole selected dataset")
    parser.add_argument("--queries", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--modes", type=str, default=None,
                        help="Comma-separated list of modes to run (e.g. 'sq8,pq')")
    parser.add_argument("--db-dir", type=str, default=None,
                        help="Persistent DB directory (default: <scratch-dir>/bigann_db)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--repeat", type=int, default=1,
                        help="Measure every operating point N times and report the median, "
                             "with the interquartile range kept in the JSON (default: 1)")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Permit a result from a modified working tree. For throwaway runs "
                             "only: a dirty-tree result cannot be reproduced from a commit.")
    parser.add_argument("--refine-factor", type=int, default=None,
                        help="Two-stage exact F32 rescore factor (e.g. 4 or 5)")
    parser.add_argument("--compaction-k", type=int, default=1024,
                        help="Global cluster count for --index-compaction (default: 1024)")
    parser.add_argument("--compaction-m", type=int, default=16,
                        help="HNSW graph degree for --index-compaction (default: 16)")
    parser.add_argument("--compaction-ef-construction", type=int, default=200,
                        help="HNSW construction beam for --index-compaction (default: 200)")
    parser.add_argument("--dim", type=int, default=None,
                        help="Vector dimension (overrides dataset default)")
    parser.add_argument("--metric", choices=["l2", "ip", "inner_product", "cosine"], default=None,
                        help="Distance metric (overrides dataset default)")
    parser.add_argument("--out", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "results", "neurips_bigann"),
                        help="Directory for the JSON/Markdown results and the ledger")
    args = parser.parse_args()

    # Derived state. The command line no longer carries these: the subcommand
    # decides the stage, --grid decides the operating points, --modes decides the
    # quantizers, and the profile decides the machine. The attributes below keep
    # the rest of the harness readable without reintroducing flags for them.
    args.stage = {"ingest": "ingest", "query": "query", "compact": "query"}.get(
        args.command, "both"
    )
    args.force_reindex = bool(args.reindex)
    args.skip_reindex = args.command in ("query", "compact") and not args.reindex
    args.rebuild_db = args.command == "ingest"
    args.index_compaction = args.command == "compact"
    args.sweep = False
    args.ef_search = 256
    args.n_probe = 32
    args.ef_sweep = "32,64,128,256"
    args.n_probe_sweep = "4,8,16,32"
    args.all_modes = False
    args.mode = None
    args.auto_warmup = False
    args.cleanup = False
    args.base = None
    args.query = None
    args.gt = None
    args.query_set = "public"
    args.compaction_target_shard_bytes = None

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    apply_profile(args)

    # Provenance gate. A result produced from a modified tree cannot be
    # reproduced from any commit, which is the defect that made the previous
    # artifact set unverifiable.
    state = git_state()
    if state["git_dirty"] and not args.allow_dirty:
        print("[ERROR] the working tree has modified files, so this result cannot be "
              "reproduced from a commit:")
        for line in _git(["status", "--porcelain"]).splitlines()[:10]:
            print(f"    {line}")
        print("        Commit the changes, or pass --allow-dirty for a throwaway run.")
        sys.exit(3)

    # Constrain threads/caches to the evaluation envelope before importing the
    # engine: BSDB_CACHE_GB is read once, when the caches are first built.
    apply_hardware(args)

    # The table is local, so a second on-disk copy of every index is pure waste.
    os.environ.setdefault("BSDB_DISK_CACHE", "off")
    os.environ.setdefault("BSDB_DISK_CACHE_GB", "0")
    os.environ["BSDB_CACHE_GB"] = str(args.cache_gb)

    os.makedirs(args.scratch_dir, exist_ok=True)

    # Puffin-packed indexes are materialized into `std::env::temp_dir()` before
    # loading, so TMPDIR must exist and must be on the fast disk (the dataset dir
    # is on the 14 TB HDD). A missing TMPDIR makes the preload fail with
    # "No such file or directory", after which every query falls back to a full
    # scan — 1.2 s/query and recall of exactly 1.0.
    tmpdir = args.tmp_dir or os.path.join(args.scratch_dir, "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir
    args.tmp_dir = tmpdir
    args.db_dir = args.db_dir or os.path.join(args.scratch_dir, "bigann_db")

    # Storage generation is part of the envelope. A Gen3 link bounds sequential
    # read throughput, so the artifact records the device and its measured link
    # rather than leaving a reader to infer it from the machine name.
    args.storage = storage_envelope(args.scratch_dir)

    base_path, q_path, gt_path, n, tag, dataset_count, qfile_count, dim, dtype, metric = resolve_dataset(args)
    args.n = n
    args.dim = dim
    args.dtype = dtype
    args.metric = metric
    args.query_path = q_path
    args.queries = min(args.queries, qfile_count)

    if args.modes:
        modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    else:
        modes = ["tq8"]

    valid_modes = {"f32", "tq8", "tq4", "sq8", "pq"}
    unknown_modes = sorted(set(modes) - valid_modes)
    if unknown_modes:
        parser.error(
            f"unknown quantizer {', '.join(unknown_modes)}; "
            f"valid: {', '.join(sorted(valid_modes))}"
        )

    if args.grid:
        grid = parse_grid(args.grid)
    elif args.sweep:
        efs = [int(v) for v in args.ef_sweep.split(",") if v.strip()]
        nps = [int(v) for v in args.n_probe_sweep.split(",") if v.strip()]
        grid = [(ef, npr) for ef in efs for npr in nps]
    else:
        grid = [(args.ef_search, args.n_probe)]

    print("=" * 90)
    print("NEURIPS BIGANN BENCHMARK HARNESS - BENOSTREAMDB")
    print(f"Vectors: {n:,} | Queries: {args.queries:,} | Top-k: {args.k} | M: {args.m} "
          f"| ef_c: {args.ef_construction}")
    print(f"Operating points: {len(grid)} "
          f"({'x'.join(f'{ef}/{npr}' for ef, npr in grid) if args.sweep else f'{grid[0][0]}/{grid[0][1]}'})")
    print(f"Modes: {[m.upper() for m in modes]} | stage: {args.stage}")
    print(f"Raw data:  {base_path}")
    print(f"Table DB:  {args.db_dir}  (ingest once, reindex per mode)")
    _st = args.storage
    print(
        f"Storage:   {_st.get('device', 'unknown')} {_st.get('model') or ''} | "
        f"PCIe {_st.get('pcie_generation', '?')} x{_st.get('link_width', '?')} "
        f"({_st.get('link_speed') or 'link speed unknown'}) | "
        f"rotational={_st.get('rotational')} | "
        f"{_st.get('fstype') or 'fs?'} on {_st.get('source') or '?'} "
        f"at {_st.get('mount', '?')}"
        + (f" | declared: {args.storage_note}" if getattr(args, "storage_note", None) else "")
    )
    print("=" * 90 + "\n", flush=True)

    # Validate the inputs before doing any work: an incomplete dataset is the
    # most common way to start a run that dies an hour later.
    missing = []
    if not os.path.exists(base_path):
        missing.append((base_path, "required"))
    if not os.path.exists(q_path):
        missing.append((q_path, "required"))
    if gt_path and not os.path.exists(gt_path):
        missing.append((gt_path, "optional — recall reported as null without it"))
    if missing:
        print(f"[ERROR] {args.dataset} dataset is incomplete in {args.data_dir}:")
        for path, role in missing:
            print(f"    missing  {os.path.basename(path):<24} ({role})")
        print(
            "\nDownload the NeurIPS BigANN tarball for this size and unpack it into\n"
            f"  {args.data_dir}\n"
            "expected files: base.<N>.u8bin, query.public.10K.u8bin, GT.<N>\n"
            "Then:  ingest --dataset <n> --profile d8lds   followed by   query --grid 256/32,64/4"
        )
        sys.exit(2)

    if args.stage == "query" and not os.path.exists(os.path.join(args.db_dir, "_manifest")):
        print(
            f"[ERROR] no table at {args.db_dir}; the query command reuses an existing ingest.\n"
            f"        Run:  ingest --dataset {args.dataset} --profile d8lds"
        )
        sys.exit(2)

    import benostreamdb as bsdb

    records = []

    need_ingest = args.stage in ("both", "ingest") and (
        args.rebuild_db or not os.path.exists(os.path.join(args.db_dir, "_manifest"))
    )
    if need_ingest:
        print("--- Stage: INGEST (write the table once, no index) ---", flush=True)
        ingest_stats = ingest_stage(bsdb, args, base_path, dim, dtype)
        print(json.dumps(ingest_stats), flush=True)
        print(
            f"    -> Ingested {ingest_stats['n_train']:,} vectors in "
            f"{ingest_stats['ingest_time_s']}s "
            f"({ingest_stats['ingest_rate_vps']:,.0f} vec/s), "
            f"DB {ingest_stats['data_size_mb']/1024:.2f} GB\n",
            flush=True,
        )
    elif args.stage in ("both", "query"):
        print(f"--- Reusing existing table at {args.db_dir} ---\n", flush=True)

    if args.stage in ("both", "query"):
        test_queries, q = load_queries(q_path, args.queries, dim, dtype)
        gt_indices = load_ground_truth(gt_path, n, dataset_count, q, args.k)
        gt_source = "official file" if gt_indices is not None else None
        if gt_indices is None and gt_path and os.path.exists(gt_path):
            print(f"    [!] {os.path.basename(gt_path)} does not match --limit {n:,}; "
                  "ignoring it", flush=True)
        if gt_indices is None and n <= 2_000_000:
            print("    -> Computing exact ground truth for the subset ...", flush=True)
            gt_indices = exact_ground_truth(base_path, test_queries, n, args.k, dim, dtype, metric)
            gt_source = f"exact brute force over {n:,} vectors"
        if gt_indices is None:
            gt_source = "unavailable"
            print("    [!] No ground truth available; recall will be reported as null.\n",
                  flush=True)

        for mode in modes:
            print(f"--- Stage: QUERY / {mode.upper()} ---", flush=True)

            # Reuse is only safe when the index on disk is the index this run
            # claims to measure. Without this check a `--skip-reindex` run can
            # report TQ8 while querying whatever the last build left behind.
            if args.skip_reindex and not args.force_reindex:
                manifest = read_index_mode(args.db_dir)
                mismatch = []
                if not manifest:
                    mismatch.append(f"no index manifest at {index_mode_marker(args.db_dir)}")
                else:
                    for key, want in (("mode", mode), ("complexity", args.m),
                                      ("quality", args.ef_construction),
                                      ("dim", dim), ("n_vectors", n)):
                        got = manifest.get(key)
                        if got is not None and got != want:
                            mismatch.append(f"{key}={got!r} on disk, {want!r} requested")
                if mismatch:
                    print("[ERROR] refusing to reuse an index that does not match this run:")
                    for problem in mismatch:
                        print(f"    {problem}")
                    print("        Pass --force-reindex to rebuild, or drop --skip-reindex.")
                    sys.exit(4)

            t0 = time.time()
            table, meta = reindex_and_preload(bsdb, args, mode)

            # Untimed warmup pass to prefault active Voronoi index shards into RAM
            nw = min(getattr(args, "warmup_queries", 500), len(test_queries))
            if nw > 0:
                if hasattr(table, "warmup"):
                    print(f"    -> Warming index cache via native Tier C centroid probing ({nw} probes) ...", flush=True)
                    t_warm_start = time.time()
                    wstats = table.warmup(n_probes=nw, max_memory_gb=args.preload_gb)
                    print(f"    -> Warmed in {time.time() - t_warm_start:.2f}s: {wstats}", flush=True)
                else:
                    print(f"    -> Warming index cache with {nw} untimed query probes ...", flush=True)
                    t_warm_start = time.time()
                    warm_subset = [test_queries[i].tolist() for i in range(nw)]
                    with ThreadPoolExecutor(max_workers=args.threads) as executor:
                        list(executor.map(
                            lambda q_vec: table.to_arrow(
                                vector_filter={"column": "embedding", "query": q_vec, "k": args.k, "ef_search": 64},
                                columns=["id"]
                            ),
                            warm_subset
                        ))
                    print(f"    -> Warmed in {time.time() - t_warm_start:.2f}s", flush=True)

            for ef, npr in grid:
                runs = [
                    search_pass(table, args, test_queries, gt_indices, ef, npr, mode=mode)
                    for _ in range(args.repeat)
                ]
                rec = aggregate_repeats(runs)
                rec.update(
                    {
                        "dataset": f"{args.dataset}-{n}",
                        "mode": mode,
                        "n_train": n,
                        "dim": dim,
                        "metric": metric,
                        "hardware": args.hardware,
                        "reindex_s": meta["reindex_s"],
                        "compaction_s": meta.get("compaction_s", 0.0),
                        "preload_ms": meta["preload_ms"],
                        "index_size_mb": meta["index_size_mb"],
                        "data_size_mb": meta["data_size_mb"],
                        "db_size_mb": meta["db_size_mb"],
                    }
                )
                records.append(rec)
                print(json.dumps(rec), flush=True)
                r = rec["recall_at_k"]
                print(
                    f"    ef_s={ef:<4} n_probe={npr:<3} -> "
                    f"recall@{args.k}={'n/a' if r is None else f'{r:.4f}'}  "
                    f"{rec['pure_qps']:>8,.1f} QPS  "
                    f"p50 {rec['p50_ms']:>7.2f} ms  p99 {rec['p99_ms']:>8.2f} ms",
                    flush=True,
                )
            clear_evaluation_caches(table, [meta.get("db_path", args.db_dir), getattr(args, "scratch_dir", None)], label=f"post-{mode.upper()}")
            del table
            print(f"--- {mode.upper()} done in {time.time() - t0:.1f}s "
                  f"(reindex {meta['reindex_s']}s, index {meta['index_size_mb']/1024:.2f} GB)\n",
                  flush=True)

    if records:
        meta = {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "engine": {
                "version": getattr(bsdb, "__version__", "unknown"),
                "git_rev": state["git_rev"],
                "git_rev_short": state["git_rev_short"],
                "git_dirty": state["git_dirty"],
                "git_dirty_paths": state["git_dirty_paths"],
            },
            "dataset": {
                "name": f"{args.dataset}",
                "base": base_path,
                "queries": q_path,
                "gt": gt_path,
                "gt_source": gt_source,
                "n_vectors": n,
                "dataset_vectors": dataset_count,
                "dim": dim,
                "metric": metric,
            },
            "harness": {
                "script": os.path.basename(__file__),
                # v2: index_size_mb is the Puffin sidecar only. Runs before
                # 2026-10-31 recorded the whole table directory under that key.
                "metric_version": 2,
                "stage": args.stage,
                "all_modes": args.all_modes,
                "m": args.m,
                "ef_construction": args.ef_construction,
                "flush_every": args.flush_every,
                "k": args.k,
                "queries": args.queries,
                "repeat": args.repeat,
                "parquet_page_bytes": parquet_page_bytes(),
                "reused_index": bool(args.skip_reindex),
                "allow_dirty": bool(args.allow_dirty),
            },
            "index_compaction": {
                "enabled": bool(getattr(args, "index_compaction", False)),
                "k": args.compaction_k,
                "m": args.compaction_m,
                "ef_construction": args.compaction_ef_construction,
            },
            "envelope": {
                "hardware": args.hardware,
                "threads": args.threads,
                "cache_gb": args.cache_gb,
                "preload_gb": args.preload_gb,
                "storage": args.storage,
                "storage_note": getattr(args, "storage_note", None),
            },
            "grid": [{"ef_search": ef, "n_probe": npr} for ef, npr in grid],
        }
        save_results(args, records, meta)

    if args.cleanup and os.path.exists(args.db_dir):
        print(f"Removing DB at {args.db_dir}", flush=True)
        shutil.rmtree(args.db_dir)

    if records:
        labels = {
            "f32": "Uncompressed HNSW (f32)",
            "sq8": "Native SQ8 (no rotation)",
            "tq8": "TurboQuant TQ8 ($H \\cdot D$)",
            "tq4": "TurboQuant TQ4 ($H \\cdot D$ + 2B scale)",
            "pq": "Product Quantization (PQ)",
        }
        md = [
            "| Scale | Quantizer | ef_search | n_probe | Recall@10 | QPS | p50 (ms) | p99 (ms) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in sorted(records, key=lambda x: (x["mode"], -x["pure_qps"])):
            recall = "n/a" if r["recall_at_k"] is None else f"**{r['recall_at_k']:.4f}**"
            md.append(
                f"| {r['n_train']:,} | {labels.get(r['mode'], r['mode'])} | "
                f"{r['ef_search']} | {r['n_probe']} | {recall} | "
                f"{r['pure_qps']:,.1f} | {r['p50_ms']:.2f} | {r['p99_ms']:.2f} |"
            )

        print("\n" + "=" * 90)
        print(f"FINAL BIGANN RESULTS  (threads={args.threads}, "
              f"BSDB_CACHE_GB={args.cache_gb}, hardware={args.hardware})")
        print("=" * 90)
        print("\n".join(md), flush=True)

        # The leaderboard reports recall at ~0.90-0.99; surface the fastest point
        # that clears a high-recall bar so the paper can quote a comparable row.
        for bar in (0.90, 0.95, 0.98):
            for mode in modes:
                pts = [
                    r for r in records
                    if r["mode"] == mode and r["recall_at_k"] is not None
                    and r["recall_at_k"] >= bar
                ]
                if pts:
                    best = max(pts, key=lambda r: r["pure_qps"])
                    print(
                        f"  recall>={bar:.2f} [{mode.upper()}]: "
                        f"{best['pure_qps']:,.1f} QPS at "
                        f"ef_s={best['ef_search']}, n_probe={best['n_probe']} "
                        f"(recall {best['recall_at_k']:.4f})",
                        flush=True,
                    )
                    break


if __name__ == "__main__":
    main()
