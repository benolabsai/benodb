#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Competitor runner for the BenoDB benchmark plan.

Runs a competitor under the same resource envelope and parameter set as the
BenoDB harness (`benchmarks/ann_benchmarks/run.py`) and emits the same
metrics, so results are directly comparable.

Workload families:

* ``vector`` — ANN over an ANN-Benchmarks HDF5 dataset.
  Engines: ``faiss``, ``hnswlib`` (embedded), ``pgvector``, ``lancedb``,
  ``opensearch`` (server-backed).
* ``graph``  — a graph algorithm over an edge list.
  Engines: ``networkx`` (reference), ``neo4j`` (GDS).
* ``sql``    — an analytical query. Engines: ``duckdb``.

Adapters degrade gracefully: if the client library is not installed, or a
server is unreachable, the runner reports it rather than crashing.

Usage:
    python benchmarks/competitors/run_competitor.py --engine faiss \\
        --dataset sift-128-euclidean --cores 8 --ram-gb 16 \\
        --m 16 --ef-construction 200 --ef-search 200 --queries 500

    python benchmarks/competitors/run_competitor.py --engine pgvector \\
        --dsn "postgresql://user:pass@localhost/bench" --dataset glove-100-angular

    python benchmarks/competitors/run_competitor.py --engine opensearch \\
        --host http://localhost:9200 --dataset sift-128-euclidean

    python benchmarks/competitors/run_competitor.py --engine networkx \\
        --graph-edges edges.txt --algorithm pagerank

    python benchmarks/competitors/run_competitor.py --engine duckdb \\
        --sql "SELECT count(*) FROM read_parquet('hits.parquet')"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import os
import platform
import shutil
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

ANN_BASE = "http://ann-benchmarks.com"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DATASET_METRIC = {
    "sift-128-euclidean": "l2",
    "gist-960-euclidean": "l2",
    "fashion-mnist-784-euclidean": "l2",
    "mnist-784-euclidean": "l2",
    "glove-100-angular": "cosine",
    "glove-200-angular": "cosine",
    "nytimes-256-angular": "cosine",
    "lastfm-64-dot": "inner_product",
    "bigann-10m": "l2",
    "bigann-10M": "l2",
    "bigann": "l2",
    "bigann-100m": "l2",
    "bigann-100M": "l2",
}


# --------------------------------------------------------------------------- #
# Resource envelope
# --------------------------------------------------------------------------- #
# Engines whose runtime reserves a large *virtual* address space (CUDA/RAPIDS
# managed memory, wgpu). `RLIMIT_AS` limits virtual address space, not resident
# memory, so applying the RAM envelope to these engines makes the `dlopen` of
# their libraries fail with a misleading "cannot open shared object file"
# (the `mmap` returns ENOMEM). The container cgroup
# (`deploy.resources.limits.memory`) already enforces the physical RAM envelope,
# so `RLIMIT_AS` is both redundant and wrong for GPU engines.
_GPU_ENGINES = {"cugraph"}


def apply_envelope(
    cores: int, ram_gb: float, engine: str = "", device: str = "cpu"
) -> None:
    """Best-effort process limits so the competitor matches BenoDB.

    Threads: set the common BLAS/OMP env vars (must be set before numpy/faiss
    import to take effect; the runner sets them as early as possible). RAM: use
    ``RLIMIT_AS`` when ``ram_gb`` is given, so an over-budget run fails loudly
    instead of silently using more memory than BenoDB was given — except
    for GPU engines, where virtual address space far exceeds physical RAM (see
    ``_GPU_ENGINES``).
    """
    if cores > 0:
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[var] = str(cores)
    if ram_gb > 0 and engine not in _GPU_ENGINES and device != "gpu":
        try:
            import resource

            limit = int(ram_gb * 1024**3)
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        except Exception as exc:  # pragma: no cover - platform dependent
            print(f"warning: could not set RLIMIT_AS: {exc}", flush=True)


# --------------------------------------------------------------------------- #
# Vector adapters
# --------------------------------------------------------------------------- #
@dataclass
class VectorAdapter:
    """Build a vector index and answer top-k queries.

    ``search`` receives ``ef_search`` and ``metric`` explicitly (not via hidden
    attribute state) so every adapter is driven identically.
    """

    name: str
    module: str
    server_side: bool = field(default=False)
    _impl: Any = field(default=None, repr=False)

    def available(self) -> bool:
        try:
            importlib.import_module(self.module)
            return True
        except Exception:
            return False

    def build(self, train: np.ndarray, params: dict, ctx: dict):  # -> handle
        raise NotImplementedError

    def search(
        self, handle, queries: np.ndarray, k: int, ef_search: int, metric: str
    ) -> np.ndarray:
        raise NotImplementedError

    def search_full_row(
        self, handle, queries: np.ndarray, k: int, ef_search: int, metric: str
    ) -> Any:
        """Execute ANN search AND retrieve the full row payload.

        The return type is `Any` on purpose: the payload is engine-specific. The
        embedded engines return an ``ndarray`` of ids, while the server engines
        (Milvus, Weaviate, Qdrant) return the stored payload alongside each hit as a
        list of objects. Annotating this ``-> np.ndarray`` made every server-side
        override an incompatible override.
        """
        return self.search(handle, queries, k, ef_search, metric)

    def index_bytes(self, handle) -> int:
        return 0


def _vec_literal(v: np.ndarray) -> str:
    """pgvector text literal: ``[a,b,c]``."""
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total


class FaissAdapter(VectorAdapter):
    def __init__(self, index_type: str = "flat") -> None:
        name = {
            "hnsw_sq": "faiss_sq",
            "sq": "faiss_sq",
            "ivf_sq8": "faiss_ivfsq8",
            "ivfpq": "faiss_ivfpq",
        }.get(index_type, "faiss")
        super().__init__(name=name, module="faiss")
        self.index_type = index_type
        self.index_precision = {
            "hnsw_sq": "sq",
            "sq": "sq",
            "ivf_sq8": "sq",
            "ivfpq": "pq",
        }.get(index_type, "f32")
        self._train: Optional[np.ndarray] = None
        self._ivf = None
        self._nprobe = 0
        self._refine = False
        self._base_u8: Optional[np.ndarray] = None

    def build(self, train, params, ctx):
        import faiss

        dim = train.shape[1]
        metric = params.get("metric", "l2")
        faiss_metric = (
            faiss.METRIC_INNER_PRODUCT
            if metric in ("inner_product", "cosine")
            else faiss.METRIC_L2
        )
        data = np.ascontiguousarray(train)
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if metric == "cosine":
            faiss.normalize_L2(data)

        if self.index_type in ("hnsw_sq", "sq"):
            # faiss' runtime signature is IndexHNSWSQ(d, qtype, M, metric); the stub
            # types the 4th positional as a ScalarQuantizer, which is the 2nd arg.
            index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, params["m"], faiss_metric)  # type: ignore[arg-type]
            index.hnsw.efConstruction = params["ef_construction"]
            index.train(data)
            index.add(data)
        elif self.index_type in ("ivf_sq8", "ivfpq"):
            nlist = int(params.get("nlist", 1024) or 1024)
            self._nprobe = int(params.get("nprobe", 0) or 0)
            quantizer = (
                faiss.IndexFlatIP(dim)
                if faiss_metric == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(dim)
            )
            if self.index_type == "ivf_sq8":
                index = faiss.IndexIVFScalarQuantizer(
                    quantizer, dim, nlist, faiss.ScalarQuantizer.QT_8bit, faiss_metric
                )
            else:
                pq_m = int(params.get("pq_m", 16) or 16)
                pq_nbits = int(params.get("pq_nbits", 8) or 8)
                index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, pq_nbits, faiss_metric)
            index.train(data)
            if params.get("mmap_invlists"):
                base = ctx.get("path") or tempfile.mkdtemp(prefix="faiss_ivf_")
                os.makedirs(base, exist_ok=True)
                inv = faiss.OnDiskInvertedLists(
                    index.nlist, index.code_size, os.path.join(base, "invlists.dat")
                )
                index.replace_invlists(inv)
            index.add(data)
            self._ivf = index
            index.nprobe = (
                self._nprobe
                if self._nprobe > 0
                else max(1, int(params.get("ef_search", 16) or 16) // 10)
            )
            if self.index_type == "ivfpq" and params.get("refine", False):
                # Refine manually against the source vectors instead of wrapping
                # faiss.IndexRefineFlat. IndexRefineFlat copies the full float32
                # base into its internal IndexFlat, doubling the resident set; on
                # BigANN 10M that is 5.12 GiB of float32 for the upcast plus a
                # second 5.12 GiB copy, which OOMs the 16 GiB cgroup. BigANN is
                # uint8, so exact L2 refinement against the 1.28 GiB uint8 source
                # is both lossless and four times smaller.
                self._refine = True
                self._base_u8 = train if train.dtype in (np.uint8, np.int8) else data
        else:
            index = faiss.IndexHNSWFlat(dim, params["m"], faiss_metric)
            index.hnsw.efConstruction = params["ef_construction"]
            index.add(data)

        self._train = None if self._refine else data
        self._used_gpu = False
        if ctx.get("device") == "gpu":
            # Requires faiss-gpu; falls back to the CPU index when unavailable.
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                self._used_gpu = True
            except Exception as exc:
                print(f"faiss: GPU unavailable, using CPU index ({exc})", flush=True)
        if self._refine:
            return (index, self._base_u8)
        return (index, data)

    def search(self, handle, queries, k, ef_search, metric):
        import faiss

        index = handle[0] if isinstance(handle, tuple) else handle
        q = np.ascontiguousarray(queries)
        if q.dtype != np.float32:
            q = q.astype(np.float32)
        if metric == "cosine":
            faiss.normalize_L2(q)
        if self.index_type in ("ivf_sq8", "ivfpq"):
            nlist = int(getattr(self._ivf, "nlist", 0) or 0)
            nprobe = self._nprobe if self._nprobe > 0 else max(1, int(ef_search) // 10)
            nprobe = min(nlist, nprobe) if nlist else nprobe
            self._ivf.nprobe = nprobe
            if self._refine:
                k_factor = int(os.environ.get("REFINE_FACTOR", "4") or 4)
                _, cand = self._ivf.search(q, max(k, k * k_factor))
                ids = self._refine_topk(q, cand, k, metric)
            else:
                _, ids = index.search(q, k)
        else:
            index.hnsw.efSearch = ef_search
            _, ids = index.search(q, k)
        return ids

    def search_full_row(self, handle, queries, k, ef_search, metric):
        index, data = handle if isinstance(handle, tuple) else (handle, self._train)
        ids = self.search(index, queries, k, ef_search, metric)
        if data is not None and len(ids) > 0:
            return data[ids[0]]
        return ids

    def _refine_topk(self, q, cand_ids, k, metric):
        """Re-score IVF-PQ candidates exactly against the source vectors."""
        base = self._base_u8 if self._base_u8 is not None else self._train
        cids = np.asarray(cand_ids[0], dtype=np.int64)
        cids = cids[cids >= 0]
        if len(cids) == 0 or base is None:
            return np.asarray(cand_ids, dtype=np.int64)[:, :k]
        vecs = base[cids]
        if vecs.dtype != np.float32:
            vecs = vecs.astype(np.float32)
        qv = q[0]
        if qv.dtype != np.float32:
            qv = qv.astype(np.float32)
        if metric == "l2":
            d = ((vecs - qv) ** 2).sum(axis=1)
        else:
            d = -(vecs @ qv)
        order = np.argsort(d)[:k]
        return cids[order][None, :]

    def index_bytes(self, handle):
        import faiss
        index = handle[0] if isinstance(handle, tuple) else handle
        try:
            return int(len(faiss.serialize_index(index)))
        except Exception:
            try:
                return int(index.ntotal) * int(index.d) * 4
            except Exception:
                return 0


class HnswlibAdapter(VectorAdapter):
    def __init__(self) -> None:
        super().__init__(name="hnswlib", module="hnswlib")

    def build(self, train, params, ctx):
        import hnswlib

        # hnswlib supports l2/cosine/ip; map the workload metric directly so a
        # dot-product dataset is not silently built and queried in L2 space.
        space = {
            "cosine": "cosine",
            "inner_product": "ip",
        }.get(params.get("metric") or "l2", "l2")
        # hnswlib accepts the space string at runtime; the stub narrows it to a Literal.
        index = hnswlib.Index(space=space, dim=train.shape[1])  # type: ignore[arg-type]
        index.init_index(
            max_elements=train.shape[0],
            ef_construction=params["ef_construction"],
            M=params["m"],
        )
        index.add_items(train, np.arange(train.shape[0]))
        return index

    def search(self, handle, queries, k, ef_search, metric):
        handle.set_ef(ef_search)
        labels, _ = handle.knn_query(queries, k=k)
        return labels

    def search_full_row(self, handle, queries, k, ef_search, metric):
        handle.set_ef(ef_search)
        labels, _ = handle.knn_query(queries, k=k)
        if len(labels) > 0:
            return handle.get_items(labels[0])
        return labels

    def index_bytes(self, handle):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".hnsw")
        tmp.close()
        try:
            handle.save_index(tmp.name)
            return int(os.path.getsize(tmp.name))
        except Exception:
            return 0
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


class PgvectorAdapter(VectorAdapter):
    def __init__(self) -> None:
        super().__init__(name="pgvector", module="psycopg", server_side=True)

    def build(self, train, params, ctx):
        import psycopg

        dsn = ctx.get("dsn") or os.environ.get("PGVECTOR_DSN")
        if not dsn:
            raise RuntimeError("pgvector requires --dsn or PGVECTOR_DSN")
        conn = psycopg.connect(dsn, autocommit=True)
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("DROP TABLE IF EXISTS bdb_bench")
        dim = train.shape[1]
        cur.execute(
            f"CREATE TABLE bdb_bench (id int primary key, embedding vector({dim}))"
        )
        metric = params.get("metric", "l2")
        opclass = (
            "vector_ip_ops"
            if metric == "inner_product"
            else ("vector_cosine_ops" if metric == "cosine" else "vector_l2_ops")
        )
        with cur.copy("COPY bdb_bench (id, embedding) FROM STDIN") as copy:
            for i, row in enumerate(train):
                copy.write_row((int(i), _vec_literal(row)))
        # HNSW build is memory-bound; the Postgres default (64MB) throttles it.
        cur.execute("SET maintenance_work_mem = '2GB'")
        # DDL cannot take bind parameters, so inline the (validated) integers.
        cur.execute(
            f"CREATE INDEX ON bdb_bench USING hnsw (embedding {opclass}) "
            f"WITH (m = {int(params['m'])}, ef_construction = {int(params['ef_construction'])})"
        )
        return conn

    def search(self, handle, queries, k, ef_search, metric):
        # pgvector operators: `<->` is L2, `<=>` is cosine, `<#>` is (negative)
        # inner product. The operator MUST match the index opclass, or Postgres
        # cannot use the HNSW index and silently falls back to a sequential scan.
        op = {
            "l2": "<->",
            "cosine": "<=>",
            "inner_product": "<#>",
        }.get(metric, "<->")
        cur = handle.cursor()
        cur.execute(f"SET hnsw.ef_search = {int(ef_search)}")
        out = []
        for q in queries:
            cur.execute(
                f"SELECT id FROM bdb_bench ORDER BY embedding {op} %s::vector LIMIT %s",
                (_vec_literal(q), int(k)),
            )
            out.append([r[0] for r in cur.fetchall()])
        return np.array(out)

    def search_full_row(self, handle, queries, k, ef_search, metric):
        op = {
            "l2": "<->",
            "cosine": "<=>",
            "inner_product": "<#>",
        }.get(metric, "<->")
        cur = handle.cursor()
        cur.execute(f"SET hnsw.ef_search = {int(ef_search)}")
        out = []
        for q in queries:
            cur.execute(
                f"SELECT id, embedding FROM bdb_bench ORDER BY embedding {op} %s::vector LIMIT %s",
                (_vec_literal(q), int(k)),
            )
            out.append(cur.fetchall())
        return out

    def index_bytes(self, handle):
        cur = handle.cursor()
        cur.execute("SELECT pg_total_relation_size('bdb_bench')")
        return int(cur.fetchone()[0])


class DiskANNAdapter(VectorAdapter):
    """Microsoft DiskANN disk-backed Vamana index.

    Implements out-of-core graph traversal over SSDs using Microsoft's
    diskannpy bindings. Builds a compressed disk layout with in-memory centroids
    and PQ-quantized routing codes, paired with disk-resident neighborhood lists.
    """

    def __init__(self, index_type: str = "disk") -> None:
        name = "diskann" if index_type == "disk" else "diskann_mem"
        super().__init__(name=name, module="diskannpy")
        self.index_type = index_type
        self.index_precision = "pq-128" if index_type == "disk" else "f32"
        self._dir: Optional[str] = None
        self.last_build_s: float = 0.0

    def build(self, train: np.ndarray, params: dict, ctx: dict):
        import diskannpy as dap

        index_dir = ctx.get("path") or tempfile.mkdtemp(prefix="diskann_bench_")
        os.makedirs(index_dir, exist_ok=True)
        self._dir = index_dir
        metric_str = params.get("metric", "l2")
        dap_metric = "mips" if metric_str in ("inner_product", "cosine") else "l2"
        v_dtype = train.dtype if isinstance(train, np.ndarray) else np.float32
        if v_dtype not in (np.float32, np.uint8, np.int8):
            v_dtype = np.float32

        data = np.ascontiguousarray(train, dtype=v_dtype)
        if metric_str == "cosine":
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            data = (data / norms).astype(np.float32)
            v_dtype = np.float32

        num_threads = int(os.environ.get("BENCH_CPUS", "8") or 8)
        search_mem = float(os.environ.get("DISKANN_SEARCH_MEM_GB", "4.0") or 4.0)
        build_mem = float(os.environ.get("DISKANN_BUILD_MEM_GB", "12.0") or 12.0)
        r = int(params.get("m", 32) or 32)
        l_build = int(params.get("ef_construction", 64) or 64)
        stats_file = os.path.join(index_dir, "build_stats.json")

        if self.index_type == "disk":
            disk_index_file = os.path.join(index_dir, "ann_disk.index")
            if not os.path.exists(disk_index_file):
                t_b0 = time.time()
                dap.build_disk_index(
                    data=data,
                    distance_metric=dap_metric,
                    index_directory=index_dir,
                    complexity=l_build,
                    graph_degree=r,
                    search_memory_maximum=search_mem,
                    build_memory_maximum=build_mem,
                    num_threads=num_threads,
                    vector_dtype=v_dtype,
                )
                self.last_build_s = round(time.time() - t_b0, 3)
                try:
                    with open(stats_file, "w") as f:
                        json.dump({"build_s": self.last_build_s}, f)
                except Exception:
                    pass
            else:
                if os.path.exists(stats_file):
                    try:
                        with open(stats_file) as f:
                            self.last_build_s = float(json.load(f).get("build_s", 0.0))
                    except Exception:
                        pass
            index = dap.StaticDiskIndex(
                index_directory=index_dir,
                num_threads=num_threads,
                num_nodes_to_cache=1000,
                distance_metric=dap_metric,
                vector_dtype=v_dtype,
                dimensions=data.shape[1],
            )
        else:
            mem_index_file = os.path.join(index_dir, "ann_mem.index")
            if not os.path.exists(mem_index_file):
                t_b0 = time.time()
                dap.build_memory_index(
                    data=data,
                    distance_metric=dap_metric,
                    vector_dtype=v_dtype,
                    graph_degree=r,
                    complexity=l_build,
                    num_threads=num_threads,
                    index_directory=index_dir,
                )
                self.last_build_s = round(time.time() - t_b0, 3)
                try:
                    with open(stats_file, "w") as f:
                        json.dump({"build_s": self.last_build_s}, f)
                except Exception:
                    pass
            else:
                if os.path.exists(stats_file):
                    try:
                        with open(stats_file) as f:
                            self.last_build_s = float(json.load(f).get("build_s", 0.0))
                    except Exception:
                        pass
            index = dap.StaticMemoryIndex(
                distance_metric=dap_metric,
                vector_dtype=v_dtype,
                index_directory=index_dir,
                num_threads=num_threads,
            )
        return (index, index_dir, dap_metric, v_dtype)

    def search(
        self, handle, queries: np.ndarray, k: int, ef_search: int, metric: str
    ) -> np.ndarray:
        index, _index_dir, _dap_metric, v_dtype = handle
        q = np.ascontiguousarray(queries, dtype=v_dtype)
        if metric == "cosine":
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            q = (q / norms).astype(np.float32)
        num_threads = int(os.environ.get("BENCH_CPUS", "8") or 8)
        beam_width = int(os.environ.get("DISKANN_BEAM_WIDTH", "2") or 2)
        res = index.batch_search(
            q,
            k_neighbors=int(k),
            complexity=int(ef_search),
            num_threads=num_threads,
            beam_width=beam_width,
        )
        return np.asarray(res.identifiers)

    def search_full_row(
        self, handle, queries: np.ndarray, k: int, ef_search: int, metric: str
    ) -> Any:
        return self.search(handle, queries, k, ef_search, metric)

    def index_bytes(self, handle) -> int:
        index_dir = handle[1]
        return _dir_size(index_dir)


class LanceDbAdapter(VectorAdapter):
    """LanceDB vector index.

    ``index_type`` selects the ANN family, so both of LanceDB's relevant
    algorithms can be reported side by side:

    * ``ivf_pq``  — LanceDB's **default** disk-ANN index (IVF + product
      quantization). This is what ``create_index`` builds with no config, so it
      is the representative "what a LanceDB user gets" baseline.
    * ``hnsw_flat`` — LanceDB's uncompressed HNSW over raw float32 vectors
      (Index Precision: f32). Parity with plain-float HNSW in FAISS/HNSWlib.
    * ``hnsw_sq``   — LanceDB's HNSW with scalar quantization (8-bit integer vectors,
      Index Precision: sq).
    """

    def __init__(self, index_type: str = "hnsw_flat") -> None:
        name = "lancedb" if index_type in ("hnsw_flat", "flat") else "lancedb_sq"
        super().__init__(name=name, module="lancedb")
        self.index_type = index_type
        self.index_precision = "f32" if index_type in ("hnsw_flat", "flat") else "sq"

    def build(self, train, params, ctx):
        import lancedb
        import pyarrow as pa

        uri = ctx.get("path") or tempfile.mkdtemp(prefix="lancedb_bench_")
        db = lancedb.connect(uri)
        tbl = pa.table(
            {
                "id": pa.array(np.arange(len(train), dtype=np.int64)),
                "vector": pa.array(
                    [row.tolist() for row in train], type=pa.list_(pa.float32())
                ),
            }
        )
        t = db.create_table("bench", data=tbl, mode="overwrite")
        # LanceDB distance types are l2 / cosine / dot. A dot-product dataset
        # must use "dot" (not l2), or the ranking uses the wrong metric.
        # Declared Any: the lancedb stubs narrow `distance_type` to a Literal, and the
        # value is only ever one of the three literals below, so the narrowing adds
        # three call-site ignores and no safety.
        metric: Any = {
            "cosine": "cosine",
            "inner_product": "dot",
        }.get(str(params.get("metric") or ""), "l2")
        # New unified API: the first positional arg is the vector column name.
        if self.index_type in ("hnsw_flat", "flat"):
            from lancedb.index import HnswFlat

            t.create_index(
                "vector",
                config=HnswFlat(
                    distance_type=metric,
                    m=int(params["m"]),
                    ef_construction=int(params["ef_construction"]),
                ),
            )
        elif self.index_type in ("hnsw_sq", "sq"):
            from lancedb.index import HnswSq

            t.create_index(
                "vector",
                config=HnswSq(
                    distance_type=metric,
                    m=int(params["m"]),
                    ef_construction=int(params["ef_construction"]),
                ),
            )
        else:
            # IVF_PQ with LanceDB's defaults (num_partitions / num_sub_vectors
            # auto-selected) — the disk-ANN index LanceDB builds by default.
            from lancedb.index import IvfPq

            t.create_index("vector", config=IvfPq(distance_type=metric))
        return (t, uri)

    def search(self, handle, queries, k, ef_search, metric):
        t, _uri = handle
        out = []
        for q in queries:
            # Project only `id`: the other engines return ids only, so fetching
            # the vector payload here would make LanceDB do strictly more work.
            qb = t.search(q.tolist()).select(["id"]).limit(int(k))
            # Apply the same recall knob the other engines get: the HNSW index
            # uses `ef`, IVF_PQ uses `nprobes`. Guarded so a client without
            # either method still runs instead of failing the whole engine.
            try:
                if self.index_type in ("hnsw_flat", "hnsw_sq", "flat", "sq"):
                    qb = qb.ef(int(ef_search))
                else:
                    qb = qb.nprobes(max(1, int(ef_search) // 10))
            except (AttributeError, TypeError):
                pass
            # Two-stage exact rescoring, the same concept as BenoDB's
            # `refine_factor`: LanceDB re-ranks the candidate set from the
            # full-precision payload instead of the quantized codes. Only meaningful
            # for a quantized index, so it is guarded and opt-in via REFINE_FACTOR.
            _rf = int(os.environ.get("REFINE_FACTOR", "1") or 1)
            if _rf > 1:
                try:
                    qb = qb.refine_factor(_rf)
                except (AttributeError, TypeError):
                    pass
            rows = qb.to_list()
            out.append([int(r["id"]) for r in rows])
        return np.array(out)

    def search_full_row(self, handle, queries, k, ef_search, metric):
        t, _uri = handle
        out = []
        for q in queries:
            # Full row: fetch id and vector from Lance file on disk
            qb = t.search(q.tolist()).limit(int(k))
            try:
                if self.index_type in ("hnsw_flat", "hnsw_sq", "flat", "sq"):
                    qb = qb.ef(int(ef_search))
                else:
                    qb = qb.nprobes(max(1, int(ef_search) // 10))
            except (AttributeError, TypeError):
                pass
            _rf = int(os.environ.get("REFINE_FACTOR", "1") or 1)
            if _rf > 1:
                try:
                    qb = qb.refine_factor(_rf)
                except (AttributeError, TypeError):
                    pass
            out.append(qb.to_list())
        return out

    def index_bytes(self, handle):
        _t, uri = handle
        return _dir_size(uri)


def _opensearch_client(host: str):
    """Return (client, helpers) for the OpenSearch server at `host`.

    The compose stack runs OpenSearch, so we use the opensearch-py client; the
    elasticsearch 8.x client hard-rejects non-Elasticsearch servers with
    "UnsupportedProductError".
    """
    from opensearchpy import OpenSearch, helpers

    return OpenSearch(host), helpers


class OpenSearchAdapter(VectorAdapter):
    """OpenSearch (dense_vector + HNSW kNN)."""

    def __init__(self) -> None:
        super().__init__(name="opensearch", module="opensearchpy", server_side=True)

    def build(self, train, params, ctx):
        host = ctx.get("host") or os.environ.get("ES_URL") or "http://localhost:9200"
        es, helpers = _opensearch_client(host)
        dim = train.shape[1]
        metric = params.get("metric", "l2")
        # OpenSearch kNN space types differ from Elasticsearch's similarity names.
        space_type = {
            "l2": "l2",
            "cosine": "cosinesimil",
            "inner_product": "innerproduct",
        }.get(metric, "l2")
        # `ignore_unavailable` is valid for the Elasticsearch/OpenSearch REST API but
        # missing from the client stubs.
        es.indices.delete(index="bdb-bench", ignore_unavailable=True)  # type: ignore[call-arg]
        es.indices.create(
            index="bdb-bench",
            body={
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": int(params.get("ef_search", 100)),
                    }
                },
                "mappings": {
                    "properties": {
                        "vector": {
                            "type": "knn_vector",
                            "dimension": dim,
                            "method": {
                                "name": "hnsw",
                                "space_type": space_type,
                                "engine": "lucene",
                                "parameters": {
                                    "m": int(params["m"]),
                                    "ef_construction": int(params["ef_construction"]),
                                },
                            },
                        }
                    }
                },
            },
        )
        def _row_source(i, row):
            vec = row.tolist()
            # OpenSearch rejects an all-zero vector for cosinesimil ("zero vector
            # is not supported"); nudge it to a tiny vector. A zero vector is
            # orthogonal to everything, so this preserves the similarity result.
            if space_type == "cosinesimil" and not any(vec):
                vec = list(vec)
                vec[0] = 1e-6
            return {"_index": "bdb-bench", "_id": int(i), "_source": {"vector": vec}}

        actions = (_row_source(i, row) for i, row in enumerate(train))
        helpers.bulk(es, actions)
        es.indices.refresh(index="bdb-bench")
        return es

    def search(self, handle, queries, k, ef_search, metric):
        es = handle
        out = []
        for q in queries:
            body = {
                "size": int(k),
                "query": {
                    "knn": {
                        "vector": {
                            "vector": q.tolist(),
                            "k": int(k),
                            "method_parameters": {"ef_search": int(ef_search)},
                        }
                    }
                },
                "_source": False,
            }
            res = es.search(index="bdb-bench", body=body)
            out.append([int(h["_id"]) for h in res["hits"]["hits"]])
        return np.array(out)

    def search_full_row(self, handle, queries, k, ef_search, metric):
        es = handle
        out = []
        for q in queries:
            body = {
                "size": int(k),
                "query": {
                    "knn": {
                        "vector": {
                            "vector": q.tolist(),
                            "k": int(k),
                            "method_parameters": {"ef_search": int(ef_search)},
                        }
                    }
                },
                "_source": True,
            }
            res = es.search(index="bdb-bench", body=body)
            out.append(res["hits"]["hits"])
        return out

    def index_bytes(self, handle):
        # OpenSearch exposes the on-disk index size via the stats API.
        try:
            stats = handle.indices.stats(index="bdb-bench")
            return int(stats["indices"]["bdb-bench"]["total"]["store"]["size_in_bytes"])
        except Exception:
            return 0


class QdrantAdapter(VectorAdapter):
    """Qdrant (dense-vector collection + HNSW) via the qdrant-client."""

    def __init__(self, index_type: str = "flat") -> None:
        name = "qdrant_sq" if index_type in ("hnsw_sq", "sq") else "qdrant"
        super().__init__(name=name, module="qdrant_client", server_side=True)
        self.index_type = index_type
        self.index_precision = "sq" if index_type in ("hnsw_sq", "sq") else "f32"

    def build(self, train, params, ctx):
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, HnswConfigDiff, PointStruct, VectorParams,
            ScalarQuantization, ScalarQuantizationConfig, ScalarType
        )

        url = ctx.get("host") or os.environ.get("QDRANT_URL") or "http://localhost:6333"
        client = QdrantClient(url=url, timeout=300)
        metric = params.get("metric", "l2")
        distance = {
            "cosine": Distance.COSINE,
            "inner_product": Distance.DOT,
        }.get(metric, Distance.EUCLID)
        quant_config = None
        if self.index_type in ("hnsw_sq", "sq"):
            quant_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )
        client.recreate_collection(
            collection_name="bdb_bench",
            vectors_config=VectorParams(size=int(train.shape[1]), distance=distance),
            hnsw_config=HnswConfigDiff(
                m=int(params["m"]),
                ef_construct=int(params["ef_construction"]),
            ),
            quantization_config=quant_config,
        )
        batch = 512
        for start in range(0, len(train), batch):
            end = min(start + batch, len(train))
            client.upsert(
                collection_name="bdb_bench",
                points=[
                    PointStruct(id=int(i), vector=train[i].tolist())
                    for i in range(start, end)
                ],
                wait=True,
            )
        return client

    def search(self, handle, queries, k, ef_search, metric):
        from qdrant_client.models import SearchParams

        search_params = SearchParams(hnsw_ef=int(ef_search), exact=False)
        out = []
        for q in queries:
            vec = q.tolist()
            try:  # qdrant-client >= 1.10
                res = handle.query_points(
                    collection_name="bdb_bench",
                    query=vec,
                    limit=int(k),
                    search_params=search_params,
                ).points
            except AttributeError:  # older qdrant-client
                res = handle.search(
                    collection_name="bdb_bench",
                    query_vector=vec,
                    limit=int(k),
                    search_params=search_params,
                )
            out.append([int(p.id) for p in res])
        return np.array(out)

    def search_full_row(self, handle, queries, k, ef_search, metric):
        from qdrant_client.models import SearchParams

        search_params = SearchParams(hnsw_ef=int(ef_search), exact=False)
        out = []
        for q in queries:
            vec = q.tolist()
            try:
                res = handle.query_points(
                    collection_name="bdb_bench",
                    query=vec,
                    limit=int(k),
                    search_params=search_params,
                    with_payload=True,
                    with_vectors=True,
                ).points
            except Exception:
                res = handle.search(
                    collection_name="bdb_bench",
                    query_vector=vec,
                    limit=int(k),
                    search_params=search_params,
                    with_payload=True,
                    with_vectors=True,
                )
            out.append(res)
        return out

    def index_bytes(self, handle):
        try:
            info = handle.get_collection("bdb_bench")
            dim = int(info.config.params.vectors.size)
            bytes_per_dim = 1 if self.index_precision == "sq" else 4
            return dim * int(info.points_count or 0) * bytes_per_dim
        except Exception:
            return 0


class MilvusAdapter(VectorAdapter):
    """Milvus (dense collection + HNSW / HNSW_SQ) via pymilvus."""

    def __init__(self, index_type: str = "hnsw") -> None:
        quantized = index_type in ("hnsw_sq", "ivf_sq8")
        name = "milvus_sq" if quantized else "milvus"
        super().__init__(name=name, module="pymilvus", server_side=True)
        self.index_type = index_type
        self.index_precision = "sq" if quantized else "f32"
        self._nlist = 0

    def build(self, train, params, ctx):
        from pymilvus import MilvusClient
        from pymilvus.milvus_client.index import IndexParams

        # `tcp://` selects gRPC (the low-latency benchmark path); `http://` selects
        # the RESTful API whose per-request overhead (~200 ms p50) dominated the
        # earlier "misconfigured" numbers.
        uri = ctx.get("host") or os.environ.get("MILVUS_URI") or "tcp://localhost:19530"
        client = MilvusClient(uri=uri)
        metric = {
            "cosine": "COSINE",
            "inner_product": "IP",
        }.get(str(params.get("metric") or ""), "L2")
        name = "bdb_bench"
        if client.has_collection(name):
            client.drop_collection(name)

        # pymilvus 3.x requires an `IndexParams` object (not a dict). Declare the
        # index at collection creation: `create_collection` auto-creates an
        # AUTOINDEX on the vector field when `index_params` is omitted, and a
        # second index on the same field is rejected — so the index must be passed
        # in here, not added via a later `create_index` call. `Strong` (rather than
        # the default `Bounded`) keeps the build-then-search sequence deterministic.
        index_params = IndexParams()
        if self.index_type in ("hnsw_sq", "ivf_sq8"):
            # HNSW_SQ: graph traversal over 8-bit scalar-quantized vectors. Chosen
            # over IVF_SQ8 because IVF partitions on L2 centroids, which is a poor
            # router for a dot-product metric: on lastfm-64-dot IVF_SQ8 with
            # metric_type=IP returned 0.8608 recall while the same collection served
            # by HNSW (f32) returned 0.9954, and the loss was the routing, not the
            # quantization -- a true top-IP neighbour lands in a cluster whose L2
            # centroid is far away, so a small nprobe excludes it. A graph keeps the
            # metric in the traversal, so only the quantizer's error remains.
            index_params.add_index(
                field_name="vector",
                index_type="HNSW_SQ",
                index_name="bdb_hnsw_sq",
                metric_type=metric,
                params={
                    "M": int(params["m"]),
                    "efConstruction": int(params["ef_construction"]),
                },
            )
        else:
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                index_name="bdb_hnsw",
                metric_type=metric,
                params={"M": int(params["m"]), "efConstruction": int(params["ef_construction"])},
            )
        # `create_collection(index_params=...)` does not stick: the live collection
        # reported `index_type: AUTOINDEX`, and AUTOINDEX never reads the HNSW `ef`
        # search parameter. That is why recall was byte-identical (0.6796 on
        # nytimes-256) for ef=200, ef=800, and after an explicit flush, while QPS
        # moved only with noise. Create the collection, drop the auto index, then
        # declare the intended index after the data is in.
        client.create_collection(
            collection_name=name,
            dimension=int(train.shape[1]),
            metric_type=metric,
            auto_id=False,
            consistency_level="Strong",
        )
        client.release_collection(name)
        for auto_index in client.list_indexes(name):
            client.drop_index(name, index_name=auto_index)
        batch = 512
        for start in range(0, len(train), batch):
            end = min(start + batch, len(train))
            client.insert(
                collection_name=name,
                data=[{"id": int(i), "vector": train[i].tolist()} for i in range(start, end)],
            )
        # Seal the growing segments before loading. Without an explicit flush the
        # newest inserts still live in an unsealed growing segment, which
        # `load_collection` does not make searchable: queries silently return fewer
        # than k neighbours. That asymmetry is why only some datasets lost recall
        # (nytimes-256 0.68 vs qdrant 0.998) while others looked healthy — the loss
        # tracks how much of the insert stream Milvus happened to auto-seal.
        client.flush(name)
        client.create_index(name, index_params=index_params)
        # Wait for the index build to finish; searching a half-built index returns
        # whatever is already indexed.
        deadline = time.time() + 600
        while time.time() < deadline:
            names = client.list_indexes(name)
            if names:
                info = client.describe_index(name, names[0])
                if str(info.get("state", "")).lower() == "finished":
                    break
            time.sleep(0.2)
        client.load_collection(name)
        # `load_collection` acknowledges the request; it is not a barrier. Searching
        # before every segment is queryable returns fewer than `k` neighbours, which
        # scores as low recall rather than as an error. On the 20k slices only the
        # cosine datasets showed it (milvus 0.68–0.85 vs qdrant 0.998), because COSINE
        # indexing is the slowest to become queryable, so wait for Loaded explicitly.
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                state = client.get_load_state(collection_name=name)
            except Exception:
                break  # older client: nothing to poll, fall through and search
            if isinstance(state, dict):
                state = state.get("state", state)
            label = getattr(state, "name", str(state))
            if str(label).strip().lower() == "loaded":
                break
            time.sleep(0.1)
        return (client, name, metric, int(train.shape[1]), int(len(train)))

    def _search_params(self, mtype, ef_search):
        if self.index_type == "ivf_sq8":
            nprobe = min(self._nlist, max(1, int(ef_search) // 10))
            return {"metric_type": mtype, "params": {"nprobe": nprobe}}
        # HNSW and HNSW_SQ both search with `ef`, so the quantized variant is
        # compared at the same traversal budget as the f32 one.
        return {"metric_type": mtype, "params": {"ef": int(ef_search)}}

    def search(self, handle, queries, k, ef_search, metric):
        client, name, mtype, _dim, _n = handle
        out = []
        for q in queries:
            hits = client.search(
                collection_name=name,
                data=[q.tolist()],
                limit=int(k),
                search_params=self._search_params(mtype, ef_search),
            )
            out.append([int(h["id"]) for h in hits[0]])
        return np.array(out)

    def search_full_row(self, handle, queries, k, ef_search, metric):
        client, name, mtype, _dim, _n = handle
        out = []
        for q in queries:
            hits = client.search(
                collection_name=name,
                data=[q.tolist()],
                limit=int(k),
                output_fields=["id", "vector"],
                search_params=self._search_params(mtype, ef_search),
            )
            out.append(hits[0])
        return out

    def index_bytes(self, handle):
        # Milvus does not expose per-index bytes; report the raw vector payload
        # (dim * count * 4B) as a comparable floor.
        _client, _name, _metric, dim, n = handle
        return int(dim) * int(n) * 4


class WeaviateAdapter(VectorAdapter):
    """Weaviate (HNSW) via the v4 weaviate-client."""

    def __init__(self, index_type: str = "flat") -> None:
        name = "weaviate_sq" if index_type in ("hnsw_sq", "sq") else "weaviate"
        super().__init__(name=name, module="weaviate", server_side=True)
        self.index_type = index_type
        self.index_precision = "sq" if index_type in ("hnsw_sq", "sq") else "f32"

    def build(self, train, params, ctx):
        import weaviate
        from weaviate.classes.config import Configure, DataType, Property, VectorDistances

        url = ctx.get("host") or os.environ.get("WEAVIATE_URL") or "http://localhost:8080"
        url = url.replace("http://", "").replace("https://", "").rstrip("/")
        host, _, port = url.partition(":")
        client = weaviate.connect_to_custom(
            http_host=host or "localhost",
            http_port=int(port or 8080),
            http_secure=False,
            grpc_host=host or "localhost",
            grpc_port=50051,
            grpc_secure=False,
        )
        dist = {
            "cosine": VectorDistances.COSINE,
            "inner_product": VectorDistances.DOT,
        }.get(str(params.get("metric") or ""), VectorDistances.L2_SQUARED)
        name = "BsdbBench"
        if client.collections.exists(name):
            client.collections.delete(name)
        quantizer = None
        if self.index_type in ("hnsw_sq", "sq"):
            quantizer = Configure.VectorIndex.Quantizer.sq(training_limit=100000)
        # weaviate-client 4.24.1's `near_vector` has no `ef` argument (verified by
        # introspecting the bound method), so the previous `near_vector(..., ef=...)`
        # always raised TypeError and fell through to the fallback that sends no ef at
        # all — every query then used Weaviate's dynamic ef (min 100 / max 500), which
        # is why recall was insensitive to EF_SEARCH. Pin the beam width at the
        # collection instead: min == max makes the ef the harness asks for the ef that
        # is actually used.
        ef_target = max(int(os.environ.get("EF_SEARCH", "200")), 1)
        client.collections.create(
            name=name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=dist,
                ef_construction=int(params["ef_construction"]),
                max_connections=int(params["m"]),
                dynamic_ef_min=ef_target,
                dynamic_ef_max=ef_target,
                quantizer=quantizer,
            ),
            properties=[Property(name="rid", data_type=DataType.INT)],
        )
        coll = client.collections.get(name)
        with coll.batch.dynamic() as batch:
            for i in range(len(train)):
                batch.add_object(properties={"rid": int(i)}, vector=train[i].tolist())
        return (client, coll, int(train.shape[1]), int(len(train)))

    def search(self, handle, queries, k, ef_search, metric):
        _client, coll, _dim, _n = handle
        out = []
        for q in queries:
            try:
                res = coll.query.near_vector(
                    near_vector=q.tolist(),
                    limit=int(k),
                    return_properties=["rid"],
                    ef=int(ef_search),
                )
            except TypeError:  # client without a per-query `ef`
                res = coll.query.near_vector(
                    near_vector=q.tolist(),
                    limit=int(k),
                    return_properties=["rid"],
                )
            out.append([int(o.properties["rid"]) for o in res.objects])
        return np.array(out)

    def search_full_row(self, handle, queries, k, ef_search, metric):
        _client, coll, _dim, _n = handle
        out = []
        for q in queries:
            try:
                res = coll.query.near_vector(
                    near_vector=q.tolist(),
                    limit=int(k),
                    return_properties=["rid"],
                    return_metadata=["distance"],
                    include_vector=True,
                    ef=int(ef_search),
                )
            except Exception:
                res = coll.query.near_vector(
                    near_vector=q.tolist(),
                    limit=int(k),
                    return_properties=["rid"],
                    include_vector=True,
                )
            out.append(res.objects)
        return out

    def index_bytes(self, handle):
        # Weaviate's schema API exposes object counts, not index bytes; report
        # the raw vector payload (dim * count * bytes_per_dim) as a comparable floor.
        _client, _coll, dim, n = handle
        bytes_per_dim = 1 if self.index_precision == "sq" else 4
        return int(dim) * int(n) * bytes_per_dim


def vector_adapters() -> dict[str, VectorAdapter]:
    return {
        "diskann": DiskANNAdapter("disk"),
        "diskann_mem": DiskANNAdapter("memory"),
        "faiss": FaissAdapter("flat"),
        "faiss_sq": FaissAdapter("hnsw_sq"),
        "faiss_ivfsq8": FaissAdapter("ivf_sq8"),
        "faiss_ivfpq": FaissAdapter("ivfpq"),
        "hnswlib": HnswlibAdapter(),
        "pgvector": PgvectorAdapter(),
        # LanceDB HNSW: uncompressed float32 (HnswFlat) and scalar-quantized (HnswSq)
        "lancedb": LanceDbAdapter("hnsw_flat"),
        "lancedb_sq": LanceDbAdapter("hnsw_sq"),
        "opensearch": OpenSearchAdapter(),
        "qdrant": QdrantAdapter("flat"),
        "qdrant_sq": QdrantAdapter("hnsw_sq"),
        "milvus": MilvusAdapter(),
        # HNSW_SQ, not IVF_SQ8: see the note in `MilvusAdapter.build`. IVF routing
        # measured the L2 centroid layout rather than the quantization on the
        # dot-product dataset.
        "milvus_sq": MilvusAdapter("hnsw_sq"),
        "weaviate": WeaviateAdapter("flat"),
        "weaviate_sq": WeaviateAdapter("hnsw_sq"),
    }


# Engines with a real GPU execution path. The recorded `device` must reflect
# where the algorithm actually ran, not which container launched the run: only
# these use the GPU when the run's device is "gpu"; every other engine (pgvector,
# OpenSearch, Qdrant, Milvus, Weaviate, LanceDB, hnswlib, ...) is CPU-only.
def _vector_backend(adapter: "VectorAdapter", device: str) -> str:
    if device != "gpu":
        return "cpu"
    name = getattr(adapter, "name", "")
    if name == "faiss":
        return "gpu" if getattr(adapter, "_used_gpu", False) else "cpu"
    if name in ("benodb", "bdb"):
        return "gpu"
    return "cpu"


# --------------------------------------------------------------------------- #
# Vector workload
# --------------------------------------------------------------------------- #
def download(dataset: str) -> str:
    if dataset.lower().startswith("bigann"):
        return dataset
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{dataset}.hdf5")
    if os.path.exists(path):
        return path
    url = f"{ANN_BASE}/{dataset}.hdf5"
    print(f"downloading {url} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(path, "wb") as out:
        shutil.copyfileobj(resp, out)
    return path


def load_dataset(path: str, data_dir: Optional[str] = None):
    if path.lower().startswith("bigann") or (data_dir and os.path.exists(os.path.join(data_dir, "base.10M.u8bin"))):
        cands = [
            data_dir,
            os.environ.get("BENCH_DATA_DIR"),
            "/opt/bench/data",
            "/home/ralbright/data/datasets/bigann",
            CACHE_DIR,
        ]
        chosen_dir = None
        for c in cands:
            if c and os.path.exists(os.path.join(c, "base.10M.u8bin")):
                chosen_dir = c
                break
        if not chosen_dir:
            raise FileNotFoundError(
                f"Could not find BigANN dataset in any candidate directory: {cands}"
            )
        base_path = os.path.join(chosen_dir, "base.10M.u8bin")
        query_path = os.path.join(chosen_dir, "query.public.10K.u8bin")
        gt_path = os.path.join(chosen_dir, "GT.10M")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(chosen_dir, "bigann-10M")

        train = np.memmap(base_path, dtype=np.uint8, mode="r", offset=8, shape=(10_000_000, 128))
        test = np.memmap(query_path, dtype=np.uint8, mode="r", offset=8, shape=(10_000, 128))

        with open(gt_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=2)
            n_q, file_k = int(header[0]), int(header[1])
            neighbors = np.fromfile(f, dtype=np.uint32, count=n_q * file_k).reshape((n_q, file_k))
        return train, test, neighbors

    import h5py

    with h5py.File(path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int64)
    return train, test, neighbors


def exact_ground_truth(train, test, q, k, metric):
    """Exact top-k over a subsetted train set, using the dataset metric.

    The HDF5 `neighbors` field is only valid for the full train set. An L2
    brute force on an angular dataset produces a different neighbour set, so
    recall from that truth is not comparable to engines that search cosine/IP.
    """
    truth = []
    if metric == "cosine":
        train_n = train.astype(np.float32) / (np.linalg.norm(train, axis=1, keepdims=True) + 1e-12)
        for i in range(q):
            query = test[i].astype(np.float32) / (np.linalg.norm(test[i]) + 1e-12)
            truth.append(np.argsort(1.0 - train_n @ query)[:k])
    elif metric in ("inner_product", "ip", "dot"):
        for i in range(q):
            truth.append(np.argsort(-(train.astype(np.float32) @ test[i].astype(np.float32)))[:k])
    else:
        for i in range(q):
            truth.append(np.argsort(np.linalg.norm(train.astype(np.float32) - test[i].astype(np.float32), axis=1))[:k])
    return np.array(truth)


def run_vector(adapter: VectorAdapter, args) -> dict:
    metric = args.metric or DATASET_METRIC.get(args.dataset, "l2")
    data_dir = getattr(args, "data_dir", None)
    train, test, neighbors = load_dataset(download(args.dataset), data_dir)
    if args.limit and args.limit < len(train):
        train = train[: args.limit]
        neighbors = None  # ground truth invalid for a subset
    n, dim = train.shape

    params = {
        "m": args.m,
        "ef_construction": args.ef_construction,
        "ef_search": args.ef_search,
        "metric": metric,
        "nlist": args.nlist,
        "nprobe": args.nprobe,
        "pq_m": args.pq_m,
        "pq_nbits": args.pq_nbits,
        "refine": args.refine,
        "mmap_invlists": args.mmap_invlists,
    }
    ctx = {
        "dsn": args.dsn,
        "host": args.host,
        "path": args.path,
        "device": args.device,
    }

    t0 = time.time()
    handle = adapter.build(train, params, ctx)
    build_s = time.time() - t0
    if hasattr(adapter, "last_build_s") and adapter.last_build_s > 0:
        build_s = adapter.last_build_s

    if neighbors is None:
        q = min(args.queries, len(test))
        neighbors = exact_ground_truth(train, test, q, args.k, metric)

    q = min(args.queries, len(test))
    k = args.k

    # Warm both paths before either is timed.
    #
    # The two loops used to run back to back with no warm-up: the index-only pass
    # paid first-touch page faults on the freshly built index plus the CPU's
    # frequency ramp, and the full-row pass inherited an already-warm process. The
    # result was a row pass reporting *higher* throughput than the index-only pass
    # it contains (faiss: 3448 index QPS vs 6848 row QPS), which cannot happen —
    # `search_full_row` is `search` plus a payload lookup.
    warmup = min(50, q)
    for i in range(warmup):
        batch = test[i : i + 1]
        adapter.search(handle, batch, k, args.ef_search, metric)
        adapter.search_full_row(handle, batch, k, args.ef_search, metric)

    pure_latencies = []
    full_latencies = []
    hits = 0.0
    # Alternate which path is measured first within the loop, so any residual drift
    # (thermal, allocator warmth, page-cache growth) lands on both columns instead
    # of systematically favouring whichever ran second.
    for i in range(q):
        batch = test[i : i + 1]
        if i % 2 == 0:
            t1 = time.perf_counter()
            ids = adapter.search(handle, batch, k, args.ef_search, metric)
            pure_latencies.append(time.perf_counter() - t1)
            t1 = time.perf_counter()
            _ = adapter.search_full_row(handle, batch, k, args.ef_search, metric)
            full_latencies.append(time.perf_counter() - t1)
        else:
            t1 = time.perf_counter()
            _ = adapter.search_full_row(handle, batch, k, args.ef_search, metric)
            full_latencies.append(time.perf_counter() - t1)
            t1 = time.perf_counter()
            ids = adapter.search(handle, batch, k, args.ef_search, metric)
            pure_latencies.append(time.perf_counter() - t1)
        got = set(int(x) for x in ids[0].tolist())
        truth_i = set(int(x) for x in neighbors[i][:k].tolist())
        hits += len(got & truth_i) / k

    recall = hits / q
    pure_total = sum(pure_latencies) or 1e-9
    full_total = sum(full_latencies) or 1e-9
    pure_p50 = float(np.percentile(pure_latencies, 50) * 1000)
    pure_p99 = float(np.percentile(pure_latencies, 99) * 1000)
    full_p50 = float(np.percentile(full_latencies, 50) * 1000)
    full_p99 = float(np.percentile(full_latencies, 99) * 1000)

    return {
        "engine": adapter.name,
        "index_type": getattr(adapter, "index_type", "hnsw"),
        "index_precision": getattr(adapter, "index_precision", "f32"),
        "dataset": args.dataset,
        "workload": "vector_ann",
        "device": _vector_backend(adapter, args.device),
        "n": int(n),
        "dim": int(dim),
        "index": params,
        "queries": q,
        "k": k,
        "build_s": round(build_s, 3),
        "index_mb": round(adapter.index_bytes(handle) / 1e6, 2),
        "recall_at_k": round(recall, 4),
        "pure_index_qps": round(q / pure_total, 1),
        # Recorded for every engine so the report's refine column is meaningful:
        # engines with no refinement concept (faiss, hnswlib, pgvector, opensearch)
        # carry 1, i.e. "off", and the report prints False.
        "refine_factor": int(os.environ.get("REFINE_FACTOR", "1") or 1),
        "pure_p50_ms": round(pure_p50, 3),
        "pure_p99_ms": round(pure_p99, 3),
        # `full_row_qps` is kept as an explicit alias because older artifacts were
        # written with it, but `qps` is the field the report reads for Row QPS and
        # it must be the row-path rate. It used to be `q / pure_total` -- the
        # index-only rate -- while `p50_ms`/`p99_ms` beside it came from the row
        # loop, so every competitor's "Row QPS" was the index-only throughput
        # printed next to row-path latencies, and was bit-identical to
        # `pure_index_qps` (faiss on sift: 9554.2 in both columns).
        "full_row_qps": round(q / full_total, 1),
        "full_p50_ms": round(full_p50, 3),
        "full_p99_ms": round(full_p99, 3),
        "qps": round(q / full_total, 1),
        "p50_ms": round(full_p50, 3),
        "p99_ms": round(full_p99, 3),
    }


# --------------------------------------------------------------------------- #
# Graph workload
# --------------------------------------------------------------------------- #
def _graph_networkx(args, edges) -> dict:
    import networkx as nx

    g = nx.from_edgelist((int(a), int(b)) for a, b in edges)
    t0 = time.time()
    if args.algorithm == "pagerank":
        result = nx.pagerank(g)
        n_out = len(result)
    elif args.algorithm == "connected_components":
        result = list(nx.connected_components(g))
        n_out = len(result)
    elif args.algorithm == "shortest_path":
        result = nx.single_source_shortest_path_length(g, args.source)
        n_out = len(result)
    else:
        raise RuntimeError(f"unknown algorithm {args.algorithm}")
    return {
        "engine": "networkx",
        "workload": "graph",
        "available": True,
        "algorithm": args.algorithm,
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "seconds": round(time.time() - t0, 3),
        "result_size": n_out,
    }


def _graph_neo4j(args, edges) -> dict:
    from neo4j import GraphDatabase

    uri = args.uri or os.environ.get("NEO4J_URI") or "bolt://localhost:7687"
    user = args.user or os.environ.get("NEO4J_USER") or "neo4j"
    password = args.password or os.environ.get("NEO4J_PASSWORD") or "neo4j"
    driver = GraphDatabase.driver(uri, auth=(user, password))

    algo = args.algorithm
    # Run the algorithm *inside the Neo4j JVM* via GDS `.mutate`, which returns a
    # single summary row, instead of `.stream`, which returns one row per node.
    # Timing `.stream` from Python measures the neo4j driver marshalling 100k+
    # Bolt records (PageRank on web-Google read as ~3.7 s) rather than the graph
    # algorithm (~0.1 s of JVM compute). `.mutate` writes the result to an
    # in-memory property and returns computeMillis / nodePropertiesWritten /
    # componentCount, so `seconds` is the native GDS execution time.
    mutate_procs = {
        "pagerank": "gds.pageRank.mutate",
        "connected_components": "gds.wcc.mutate",
    }
    if algo in mutate_procs:
        graph_call = f"CALL {mutate_procs[algo]}('bench', {{mutateProperty: 'bench_score'}})"
        params: dict = {}
    elif algo == "shortest_path":
        graph_call = (
            "CALL gds.shortestPath.dijkstra.stream('bench', "
            "{sourceNode: $src, targetNode: $tgt})"
        )
        src = int(args.source)
        params = {"src": src, "tgt": int(args.target) if args.target is not None else src}
    else:
        return {
            "engine": "neo4j",
            "workload": "graph",
            "available": False,
            "error": f"unknown algorithm {algo}",
        }

    # Load + project are setup, not query time. Teardown + constraint are harness
    # hygiene, kept out of `load_s` so a stale prior graph cannot inflate it.
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n").consume()
            session.run("DROP CONSTRAINT node_id IF EXISTS").consume()
            session.run("CREATE CONSTRAINT node_id FOR (n:Node) REQUIRE n.id IS UNIQUE").consume()

            t_load = time.time()
            edges_list = [(int(a), int(b)) for a, b in edges]
            batch_size = 10000
            for i in range(0, len(edges_list), batch_size):
                session.run(
                    "UNWIND $rows AS r MERGE (a:Node {id: r[0]}) "
                    "MERGE (b:Node {id: r[1]}) MERGE (a)-[:E]->(b)",
                    rows=edges_list[i:i + batch_size],
                ).consume()
            session.run("CALL gds.graph.drop('bench', false)").consume()
            session.run("CALL gds.graph.project('bench', 'Node', 'E')").consume()
            load_s = time.time() - t_load

            t0 = time.time()
            if algo == "shortest_path":
                # neo4j 5 requires LiteralString|Query; the query is built at runtime.
                out_rows = list(session.run(graph_call, **params))  # type: ignore[arg-type]
                n_out = len(out_rows)
                compute_ms = None
            else:
                summary = session.run(graph_call, **params).single()  # type: ignore[arg-type]
                data = dict(summary) if summary is not None else {}
                compute_ms = data.get("computeMillis")
                n_out = (
                    data.get("componentCount")
                    if algo == "connected_components"
                    else data.get("nodePropertiesWritten")
                )
            seconds = time.time() - t0

        return {
            "engine": "neo4j",
            "workload": "graph",
            "available": True,
            "algorithm": algo,
            "edges": len(edges),
            "load_s": round(load_s, 3),
            "seconds": round(seconds, 3),
            "compute_ms": compute_ms,
            "result_size": n_out,
            "layer": "neo4j GDS (native JVM)",
        }
    except Exception as exc:  # server unreachable / GDS missing
        return {"engine": "neo4j", "workload": "graph", "available": False, "error": str(exc)[:200]}
    finally:
        driver.close()


def _graph_benodb(args, edges) -> dict:
    import benodb as bdb
    import pyarrow as pa
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="benodb_graph_bench_")
    try:
        t_build_0 = time.time()
        schema = pa.schema([
            ("source", pa.uint64()),
            ("target", pa.uint64()),
        ])
        table = bdb.Table.create(tmpdir, schema)
        table.add_index(
            "source", {"type": "graph", "src_column": "source", "dst_column": "target"}
        )
        table.add_index(
            "target", {"type": "graph", "src_column": "target", "dst_column": "source"}
        )
        arr_src = pa.array(edges[:, 0].astype(np.uint64), type=pa.uint64())
        arr_dst = pa.array(edges[:, 1].astype(np.uint64), type=pa.uint64())
        batch = pa.Table.from_arrays([arr_src, arr_dst], names=["source", "target"])
        table.insert(batch)
        table.commit()
        table.wait_for_background_tasks()
        build_s = round(time.time() - t_build_0, 3)

        damping = float(getattr(args, "damping", 0.85))
        iterations = int(getattr(args, "iterations", 30))

        t0 = time.time()
        if args.algorithm == "pagerank":
            res = table.pagerank(damping=damping, iterations=iterations).to_pandas()
            n_out = len(res)
        elif args.algorithm == "connected_components":
            res = table.connected_components().to_pandas()
            n_out = int(res["component"].nunique()) if "component" in res.columns else len(res)
        elif args.algorithm == "shortest_path":
            target = getattr(args, "target", None)
            if target is not None:
                res = table.shortest_path(int(args.source), int(target))
                n_out = len(res)
            else:
                res = table.graph_neighbors(int(args.source), hops=int(getattr(args, "hops", 2)))
                n_out = len(res)
        else:
            raise RuntimeError(f"unknown algorithm {args.algorithm}")
        elapsed = round(time.time() - t0, 3)
        return {
            "engine": "benodb",
            "workload": "graph",
            "available": True,
            "algorithm": args.algorithm,
            "edges": len(edges),
            "build_seconds": build_s,
            "seconds": elapsed,
            "result_size": n_out,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _graph_cugraph(args, edges) -> dict:
    if not importlib.util.find_spec("cugraph") or not importlib.util.find_spec("cudf"):
        return {
            "engine": "cugraph",
            "workload": "graph",
            "available": False,
            "error": "cugraph and cudf required (pip/conda install cugraph)",
        }
    # Configure RMM *before* importing cudf/cugraph. Managed (unified) memory lets
    # the device pool spill to host RAM instead of OOMing — important on a
    # workstation GPU where the desktop compositor (Wayland) holds VRAM.
    try:
        import rmm

        rmm.reinitialize(managed_memory=True)
    except Exception:
        pass
    import cugraph
    import cudf

    t_build_0 = time.time()
    gdf = cudf.DataFrame({
        "src": edges[:, 0],
        "dst": edges[:, 1],
    })
    G = cugraph.Graph(directed=True)
    # renumber=True maps vertex IDs to a contiguous [0, N-1] block. Without it a
    # sparse/high ID space makes cuGraph allocate an array sized to the max ID,
    # which OOMs immediately.
    G.from_cudf_edgelist(gdf, source="src", destination="dst", renumber=True)
    build_s = round(time.time() - t_build_0, 3)

    damping = float(getattr(args, "damping", 0.85))
    iterations = int(getattr(args, "iterations", 30))

    t0 = time.time()
    if args.algorithm == "pagerank":
        res = cugraph.pagerank(G, alpha=damping, max_iter=iterations)
        n_out = len(res)
    elif args.algorithm == "connected_components":
        res = cugraph.weakly_connected_components(G)
        n_out = len(res)
    elif args.algorithm == "shortest_path":
        target = getattr(args, "target", None)
        res = cugraph.bfs(G, start=int(args.source))
        if target is not None:
            res_target = res[res["vertex"] == int(target)]
            n_out = len(res_target)
        else:
            n_out = len(res)
    else:
        raise RuntimeError(f"unknown algorithm {args.algorithm}")
    return {
        "engine": "cugraph",
        "workload": "graph",
        "available": True,
        "algorithm": args.algorithm,
        "edges": len(edges),
        "build_seconds": build_s,
        "seconds": round(time.time() - t0, 3),
        "result_size": n_out,
    }


def _graph_memgraph(args, edges) -> dict:
    """Memgraph via its MAGE query modules (bolt; native C++ engine, no JVM).

    Like Neo4j, the algorithm runs server-side, so we wrap the procedure in an
    aggregate and return only the summary — the Python driver never marshals the
    per-node result.
    """
    from neo4j import GraphDatabase

    uri = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
    driver = GraphDatabase.driver(uri)
    algo = args.algorithm
    calls = {
        "pagerank":
            "CALL pagerank.get() YIELD node, rank RETURN count(node) AS n",
        "connected_components":
            "CALL weakly_connected_components.get() YIELD node, component_id "
            "RETURN count(DISTINCT component_id) AS n",
    }
    call = calls.get(algo)
    if call is None:
        return {"engine": "memgraph", "workload": "graph", "available": False,
                "error": f"unsupported algorithm {algo}"}
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n").consume()
            # Index the id property first: without it, each unmatched MERGE does
            # a full label scan, so the load is O(n^2) and pegs a core for
            # minutes on a 500k-edge graph.
            try:
                session.run("CREATE INDEX ON :Node(id)").consume()
            except Exception:
                pass
            rows = [(int(a), int(b)) for a, b in edges]
            t_load = time.time()
            for i in range(0, len(rows), 10000):
                session.run(
                    "UNWIND $rs AS r MERGE (a:Node {id: r[0]}) "
                    "MERGE (b:Node {id: r[1]}) MERGE (a)-[:E]->(b)",
                    rs=rows[i:i + 10000],
                ).consume()
            load_s = time.time() - t_load
            t0 = time.time()
            rec = session.run(call).single()  # type: ignore[arg-type]
            seconds = time.time() - t0
        return {
            "engine": "memgraph",
            "workload": "graph",
            "available": True,
            "algorithm": algo,
            "edges": len(edges),
            "load_s": round(load_s, 3),
            "seconds": round(seconds, 3),
            "result_size": int(rec["n"]) if rec is not None else None,
            "layer": "memgraph MAGE (native engine)",
        }
    except Exception as exc:
        return {"engine": "memgraph", "workload": "graph", "available": False,
                "error": str(exc)[:200]}
    finally:
        driver.close()


def _graph_kuzu(args, edges) -> dict:
    """Kùzu (embedded columnar graph DB) graph algorithms via the `algo` extension."""
    try:
        import kuzu
    except Exception:
        return {"engine": "kuzu", "workload": "graph", "available": False,
                "error": "pip install kuzu"}
    import shutil
    import tempfile

    algo = args.algorithm
    if algo not in ("pagerank", "connected_components"):
        return {"engine": "kuzu", "workload": "graph", "available": False,
                "error": f"unsupported algorithm {algo}"}
    tmpdir = tempfile.mkdtemp(prefix="kuzu_bench_")
    try:
        # Kùzu expects a database *file* path, and sizes its buffer pool / DB
        # from host RAM by default (mmap fails under the cgroup cap), so cap both
        # to the shared envelope.
        ram_gb = float(getattr(args, "ram_gb", 0) or 8)
        db = kuzu.Database(
            os.path.join(tmpdir, "g.kuzu"),
            buffer_pool_size=int(ram_gb * 1024**3 / 2),
            max_db_size=2 * 1024**3,
            max_num_threads=max(1, int(getattr(args, "cores", 0) or 1)),
        )
        conn = kuzu.Connection(db)
        t_load = time.time()
        conn.execute("CREATE NODE TABLE Node(id INT64, PRIMARY KEY(id))")
        conn.execute("CREATE REL TABLE E(FROM Node TO Node)")
        # Kùzu binds a list of lists (not tuples), and indexes lists 1-based.
        rows = [[int(a), int(b)] for a, b in edges]
        for i in range(0, len(rows), 10000):
            conn.execute(
                "UNWIND $rs AS r MERGE (a:Node {id: r[1]}) "
                "MERGE (b:Node {id: r[2]}) MERGE (a)-[:E]->(b)",
                {"rs": rows[i:i + 10000]},
            )
        load_s = time.time() - t_load
        # Kùzu 0.11 ships `algo` statically; algorithms run on a projected graph
        # and PageRank's procedure is `page_rank`.
        conn.execute("CALL project_graph('g', ['Node'], ['E'])")
        t0 = time.time()
        if algo == "pagerank":
            res = conn.execute("CALL page_rank('g') RETURN count(*) AS n")
        else:
            res = None
            for col in ("component_id", "component"):
                try:
                    res = conn.execute(
                        f"CALL weakly_connected_components('g') "
                        f"RETURN count(DISTINCT {col}) AS n"
                    )
                    break
                except Exception:
                    res = None
            if res is None:
                res = conn.execute(
                    "CALL weakly_connected_components('g') RETURN count(*) AS n"
                )
        n_out = None
        # kuzu returns a QueryResult with has_next()/get_next(); its bundled stubs
        # type the value as a list, so hold it as Any instead of fighting the stub.
        res_any: Any = res
        while res_any.has_next():
            n_out = res_any.get_next()[0]
        seconds = time.time() - t0
        return {
            "engine": "kuzu",
            "workload": "graph",
            "available": True,
            "algorithm": algo,
            "edges": len(edges),
            "load_s": round(load_s, 3),
            "seconds": round(seconds, 3),
            "result_size": n_out,
            "layer": "embedded (in-process C++)",
        }
    except Exception as exc:
        return {"engine": "kuzu", "workload": "graph", "available": False,
                "error": str(exc)[:200]}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# How each engine's graph latency is measured, so the report can state that
# Neo4j/Memgraph run the algorithm server-side in a native engine, while the
# others are in-process libraries.
_GRAPH_LAYER = {
    "neo4j": "neo4j GDS (native JVM)",
    "memgraph": "memgraph MAGE (native engine)",
    "benodb": "embedded (in-process Rust)",
    "bdb": "embedded (in-process Rust)",
    "networkx": "embedded (in-process Python)",
    "kuzu": "embedded (in-process C++)",
    "cugraph": "embedded (in-process GPU)",
}


def run_graph(args) -> dict:
    """Dispatch a graph engine, tagging dataset + measurement layer + backend."""
    res = _graph_dispatch(args)
    if isinstance(res, dict):
        res.setdefault("dataset", getattr(args, "dataset", None) or "graph")
        # Only cuGraph executes on the GPU; the other graph engines are CPU
        # regardless of which container launched the run.
        res.setdefault("device", "gpu" if res.get("engine") == "cugraph" else "cpu")
        if res.get("available"):
            res.setdefault("layer", _GRAPH_LAYER.get(res.get("engine", ""), "unknown"))
    return res


def _graph_dispatch(args) -> dict:
    if not args.graph_edges:
        return {
            "workload": "graph",
            "available": False,
            "error": "graph engines require --graph-edges",
        }
    edges = np.loadtxt(args.graph_edges, dtype=np.int64, ndmin=2)
    engine = args.engine
    if engine in ("benodb", "bdb"):
        return _graph_benodb(args, edges)
    if engine == "cugraph":
        return _graph_cugraph(args, edges)
    if engine == "memgraph":
        return _graph_memgraph(args, edges)
    if engine == "kuzu":
        return _graph_kuzu(args, edges)
    if engine == "neo4j":
        if not importlib.util.find_spec("neo4j"):
            return {
                "engine": "neo4j",
                "workload": "graph",
                "available": False,
                "error": "pip install neo4j (and run Neo4j with the GDS plugin)",
            }
        return _graph_neo4j(args, edges)
    if engine == "networkx":
        if not importlib.util.find_spec("networkx"):
            return {"engine": "networkx", "workload": "graph", "available": False}
        return _graph_networkx(args, edges)
    return {
        "engine": engine,
        "workload": "graph",
        "available": False,
        "error": f"unknown graph engine '{engine}'",
    }


# --------------------------------------------------------------------------- #
# Lexical workload
# --------------------------------------------------------------------------- #
def _lexical_tantivy(args, corpus, queries, qrels) -> dict:
    import tantivy
    import tempfile

    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("id", stored=True)
    schema_builder.add_text_field("text", stored=True, tokenizer_name="en_stem")
    schema = schema_builder.build()

    t0 = time.time()
    with tempfile.TemporaryDirectory() as index_dir:
        index = tantivy.Index(schema, path=index_dir)
        writer = index.writer(heap_size=1024_000_000)

        for doc_id, text in corpus:
            writer.add_document(tantivy.Document(
                id=[doc_id],
                text=[text],
            ))
        writer.commit()
        build_s = time.time() - t0

        index.reload()
        searcher = index.searcher()

        hits = 0.0
        q = min(args.queries, len(queries))
        latencies = []

        import re
        for q_id, q_text in queries[:q]:
            # Sanitize to avoid tantivy syntax errors
            safe_q = re.sub(r'[^\w\s]', ' ', q_text)
            query = index.parse_query(safe_q, ["text"])

            t1 = time.perf_counter()
            results = searcher.search(query, args.k).hits
            latencies.append(time.perf_counter() - t1)

            got = {searcher.doc(doc_address)["id"][0] for score, doc_address in results}
            truth = qrels.get(q_id, set())
            if truth:
                hits += len(got & truth) / min(args.k, len(truth))

        recall = hits / q
        total = sum(latencies) or 1e-9

        return {
            "engine": "tantivy",
            "workload": "lexical",
            "available": True,
            "queries": q,
            "build_s": round(build_s, 3),
            "recall_at_k": round(recall, 4),
            "qps": round(q / total, 1),
            "p50_ms": round(float(np.percentile(latencies, 50) * 1000), 3),
            "p99_ms": round(float(np.percentile(latencies, 99) * 1000), 3),
        }

def _lexical_opensearch(args, corpus, queries, qrels) -> dict:
    from opensearchpy import OpenSearch, helpers
    host = args.host or os.environ.get("ES_URL") or "http://localhost:9200"
    es = OpenSearch(host)

    # 1. Build index
    # `ignore_unavailable` is valid for the REST API but missing from the client stubs.
    es.indices.delete(index="bdb-lexical", ignore_unavailable=True)  # type: ignore[call-arg]
    es.indices.create(
        index="bdb-lexical",
        body={
            "mappings": {
                "properties": {
                    "text": {"type": "text", "analyzer": "english"}
                }
            }
        },
    )
    
    t0 = time.time()
    actions = (
        {
            "_index": "bdb-lexical",
            "_id": doc_id,
            "_source": {"text": text},
        }
        for doc_id, text in corpus
    )
    helpers.bulk(es, actions)
    es.indices.refresh(index="bdb-lexical")
    build_s = time.time() - t0

    # 2. Search
    hits = 0.0
    q = min(args.queries, len(queries))
    latencies = []
    
    import re
    for q_id, q_text in queries[:q]:
        safe_q = re.sub(r'[^\w\s]', ' ', q_text)
        
        t1 = time.perf_counter()
        res = es.search(
            index="bdb-lexical",
            body={
                "query": {"match": {"text": safe_q}},
                "size": args.k,
                "_source": False,
            }
        )
        latencies.append(time.perf_counter() - t1)
        
        got = {str(h["_id"]) for h in res["hits"]["hits"]}
        truth = qrels.get(q_id, set())
        if truth:
            hits += len(got & truth) / min(args.k, len(truth))

    recall = hits / q
    total = sum(latencies) or 1e-9

    return {
        "engine": args.engine,
        "workload": "lexical",
        "available": True,
        "queries": q,
        "build_s": round(build_s, 3),
        "recall_at_k": round(recall, 4),
        "qps": round(q / total, 1),
        "p50_ms": round(float(np.percentile(latencies, 50) * 1000), 3),
        "p99_ms": round(float(np.percentile(latencies, 99) * 1000), 3),
    }

def run_lexical(args) -> dict:
    if not args.beir_corpus:
        return {
            "workload": "lexical",
            "available": False,
            "error": "lexical engines require --beir-corpus",
        }

    engine = args.engine
    if engine in ("tantivy", "opensearch"):
        if engine == "tantivy" and not importlib.util.find_spec("tantivy"):
            return {
                "engine": "tantivy",
                "workload": "lexical",
                "available": False,
                "error": "pip install tantivy",
            }
        elif engine == "opensearch" and not importlib.util.find_spec("opensearchpy"):
            return {
                "engine": engine,
                "workload": "lexical",
                "available": False,
                "error": "pip install opensearch-py",
            }

        corpus = []
        with open(args.beir_corpus, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                corpus.append((d["_id"], d.get("title", "") + " " + d.get("text", "")))

        qrels = {}
        with open(args.beir_qrels, "r", encoding="utf-8") as f:
            next(f) # skip header
            for line in f:
                q_id, doc_id, score = line.strip().split('\t')
                if int(score) > 0:
                    qrels.setdefault(q_id, set()).add(doc_id)

        queries = []
        with open(args.beir_queries, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if str(d["_id"]) in qrels:
                    queries.append((str(d["_id"]), d["text"]))

        if engine == "tantivy":
            return _lexical_tantivy(args, corpus, queries, qrels)
        else:
            return _lexical_opensearch(args, corpus, queries, qrels)

    return {
        "engine": engine,
        "workload": "lexical",
        "available": False,
        "error": f"unsupported lexical engine {engine}"
    }

# --------------------------------------------------------------------------- #
# SQL workload
# --------------------------------------------------------------------------- #
def _sql_duckdb(args) -> dict:
    if not importlib.util.find_spec("duckdb"):
        return {
            "engine": "duckdb",
            "workload": "sql",
            "available": False,
            "error": "pip install duckdb",
        }
    import duckdb

    sql = args.sql
    if not sql:
        if not args.parquet:
            return {
                "engine": "duckdb",
                "workload": "sql",
                "available": False,
                "error": "provide --sql or --parquet",
            }
        sql = (
            "SELECT count(*) AS n, avg(WatchID) AS avg_id "
            f"FROM read_parquet('{args.parquet}')"
        )
    con = duckdb.connect()
    # Match the shared envelope. DuckDB otherwise defaults to every host core
    # and ~80% of host RAM, over-subscribing next to the cgroup-limited engines.
    if getattr(args, "cores", 0):
        con.execute(f"SET threads = {int(args.cores)}")
    if getattr(args, "ram_gb", 0.0):
        con.execute(f"SET memory_limit = '{float(args.ram_gb)}GB'")
    if args.parquet and "from t" in sql.lower() and "read_parquet" not in sql.lower():
        con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{args.parquet}')")
    t0 = time.time()
    result = con.execute(sql).fetchall()
    return {
        "engine": "duckdb",
        "workload": "sql",
        "available": True,
        "sql": sql,
        "seconds": round(time.time() - t0, 3),
        "rows": len(result),
    }


def _sql_datafusion(args) -> dict:
    if not importlib.util.find_spec("datafusion"):
        return {
            "engine": "datafusion",
            "workload": "sql",
            "available": False,
            "error": "pip install datafusion",
        }
    import datafusion

    # Match the shared envelope: DataFusion defaults to one partition per host
    # core, which over-subscribes inside the cgroup-limited runner.
    ctx = datafusion.SessionContext()
    if getattr(args, "cores", 0):
        try:
            cfg = datafusion.SessionConfig().with_target_partitions(int(args.cores))
            ctx = datafusion.SessionContext(cfg)
        except Exception:
            pass
    sql = args.sql
    if not sql:
        if not args.parquet:
            return {
                "engine": "datafusion",
                "workload": "sql",
                "available": False,
                "error": "provide --sql or --parquet",
            }
        ctx.register_parquet("t", args.parquet)
        sql = "SELECT count(*) AS n FROM t"
    elif args.parquet:
        ctx.register_parquet("t", args.parquet)

    t0 = time.time()
    df = ctx.sql(sql)
    batches = df.collect()
    total_rows = sum(b.num_rows for b in batches)
    return {
        "engine": "datafusion",
        "workload": "sql",
        "available": True,
        "sql": sql,
        "seconds": round(time.time() - t0, 3),
        "rows": total_rows,
    }


def _sql_clickhouse(args) -> dict:
    if not importlib.util.find_spec("clickhouse_connect"):
        return {
            "engine": "clickhouse",
            "workload": "sql",
            "available": False,
            "error": "pip install clickhouse_connect",
        }
    import clickhouse_connect

    host = getattr(args, "host", None) or os.environ.get("CLICKHOUSE_HOST", "localhost")
    port = int(getattr(args, "port", None) or os.environ.get("CLICKHOUSE_PORT", "8123"))
    user = getattr(args, "user", None) or os.environ.get("CLICKHOUSE_USER", "default")
    password = getattr(args, "password", None) or os.environ.get("CLICKHOUSE_PASSWORD", "")
    try:
        client = clickhouse_connect.get_client(host=host, port=port, username=user, password=password)
    except Exception as e:
        return {
            "engine": "clickhouse",
            "workload": "sql",
            "available": False,
            "error": f"connection error: {e}",
        }

    sql = args.sql
    if args.parquet:
        # Materialise the Parquet into a table named `t` so the shared SQL
        # (which references `t`) works across every SQL engine. The schema is
        # taken from the Parquet itself (ClickBench, TPC-H, NYC TLC, ... all
        # differ). Load via the client (not `file()`) so ClickHouse needs no
        # data volume — its entrypoint chowns any mounted dir.
        import pyarrow as pa
        import pyarrow.parquet as pq

        def _ch_type(t, nullable: bool) -> str:
            if pa.types.is_int64(t):
                base = "Int64"
            elif pa.types.is_int32(t):
                base = "Int32"
            elif pa.types.is_int16(t):
                base = "Int16"
            elif pa.types.is_float64(t):
                base = "Float64"
            elif pa.types.is_float32(t):
                base = "Float32"
            elif pa.types.is_boolean(t):
                base = "UInt8"
            elif pa.types.is_date32(t):
                base = "Date32"
            elif pa.types.is_date64(t):
                base = "Date"
            elif pa.types.is_timestamp(t):
                base = "DateTime64(6)"
            elif pa.types.is_decimal(t):
                base = f"Decimal({t.precision},{t.scale})"
            else:
                base = "String"
            # A nullable Parquet column (e.g. NYC TLC's `passenger_count`) must
            # map to a Nullable ClickHouse type, or the client cannot build the
            # native array for the NULL values.
            return f"Nullable({base})" if nullable else base

        tbl = pq.read_table(args.parquet)
        cols = tbl.schema.names
        col_defs = ", ".join(
            f"`{c}` {_ch_type(tbl.schema.field(c).type, tbl.schema.field(c).nullable)}"
            for c in cols
        )
        client.command("DROP TABLE IF EXISTS t")
        client.command(f"CREATE TABLE t ({col_defs}) ENGINE = MergeTree ORDER BY tuple()")
        client.insert(
            "t",
            list(zip(*[tbl.column(c).to_pylist() for c in cols])),
            column_names=cols,
        )
    if not sql:
        if not args.parquet:
            return {
                "engine": "clickhouse",
                "workload": "sql",
                "available": False,
                "error": "provide --sql or --parquet",
            }
        sql = "SELECT count(*) AS n FROM t"

    t0 = time.time()
    res = client.query(sql)
    return {
        "engine": "clickhouse",
        "workload": "sql",
        "available": True,
        "sql": sql,
        "seconds": round(time.time() - t0, 3),
        "rows": len(res.result_rows),
    }


def _sql_trino(args) -> dict:
    if not importlib.util.find_spec("trino"):
        return {
            "engine": "trino",
            "workload": "sql",
            "available": False,
            "error": "pip install trino",
        }
    from trino.dbapi import connect

    host = getattr(args, "host", None) or os.environ.get("TRINO_HOST", "localhost")
    port = int(getattr(args, "port", None) or os.environ.get("TRINO_PORT", "8080"))
    user = getattr(args, "user", None) or os.environ.get("TRINO_USER", "bench")
    catalog = getattr(args, "catalog", None) or os.environ.get("TRINO_CATALOG", "memory")
    schema = getattr(args, "schema", None) or os.environ.get("TRINO_SCHEMA", "default")
    # A Parquet dataset is registered as an external Hive table (the memory
    # connector cannot bulk-load large datasets), so query through `hive`.
    if args.parquet:
        catalog = "hive"

    try:
        conn = connect(host=host, port=port, user=user, catalog=catalog, schema=schema)
    except Exception as e:
        return {
            "engine": "trino",
            "workload": "sql",
            "available": False,
            "error": f"connection error: {e}",
        }

    cur = conn.cursor()
    ingest_s = None
    if args.parquet:
        # Register the shared Parquet as an external Hive table `t` (the memory
        # connector cannot bulk-load large datasets — it OOMs). The schema is
        # taken from the Parquet itself (ClickBench / TPC-H / NYC TLC differ).
        # The runner and Trino mount the same data dir at the same path, so the
        # `file://` location resolves inside Trino.
        import pyarrow as pa
        import pyarrow.parquet as pq

        def _trino_type(t) -> str:
            if pa.types.is_boolean(t):
                return "boolean"
            if pa.types.is_int64(t):
                return "bigint"
            if pa.types.is_int32(t):
                return "integer"
            if pa.types.is_int16(t):
                return "smallint"
            if pa.types.is_float64(t):
                return "double"
            if pa.types.is_float32(t):
                return "real"
            if pa.types.is_date(t):
                return "date"
            if pa.types.is_timestamp(t):
                return "timestamp"
            if pa.types.is_decimal(t):
                return f"decimal({t.precision},{t.scale})"
            return "varchar"

        t_ingest = time.time()
        tbl = pq.read_table(args.parquet)
        cols = tbl.schema.names
        col_defs = ", ".join(
            f'"{c}" {_trino_type(tbl.schema.field(c).type)}' for c in cols
        )
        location = "file://" + os.path.dirname(args.parquet)
        cur.execute("CREATE SCHEMA IF NOT EXISTS hive.default")
        cur.execute("DROP TABLE IF EXISTS hive.default.t")
        cur.execute(
            f"CREATE TABLE hive.default.t ({col_defs}) "
            f"WITH (format = 'PARQUET', external_location = '{location}')"
        )
        ingest_s = round(time.time() - t_ingest, 3)

    sql = args.sql
    if not sql:
        if not args.parquet:
            return {
                "engine": "trino",
                "workload": "sql",
                "available": False,
                "error": "provide --sql or --parquet",
            }
        sql = "SELECT count(*) AS n FROM t"

    t0 = time.time()
    cur.execute(sql)
    rows = cur.fetchall()
    return {
        "engine": "trino",
        "workload": "sql",
        "available": True,
        "sql": sql,
        "ingest_seconds": ingest_s,
        "seconds": round(time.time() - t0, 3),
        "rows": len(rows),
    }


def _sql_benodb(args) -> dict:
    import benodb as bdb
    import pyarrow.parquet as pq
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="benodb_sql_bench_")
    try:
        t_ingest_0 = time.time()
        table = bdb.Table(tmpdir)
        if args.parquet:
            pa_table = pq.read_table(args.parquet)
            table.write(pa_table)
            table.commit()
            table.wait_for_background_tasks()
        ingest_s = round(time.time() - t_ingest_0, 3)

        sql = args.sql or "SELECT count(*) FROM t"
        t0 = time.time()
        res = table.sql(sql)
        elapsed = round(time.time() - t0, 3)
        return {
            "engine": "benodb",
            "workload": "sql",
            "available": True,
            "sql": sql,
            "ingest_seconds": ingest_s,
            "seconds": elapsed,
            "rows": len(res),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_sql(args) -> dict:
    engine = args.engine
    if engine == "duckdb":
        rec = _sql_duckdb(args)
    elif engine == "datafusion":
        rec = _sql_datafusion(args)
    elif engine == "clickhouse":
        rec = _sql_clickhouse(args)
    elif engine == "trino":
        rec = _sql_trino(args)
    elif engine in ("benodb", "bdb"):
        rec = _sql_benodb(args)
    else:
        return {
            "engine": engine,
            "workload": "sql",
            "available": False,
            "error": f"unknown sql engine '{engine}'",
        }
    # The per-engine helpers do not know the dataset name; carry it through so
    # the report can group SQL rows by dataset (synth_500000, tpch_lineitem, …).
    rec.setdefault("dataset", getattr(args, "dataset", None))
    return rec


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def _total_ram_gb() -> Optional[float]:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page / 1024**3, 1)
    except Exception:
        return None


def _cgroup_mem_gb() -> Optional[float]:
    """Container memory limit from cgroup v2/v1 (None when unconstrained)."""
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path, encoding="utf-8") as f:
                raw = f.read().strip()
            if raw and raw != "max" and int(raw) < (1 << 62):
                return round(int(raw) / 1024**3, 1)
        except Exception:
            continue
    return None


def _cgroup_cpus() -> Optional[float]:
    """Container CPU limit from cgroup v2 ``cpu.max`` / v1 quota (None if none)."""
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as f:
            quota, period = f.read().split()
        if quota != "max":
            return round(int(quota) / int(period), 2)
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="utf-8") as f:
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="utf-8") as f:
            period = int(f.read().strip())
        if quota > 0 and period > 0:
            return round(quota / period, 2)
    except Exception:
        pass
    return None


def _gpu_info() -> list[str]:
    """Visible NVIDIA GPUs (empty in a CPU-only run)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:
        pass
    return []


def emit(record: dict, out: Optional[str]) -> None:
    """Attach the hardware profile so every result is self-describing.

    ``cores``/``ram_gb`` are the *cgroup-visible* limits when running in Docker
    (`os.cpu_count()` and `SC_PHYS_PAGES` reflect the container's limits), so
    the recorded envelope matches what the engine actually saw.
    """
    record.setdefault("env", {})
    record["env"].setdefault("cpu_model", _cpu_model())
    # Report the *cgroup* envelope the engine actually saw, not the host totals.
    record["env"].setdefault("cores", _cgroup_cpus() or os.cpu_count())
    record["env"].setdefault("ram_gb", _cgroup_mem_gb() or _total_ram_gb())
    record["env"].setdefault("host_cores", os.cpu_count())
    record["env"].setdefault("host_ram_gb", _total_ram_gb())
    record["env"].setdefault("os", platform.platform())
    record["env"].setdefault("python", platform.python_version())
    record["env"].setdefault("containerized", os.path.exists("/.dockerenv"))
    record["env"].setdefault("gpus", _gpu_info())
    print(json.dumps(record, indent=2), flush=True)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        with open(out.replace(".json", ".md"), "w", encoding="utf-8") as f:
            f.write(f"# Competitor: {record.get('engine')} / {record.get('workload')}\n\n")
            f.write("| Metric | Value |\n|---|---|\n")
            for key, val in record.items():
                if isinstance(val, dict):
                    val = ", ".join(f"{k}={v}" for k, v in val.items())
                f.write(f"| {key} | {val} |\n")
        print(f"wrote {out}", flush=True)


def fail(engine: str, exc: Exception) -> dict:
    return {
        "engine": engine,
        "available": False,
        "error": f"{type(exc).__name__}: {exc}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--dataset", default="sift-128-euclidean")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", type=int, default=200)
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--nprobe", type=int, default=0)
    ap.add_argument("--pq-m", type=int, default=16)
    ap.add_argument("--pq-nbits", type=int, default=8)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--mmap-invlists", action="store_true")
    ap.add_argument("--metric", default=None)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--cores", type=int, default=0)
    ap.add_argument("--ram-gb", type=float, default=0.0)
    # Server / path connection options.
    ap.add_argument("--dsn", default=None, help="pgvector connection string")
    ap.add_argument("--host", default=None, help="OpenSearch URL")
    ap.add_argument("--path", default=None, help="LanceDB directory")
    ap.add_argument("--uri", default=None, help="Neo4j bolt URI")
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    # Graph / SQL / Lexical options.
    ap.add_argument("--workload", default=None, choices=["vector", "graph", "sql", "lexical"])
    ap.add_argument("--graph-edges", default=None)
    ap.add_argument("--beir-corpus", default=None)
    ap.add_argument("--beir-queries", default=None)
    ap.add_argument("--beir-qrels", default=None)
    ap.add_argument("--algorithm", default="pagerank")
    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--damping", type=float, default=0.85)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--sql", default=None)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--data-dir", default=None, help="Directory containing binary dataset files (e.g. bigann)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    apply_envelope(args.cores, args.ram_gb, args.engine, args.device)

    try:
        if args.workload == "sql" or args.engine in ("duckdb", "datafusion", "clickhouse", "trino") or (args.engine in ("benodb", "bdb") and (args.sql or (args.parquet and not args.dataset))):
            emit(run_sql(args), args.out)
            return
        if args.workload == "graph" or args.engine in ("networkx", "neo4j", "memgraph", "kuzu", "cugraph") or (args.engine in ("benodb", "bdb") and args.graph_edges):
            emit(run_graph(args), args.out)
            return
        if args.workload == "lexical" or args.engine == "tantivy" or (args.engine == "opensearch" and args.beir_corpus):
            emit(run_lexical(args), args.out)
            return

        adapters = vector_adapters()
        adapter = adapters.get(args.engine)
        if adapter is None:
            raise SystemExit(f"unknown engine '{args.engine}' (have: {sorted(adapters)})")
        if not adapter.available():
            print(
                json.dumps(
                    {
                        "engine": args.engine,
                        "available": False,
                        "error": f"'{adapter.module}' is not installed — "
                        f"`pip install {adapter.module}`",
                    },
                    indent=2,
                )
            )
            return
        emit(run_vector(adapter, args), args.out)
    except Exception as exc:  # server/setup failures: report, do not crash
        emit(fail(args.engine, exc), args.out)


if __name__ == "__main__":
    main()
