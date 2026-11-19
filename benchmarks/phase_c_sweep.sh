#!/usr/bin/env bash
#
# Phase C: rerank I/O arms on the BigANN 10M set.
#
# Every arm is measured cold, one query in flight, because a resident table
# makes the physical read counters flat and hides the page skip.
#
# Arms:
#   coarse   128 MB pages, default row groups. Page skipping cannot help, so
#            this is the control that still rescores.
#   fine     32 KB pages, default row groups. The claim under test.
#   smallrg  32 KB pages, 960,000 row groups, about 128 MB at 136 B per row.
#   full     32 KB pages, selection disabled. Byte control only: without a
#            compacted candidate array the rescore returns graph order, so its
#            recall is recorded but never compared.
#
# Candidate counts come from --refine-factor: k=10 at 1, 4, and 16 gives 10,
# 40, and 160 candidates per query.
#
# Each writer arm ingests and builds once, then the two remaining candidate
# counts reuse the index through the manifest guard.

set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
HARNESS=${HARNESS:-benchmarks/run_neurips_bigann.py}
# Written outside the repository on purpose: a result directory inside the tree
# makes `git status` dirty, and the harness refuses to stamp a run from a dirty
# tree. Copy an arm into the repository only when it is being published.
OUT=${OUT:-/home/ralbright/data/benchmarks/phase_c}
QUERIES=${QUERIES:-20}
REPEATS=${REPEATS:-3}
LIMIT=${LIMIT:-10000000}
GRID=${GRID:-64/8}
MODES=${MODES:-tq8}

mkdir -p "$OUT"

build_and_measure() {
  local arm=$1 db=$2 page=$3 rg=$4 refine=$5
  echo "[phase-c] $arm build, refine=$refine"
  BDB_PARQUET_PAGE_BYTES="$page" \
  BDB_PARQUET_ROW_GROUP_ROWS="$rg" \
  BDB_READER_SKIP=1 \
  BDB_BENCH_EVICT=query \
  "$PY" "$HARNESS" \
    --profile local --dataset 10m --limit "$LIMIT" \
    --modes "$MODES" --grid "$GRID" \
    --refine-factor "$refine" --queries "$QUERIES" --repeat "$REPEATS" \
    --db-dir "$db" --out "$OUT/$arm"
}

query_only() {
  local arm=$1 db=$2 page=$3 rg=$4 refine=$5 skip=$6
  echo "[phase-c] $arm reuse, refine=$refine, skip=$skip"
  BDB_PARQUET_PAGE_BYTES="$page" \
  BDB_PARQUET_ROW_GROUP_ROWS="$rg" \
  BDB_READER_SKIP="$skip" \
  BDB_BENCH_EVICT=query \
  "$PY" "$HARNESS" query \
    --profile local --dataset 10m --limit "$LIMIT" \
    --modes "$MODES" --grid "$GRID" \
    --refine-factor "$refine" --queries "$QUERIES" --repeat "$REPEATS" \
    --db-dir "$db" --out "$OUT/$arm"
}

FINE_PAGE=32768
COARSE_PAGE=134217728
DEFAULT_RG=1048576
SMALL_RG=983040

[ "${REUSE:-0}" = "1" ] || rm -rf benchmarks/scratch_phasec/arms

# fine: the claim
if [ "${REUSE:-0}" = "1" ]; then
  query_only fine benchmarks/scratch_phasec/arms/fine "$FINE_PAGE" "$DEFAULT_RG" 4 1
else
  build_and_measure fine benchmarks/scratch_phasec/arms/fine "$FINE_PAGE" "$DEFAULT_RG" 4
fi
query_only fine benchmarks/scratch_phasec/arms/fine "$FINE_PAGE" "$DEFAULT_RG" 1 1
query_only fine benchmarks/scratch_phasec/arms/fine "$FINE_PAGE" "$DEFAULT_RG" 16 1

# coarse: control that still rescores
if [ "${REUSE:-0}" = "1" ]; then
  query_only coarse benchmarks/scratch_phasec/arms/coarse "$COARSE_PAGE" "$DEFAULT_RG" 4 1
else
  build_and_measure coarse benchmarks/scratch_phasec/arms/coarse "$COARSE_PAGE" "$DEFAULT_RG" 4
fi
query_only coarse benchmarks/scratch_phasec/arms/coarse "$COARSE_PAGE" "$DEFAULT_RG" 1 1
query_only coarse benchmarks/scratch_phasec/arms/coarse "$COARSE_PAGE" "$DEFAULT_RG" 16 1

# smallrg: 128 MB row groups with fine pages
if [ "${REUSE:-0}" = "1" ]; then
  query_only smallrg benchmarks/scratch_phasec/arms/smallrg "$FINE_PAGE" "$SMALL_RG" 4 1
else
  build_and_measure smallrg benchmarks/scratch_phasec/arms/smallrg "$FINE_PAGE" "$SMALL_RG" 4
fi
query_only smallrg benchmarks/scratch_phasec/arms/smallrg "$FINE_PAGE" "$SMALL_RG" 1 1
query_only smallrg benchmarks/scratch_phasec/arms/smallrg "$FINE_PAGE" "$SMALL_RG" 16 1

# full: byte control, no rescore, recall recorded but not compared
query_only full benchmarks/scratch_phasec/arms/fine "$FINE_PAGE" "$DEFAULT_RG" 4 0

echo "[phase-c] done: $OUT"
