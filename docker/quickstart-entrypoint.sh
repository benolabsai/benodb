#!/bin/bash
# ---------------------------------------------------------------------------
# BenoDB Quickstart Entrypoint
# Launches both Search (ES 7.10 + Qdrant) and Flight SQL services
# ---------------------------------------------------------------------------
set -e

echo "============================================="
echo "  BenoDB Quickstart"
echo "  ES 7.10 API:     http://${BDB_SEARCH_BIND:-0.0.0.0}:${BDB_SEARCH_PORT:-9200}"
echo "  Qdrant API:      http://${BDB_QDRANT_BIND:-0.0.0.0}:${BDB_QDRANT_PORT:-6333}"
echo "  Flight SQL gRPC: grpc://${BDB_FLIGHT_BIND:-0.0.0.0}:${BDB_FLIGHT_PORT:-50051}"
echo "============================================="

# Launch Flight SQL server in background
echo "[quickstart] Starting Flight SQL server on :${BDB_FLIGHT_PORT:-50051}..."
benodb-flight &
FLIGHT_PID=$!

# Launch Search server in background
echo "[quickstart] Starting Search server on :${BDB_SEARCH_PORT:-9200} (ES) + :${BDB_QDRANT_PORT:-6333} (Qdrant)..."
bdb-search &
SEARCH_PID=$!

# Trap signals to shut down both processes gracefully
cleanup() {
    local code="${1:-0}"
    echo "[quickstart] Shutting down..."
    kill $FLIGHT_PID 2>/dev/null || true
    kill $SEARCH_PID 2>/dev/null || true
    wait $FLIGHT_PID 2>/dev/null || true
    wait $SEARCH_PID 2>/dev/null || true
    exit "$code"
}
trap 'cleanup 0' SIGTERM SIGINT

# Wait for either process to exit
EXIT_STATUS=0
wait -n $FLIGHT_PID $SEARCH_PID || EXIT_STATUS=$?

echo "[quickstart] A service exited with status ${EXIT_STATUS}."
cleanup "$EXIT_STATUS"
