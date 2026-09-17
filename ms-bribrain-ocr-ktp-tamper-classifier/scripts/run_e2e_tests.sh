#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run e2e tests against a live server.
#
# Usage (standalone):
#   ./scripts/run_e2e_tests.sh          # starts server, runs tests, stops server
#
# Usage (Docker build):
#   Called automatically by the e2e-test stage in the Dockerfile.
# ---------------------------------------------------------------------------
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8050}"
BASE_URL="http://127.0.0.1:${PORT}"
TIMEOUT="${STARTUP_TIMEOUT:-120}"

echo "▶ Starting tamper detection server on ${HOST}:${PORT} ..."
/app/.venv/bin/uvicorn src.main:app --host "$HOST" --port "$PORT" &
SERVER_PID=$!

# Wait for readiness
elapsed=0
until curl -sf "${BASE_URL}/health/live" > /dev/null 2>&1; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [ "$elapsed" -ge "$TIMEOUT" ]; then
        echo "✘ Server did not become ready within ${TIMEOUT}s"
        kill "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
done
echo "✔ Server ready (${elapsed}s)"

# Run e2e tests
echo "▶ Running e2e tests ..."
set +e
E2E_BASE_URL="${BASE_URL}" /app/.venv/bin/pytest test/e2e/ -m e2e -v --no-cov --tb=short 2>&1
TEST_EXIT=$?
set -e

# Cleanup
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

if [ "$TEST_EXIT" -ne 0 ]; then
    echo "✘ E2E tests failed (exit code ${TEST_EXIT})"
    exit "$TEST_EXIT"
fi

echo "✔ All e2e tests passed"
