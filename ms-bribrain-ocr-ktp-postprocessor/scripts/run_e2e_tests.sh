#!/bin/bash
# ---------------------------------------------------------------------------
# run_e2e_tests.sh
#
# Starts the postprocessor in the background, waits for it to be ready,
# runs e2e tests against it, then kills the server.
# Exit code reflects test outcome.
# ---------------------------------------------------------------------------

set -euo pipefail

HOST="${APP_HOST:-0.0.0.0}"
PORT="${APP_PORT:-8090}"
HEALTH_URL="http://127.0.0.1:${PORT}/health/ready"
MAX_WAIT=120       # seconds to wait
POLL_INTERVAL=3    # seconds between polls

echo "=========================================="
echo "  E2E Test Runner - OCR Postprocessor"
echo "=========================================="

# Load environment variables from .env (mounted secret).
# Use Python to parse .env so special chars (JSON keys, etc.) are handled correctly.
if [ -f /app/.env ]; then
    eval "$(python3 -c "
import shlex
with open('/app/.env') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] in ('\"', \"'\") and value[-1] == value[0]:
            value = value[1:-1]
        print(f'export {key}={shlex.quote(value)}')
")"
fi

# ------------------------------------------------------------------
# 1. Start the service in the background
# ------------------------------------------------------------------
echo "[1/4] Starting postprocessor on ${HOST}:${PORT} ..."

uv run uvicorn src.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --timeout-keep-alive 60 \
    --log-level warning &

SERVER_PID=$!

cleanup() {
    echo ""
    echo "[cleanup] Stopping service (PID ${SERVER_PID}) ..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ------------------------------------------------------------------
# 2. Wait until the readiness endpoint responds
# ------------------------------------------------------------------
echo "[2/4] Waiting for readiness at ${HEALTH_URL} (up to ${MAX_WAIT}s) ..."

elapsed=0
while [ $elapsed -lt $MAX_WAIT ]; do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        echo "       Service is ready (took ${elapsed}s)"
        break
    fi
    sleep $POLL_INTERVAL
    elapsed=$((elapsed + POLL_INTERVAL))
done

if [ $elapsed -ge $MAX_WAIT ]; then
    echo "ERROR: Service did not become ready within ${MAX_WAIT}s"
    exit 1
fi

# ------------------------------------------------------------------
# 3. Run e2e tests
# ------------------------------------------------------------------
echo "[3/4] Running e2e tests ..."

export E2E_BASE_URL="http://127.0.0.1:${PORT}"

uv run pytest tests/e2e/ \
    -m e2e \
    --tb=short \
    -q \
    --no-cov \
    -p no:cacheprovider \
    -x

TEST_EXIT=$?

# ------------------------------------------------------------------
# 4. Report result
# ------------------------------------------------------------------
echo ""
if [ $TEST_EXIT -eq 0 ]; then
    echo "=========================================="
    echo "  E2E TESTS PASSED"
    echo "=========================================="
else
    echo "=========================================="
    echo "  E2E TESTS FAILED (exit code: $TEST_EXIT)"
    echo "=========================================="
fi

exit $TEST_EXIT
