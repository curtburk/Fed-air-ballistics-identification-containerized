#!/bin/bash
set -e
echo ""
echo "============================================================"
echo "  NAWSCL Range Surveillance - Starting"
echo "============================================================"
echo ""

APP_PORT=${PORT:-8000}
VLLM_URL=${VLLM_URL:-http://172.17.0.1:8091}

echo "  vLLM endpoint: $VLLM_URL"
echo "  Starting FastAPI application on port $APP_PORT..."
echo ""

cd /app
exec python3 -m uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port "$APP_PORT" \
    --log-level info
