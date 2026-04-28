#!/bin/bash
# =============================================================================
# NAWSCL Range Surveillance - Aircraft Classification - Start Script
# Detects host IP and launches the Docker container
# =============================================================================

set -e

# -- Pre-flight checks -------------------------------------------------------

# Remove existing container if present
if docker ps -a --format '{{.Names}}' | grep -q "^nawscl-aircraft-classification$"; then
    echo "Removing existing container..."
    docker rm -f nawscl-aircraft-classification
fi

# Check that model exists
MODEL_DIR="./models/Qwen3-VL-8B-Instruct-FP8"
if [ ! -d "$MODEL_DIR" ]; then
    echo ""
    echo "Model not found: $MODEL_DIR"
    echo ""
    echo "   Download the model first:"
    echo "     ./download_models.sh"
    echo ""
    exit 1
fi

# Check Docker is running
if ! docker info &>/dev/null; then
    echo ""
    echo "Docker daemon is not running."
    echo "   Start it with: sudo systemctl start docker"
    echo ""
    exit 1
fi

# -- Detect host LAN IP ------------------------------------------------------

if [ -z "$HOST_IP" ]; then
    HOST_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')
fi
if [ -z "$HOST_IP" ]; then
    HOST_IP=$(hostname -I | awk '{print $1}')
fi

export HOST_IP

echo ""
echo "============================================================"
echo "  NAWSCL Range Surveillance - Aircraft Classification"
echo "  Qwen3-VL-8B-Instruct-FP8 | HP ZGX Nano | vLLM"
echo "============================================================"
echo "  Host IP: $HOST_IP"
echo ""
echo "  Demo:    http://$HOST_IP:8000"
echo "  Health:  http://$HOST_IP:8000/api/health"
echo ""
echo "  Note: First startup takes 2-3 minutes for model loading."
echo "============================================================"
echo ""

# Pass all arguments through (e.g., --build, -d)
docker compose up "$@"
