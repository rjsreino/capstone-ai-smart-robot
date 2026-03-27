#!/bin/bash
# Start ROVY Cloud Server (runs on PC)
# This handles:
# - REST API (port 8000) - for mobile app
# - WebSocket (port 8765) - for robot communication
# - AI processing: LLM, Vision, STT, TTS

cd "$(dirname "$0")"

echo "================================"
echo "  ROVY CLOUD SERVER (PC)"
echo "================================"
echo ""
echo "This runs on your PC with GPU"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found"
    exit 1
fi

# Get IP
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$IP" ]; then
    IP="localhost"
fi

echo "Starting Cloud Server..."
echo ""
echo "  REST API:    http://$IP:8000"
echo "  WebSocket:   ws://$IP:8765"
echo "  Tailscale:   http://100.121.110.125:8000"
echo ""
echo "Logs will appear below:"
echo ""

python3 main.py

