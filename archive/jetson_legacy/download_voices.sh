#!/bin/bash
# Download additional Piper voices for testing
# These are male voices that may sound more natural

VOICE_DIR="$HOME/.local/share/piper-voices"
mkdir -p "$VOICE_DIR"

echo "========================================"
echo "Downloading Additional Piper Voices"
echo "========================================"
echo ""

BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US"

# Download Joe (medium quality, natural male voice)
echo "[1/3] Downloading Joe (medium)..."
if [ ! -f "$VOICE_DIR/en_US-joe-medium.onnx" ]; then
    wget -q --show-progress "$BASE_URL/joe/medium/en_US-joe-medium.onnx" -O "$VOICE_DIR/en_US-joe-medium.onnx"
    wget -q "$BASE_URL/joe/medium/en_US-joe-medium.onnx.json" -O "$VOICE_DIR/en_US-joe-medium.onnx.json"
    echo "✅ Joe downloaded"
else
    echo "✅ Joe already exists"
fi

# Download HFC Male (medium quality, clear male voice)
echo ""
echo "[2/3] Downloading HFC Male (medium)..."
if [ ! -f "$VOICE_DIR/en_US-hfc_male-medium.onnx" ]; then
    wget -q --show-progress "$BASE_URL/hfc_male/medium/en_US-hfc_male-medium.onnx" -O "$VOICE_DIR/en_US-hfc_male-medium.onnx"
    wget -q "$BASE_URL/hfc_male/medium/en_US-hfc_male-medium.onnx.json" -O "$VOICE_DIR/en_US-hfc_male-medium.onnx.json"
    echo "✅ HFC Male downloaded"
else
    echo "✅ HFC Male already exists"
fi

# Download Ryan (high quality, natural male voice)
echo ""
echo "[3/3] Downloading Ryan (high)..."
if [ ! -f "$VOICE_DIR/en_US-ryan-high.onnx" ]; then
    wget -q --show-progress "$BASE_URL/ryan/high/en_US-ryan-high.onnx" -O "$VOICE_DIR/en_US-ryan-high.onnx"
    wget -q "$BASE_URL/ryan/high/en_US-ryan-high.onnx.json" -O "$VOICE_DIR/en_US-ryan-high.onnx.json"
    echo "✅ Ryan downloaded"
else
    echo "✅ Ryan already exists"
fi

echo ""
echo "========================================"
echo "✅ Download Complete!"
echo "========================================"
echo ""
echo "Installed voices:"
ls -lh "$VOICE_DIR"/*.onnx | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Test with: python3 test_tts_compare.py"

