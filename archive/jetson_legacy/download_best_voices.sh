#!/bin/bash
# Download the MOST NATURAL Piper voices based on community feedback
# These are rated as the most human-like and natural sounding

VOICE_DIR="$HOME/.local/share/piper-voices"
mkdir -p "$VOICE_DIR"

echo "========================================"
echo "Downloading BEST Natural Piper Voices"
echo "========================================"
echo ""

BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US"

# 1. Danny - One of the most natural male voices (British-leaning American)
echo "[1/4] Downloading Danny (low) - Most natural male voice..."
if [ ! -f "$VOICE_DIR/en_US-danny-low.onnx" ]; then
    wget -q --show-progress "$BASE_URL/danny/low/en_US-danny-low.onnx" -O "$VOICE_DIR/en_US-danny-low.onnx"
    wget -q "$BASE_URL/danny/low/en_US-danny-low.onnx.json" -O "$VOICE_DIR/en_US-danny-low.onnx.json"
    echo "✅ Danny downloaded"
else
    echo "✅ Danny already exists"
fi

# 2. Libritts_r - Improved LibriTTS (more natural, less robotic)
echo ""
echo "[2/4] Downloading LibriTTS_r (high) - Improved natural voice..."
if [ ! -f "$VOICE_DIR/en_US-libritts_r-medium.onnx" ]; then
    wget -q --show-progress "$BASE_URL/libritts_r/medium/en_US-libritts_r-medium.onnx" -O "$VOICE_DIR/en_US-libritts_r-medium.onnx" 2>/dev/null || echo "LibriTTS_r not available"
    wget -q "$BASE_URL/libritts_r/medium/en_US-libritts_r-medium.onnx.json" -O "$VOICE_DIR/en_US-libritts_r-medium.onnx.json" 2>/dev/null
fi

# 3. Kristin - Very natural female voice (in case you want variety)
echo ""
echo "[3/4] Downloading Kristin (medium) - Natural female..."
if [ ! -f "$VOICE_DIR/en_US-kristin-medium.onnx" ]; then
    wget -q --show-progress "$BASE_URL/kristin/medium/en_US-kristin-medium.onnx" -O "$VOICE_DIR/en_US-kristin-medium.onnx" 2>/dev/null || echo "Kristin not available"
    wget -q "$BASE_URL/kristin/medium/en_US-kristin-medium.onnx.json" -O "$VOICE_DIR/en_US-kristin-medium.onnx.json" 2>/dev/null
fi

# 4. Norman - Deep male voice, professional
echo ""
echo "[4/4] Downloading Norman (medium) - Deep professional male..."
if [ ! -f "$VOICE_DIR/en_US-norman-medium.onnx" ]; then
    wget -q --show-progress "$BASE_URL/norman/medium/en_US-norman-medium.onnx" -O "$VOICE_DIR/en_US-norman-medium.onnx" 2>/dev/null || echo "Norman not available"
    wget -q "$BASE_URL/norman/medium/en_US-norman-medium.onnx.json" -O "$VOICE_DIR/en_US-norman-medium.onnx.json" 2>/dev/null
fi

echo ""
echo "========================================"
echo "✅ Download Complete!"
echo "========================================"
echo ""
echo "📊 All installed voices:"
ls -lh "$VOICE_DIR"/*.onnx 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "🎤 RECOMMENDED: Danny (low) - Most natural male voice"
echo ""
echo "Test with: python3 test_tts_compare.py"

