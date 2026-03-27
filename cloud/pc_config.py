"""
PC Server Configuration
Runs AI models, receives data from Raspberry Pi client
"""
import os

# ===========================================
# NETWORK
# ===========================================
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8765

# Tailscale IPs (for reference)
PC_SERVER_IP = "100.121.110.125"      # This PC
RASPBERRY_PI_IP = "100.72.107.106"    # Raspberry Pi client

# ===========================================
# LOCAL LLM MODELS (GGUF format)
# ===========================================
# Set via environment variables or edit paths below

# Text model (Gemma, Llama, Mistral, etc.)
TEXT_MODEL_PATH = os.getenv("ROVY_TEXT_MODEL", None)  # Auto-detect if None

# Vision model (LLaVA, Phi-3-Vision)
VISION_MODEL_PATH = os.getenv("ROVY_VISION_MODEL", None)
VISION_MMPROJ_PATH = os.getenv("ROVY_VISION_MMPROJ", None)

# GPU settings
N_GPU_LAYERS = -1  # -1 = all layers on GPU, 0 = CPU only
N_CTX = 2048       # Context window size

# ===========================================
# WHISPER (Speech Recognition)
# ===========================================
WHISPER_MODEL = "base"  # tiny, base, small, medium, large

# ===========================================
# TTS (Text-to-Speech)
# ===========================================
TTS_ENGINE = "espeak"  # espeak or piper
PIPER_VOICE_PATH = None  # Path to Piper .onnx voice file

# ===========================================
# FACE RECOGNITION
# ===========================================
KNOWN_FACES_DIR = "known_faces"
FACE_TOLERANCE = 0.6  # Lower = stricter matching

# ===========================================
# ASSISTANT
# ===========================================
ASSISTANT_NAME = "Rovy"
WAKE_WORDS = ["hey rovy", "rovy", "hey robot"]
MAX_RESPONSE_TOKENS = 150
TEMPERATURE = 0.7

