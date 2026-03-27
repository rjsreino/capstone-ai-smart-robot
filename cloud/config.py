"""
Rovy Cloud Server Configuration
Unified config for REST API (mobile app) + WebSocket (robot) + AI processing
"""
import os

# =============================================================================
# Network Configuration
# =============================================================================

# Server binding
HOST = "0.0.0.0"
API_PORT = 8000        # FastAPI REST for mobile app
WS_PORT = 8765         # WebSocket for robot

# Tailscale IPs (update these for your network)
PC_SERVER_IP = os.getenv("ROVY_PC_IP", "100.121.110.125")
ROBOT_IP = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")

# =============================================================================
# AI Model Configuration (Qwen VLM)
# =============================================================================

# Qwen VLM Configuration
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2-VL-7B-Instruct")  # Model for text and vision queries
QWEN_VISION_MODEL = os.getenv("QWEN_VISION_MODEL", "Qwen/Qwen2-VL-7B-Instruct")  # Same model for vision
QWEN_TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.7"))  # Higher = more creative/personality

# =============================================================================
# Speech Configuration
# =============================================================================

# Speech-to-text (Whisper)
# Use local Whisper model (no API required)
USE_LOCAL_WHISPER = True

# Local Whisper model
# Model size affects accuracy: tiny < base < small < medium < large
# For multilingual accuracy, use "small" or "medium"
WHISPER_MODEL = os.getenv("ROVY_WHISPER_MODEL", "small")  # tiny, base, small, medium, large

# Language detection settings
WHISPER_LANGUAGE = os.getenv("ROVY_WHISPER_LANGUAGE", None)  # None = auto-detect, or specify like "en", "es", etc.
WHISPER_TASK = os.getenv("ROVY_WHISPER_TASK", "transcribe")  # "transcribe" or "translate" (to English)

# Text-to-speech
TTS_ENGINE = os.getenv("ROVY_TTS_ENGINE", "piper")  # piper, espeak

# Piper voice paths for different languages
# Each language can have its own voice model
# Download voices from: https://github.com/rhasspy/piper/blob/master/VOICES.md
PIPER_VOICES = {
    "en": os.path.expanduser("~/rovy_client/models/piper/en_US-hfc_male-medium.onnx"),
    "es": os.path.expanduser("~/rovy_client/models/piper/es_ES-davefx-medium.onnx"),
    "fr": os.path.expanduser("~/rovy_client/models/piper/fr_FR-siwis-medium.onnx"),
    "de": os.path.expanduser("~/rovy_client/models/piper/de_DE-thorsten-medium.onnx"),
    "it": os.path.expanduser("~/rovy_client/models/piper/it_IT-riccardo-x_low.onnx"),
    "pt": os.path.expanduser("~/rovy_client/models/piper/pt_BR-faber-medium.onnx"),
    "ru": os.path.expanduser("~/rovy_client/models/piper/ru_RU-dmitri-medium.onnx"),
    "zh": os.path.expanduser("~/rovy_client/models/piper/zh_CN-huayan-medium.onnx"),
    "vi": os.path.expanduser("~/rovy_client/models/piper/vi_VN-vais1000-medium.onnx"),
    "hi": os.path.expanduser("~/rovy_client/models/piper/hi_IN-pratham-medium.onnx"),
    "ne": os.path.expanduser("~/rovy_client/models/piper/ne_NP-chitwan-medium.onnx"),
    "fa": os.path.expanduser("~/rovy_client/models/piper/fa_IR-amir-medium.onnx"),
    # Korean (ko) is not available in Piper TTS
}

# Legacy single voice path (for backward compatibility)
PIPER_VOICE_PATH = os.getenv("ROVY_PIPER_VOICE", PIPER_VOICES.get("en"))

# =============================================================================
# Assistant Configuration
# =============================================================================

ASSISTANT_NAME = "Rovy"
WAKE_WORDS = ["hey rovy", "rovy", "hey robot"]

# =============================================================================
# Face Recognition
# =============================================================================

KNOWN_FACES_DIR = os.getenv("ROVY_KNOWN_FACES", "known-faces")
FACE_RECOGNITION_THRESHOLD = float(os.getenv("ROVY_FACE_THRESHOLD", "0.6"))

# =============================================================================
# Robot Hardware (sent to robot client)
# =============================================================================

ROVER_SERIAL_PORT = "/dev/ttyACM0"
ROVER_BAUDRATE = 115200

# Camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
JPEG_QUALITY = 80

# Audio
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

# =============================================================================
# Connection Settings
# =============================================================================

RECONNECT_DELAY = 5
MAX_RECONNECT_ATTEMPTS = 0  # 0 = infinite

# =============================================================================
# Spotify Configuration
# =============================================================================

SPOTIFY_ENABLED = os.getenv("SPOTIFY_ENABLED", "true").lower() == "true"
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "93138e86ecf24daea4b07df74c7cb8e9")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "f8f131ad542a4cf2a021aae8bdbc5763")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
SPOTIFY_DEVICE_NAME = os.getenv("SPOTIFY_DEVICE_NAME", "ROVY")  # Raspotify device name

# =============================================================================
# Logging
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

