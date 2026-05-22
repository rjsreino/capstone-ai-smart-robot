import os
import pyaudio

# =========================
# MODEL
# =========================

MODEL_PATH = "models/yolov8n.pt"

# =========================
# CAMERA
# =========================

CAMERA_INDEX = int(
    os.getenv("VISION_CAMERA_INDEX", "0")
)

FRAME_WIDTH = 640
FRAME_HEIGHT = 360

IMG_SIZE = 320

# =========================
# YOLO
# =========================

CONF_THRESHOLD = 0.20

ALLOWED_CLASSES = {
    "person",
    "chair",
    "couch",
    "bench",
    "dining table",
    "bottle",
    "backpack",
    "potted plant",
    "cell phone",
    "cup",
    "laptop",
    "book",
    "handbag",
    "suitcase",
    "bed",
    "tv",
    "keyboard",
    "mouse",
    "remote",
}

MIN_AREA_RATIO = 0.03
MIN_PERSON_AREA_RATIO = 0.10

# =========================
# AUDIO
# =========================

MIC_INDEX = 1

SAMPLE_RATE = 16000
CHANNELS = 1

FORMAT = pyaudio.paInt16

FRAME_SIZE = 640

SD_WAKE_DEVICE_INDEX = 1
DEVICE_INDEX = 1

# =========================
# WAKEWORD
# =========================

WAKEWORD_NAME = "jarvis"

WAKEWORD_THRESHOLD = 0.35
WAKEWORD_COOLDOWN = 1.0
WAKE_RESTART_DELAY = 1.5

# =========================
# RECORDING
# =========================

CHUNK_DURATION = 0.1

SILENCE_THRESHOLD = 0.1
SILENCE_DURATION = 2.5

START_SPEECH_THRESHOLD = 0.3

MIN_COMMAND_SECONDS = 0.5
MAX_COMMAND_SECONDS = 6.0

# =========================
# DEPTH
# =========================

USE_ZED_DEPTH = False

DEPTH_COLLISION_THRESHOLD = 0.8

# =========================
# PROACTIVE AI
# =========================

GLOBAL_COOLDOWN = 2.5
MESSAGE_COOLDOWN = 4.0

CLEAR_PATH_COOLDOWN = 7.0

TEXT_DETECTED_COOLDOWN = 10.0

VISION_ANNOUNCE_INTERVAL = 2.0

TEXT_AUTO_READ_MAX_CHARS = 120

ENABLE_AUTO_TEXT_READ = False

USER_INTERACTION_COOLDOWN = 5.0

# =========================
# WINDOW
# =========================

WINDOW_NAME = "Proactive Live Vision Assistant"

# =========================
# TTS
# =========================

TTS_VOICE = "en-US-GuyNeural"

# =========================
# SOUNDDEVICE
# =========================

SD_SAMPLE_RATE = 16000