import queue
import threading

# =========================
# SYSTEM STATE
# =========================

running = True

tts_playing = False
voice_interaction_active = False

# =========================
# FRAME / DETECTION
# =========================

latest_detections = []
latest_frame = None

frame_lock = threading.Lock()

# =========================
# SPEECH
# =========================

speech_queue = queue.Queue()

speech_lock = threading.Lock()

# =========================
# MEMORY
# =========================

last_response_text = ""
last_spoken_text = ""

# =========================
# TIMERS
# =========================

last_spoken_time = 0.0
last_global_announce_time = 0.0

last_spoken_times = {}

last_path_clear_time = 0.0
last_text_detect_time = 0.0

last_user_interaction_time = 0.0
last_wakeword_time = 0.0
last_manual_speech_time = 0.0

# =========================
# WAKEWORD
# =========================

wake_block_until = 0.0