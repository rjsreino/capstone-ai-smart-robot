import re
import sys
import time
import signal
import subprocess
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# =========================
# CONFIG
# =========================
WAKE_WORDS = [
    "kevin",
    "hey kevin",
    "hello kevin",
    "okay kevin",
]

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
INPUT_DEVICE = None 

# Audio block sizes
GATE_BLOCK_SECONDS = 0.20
RECORD_BLOCK_SECONDS = 0.25

# Adaptive noise settings
NOISE_CALIBRATION_BLOCKS = 12
START_MARGIN = 90.0
SILENCE_MARGIN = 40.0
MIN_START_THRESHOLD = 140.0
MAX_SILENCE_THRESHOLD = 260.0

# Recording behavior
SILENCE_DURATION = 1.4
MIN_COMMAND_SECONDS = 0.8
MAX_COMMAND_SECONDS = 12.0
PREBUFFER_BLOCKS = 3

# Whisper
WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# TTS
EDGE_TTS_VOICE = "en-US-GuyNeural"
USE_EDGE_TTS = True

# External scripts
SCRIPT_LIVE_VISION = "live_vision_assistant.py"
SCRIPT_FIND_OBJECT = "find_object.py"
SCRIPT_HAZARD = "hazard_detection.py" 

# =========================
# MODEL LOAD
# =========================
model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)

# =========================
# RUNTIME STATE
# =========================
@dataclass
class ConversationState:
    last_command: str = ""
    last_object: str = ""
    last_intent: str = ""
    last_interaction_ts: float = 0.0
    recent_transcripts: List[str] = field(default_factory=list)

    def remember(self, transcript: str, command: str, intent: str, target_object: str = ""):
        self.last_command = command
        self.last_intent = intent
        self.last_interaction_ts = time.time()
        if target_object:
            self.last_object = target_object

        if transcript:
            self.recent_transcripts.append(transcript)
            self.recent_transcripts = self.recent_transcripts[-5:]


STATE = ConversationState()

# =========================
# SIGNAL HANDLER
# =========================
def handle_interrupt(sig, frame):
    print("\nExiting...")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_interrupt)

# =========================
# TEXT HELPERS
# =========================
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_wake_word(text: str) -> bool:
    normalized = normalize_text(text)
    for wake in WAKE_WORDS:
        pattern = r"\b" + re.escape(wake) + r"\b"
        if re.search(pattern, normalized):
            return True
    return False


def remove_wake_word(text: str) -> str:
    cleaned = normalize_text(text)

    for wake in sorted(WAKE_WORDS, key=len, reverse=True):
        pattern = r"\b" + re.escape(wake) + r"\b"
        if re.search(pattern, cleaned):
            cleaned = re.sub(pattern, "", cleaned, count=1).strip()
            break

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # normalize search phrasing
    for prefix in (
        "look for a ",
        "look for an ",
        "look for the ",
        "find a ",
        "find an ",
        "find the ",
    ):
        if cleaned.startswith(prefix):
            cleaned = "look for " + cleaned[len(prefix):]
            break

    return cleaned.strip()


# =========================
# TTS
# =========================
def speak(text: str):
    print(f"Assistant: {text}")

    if not USE_EDGE_TTS:
        return

    try:
        subprocess.run(
            [
                "edge-playback",
                "--voice",
                EDGE_TTS_VOICE,
                "--text",
                text,
            ],
            check=True,
        )
    except FileNotFoundError:
        print("edge-playback command not found. Install edge-tts.")
    except subprocess.CalledProcessError as e:
        print(f"edge-playback failed: {e}")


# =========================
# AUDIO HELPERS
# =========================
def compute_volume(audio_block: np.ndarray) -> float:
    if audio_block.ndim > 1:
        audio_block = audio_block[:, 0]
    return float(np.abs(audio_block.astype(np.int32)).mean())


def record_block(seconds: float) -> np.ndarray:
    frames = int(seconds * SAMPLE_RATE)
    audio = sd.rec(
        frames,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        device=INPUT_DEVICE,
        blocking=True,
    )
    return audio.copy()


def concat_blocks(blocks: List[np.ndarray]) -> np.ndarray:
    if not blocks:
        return np.zeros((0, CHANNELS), dtype=np.int16)
    return np.concatenate(blocks, axis=0)


def approx_seconds_from_audio(audio: np.ndarray) -> float:
    return len(audio) / float(SAMPLE_RATE)


def calibrate_noise_floor() -> float:
    print("Calibrating room noise...")
    volumes = []
    for _ in range(NOISE_CALIBRATION_BLOCKS):
        block = record_block(GATE_BLOCK_SECONDS)
        volumes.append(compute_volume(block))
    noise_floor = float(np.median(volumes)) if volumes else 0.0
    print(f"Estimated noise floor: {noise_floor:.1f}")
    return noise_floor


def get_dynamic_thresholds(noise_floor: float) -> Tuple[float, float]:
    start_threshold = max(MIN_START_THRESHOLD, noise_floor + START_MARGIN)
    silence_threshold = min(MAX_SILENCE_THRESHOLD, noise_floor + SILENCE_MARGIN)
    return start_threshold, silence_threshold


def listen_for_speech_gate(noise_floor: float) -> Tuple[List[np.ndarray], float]:
    print("Listening for speech...")

    prebuffer: List[np.ndarray] = []
    smoothed_noise = noise_floor

    while True:
        block = record_block(GATE_BLOCK_SECONDS)
        volume = compute_volume(block)

        # update background estimate only when likely near noise level
        if volume < smoothed_noise + 60:
            smoothed_noise = 0.92 * smoothed_noise + 0.08 * volume

        start_threshold, _ = get_dynamic_thresholds(smoothed_noise)

        prebuffer.append(block)
        if len(prebuffer) > PREBUFFER_BLOCKS:
            prebuffer.pop(0)

        if volume > start_threshold:
            print(f"Speech started. volume={volume:.1f}, threshold={start_threshold:.1f}")
            return prebuffer.copy(), smoothed_noise


def record_until_silence(prebuffer: List[np.ndarray], noise_floor: float) -> np.ndarray:
    print("Recording...")

    frames = prebuffer.copy()
    silent_chunks = 0
    silence_chunks_needed = max(1, int(SILENCE_DURATION / RECORD_BLOCK_SECONDS))
    _, silence_threshold = get_dynamic_thresholds(noise_floor)

    while True:
        block = record_block(RECORD_BLOCK_SECONDS)
        frames.append(block)

        volume = compute_volume(block)
        duration = approx_seconds_from_audio(concat_blocks(frames))

        if volume < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0

        if silent_chunks >= silence_chunks_needed:
            break

        if duration >= MAX_COMMAND_SECONDS:
            print("Reached max command duration.")
            break

    return concat_blocks(frames)


# =========================
# STT
# =========================
def transcribe_audio(audio: np.ndarray) -> str:
    if audio.ndim > 1:
        audio = audio[:, 0]

    audio_f32 = audio.astype(np.float32) / 32768.0

    segments, _ = model.transcribe(
        audio_f32,
        language="en",
        vad_filter=True,
        beam_size=5,
    )

    text_parts = [segment.text.strip() for segment in segments if segment.text.strip()]
    return " ".join(text_parts).strip()


# =========================
# SCRIPT UTILITIES
# =========================
def run_python_script(script_name: str, *args: str):
    try:
        subprocess.run([sys.executable, script_name, *args], check=False)
    except Exception as e:
        print(f"Failed to launch {script_name}: {e}")


def script_exists(script_name: str) -> bool:
    try:
        with open(script_name, "r", encoding="utf-8"):
            return True
    except OSError:
        return False


# =========================
# INTENT PARSER
# =========================
def extract_object_from_command(command: str) -> str:
    command = normalize_text(command)

    patterns = [
        r"^look for\s+(.+)$",
        r"^find\s+(.+)$",
        r"^search for\s+(.+)$",
    ]

    for pattern in patterns:
        m = re.match(pattern, command)
        if m:
            target = m.group(1).strip()
            for prefix in ("a ", "an ", "the "):
                if target.startswith(prefix):
                    target = target[len(prefix):].strip()
                    break
            return target

    return ""


def parse_intent(command: str) -> Tuple[str, Optional[str]]:
    cmd = normalize_text(command)

    if not cmd:
        return "empty", None

    if cmd in ("hi", "hello", "hey"):
        return "greeting", None

    if "time" in cmd:
        return "time", None

    if "your name" in cmd or cmd == "name" or "who are you" in cmd:
        return "identity", None

    if (
        "assistant mode" in cmd
        or "vision mode" in cmd
        or "live vision" in cmd
        or "start vision" in cmd
        or "open vision" in cmd
        or "camera mode" in cmd
        or "assist me" in cmd
        or "open camera" in cmd
        or "see mode" in cmd
    ):
        return "vision_mode", None

    if (
        "what do you see" in cmd
        or "describe scene" in cmd
        or "describe what you see" in cmd
        or "describe the scene" in cmd
    ):
        return "scene_description", None

    if "detect obstacle" in cmd or "detect hazards" in cmd or "hazard" in cmd or "obstacle" in cmd:
        return "hazard_detection", None

    target_object = extract_object_from_command(cmd)
    if target_object:
        return "object_search", target_object

    if "find it again" in cmd or "look for it again" in cmd or "search again" in cmd:
        return "repeat_last_object", None

    if cmd in ("stop", "exit", "quit", "shutdown"):
        return "stop", None

    return "unknown", None


# =========================
# COMMAND HANDLER
# =========================
def handle_command(command: str, raw_transcript: str):
    intent, payload = parse_intent(command)

    if intent == "empty":
        speak("Hi, I'm Kevin. Ready.")
        STATE.remember(raw_transcript, command, intent)
        return

    if intent == "greeting":
        speak("Hello.")
        STATE.remember(raw_transcript, command, intent)
        return

    if intent == "time":
        import datetime as dt
        now = dt.datetime.now().strftime("%H:%M")
        speak(f"The time is {now}.")
        STATE.remember(raw_transcript, command, intent)
        return

    if intent == "identity":
        speak("I'm Kevin.")
        STATE.remember(raw_transcript, command, intent)
        return

    if intent == "vision_mode":
        speak("Starting live vision assistant.")
        STATE.remember(raw_transcript, command, intent)
        run_python_script(SCRIPT_LIVE_VISION)
        return

    if intent == "scene_description":
        speak("Starting scene description.")
        STATE.remember(raw_transcript, command, intent)
        run_python_script(SCRIPT_LIVE_VISION, "--mode", "describe")
        return

    if intent == "hazard_detection":
        speak("Starting hazard detection.")
        STATE.remember(raw_transcript, command, intent)
        if script_exists(SCRIPT_HAZARD):
            run_python_script(SCRIPT_HAZARD)
        else:
            run_python_script(SCRIPT_LIVE_VISION, "--mode", "hazard")
        return

    if intent == "object_search":
        target_object = payload.strip() if payload else ""
        if not target_object:
            speak("Please say the object name.")
            return

        speak(f"Looking for {target_object}.")
        STATE.remember(raw_transcript, command, intent, target_object=target_object)
        run_python_script(SCRIPT_FIND_OBJECT, target_object)
        return

    if intent == "repeat_last_object":
        if not STATE.last_object:
            speak("I do not have a previous object to search for.")
            return

        speak(f"Looking again for {STATE.last_object}.")
        STATE.remember(raw_transcript, command, intent, target_object=STATE.last_object)
        run_python_script(SCRIPT_FIND_OBJECT, STATE.last_object)
        return

    if intent == "stop":
        speak("Stopping.")
        raise SystemExit

    speak("Command not recognized.")
    STATE.remember(raw_transcript, command, "unknown")


# =========================
# MAIN LOOP
# =========================
def main():
    base_noise_floor = calibrate_noise_floor()

    while True:
        try:
            prebuffer, live_noise_floor = listen_for_speech_gate(base_noise_floor)

            audio = record_until_silence(prebuffer, live_noise_floor)
            duration = approx_seconds_from_audio(audio)

            if duration < MIN_COMMAND_SECONDS:
                print("Too short.")
                time.sleep(0.10)
                continue

            transcript = transcribe_audio(audio)
            print(f"Transcript: {transcript}")

            if not transcript:
                print("No speech recognized.")
                time.sleep(0.10)
                continue

            if contains_wake_word(transcript):
                print("Wake word detected.")
                command = remove_wake_word(transcript)

                if not command:
                    speak("Hi, I'm ready to assist you.")
                    STATE.remember(transcript, "", "empty")
                else:
                    handle_command(command, transcript)
            else:
                print("Wake word not found. Ignored.")

        except KeyboardInterrupt:
            print("Stopping...")
            break
        except SystemExit:
            break
        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(0.25)


if __name__ == "__main__":
    main()