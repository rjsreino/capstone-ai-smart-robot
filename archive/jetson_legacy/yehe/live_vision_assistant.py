from ultralytics import YOLO
import cv2
import time
import threading
import subprocess
import io
import wave
import signal
import sys
import re
import os
import queue
from typing import Optional
import sounddevice as sd

import easyocr
import numpy as np
import pyaudio
import soundfile as sf
import whisper
from openwakeword.model import Model

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from llm_reasoner import ask_llm
    print("[LLM] Connected to llm_reasoner.py")
except Exception as e:
    print("[LLM ERROR] Could not import llm_reasoner.py:", e)

    def ask_llm(command: str, detections: list[dict]) -> str:
        return "LLM is not connected right now."


# =========================
# CONFIG
# =========================
MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = int(os.getenv("VISION_CAMERA_INDEX", "1"))
CONF_THRESHOLD = 0.30
IMG_SIZE = 320
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
WINDOW_NAME = "Proactive Live Vision Assistant"
MIC_INDEX = 1
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAME_SIZE = 1280
SD_WAKE_DEVICE_INDEX = 1
DEVICE_INDEX = 1

CHUNK_DURATION = 0.1
SILENCE_THRESHOLD = 1800
SILENCE_DURATION = 0.8
START_SPEECH_THRESHOLD = 2500
MIN_COMMAND_SECONDS = 0.5
WAKEWORD_THRESHOLD = 0.5
WAKEWORD_COOLDOWN = 5.0
MAX_COMMAND_SECONDS = 6.0

WAKE_RESTART_DELAY = 5.0
wake_block_until = 0.0

MIN_AREA_RATIO = 0.03
MIN_PERSON_AREA_RATIO = 0.10
voice_interaction_active = False
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

# Proactive behavior tuning
GLOBAL_COOLDOWN = 2.5
MESSAGE_COOLDOWN = 4.0
CLEAR_PATH_COOLDOWN = 7.0
TEXT_DETECTED_COOLDOWN = 10.0
VISION_ANNOUNCE_INTERVAL = 2.0
TEXT_AUTO_READ_MAX_CHARS = 120
ENABLE_AUTO_TEXT_READ = False
SD_WAKE_DEVICE_INDEX = 1
SD_SAMPLE_RATE = 16000


# =========================
# GLOBAL STATE
# =========================
running = True

latest_detections = []
latest_frame = None
frame_lock = threading.Lock()

speech_queue = queue.Queue()
speech_lock = threading.Lock()

last_response_text = ""
last_spoken_text = ""
last_spoken_time = 0.0
last_global_announce_time = 0.0
last_spoken_times = {}
last_path_clear_time = 0.0
last_text_detect_time = 0.0
last_user_interaction_time = 0.0
last_wakeword_time = 0.0
USER_INTERACTION_COOLDOWN = 5.0


# =========================
# AUDIO / DEVICE HELPERS
# =========================
def get_input_device_index(pa: pyaudio.PyAudio) -> Optional[int]:
    if MIC_INDEX >= 0:
        print(f"[MIC] Using hardcoded microphone index: {MIC_INDEX}")
        return MIC_INDEX

    blocked_keywords = [
    "voicemod",
    "steelseries",
    "sonar",
    "steam",
    "intelligo",
    "virtual",
    "vad",
    "speaker",
    "output",
    "microphone (realtek hd audio mic input)",
]

    try:
        default_info = pa.get_default_input_device_info()
        name = default_info["name"].lower()

        if not any(word in name for word in blocked_keywords):
            print(
                f"[MIC] Using Windows default microphone: "
                f"{default_info['index']} | {default_info['name']}"
            )
            return int(default_info["index"])

        print(f"[MIC] Default mic is virtual/skipped: {default_info['name']}")

    except Exception as e:
        print("[MIC] No Windows default mic. Using fallback scan.")

    print("\n=== INPUT DEVICES ===")

    fallback_devices = []

    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
            name = info["name"].lower()
            max_channels = int(info.get("maxInputChannels", 0))

            if max_channels <= 0:
                continue

            print(
                f"INDEX {i} | {info['name']} | "
                f"INPUTS={max_channels} | "
                f"RATE={int(info['defaultSampleRate'])}"
            )

            if any(word in name for word in blocked_keywords):
                print(f"[MIC] Skipping virtual/bad device: {i}")
                continue

            fallback_devices.append(i)

        except Exception:
            continue

    print("=====================\n")

    for i in fallback_devices:
        try:
            info = pa.get_device_info_by_index(i)

            pa.is_format_supported(
                rate=int(info["defaultSampleRate"]),
                input_device=i,
                input_channels=1,
                input_format=FORMAT,
            )

            print(f"[MIC] Selected fallback physical mic: {i} | {info['name']}")
            return i

        except Exception as e:
            print(f"[MIC] Skipping unsupported physical mic {i}: {e}")

    print("[MIC ERROR] No usable physical microphone found.")
    return None
# =========================
# MODELS
# =========================
wakeword_model = Model(
    wakeword_models=["hey_jarvis"],
    inference_framework="onnx"
)

whisper_model = whisper.load_model("tiny").to(device)
try:
    ocr_reader = easyocr.Reader(["en"], gpu=True)
except Exception:
    ocr_reader = easyocr.Reader(["en"], gpu=False)


# =========================
# SYSTEM / EXIT
# =========================
def handle_interrupt(sig, frame):
    global running
    running = False
    print("\nExiting...")
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
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






def is_repeat_command(command: str) -> bool:
    repeat_keywords = [
        "repeat",
        "say again",
        "one more time",
        "repeat it",
        "repeat that",
        "can you repeat",
        "say it again",
    ]
    return any(keyword in command for keyword in repeat_keywords)


def remember_response(text: str):
    global last_response_text
    last_response_text = text


# =========================
# SPEECH OUTPUT
# =========================
def speak_windows_tts(text: str):
    global last_spoken_text, last_spoken_time

    text = str(text).strip()
    if not text:
        return

    now = time.time()
    if text == last_spoken_text and (now - last_spoken_time) < 1.0:
        return

    last_spoken_text = text
    last_spoken_time = now

    safe_text = text.replace("'", "''")

    ps_command = (
        "Add-Type -AssemblyName System.Speech;"
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$speaker.Rate = 0;"
        f"$speaker.Speak('{safe_text}');"
    )

    try:
        subprocess.run(
            ["powershell", "-Command", ps_command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        print(f"[TTS] {text}")


def enqueue_speech(text: str):
    text = str(text).strip()
    if not text:
        return
    speech_queue.put(text)


def speech_worker():
    global running

    while running:
        try:
            text = speech_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        with speech_lock:
            speak_windows_tts(text)

        speech_queue.task_done()


# =========================
# AUDIO INPUT
# =========================
def listen_for_wake_word(pa: pyaudio.PyAudio, device_index: Optional[int]) -> bool:
    global last_wakeword_time
    global wake_block_until

    print("[WAKE] Listening for Hey Jarvis using sounddevice...")

    try:
        with sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SIZE,
            device=SD_WAKE_DEVICE_INDEX
        ) as stream:

            while running:
                audio, _ = stream.read(FRAME_SIZE)
                pcm = np.squeeze(audio)

                now = time.time()

                if now < wake_block_until:
                    continue

                prediction = wakeword_model.predict(pcm)
                score = prediction.get("hey_jarvis", 0)

                print(f"[WAKE SCORE] {score:.2f}")

                if score > WAKEWORD_THRESHOLD and now - last_wakeword_time > WAKEWORD_COOLDOWN:
                    print(f"[WAKE] Hey Jarvis detected ({score:.2f})")
                    last_wakeword_time = now
                    return True

    except Exception as e:
        print(f"[WAKE ERROR] sounddevice failed: {e}")
        time.sleep(1.0)
        return False

    return False


def record_until_silence(
    pa: pyaudio.PyAudio,
    device_index: Optional[int],
    sample_rate: int = SAMPLE_RATE,
    chunk_duration: float = CHUNK_DURATION,
    silence_threshold: int = SILENCE_THRESHOLD,
    silence_duration: float = SILENCE_DURATION,
) -> io.BytesIO:

    sample_rate = SD_SAMPLE_RATE
    chunk_size = int(sample_rate * chunk_duration)
    silence_chunks_needed = max(1, int(silence_duration / chunk_duration))

    frames = []
    silent_chunks = 0
    speech_detected = False

    print("[REC] Listening for command...")

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=chunk_size,
        device=SD_WAKE_DEVICE_INDEX
    ) as stream:

        wait_start_time = time.time()
        start_time = None

        while running:
            audio, _ = stream.read(chunk_size)
            pcm = np.squeeze(audio).astype(np.int16)

            volume = np.abs(pcm).mean()
            print(f"[VOL] {volume}")

            if not speech_detected:
                if volume > START_SPEECH_THRESHOLD:
                    speech_detected = True
                    silent_chunks = 0
                    start_time = time.time()
                    frames.append(pcm.tobytes())

                if time.time() - wait_start_time > 5.0:
                    print("[REC] No speech detected.")
                    return io.BytesIO()

            else:
                frames.append(pcm.tobytes())

                if volume > silence_threshold:
                    silent_chunks = 0
                else:
                    silent_chunks += 1

                if silent_chunks >= silence_chunks_needed:
                    break

                if start_time and time.time() - start_time > MAX_COMMAND_SECONDS:
                    print("[REC] Max command time reached.")
                    break

    if not speech_detected or not frames:
        return io.BytesIO()

    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    wav_buffer.seek(0)
    return wav_buffer


def transcribe_audio(wav_buffer: io.BytesIO) -> str:
    audio_array, sample_rate = sf.read(wav_buffer, dtype="float32")

    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)

    if sample_rate != SAMPLE_RATE:
        import librosa
        audio_array = librosa.resample(
            audio_array,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE
        )

    if audio_array.size == 0:
        return ""

    audio_array = whisper.pad_or_trim(audio_array)
    mel = whisper.log_mel_spectrogram(audio_array).to(whisper_model.device)

    options = whisper.DecodingOptions(language="en", fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    return result.text.strip()

def clear_speech_queue():
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except queue.Empty:
            break

# =========================
# VISION HELPERS
# =========================
def get_position_label(center_x: float, frame_width: int) -> str:
    if center_x < frame_width / 3:
        return "left"
    if center_x < 2 * frame_width / 3:
        return "center"
    return "right"


def get_distance_label(area_ratio: float) -> str:
    if area_ratio > 0.18:
        return "very close"
    if area_ratio > 0.10:
        return "close"
    if area_ratio > 0.05:
        return "medium"
    return "far"


# =========================
# OCR
# =========================
def extract_text_from_frame(frame) -> str:
    if frame is None:
        return ""

    h, w = frame.shape[:2]

    x1 = int(w * 0.15)
    x2 = int(w * 0.85)
    y1 = int(h * 0.15)
    y2 = int(h * 0.75)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return ""

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    results = ocr_reader.readtext(thresh)

    filtered_lines = []
    seen = set()

    for _, text, conf in results:
        text = text.strip()
        if len(text) < 3:
            continue
        if conf < 0.30:
            continue

        lowered = text.lower()
        if lowered in seen:
            continue

        seen.add(lowered)
        filtered_lines.append(text)

    return " ".join(filtered_lines)


# =========================
# PROACTIVE EVENT LOGIC
# =========================
def should_announce(message: str, now: float) -> bool:
    global last_global_announce_time

    if now - last_global_announce_time < GLOBAL_COOLDOWN:
        return False

    last_time = last_spoken_times.get(message, 0.0)
    if now - last_time < MESSAGE_COOLDOWN:
        return False

    last_spoken_times[message] = now
    last_global_announce_time = now
    return True


def evaluate_scene_events(detections: list[dict]) -> list[dict]:
    events = []
    center_objects = [d for d in detections if d["position"] == "center"]

    for d in detections:
        class_name = d["class_name"]
        position = d["position"]
        distance = d["distance"]

        if position == "center" and distance == "very close":
            events.append({
                "priority": 100,
                "message": f"Warning. {class_name} very close ahead."
            })
        elif position == "center" and distance == "close":
            events.append({
                "priority": 85,
                "message": f"Caution. {class_name} ahead."
            })
        elif class_name == "person" and position == "center" and distance in {"close", "medium"}:
            events.append({
                "priority": 80,
                "message": "Person ahead."
            })
        elif position in {"left", "right"} and distance == "very close":
            events.append({
                "priority": 75,
                "message": f"Warning. {class_name} very close on your {position}."
            })
        elif position in {"left", "right"} and distance == "close":
            events.append({
                "priority": 60,
                "message": f"{class_name} on your {position}."
            })

    if not center_objects:
        events.append({
            "priority": 15,
            "message": "Path ahead appears clear."
        })

    return sorted(events, key=lambda x: x["priority"], reverse=True)


def maybe_announce_clear_path(now: float):
    global last_path_clear_time

    if now - last_path_clear_time < CLEAR_PATH_COOLDOWN:
        return

    msg = "Path ahead appears clear."
    if should_announce(msg, now):
        remember_response(msg)
        enqueue_speech(msg)
        last_path_clear_time = now


def maybe_announce_text(frame, now: float):
    global last_text_detect_time

    if frame is None:
        return

    if now - last_text_detect_time < TEXT_DETECTED_COOLDOWN:
        return

    extracted_text = extract_text_from_frame(frame)
    if not extracted_text:
        return

    if ENABLE_AUTO_TEXT_READ:
        spoken_text = extracted_text[:TEXT_AUTO_READ_MAX_CHARS].strip()
        if len(extracted_text) > TEXT_AUTO_READ_MAX_CHARS:
            spoken_text += " ..."
        msg = f"Text says: {spoken_text}"
    else:
        msg = "Text detected in front of you."

    if should_announce(msg, now):
        remember_response(msg)
        enqueue_speech(msg)
        last_text_detect_time = now


def auto_announce(detections: list[dict], frame):
    global last_user_interaction_time
    global voice_interaction_active

    if voice_interaction_active:
        return

    if time.time() - last_user_interaction_time < USER_INTERACTION_COOLDOWN:
        return

    now = time.time()
    events = evaluate_scene_events(detections)

    if events:
        best_event = events[0]
        message = best_event["message"]

        if message == "Path ahead appears clear.":
            maybe_announce_clear_path(now)
        else:
            if should_announce(message, now):
                remember_response(message)
                enqueue_speech(message)

    maybe_announce_text(frame, now)


# =========================
# QUERY HANDLER
# =========================
def handle_vision_query(command: str, detections: list[dict], frame):
    command = normalize_text(command)

    if command in {"stop", "exit", "quit"}:
        return "__EXIT__"

    if "stop vision mode" in command or "exit vision mode" in command:
        return "__EXIT__"

    if is_repeat_command(command):
        if last_response_text:
            return last_response_text
        return "There is nothing recent to repeat."

    llm_input = []

    for detection in detections:
        llm_input.append({
            "class": detection["class_name"],
            "position": detection["position"],
            "distance": detection["distance"],
            "confidence": round(detection["confidence"], 2),
        })

    try:
        answer = ask_llm(command, llm_input)
        remember_response(answer)
        return answer

    except Exception as e:
        print("[LLM ERROR]", e)
        answer = "I could not process that question right now."
        remember_response(answer)
        return answer


# =========================
# VISION LOOP
# =========================
def vision_loop():
    global latest_detections, latest_frame, running

    model = YOLO(MODEL_PATH)
    model.to(device)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Failed to open camera")
        running = False
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or FRAME_WIDTH
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_HEIGHT
    frame_area = max(actual_width * actual_height, 1)

    prev_time = time.time()
    last_auto_announce_time = 0.0

    while running:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            running = False
            break

        results = model(
            frame,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            device=device,
            verbose=False
        )

        result = results[0]
        annotated = frame.copy()
        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                class_name = model.names[cls_id].lower()

                if class_name not in ALLOWED_CLASSES:
                    continue

                box_area = max(0, x2 - x1) * max(0, y2 - y1)
                area_ratio = box_area / frame_area

                if area_ratio < MIN_AREA_RATIO:
                    continue

                if class_name == "person" and area_ratio < MIN_PERSON_AREA_RATIO:
                    continue

                center_x = (x1 + x2) / 2.0
                position = get_position_label(center_x, actual_width)
                distance = get_distance_label(area_ratio)

                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "position": position,
                    "distance": distance,
                    "area_ratio": area_ratio,
                    "box": (x1, y1, x2, y2),
                })

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated,
                    f"{class_name} {conf:.2f} | {position} | {distance}",
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

        detections.sort(key=lambda d: d["area_ratio"], reverse=True)

        with frame_lock:
            latest_detections = detections
            latest_frame = frame.copy()

        now = time.time()
        if now - last_auto_announce_time >= VISION_ANNOUNCE_INTERVAL:
            auto_announce(detections, frame)
            last_auto_announce_time = now

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        cv2.putText(
            annotated,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            "Proactive Live Vision Assistant",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            "Auto guidance ON | Say: Hey Jarvis ... | Press Q or ESC to exit.",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow(WINDOW_NAME, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================
# VOICE LOOP
# =========================
def voice_loop():
    global running
    global last_user_interaction_time
    global voice_interaction_active
    global wake_block_until

    pa = pyaudio.PyAudio()
    device_index = get_input_device_index(pa)

    print(f"[INFO] Using microphone device index: {device_index}")

    if device_index is None:
        print("[VOICE ERROR] No valid microphone found.")
        return

    try:
        while running:
            heard = listen_for_wake_word(pa, device_index)

            if not heard or not running:
                continue

            print("[VOICE] Wake word accepted.")

            voice_interaction_active = True
            last_user_interaction_time = time.time()

            clear_speech_queue()

            try:
                wakeword_model.reset()
            except Exception:
                pass

            with speech_lock:
                speak_windows_tts("Yes?")

            time.sleep(1.0)

            try:
                raw_audio = record_until_silence(
                    pa,
                    device_index,
                    SAMPLE_RATE
                )
            except Exception as e:
                print(f"[VOICE ERROR] Audio capture failed: {e}")

                voice_interaction_active = False
                wake_block_until = time.time() + WAKE_RESTART_DELAY

                try:
                    wakeword_model.reset()
                except Exception:
                    pass

                time.sleep(WAKE_RESTART_DELAY)
                continue

            try:
                raw_audio.seek(0)

                if raw_audio.getbuffer().nbytes == 0:
                    print("[VOICE] Empty audio buffer.")

                    voice_interaction_active = False
                    wake_block_until = time.time() + WAKE_RESTART_DELAY

                    try:
                        wakeword_model.reset()
                    except Exception:
                        pass

                    time.sleep(WAKE_RESTART_DELAY)
                    continue

            except Exception as e:
                print(f"[VOICE ERROR] Invalid audio buffer: {e}")

                voice_interaction_active = False
                wake_block_until = time.time() + WAKE_RESTART_DELAY

                try:
                    wakeword_model.reset()
                except Exception:
                    pass

                time.sleep(WAKE_RESTART_DELAY)
                continue

            try:
                transcript = transcribe_audio(raw_audio)
                print(f"[TRANSCRIPT RAW] '{transcript}'")

            except Exception as e:
                print(f"[VOICE ERROR] Transcription failed: {e}")

                voice_interaction_active = False
                wake_block_until = time.time() + WAKE_RESTART_DELAY

                try:
                    wakeword_model.reset()
                except Exception:
                    pass

                time.sleep(WAKE_RESTART_DELAY)
                continue

            if not transcript or not transcript.strip():
                print("[VOICE] No valid speech detected.")

                voice_interaction_active = False
                wake_block_until = time.time() + WAKE_RESTART_DELAY

                try:
                    wakeword_model.reset()
                except Exception:
                    pass

                time.sleep(WAKE_RESTART_DELAY)
                continue

            normalized_transcript = normalize_text(transcript)

            print(f"[HEARD] {normalized_transcript}")

            last_user_interaction_time = time.time()

            with frame_lock:
                detections_copy = list(latest_detections)
                frame_copy = None if latest_frame is None else latest_frame.copy()

            try:
                answer = handle_vision_query(
                    normalized_transcript,
                    detections_copy,
                    frame_copy
                )

            except Exception as e:
                print(f"[VOICE ERROR] Query handling failed: {e}")
                answer = "I could not process that request."

            print(f"[ASSISTANT] {answer}")

            if answer == "__EXIT__":
                with speech_lock:
                    speak_windows_tts("Stopping live vision assistant.")

                running = False
                break

            clear_speech_queue()

            try:
                with speech_lock:
                    speak_windows_tts(answer)

            except Exception as e:
                print(f"[VOICE ERROR] TTS failed: {e}")

            last_user_interaction_time = time.time()
            voice_interaction_active = False

            wake_block_until = time.time() + WAKE_RESTART_DELAY

            try:
                wakeword_model.reset()
            except Exception:
                pass

            print("[VOICE] Cooling down before wake word restart...")
            time.sleep(WAKE_RESTART_DELAY)

            try:
                wakeword_model.reset()
            except Exception:
                pass

    finally:
        voice_interaction_active = False

        try:
            wakeword_model.reset()
        except Exception:
            pass

        try:
            pa.terminate()
        except Exception:
            pass

        print("[VOICE] Voice loop terminated.")


# =========================
# MAIN
# =========================
def main():
    global running

    enqueue_speech("Proactive live vision assistant started. Automatic guidance is active. Say Hey Jarvis before optional voice commands.")

    speaker_thread = threading.Thread(target=speech_worker, daemon=True)
    vision_thread = threading.Thread(target=vision_loop, daemon=True)
    voice_thread = threading.Thread(target=voice_loop, daemon=True)

    speaker_thread.start()
    vision_thread.start()
    voice_thread.start()

    try:
        while running:
            time.sleep(0.1)
    finally:
        running = False
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[INFO] Proactive live vision assistant terminated.")


if __name__ == "__main__":
    main()
