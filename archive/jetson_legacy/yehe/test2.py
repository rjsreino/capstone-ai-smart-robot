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

import easyocr
import numpy as np
import pyaudio
import soundfile as sf
import whisper

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from llm_reasoner import ask_llm
except Exception:
    def ask_llm(command: str, detections: list[dict]) -> str:
        if not detections:
            return "I do not detect any major object right now."
        top = detections[0]
        return (
            "Advanced reasoning is unavailable right now. "
            f"The nearest visible object is a {top['class']} on the {top['position']}."
        )


# =========================
# CONFIG
# =========================
MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = int(os.getenv("VISION_CAMERA_INDEX", "0"))
CONF_THRESHOLD = 0.30
IMG_SIZE = 320
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
WINDOW_NAME = "Proactive Live Vision Assistant"

MIC_INDEX = int(os.getenv("VISION_MIC_INDEX", "-1"))  # -1 = auto detect default input
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAME_SIZE = 1280

CHUNK_DURATION = 0.1
SILENCE_THRESHOLD = 200
SILENCE_DURATION = 0.5
START_SPEECH_THRESHOLD = 300
MIN_COMMAND_SECONDS = 0.2

WAKE_WORDS = [
    "kevin",
    "hey kevin",
    "hello kevin",
]

MIN_AREA_RATIO = 0.03
MIN_PERSON_AREA_RATIO = 0.10

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


# =========================
# AUDIO / DEVICE HELPERS
# =========================
def get_input_device_index(pa: pyaudio.PyAudio) -> Optional[int]:
    if MIC_INDEX >= 0:
        return MIC_INDEX

    try:
        default_info = pa.get_default_input_device_info()
        return int(default_info["index"])
    except Exception:
        pass

    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0)) > 0:
                return i
        except Exception:
            continue

    return None


# =========================
# MODELS
# =========================
whisper_model = whisper.load_model("base").to(device)

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


def contains_wake_word(text: str) -> bool:
    normalized = normalize_text(text)

    for wake_word in sorted(WAKE_WORDS, key=len, reverse=True):
        pattern = r"\b" + re.escape(wake_word) + r"\b"
        if re.search(pattern, normalized):
            return True

    return False


def remove_wake_word(text: str) -> str:
    cleaned = normalize_text(text)

    for wake_word in sorted(WAKE_WORDS, key=len, reverse=True):
        pattern = r"\b" + re.escape(wake_word) + r"\b"
        if re.search(pattern, cleaned):
            cleaned = re.sub(pattern, "", cleaned, count=1).strip()
            break

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


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
def listen_for_speech_gate(pa: pyaudio.PyAudio, device_index: Optional[int]) -> bool:
    if device_index is None:
        print("[ERROR] No microphone input device found.")
        time.sleep(1.0)
        return False

    try:
        stream = pa.open(
            rate=SAMPLE_RATE,
            channels=CHANNELS,
            format=FORMAT,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=FRAME_SIZE
        )
    except Exception as e:
        print(f"[ERROR] Failed to open microphone: {e}")
        time.sleep(1.0)
        return False

    try:
        while running:
            data = stream.read(FRAME_SIZE, exception_on_overflow=False)
            pcm = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(pcm).mean()

            if volume > START_SPEECH_THRESHOLD:
                return True
    finally:
        stream.stop_stream()
        stream.close()

    return False


def record_until_silence(
    pa: pyaudio.PyAudio,
    device_index: Optional[int],
    sample_rate: int = SAMPLE_RATE,
    chunk_duration: float = CHUNK_DURATION,
    silence_threshold: int = SILENCE_THRESHOLD,
    silence_duration: float = SILENCE_DURATION,
) -> io.BytesIO:
    if device_index is None:
        raise RuntimeError("No microphone input device found.")

    chunk_size = int(sample_rate * chunk_duration)

    stream = pa.open(
        rate=sample_rate,
        channels=CHANNELS,
        format=FORMAT,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=chunk_size
    )

    frames = []
    silent_chunks = 0
    speech_detected = False
    silence_chunks_needed = max(1, int(silence_duration / chunk_duration))

    try:
        while running:
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)

            audio_np = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_np).mean()

            if volume > silence_threshold:
                speech_detected = True
                silent_chunks = 0
            else:
                if speech_detected:
                    silent_chunks += 1

            if speech_detected and silent_chunks >= silence_chunks_needed:
                break
    finally:
        stream.stop_stream()
        stream.close()

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(CHANNELS)
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

    if (
        "read" in command
        or "text" in command
        or "sign" in command
        or "menu" in command
        or "room number" in command
        or "what does this say" in command
    ):
        extracted_text = extract_text_from_frame(frame)
        if extracted_text:
            answer = f"The text says: {extracted_text}"
            remember_response(answer)
            return answer
        answer = "I cannot read any clear text right now."
        remember_response(answer)
        return answer

    if (
        "what is in front of me" in command
        or "what s in front of me" in command
        or "what is ahead of me" in command
        or "what is right in front of me" in command
    ):
        center_objects = [d for d in detections if d["position"] == "center"]

        if center_objects:
            main = center_objects[0]
            answer = (
                f"There is a {main['class_name']} in front of you. "
                f"It is {main['distance']}."
            )
            remember_response(answer)
            return answer

        if detections:
            main = detections[0]
            answer = (
                f"The nearest visible object is a {main['class_name']} "
                f"on the {main['position']}."
            )
            remember_response(answer)
            return answer

        answer = "I do not detect any major object in front of you."
        remember_response(answer)
        return answer

    if "what do you see" in command or "describe scene" in command:
        if not detections:
            answer = "I do not detect any major object."
            remember_response(answer)
            return answer

        unique_names = []
        seen = set()

        for detection in detections:
            class_name = detection["class_name"]
            if class_name not in seen:
                seen.add(class_name)
                unique_names.append(class_name)

        if len(unique_names) == 1:
            answer = f"I see a {unique_names[0]}."
        elif len(unique_names) == 2:
            answer = f"I see a {unique_names[0]} and a {unique_names[1]}."
        else:
            answer = (
                "I see "
                + ", ".join(f"a {name}" for name in unique_names[:-1])
                + f", and a {unique_names[-1]}."
            )

        remember_response(answer)
        return answer

    if "is the path clear" in command or "can i move forward" in command:
        blockers = [d for d in detections if d["position"] == "center"]

        if blockers:
            main = blockers[0]
            answer = f"The path ahead is blocked by a {main['class_name']}."
            remember_response(answer)
            return answer

        answer = "The path ahead appears clear."
        remember_response(answer)
        return answer

    try:
        llm_input = []
        for detection in detections:
            llm_input.append({
                "class": detection["class_name"],
                "position": detection["position"],
                "distance": detection["distance"],
                "confidence": round(detection["confidence"], 2),
            })

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
            "Auto guidance ON | Say: Kevin ... | Press Q or ESC to exit.",
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

    pa = pyaudio.PyAudio()
    device_index = get_input_device_index(pa)
    print(f"[INFO] Using microphone device index: {device_index}")

    try:
        while running:
            heard = listen_for_speech_gate(pa, device_index)
            if not heard or not running:
                continue

            try:
                raw_audio = record_until_silence(pa, device_index, SAMPLE_RATE)
            except Exception as e:
                print(f"[ERROR] Audio capture failed: {e}")
                time.sleep(0.2)
                continue

            raw_audio.seek(0)
            try:
                audio_array, sr = sf.read(raw_audio, dtype="int16")
                if len(audio_array.shape) > 1:
                    audio_array = np.mean(audio_array, axis=1)
                approx_seconds = len(audio_array) / float(sr)
            except Exception:
                approx_seconds = 0.0
            finally:
                raw_audio.seek(0)

            if approx_seconds < MIN_COMMAND_SECONDS:
                time.sleep(0.2)
                continue

            try:
                transcript = transcribe_audio(raw_audio)
            except Exception as e:
                print(f"Transcription error: {e}")
                time.sleep(0.2)
                continue

            if not transcript:
                continue

            normalized_transcript = normalize_text(transcript)
            print(f"[HEARD] {normalized_transcript}")

            if not contains_wake_word(normalized_transcript):
                print("[IGNORED] Wake word not found.")
                continue
            global last_user_interaction_time
            last_user_interaction_time = time.time()

            command = remove_wake_word(normalized_transcript)
            
            if not command:
                speak_windows_tts("Yes?")
                continue

            if not command:
                enqueue_speech("Ready.")
                continue

            with frame_lock:
                detections_copy = list(latest_detections)
                frame_copy = None if latest_frame is None else latest_frame.copy()

            answer = handle_vision_query(command, detections_copy, frame_copy)

            if answer == "__EXIT__":
                enqueue_speech("Stopping live vision assistant.")
                running = False
                break

            enqueue_speech(answer)

    finally:
        pa.terminate()


# =========================
# MAIN
# =========================
def main():
    global running

    enqueue_speech("Proactive live vision assistant started. Automatic guidance is active. Say Kevin before optional voice commands.")

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