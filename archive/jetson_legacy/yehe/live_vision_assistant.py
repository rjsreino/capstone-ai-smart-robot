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

from llm_reasoner import ask_llm

import numpy as np
import pyaudio
import soundfile as sf
import whisper

# =========================
# CAMERA / MODEL CONFIG
# =========================
MODEL_PATH = "yolov8s.pt"
CAMERA_INDEX = 1
CONF_THRESHOLD = 0.30
IMG_SIZE = 640
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
WINDOW_NAME = "Live Vision Assistant"

# =========================
# AUDIO INPUT CONFIG
# =========================
MIC_INDEX = 1
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAME_SIZE = 1280

CHUNK_DURATION = 0.5
SILENCE_THRESHOLD = 200
SILENCE_DURATION = 1.5
START_SPEECH_THRESHOLD = 300
MIN_COMMAND_SECONDS = 0.8

WAKE_WORDS = [
    "hey kevin",
    "hello kevin",
    "kevin"
]

# =========================
# DETECTION FILTERS
# =========================
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
    "remote"
}

# =========================
# STATE
# =========================
latest_detections = []
frame_lock = threading.Lock()
running = True

whisper_model = whisper.load_model("base")


# =========================
# UTILS
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


def speak_windows_tts(text: str):
    text = str(text).strip()
    if not text:
        return

    print(f"Assistant: {text}")

    safe_text = text.replace("'", "''")
    ps_command = (
        "Add-Type -AssemblyName System.Speech;"
        "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$speak.Rate = 0;"
        f"$speak.Speak('{safe_text}');"
    )

    subprocess.run(
        ["powershell", "-Command", ps_command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_wake_word(text: str) -> bool:
    normalized = normalize_text(text)
    return any(w in normalized for w in WAKE_WORDS)


def remove_wake_word(text: str) -> str:
    cleaned = normalize_text(text)

    for w in WAKE_WORDS:
        if w in cleaned:
            cleaned = cleaned.replace(w, "", 1)
            break

    return cleaned.strip()


def get_position_label(center_x: float, frame_width: int) -> str:
    if center_x < frame_width / 3:
        return "left"
    elif center_x < 2 * frame_width / 3:
        return "center"
    return "right"


def get_distance_label(area_ratio: float) -> str:
    if area_ratio > 0.18:
        return "very close"
    elif area_ratio > 0.10:
        return "close"
    elif area_ratio > 0.05:
        return "medium"
    return "far"


def listen_for_speech_gate(pa: pyaudio.PyAudio) -> bool:
    stream = pa.open(
        rate=SAMPLE_RATE,
        channels=1,
        format=FORMAT,
        input=True,
        input_device_index=MIC_INDEX,
        frames_per_buffer=FRAME_SIZE
    )

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
    sample_rate: int = SAMPLE_RATE,
    chunk_duration: float = CHUNK_DURATION,
    silence_threshold: int = SILENCE_THRESHOLD,
    silence_duration: float = SILENCE_DURATION
) -> io.BytesIO:
    chunk_size = int(sample_rate * chunk_duration)

    stream = pa.open(
        rate=sample_rate,
        channels=1,
        format=FORMAT,
        input=True,
        input_device_index=MIC_INDEX,
        frames_per_buffer=chunk_size
    )

    frames = []
    silence_chunks_needed = int(silence_duration / chunk_duration)
    silent_chunks = 0

    try:
        while running:
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)

            audio_np = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_np).mean()

            if volume < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks >= silence_chunks_needed:
                break
    finally:
        stream.stop_stream()
        stream.close()

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

    if sample_rate != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

    audio_array = whisper.pad_or_trim(audio_array)
    mel = whisper.log_mel_spectrogram(audio_array).to(whisper_model.device)

    options = whisper.DecodingOptions(language="en", fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    return result.text.strip()


# =========================
# VISION LOOP
# =========================
def vision_loop():
    global latest_detections, running

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Failed to open camera")
        running = False
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = max(actual_width * actual_height, 1)

    prev_time = time.time()

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
                    "box": (x1, y1, x2, y2)
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
            "Live Vision Assistant Mode",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            "Ask questions directly. Press Q or ESC to exit.",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        with frame_lock:
            latest_detections = detections

        cv2.imshow(WINDOW_NAME, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================
# LLM REASONING
# =========================
def handle_vision_query(command: str, detections: list[dict]) -> str:
    command = normalize_text(command)

    if command in {"stop", "exit", "quit"}:
        return "__EXIT__"

    if "stop vision mode" in command or "exit vision mode" in command:
        return "__EXIT__"

    try:
        llm_input = []
        for d in detections:
            llm_input.append({
                "class": d["class_name"],
                "position": d["position"],
                "distance": d["distance"],
                "confidence": round(d["confidence"], 2)
            })

        if not llm_input:
            llm_input = []

        return ask_llm(command, llm_input)

    except Exception as e:
        print("[LLM ERROR]", e)
        return "I could not process that question right now."


# =========================
# VOICE LOOP
# =========================
def voice_loop():
    global running

    pa = pyaudio.PyAudio()

    try:
        while running:
            heard = listen_for_speech_gate(pa)
            if not heard or not running:
                continue

            raw_audio = record_until_silence(pa, SAMPLE_RATE)

            raw_audio.seek(0, io.SEEK_END)
            size_in_bytes = raw_audio.tell()
            raw_audio.seek(0)

            approx_num_samples = size_in_bytes // 2
            approx_seconds = approx_num_samples / SAMPLE_RATE

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

            print(f"You said: {transcript}")

            command = normalize_text(transcript)

            # Kalau user masih menyebut wake word, buang saja.
            if contains_wake_word(command):
                command = remove_wake_word(command)

            if not command:
                speak_windows_tts("Ready.")
                continue

            with frame_lock:
                detections_copy = list(latest_detections)

            answer = handle_vision_query(command, detections_copy)

            if answer == "__EXIT__":
                speak_windows_tts("Stopping live vision assistant.")
                running = False
                break

            speak_windows_tts(answer)

    finally:
        pa.terminate()


# =========================
# MAIN
# =========================
def main():
    global running

    speak_windows_tts("Live vision assistant started. You can ask questions now.")

    vision_thread = threading.Thread(target=vision_loop, daemon=True)
    voice_thread = threading.Thread(target=voice_loop, daemon=True)

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
        print("[INFO] Live vision assistant terminated.")


if __name__ == "__main__":
    main()