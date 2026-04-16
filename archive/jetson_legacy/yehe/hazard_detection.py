from ultralytics import YOLO
import cv2
import time
import subprocess
import threading
import queue

# =========================
# DETECTION CONFIG
# =========================
MODEL_PATH = "yolov8s.pt"
CAMERA_INDEX = 1
CONF_THRESHOLD = 0.30
IMG_SIZE = 640
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
WINDOW_NAME = "Hazard Detection"

HAZARD_CLASSES = {
    "person",
    "chair",
    "couch",
    "bench",
    "dining table",
    "potted plant",
    "bottle"
}

# minimum ukuran umum
MIN_AREA_RATIO = 0.03

# filter tambahan khusus person agar orang yang lewat kecil/cepat tidak langsung dianggap hazard
MIN_PERSON_AREA_RATIO = 0.10

# =========================
# AUDIO / STATE CONFIG
# =========================
audio_queue = queue.Queue(maxsize=1)

# hazard terakhir yang SUDAH diumumkan
last_spoken_class = None

# kandidat hazard yang sedang diuji kestabilannya
candidate_class = None
candidate_start_time = None

# reset setelah aman selama beberapa detik
safe_since = None
RESET_AFTER_SAFE = 1.0

# =========================
# HELPER FUNCTIONS
# =========================
def get_hazard_message(detected_hazards):
    if len(detected_hazards) == 0:
        return "SAFE: No major obstacle detected", None

    main_hazard = detected_hazards[0]["class_name"]

    if main_hazard == "person":
        return "Warning. Person detected ahead.", main_hazard

    return f"Warning. Obstacle detected. {main_hazard}.", main_hazard


def get_required_stable_time(cls):
    # person dibuat lebih ketat
    if cls == "person":
        return 1.2
    return 0.45


def speak_windows_tts_blocking(text):
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


def audio_worker():
    while True:
        text = audio_queue.get()

        if text is None:
            audio_queue.task_done()
            break

        try:
            print(f"[AUDIO] {text}")
            speak_windows_tts_blocking(text)
        except Exception as e:
            print("[AUDIO ERROR]", e)

        audio_queue.task_done()


def queue_latest_audio(text):
    while not audio_queue.empty():
        try:
            old = audio_queue.get_nowait()
            audio_queue.task_done()
            if old is None:
                break
        except queue.Empty:
            break

    try:
        audio_queue.put_nowait(text)
    except queue.Full:
        pass


def speak_if_stable(text, cls):
    global last_spoken_class, candidate_class, candidate_start_time

    if cls is None:
        return

    now = time.time()

    # kalau kandidat berubah, mulai hitung waktu dari awal
    if cls != candidate_class:
        candidate_class = cls
        candidate_start_time = now
        return

    required_time = get_required_stable_time(cls)
    stable_time = now - candidate_start_time

    # hanya bicara kalau:
    # 1. kandidat sudah stabil cukup lama
    # 2. hazard ini belum pernah diumumkan sebagai hazard aktif sekarang
    if stable_time >= required_time and cls != last_spoken_class:
        print(
            f"[SPEAK-STABLE] {text} | "
            f"stable_time={stable_time:.2f}s | required={required_time:.2f}s"
        )
        queue_latest_audio(text)
        last_spoken_class = cls


# =========================
# MAIN
# =========================
def main():
    global safe_since, last_spoken_class, candidate_class, candidate_start_time

    worker = threading.Thread(target=audio_worker, daemon=True)
    worker.start()

    print("[INFO] Loading model...")
    model = YOLO(MODEL_PATH)

    print("[INFO] Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera index {CAMERA_INDEX}")
        queue_latest_audio(None)
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = actual_width * actual_height

    print(f"[INFO] Camera opened: {actual_width}x{actual_height}")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break

        start = time.time()

        results = model(
            frame,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False
        )

        result = results[0]
        annotated = frame.copy()
        detected_hazards = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                class_name = model.names[cls_id]
                box_area = max(0, x2 - x1) * max(0, y2 - y1)
                area_ratio = box_area / frame_area

                if class_name not in HAZARD_CLASSES:
                    continue

                if area_ratio < MIN_AREA_RATIO:
                    continue

                # filter khusus person
                if class_name == "person" and area_ratio < MIN_PERSON_AREA_RATIO:
                    continue

                detected_hazards.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "area_ratio": area_ratio,
                    "box": (x1, y1, x2, y2)
                })

        # pilih hazard terbesar
        detected_hazards.sort(key=lambda item: item["area_ratio"], reverse=True)

        for hazard in detected_hazards:
            x1, y1, x2, y2 = hazard["box"]
            class_name = hazard["class_name"]
            conf = hazard["confidence"]
            area_ratio = hazard["area_ratio"]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{class_name} {conf:.2f} | area={area_ratio:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        warning_text, main_cls = get_hazard_message(detected_hazards)

        # kalau aman, tunggu sebentar lalu reset state
        if main_cls is None:
            if safe_since is None:
                safe_since = time.time()
            elif time.time() - safe_since >= RESET_AFTER_SAFE:
                last_spoken_class = None
                candidate_class = None
                candidate_start_time = None
        else:
            safe_since = None
            speak_if_stable(warning_text, main_cls)

        print(
            f"[DEBUG] main_cls={main_cls}, "
            f"candidate_class={candidate_class}, "
            f"candidate_start_time={candidate_start_time}, "
            f"last_spoken_class={last_spoken_class}"
        )

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time
        inference_ms = (time.time() - start) * 1000.0

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
            f"Inference: {inference_ms:.1f} ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        color = (0, 255, 0) if warning_text.startswith("SAFE") else (0, 0, 255)
        cv2.putText(
            annotated,
            warning_text,
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.imshow(WINDOW_NAME, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    queue_latest_audio(None)
    print("[INFO] Program terminated cleanly")


if __name__ == "__main__":
    main()