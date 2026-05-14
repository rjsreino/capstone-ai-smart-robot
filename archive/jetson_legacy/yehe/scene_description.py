from ultralytics import YOLO
import cv2
import time
import subprocess
from collections import Counter

MODEL_PATH = "yolov8s.pt"
CAMERA_INDEX = 1
CONF_THRESHOLD = 0.30
IMG_SIZE = 640
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
WINDOW_NAME = "Scene Description"

SCAN_DURATION = 5.0  # seconds


def speak_windows_tts(text):
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


def make_scene_sentence(counter):
    if not counter:
        return "I do not see any major object."

    parts = []
    for obj, count in counter.items():
        if count == 1:
            parts.append(f"one {obj}")
        else:
            parts.append(f"{count} {obj}s")

    if len(parts) == 1:
        return f"I see {parts[0]}."

    if len(parts) == 2:
        return f"I see {parts[0]} and {parts[1]}."

    return "I see " + ", ".join(parts[:-1]) + f", and {parts[-1]}."


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Failed to open camera")
        speak_windows_tts("Failed to open camera.")
        return

    start_time = time.time()
    seen_objects = Counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break

        results = model(
            frame,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False
        )

        result = results[0]
        annotated = frame.copy()
        current_frame_objects = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                class_name = model.names[cls_id].lower()
                current_frame_objects.append(class_name)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    annotated,
                    f"{class_name} {conf:.2f}",
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        # simpan object unik per frame
        for obj in set(current_frame_objects):
            seen_objects[obj] += 1

        elapsed = time.time() - start_time

        cv2.putText(
            annotated,
            f"Scanning scene...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Time left: {max(0, SCAN_DURATION - elapsed):.1f}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(WINDOW_NAME, annotated)

        if elapsed >= SCAN_DURATION:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ambil hanya object yang cukup sering muncul
    filtered_counter = Counter()
    for obj, count in seen_objects.items():
        if count >= 2:
            filtered_counter[obj] = 1

    sentence = make_scene_sentence(filtered_counter)
    print("[SCENE]", sentence)
    speak_windows_tts(sentence)


if __name__ == "__main__":
    main()