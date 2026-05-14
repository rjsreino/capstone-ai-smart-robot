from ultralytics import YOLO
import cv2
import sys
import time
import subprocess
import threading

MODEL_PATH = "yolov8s.pt"
CAMERA_INDEX = 1
CONF_THRESHOLD = 0.30
IMG_SIZE = 640
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
WINDOW_NAME = "Find Object"

SCAN_DURATION = 10.0          # total scan time
FOUND_HOLD_TIME = 3.0         # stay on screen for 3 sec after found


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


def speak_async(text):
    thread = threading.Thread(target=speak_windows_tts, args=(text,), daemon=True)
    thread.start()


def get_position_label(center_x, frame_width):
    if center_x < frame_width / 3:
        return "left"
    elif center_x < 2 * frame_width / 3:
        return "center"
    return "right"


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_object.py <object_name>")
        return

    target_object = sys.argv[1].lower().strip()

    print(f"[INFO] Target object: {target_object}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Failed to open camera")
        speak_windows_tts("Failed to open camera.")
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    start_time = time.time()

    found = False
    found_time = None
    announced = False
    best_detection = None

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

        current_best = None

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                class_name = model.names[cls_id].lower()
                center_x = (x1 + x2) / 2.0
                position = get_position_label(center_x, actual_width)
                area = max(0, x2 - x1) * max(0, y2 - y1)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{class_name} {conf:.2f}",
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                if class_name == target_object:
                    if current_best is None or area > current_best["area"]:
                        current_best = {
                            "class_name": class_name,
                            "confidence": conf,
                            "position": position,
                            "area": area
                        }

        elapsed = time.time() - start_time

        # kalau target ketemu, simpan deteksi terbaik
        if current_best is not None:
            best_detection = current_best

            if not found:
                found = True
                found_time = time.time()

            if not announced:
                message = f"{target_object} detected on the {best_detection['position']}."
                print("[RESULT]", message)
                speak_async(message)
                announced = True

        # overlay info
        cv2.putText(
            annotated,
            f"Looking for: {target_object}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        if found and best_detection is not None:
            status_text = f"FOUND: {target_object} on the {best_detection['position']}"
            status_color = (0, 255, 0)
        else:
            status_text = f"Scanning... {max(0, SCAN_DURATION - elapsed):.1f}s left"
            status_color = (0, 255, 255)

        cv2.putText(
            annotated,
            status_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        cv2.imshow(WINDOW_NAME, annotated)

        # kalau ketemu, tahan 3 detik lalu keluar
        if found and found_time is not None:
            if time.time() - found_time >= FOUND_HOLD_TIME:
                break

        # kalau tidak ketemu sampai timeout
        if not found and elapsed >= SCAN_DURATION:
            message = f"{target_object} not found."
            print("[RESULT]", message)
            speak_async(message)
            time.sleep(2.0)
            break

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()