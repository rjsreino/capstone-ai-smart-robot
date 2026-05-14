from ultralytics import YOLO
import cv2
import time

MODEL_PATH = "yolov8s.pt"
CAMERA_INDEX = 1
CONF_THRESHOLD = 0.30
IMG_SIZE = 640
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
WINDOW_NAME = "Object Detection"

SAVE_OUTPUT_VIDEO = False
OUTPUT_VIDEO_PATH = "object_detection_output.mp4"
OUTPUT_FPS = 20.0

def create_video_writer(path: str, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))

def main():
    print("[INFO] Loading model...")
    model = YOLO(MODEL_PATH)

    print("[INFO] Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera index {CAMERA_INDEX}")
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Camera opened: {actual_width}x{actual_height}")
    print(f"[INFO] Model: {MODEL_PATH}")
    print(f"[INFO] Confidence threshold: {CONF_THRESHOLD}")
    print(f"[INFO] Inference image size: {IMG_SIZE}")

    writer = None
    if SAVE_OUTPUT_VIDEO:
        writer = create_video_writer(
            OUTPUT_VIDEO_PATH, OUTPUT_FPS, actual_width, actual_height
        )
        if writer.isOpened():
            print(f"[INFO] Recording to {OUTPUT_VIDEO_PATH}")
        else:
            print("[WARNING] Failed to create video writer")
            writer = None

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

        annotated = results[0].plot()

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time
        inference_ms = (time.time() - start) * 1000.0

        cv2.putText(
            annotated,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
        cv2.putText(
            annotated,
            f"Inference: {inference_ms:.1f} ms",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.imshow(WINDOW_NAME, annotated)

        if writer is not None:
            writer.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Program terminated cleanly")

if __name__ == "__main__":
    main()