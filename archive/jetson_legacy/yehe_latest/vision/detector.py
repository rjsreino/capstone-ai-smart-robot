import cv2
import time
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np

from ultralytics import YOLO

import pyzed.sl as sl

import shared.state as state

from config.settings import *

from vision.depth import (
    get_zed_depth_distance,
)

from vision.scene_logic import (
    auto_announce,
)

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

def vision_loop():
    global latest_detections, latest_frame, running
    
    zed = None
    runtime = None
    image = None
    depth = None

    if USE_ZED_DEPTH:

        zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.coordinate_units = sl.UNIT.METER
        init_params.camera_resolution = sl.RESOLUTION.VGA

        status = zed.open(init_params)

        if status != sl.ERROR_CODE.SUCCESS:
            print("[ZED ERROR] Failed to open ZED camera")
            running = False
            return

        runtime = sl.RuntimeParameters()

        image = sl.Mat()
        depth = sl.Mat()

        print("[ZED] Depth camera initialized.")
    
    model = YOLO(MODEL_PATH)
    model.to(device)
    if USE_ZED_DEPTH:

        actual_width = 1280
        actual_height = 720

    else:

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
        if USE_ZED_DEPTH:

            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            frame = image.get_data()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        else:

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
                depth_distance = None

                if USE_ZED_DEPTH:
                    depth_distance = get_zed_depth_distance(
                        depth,
                        x1,
                        y1,
                        x2,
                        y2
                    )

                if depth_distance is not None:

                    if depth_distance < 0.8:
                        distance = "very close"
                    elif depth_distance < 1.5:
                        distance = "close"
                    elif depth_distance < 3.0:
                        distance = "medium"
                    else:
                        distance = "far"

                else:
                    distance = get_distance_label(area_ratio)
                    

                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "position": position,
                    "distance": distance,
                    "area_ratio": area_ratio,
                    "box": (x1, y1, x2, y2),
                    "depth_meters": depth_distance,
                })

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{class_name} | {position} | {distance}"

                if depth_distance is not None:
                    label += f" | {depth_distance:.2f}m"

                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

        detections.sort(key=lambda d: d["area_ratio"], reverse=True)

        with state.frame_lock:
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

    if USE_ZED_DEPTH:
        try:
            zed.close()
        except:
            pass

    else:

        try:
            cap.release()
        except:
            pass
    cv2.destroyAllWindows()