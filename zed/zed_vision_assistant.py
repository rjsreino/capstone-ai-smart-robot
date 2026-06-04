#!/usr/bin/env python3
"""
ZED Live Vision Assistant
Integrates:
1. ZED Camera Depth Safety Grid Processing & Directional HUD Guidance
2. YOLOv8 Object Detection with center-depth mapping
3. EasyOCR Text Recognition
4. Voice Interaction with OpenWakeWord (Jarvis), Whisper (tiny), and Edge-TTS
"""

import cv2
import time
import threading
import io
import wave
import sys
import os

# Ensure FFmpeg is in the path for Whisper subprocess calls on Windows (dynamic for all users)
if sys.platform == "win32":
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        ffmpeg_dir = os.path.join(
            local_app_data,
            "Microsoft",
            "WinGet",
            "Packages",
            "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe",
            "ffmpeg-8.1.1-full_build",
            "bin"
        )
        if os.path.exists(ffmpeg_dir) and ffmpeg_dir not in os.environ["PATH"]:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

import queue
import tempfile
import asyncio
import argparse
import json
import base64
import websockets
from typing import Optional, Dict

import numpy as np
import pyaudio
import sounddevice as sd
import soundfile as sf
import whisper
import torch
from openwakeword.model import Model
from ultralytics import YOLO
import easyocr
import pygame
import edge_tts

# Add current folder to path to load local files
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zed_depth_processor import ZedDepthProcessor, ZedDepthConfig
from llm_reasoner import ask_llm

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCO_MODEL_PATH = "yolov8n.pt"
DEFAULT_LANDMARK_MODEL_PATHS = (
    "runs/detect/exit_sign_only/weights/best.pt;"
    "runs/detect/exit_sign_only_v2/weights/best.pt;"
    "runs/detect/door_local_v1/weights/best.pt"
)
LANDMARK_MODEL_PATHS = [
    path.strip()
    for path in os.getenv("VICKY_YOLO_MODEL", DEFAULT_LANDMARK_MODEL_PATHS).replace(",", ";").split(";")
    if path.strip()
]
COCO_CONFIDENCE = float(os.getenv("VICKY_COCO_CONF", "0.20"))
COCO_IMAGE_SIZE = int(os.getenv("VICKY_COCO_IMGSZ", "320"))
LANDMARK_CONFIDENCE = float(os.getenv("VICKY_LANDMARK_CONF", "0.10"))
LANDMARK_IMAGE_SIZE = int(os.getenv("VICKY_LANDMARK_IMGSZ", "640"))
EXIT_SIGN_CONFIDENCE = float(os.getenv("VICKY_EXIT_SIGN_CONF", "0.45"))
EXIT_SIGN_MAX_AREA_RATIO = float(os.getenv("VICKY_EXIT_SIGN_MAX_AREA", "0.12"))
MIC_INDEX = -1             # -1 for auto-detect microphone index
SD_WAKE_DEVICE_INDEX = -1    # -1 for default sounddevice device index
WAKEWORD_NAME = "jarvis"
TTS_VOICE = "en-US-GuyNeural"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Server connection defaults
SERVER_URI = "127.0.0.1:8000"
SESSION_ID = "session_assistant_prod"

# Telemetry Queue for background streaming (max 5 frames to avoid memory bloat)
telemetry_queue = queue.Queue(maxsize=5)

def push_telemetry(frame, zones, detections, srt_ms, inference_ms, active_semantic_path=False):
    """Safely pushes a frame and telemetry info to the background thread queue (drops oldest if full)."""
    if telemetry_queue.full():
        try:
            telemetry_queue.get_nowait()
        except queue.Empty:
            pass
    try:
        telemetry_queue.put_nowait((frame.copy() if frame is not None else None, {
            "zones": zones.copy() if zones else None,
            "detections": [d.copy() for d in detections] if detections else [],
            "srt_ms": srt_ms,
            "inference_ms": inference_ms,
            "active_semantic_path": active_semantic_path,
            "timestamp": time.time()
        }))
    except queue.Full:
        pass

ALLOWED_CLASSES = {
    "person", "chair", "couch", "bench", "dining table",
    "bottle", "backpack", "potted plant", "cell phone", "cup",
    "laptop", "book", "handbag", "suitcase", "bed", "tv",
    "keyboard", "mouse", "remote",
    "door", "open door", "closed door", "doorway", "door local", "exit sign",
    "open_door", "closed_door", "door_local", "exit_sign"
}

STATIC_NAVIGATION_LABELS = {
    "door": "door",
    "open door": "doorway",
    "open_door": "doorway",
    "closed door": "door",
    "closed_door": "door",
    "doorway": "doorway",
    "door local": "doorway",
    "door_local": "doorway",
    "exit sign": "exit sign",
    "exit_sign": "exit sign",
}


def normalize_detection_label(label: str) -> str:
    return str(label or "").strip().lower().replace("_", " ").replace("-", " ")


def looks_like_exit_sign(frame: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return False

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 45, 45]), np.array([90, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 60, 80]), np.array([38, 255, 255]))
    sign_color_ratio = float(cv2.countNonZero(green_mask | yellow_mask)) / max(roi.shape[0] * roi.shape[1], 1)

    return sign_color_ratio >= 0.025


def resolve_model_path(model_path: str) -> str:
    if os.path.isabs(model_path) or model_path == COCO_MODEL_PATH:
        return model_path

    cwd_path = os.path.abspath(model_path)
    if os.path.exists(cwd_path):
        return cwd_path

    project_path = os.path.join(PROJECT_ROOT, model_path)
    if os.path.exists(project_path):
        return project_path

    return model_path

# Cooldown and threshold configurations
GLOBAL_COOLDOWN = 2.5
MESSAGE_COOLDOWN = 4.0
CLEAR_PATH_COOLDOWN = 7.0
USER_INTERACTION_COOLDOWN = 5.0

# ==========================================
# GLOBAL STATE
# ==========================================
running = True
voice_interaction_active = False
tts_playing = False
telemetry_source_active = False

latest_detections = []
latest_frame = None
latest_depth_map = None
frame_lock = threading.Lock()

speech_queue = queue.Queue()
speech_lock = threading.Lock()

# Spoken alert history & timestamps
last_response_text = ""
last_global_announce_time = 0.0
last_spoken_times = {}
last_path_clear_time = 0.0
last_user_interaction_time = 0.0
last_manual_speech_time = 0.0

# Spatial safety parameters
safe_distance_threshold = 1200.0  # mm
guidance_cmd = "STOP"
guidance_color = (0, 0, 255)
zones_data = {'left': 0.0, 'center': 0.0, 'right': 0.0}
pose_data = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
occupancy_grid = np.zeros((100, 100), dtype=np.int8).tolist()
reset_map_flag = False
semantic_objects = []
active_semantic_path = False

def trigger_map_reset():
    global reset_map_flag
    reset_map_flag = True
    print("[zva] Map reset flag set to True.")

ocr_reader = None

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_position_label(center_x: float, frame_width: int) -> str:
    if center_x < frame_width / 3:
        return "left"
    if center_x < 2 * frame_width / 3:
        return "center"
    return "right"

def get_input_device_index(pa: pyaudio.PyAudio) -> Optional[int]:
    if MIC_INDEX >= 0:
        print(f"[MIC] Using hardcoded microphone index: {MIC_INDEX}")
        return MIC_INDEX

    blocked_keywords = [
        "voicemod", "steelseries", "sonar", "steam", "intelligo",
        "virtual", "vad", "speaker", "output",
        "microphone (realtek hd audio mic input)"
    ]

    try:
        default_info = pa.get_default_input_device_info()
        name = default_info["name"].lower()
        if not any(word in name for word in blocked_keywords):
            print(f"[MIC] Using default microphone: {default_info['index']} | {default_info['name']}")
            return int(default_info["index"])
        print(f"[MIC] Default mic is virtual or skipped: {default_info['name']}")
    except Exception:
        print("[MIC] No default mic found. Scanning input devices...")

    fallback_devices = []
    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
            name = info["name"].lower()
            max_channels = int(info.get("maxInputChannels", 0))
            if max_channels <= 0:
                continue
            if any(word in name for word in blocked_keywords):
                continue
            fallback_devices.append(i)
        except Exception:
            continue

    for i in fallback_devices:
        try:
            info = pa.get_device_info_by_index(i)
            pa.is_format_supported(
                rate=int(info["defaultSampleRate"]),
                input_device=i,
                input_channels=1,
                input_format=pyaudio.paInt16,
            )
            print(f"[MIC] Selected fallback physical mic: {i} | {info['name']}")
            return i
        except Exception:
            continue

    print("[MIC WARNING] No physical microphone found matching format requirements.")
    return None

# ==========================================
# BACKGROUND WEBSOCKET TELEMETRY LOOP
# ==========================================
async def telemetry_sender_loop():
    global running
    print(f"[TELEMETRY] Starting async sender loop. Targeting: {SERVER_URI}")
    
    uri_tele = f"ws://{SERVER_URI}/ws/telemetry/stream"
    uri_video = f"ws://{SERVER_URI}/ws/video/stream"
    
    ws_tele = None
    ws_video = None
    
    while running:
        if ws_tele is None or ws_video is None:
            try:
                print(f"[TELEMETRY] Attempting to connect to FastAPI Server (telemetry/video streams)...")
                ws_tele = await websockets.connect(uri_tele, close_timeout=2)
                ws_video = await websockets.connect(uri_video, close_timeout=2)
                print("[TELEMETRY] Connected to server successfully.")
            except Exception as e:
                print(f"[TELEMETRY WARNING] Server connection failed: {e}. Retrying in 5 seconds...")
                ws_tele = None
                ws_video = None
                for _ in range(50):
                    if not running:
                        break
                    await asyncio.sleep(0.1)
                continue

        try:
            try:
                frame, data = telemetry_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
                
            if frame is not None:
                _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                try:
                    await ws_video.send(jpeg_buf.tobytes())
                except Exception as ve:
                    print(f"[TELEMETRY ERROR] Video socket send failed: {ve}")
                    ws_video = None
                    ws_tele = None
                    continue

            pose = {
                "position_meters": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rotation_degrees": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
            }
            
            zones_payload = {
                "left_clearance_mm": 5000.0,
                "center_clearance_mm": 5000.0,
                "right_clearance_mm": 5000.0,
                "escape_vector": "STOP"
            }
            if data["zones"]:
                zones_payload = {
                    "left_clearance_mm": float(data["zones"].get("left", 5000.0)),
                    "center_clearance_mm": float(data["zones"].get("center", 5000.0)),
                    "right_clearance_mm": float(data["zones"].get("right", 5000.0)),
                    "escape_vector": guidance_cmd
                }

            frustum_objects = []
            for idx, d in enumerate(data["detections"]):
                depth_m = d.get("depth_meters") or 0.0
                lateral_offset = 0.0
                if d.get("box") and frame is not None:
                    h, w, _ = frame.shape
                    x1, y1, x2, y2 = d["box"]
                    cx = (x1 + x2) / 2
                    lateral_offset = ((cx - (w / 2.0)) / (w / 2.0)) * depth_m * 0.5
                    
                frustum_objects.append({
                    "tracking_id": idx,
                    "class": d["class_name"],
                    "3d_coordinates": {"x": lateral_offset, "y": 0.0, "z": depth_m},
                    "distance_category": d["distance"]
                })

            payload = {
                "packet_metadata": {
                    "session_id": SESSION_ID,
                    "timestamp": data["timestamp"],
                    "compute_node": "local",
                    "mode_flag": "A"
                },
                "user_spatial_pose": pose,
                "spatial_depth_zones": zones_payload,
                "semantic_objects_in_frustum": frustum_objects,
                "active_semantic_path": data.get("active_semantic_path", False),
                "performance_metrics": {
                    "inference_latency_ms": data["inference_ms"],
                    "network_rtt_ms": 0.0,
                    "total_srt_ms": data["srt_ms"],
                    "hallucination_flag": False
                }
            }

            try:
                await ws_tele.send(json.dumps(payload))
            except Exception as te:
                print(f"[TELEMETRY ERROR] Telemetry socket send failed: {te}")
                ws_video = None
                ws_tele = None
                continue

            telemetry_queue.task_done()

        except Exception as loop_err:
            print(f"[TELEMETRY CRITICAL] Error in sender loop: {loop_err}")
            await asyncio.sleep(1)

    if ws_tele:
        try:
            await ws_tele.close()
        except:
            pass
    if ws_video:
        try:
            await ws_video.close()
        except:
            pass
    print("[TELEMETRY] Loop ended cleanly.")

def start_telemetry_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(telemetry_sender_loop())
    except Exception as e:
        print(f"[TELEMETRY THREAD ERROR] {e}")
    finally:
        loop.close()

# ==========================================
# AUDIO PLAYBACK & SPEECH SYNTHESIS
# ==========================================
async def async_edge_tts(text: str):
    global tts_playing
    tts_playing = True
    communicate = edge_tts.Communicate(text=text, voice=TTS_VOICE, rate="+0%")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        temp_path = f.name

    try:
        await communicate.save(temp_path)
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy() and running:
            await asyncio.sleep(0.05)
    except Exception as e:
        print(f"[TTS PLAYBACK ERROR] {e}")
    finally:
        tts_playing = False
        try:
            pygame.mixer.music.unload()
        except:
            pass
        try:
            os.remove(temp_path)
        except:
            pass

def speak(text: str):
    text = str(text).strip()
    if not text:
        return
    # Normalize standard speech contractions
    text = text.replace("it's", "it is").replace("It's", "It is")
    
    try:
        asyncio.run(async_edge_tts(text))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(async_edge_tts(text))
        loop.close()

def enqueue_speech(text: str):
    if text.strip():
        speech_queue.put(text.strip())

def speech_worker():
    global running
    while running:
        try:
            text = speech_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            with speech_lock:
                speak(text)
        except Exception as e:
            print(f"[TTS WORKER ERROR] {e}")
        
        speech_queue.task_done()

def clear_speech_queue():
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except queue.Empty:
            break

# ==========================================
# PROACTIVE SCENE EVENTS LOGIC
# ==========================================
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

def remember_response(text: str):
    global last_response_text
    last_response_text = text

def maybe_announce_clear_path(now: float):
    global last_path_clear_time
    if now - last_path_clear_time < CLEAR_PATH_COOLDOWN:
        return
    msg = "Path ahead appears clear."
    if should_announce(msg, now):
        remember_response(msg)
        enqueue_speech(msg)
        last_path_clear_time = now

def evaluate_scene_events(detections: list[dict]) -> list[dict]:
    events = []
    center_objects = [d for d in detections if d["position"] == "center"]

    for d in detections:
        class_name = d["class_name"]
        position = d["position"]
        distance = d["distance"]
        depth_meters = d.get("depth_meters")

        if position == "center" and depth_meters is not None and depth_meters < 0.8:
            events.append({
                "priority": 200,
                "message": f"Immediate obstacle ahead: a {class_name}."
            })
            continue

        if position == "center" and distance == "very close":
            events.append({
                "priority": 100,
                "message": f"Warning: {class_name} very close ahead."
            })
        elif position == "center" and distance == "close":
            events.append({
                "priority": 85,
                "message": f"Caution: {class_name} ahead."
            })
        elif class_name == "person" and position == "center" and distance in {"close", "medium"}:
            events.append({
                "priority": 80,
                "message": "Person detected ahead."
            })
        elif position in {"left", "right"} and distance == "very close":
            events.append({
                "priority": 75,
                "message": f"Warning: {class_name} very close on your {position}."
            })
        elif position in {"left", "right"} and distance == "close":
            events.append({
                "priority": 60,
                "message": f"A {class_name} is on your {position}."
            })

    if not center_objects:
        events.append({
            "priority": 15,
            "message": "Path ahead appears clear."
        })

    return sorted(events, key=lambda x: x["priority"], reverse=True)

def auto_announce(detections: list[dict]):
    global last_user_interaction_time, voice_interaction_active, last_manual_speech_time, guidance_cmd
    if time.time() - last_manual_speech_time < 3.0:
        return
    if voice_interaction_active:
        return
    if time.time() - last_user_interaction_time < USER_INTERACTION_COOLDOWN:
        return

    now = time.time()
    
    if guidance_cmd == "ENTER DOORWAY":
        msg = "I detect an open doorway approximately 2 meters ahead in the center corridor. The surrounding side walls are tight; let's head straight through the opening to explore further."
        if should_announce(msg, now):
            remember_response(msg)
            enqueue_speech(msg)
        return

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

# ==========================================
# EASYOCR TEXT RECOGNITION
# ==========================================
def extract_text_from_frame(frame) -> str:
    if frame is None or ocr_reader is None:
        return ""

    h, w = frame.shape[:2]
    # Crop to a central Region of Interest (ROI) to speed up OCR and filter background noise
    x1, x2 = int(w * 0.15), int(w * 0.85)
    y1, y2 = int(h * 0.15), int(h * 0.75)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return ""

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        results = ocr_reader.readtext(thresh)
    except Exception as e:
        print(f"[OCR ERROR] Reading frame failed: {e}")
        return ""

    filtered_lines = []
    seen = set()

    for _, text, conf in results:
        text = text.strip()
        if len(text) < 3 or conf < 0.30:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        filtered_lines.append(text)

    return " ".join(filtered_lines)

# ==========================================
# VISION PIPELINE LOOP
# ==========================================
def vision_loop():
    global latest_detections, latest_frame, latest_depth_map, running
    global guidance_cmd, guidance_color, zones_data, safe_distance_threshold
    global pose_data, occupancy_grid, reset_map_flag, semantic_objects, active_semantic_path

    # Configure ZED 1 Camera for USB 2.0 fallback connection (VGA @ 15fps)
    config = ZedDepthConfig(
        resolution="vga",
        fps=15,
        depth_mode="PERFORMANCE", # Reliable fallback bypasses neural/TRT hangs
        min_depth=400,
        max_depth=5000
    )
    
    processor = ZedDepthProcessor(config)
    try:
        processor.start()
    except Exception as e:
        print(f"[VISION FATAL] Could not initialize ZED camera: {e}")
        running = False
        return

    # Initialize YOLO models. COCO remains active for safety objects; custom model adds semantic landmarks.
    print(f"[YOLO] Loading COCO safety model: {COCO_MODEL_PATH}")
    coco_model = YOLO(COCO_MODEL_PATH)
    coco_model.to(device)
    detection_models = [("coco", coco_model)]

    for index, landmark_model_path in enumerate(LANDMARK_MODEL_PATHS, start=1):
        resolved_landmark_model_path = resolve_model_path(landmark_model_path)
        print(f"[YOLO] Loading landmark model {index}: {resolved_landmark_model_path}")
        landmark_model = YOLO(resolved_landmark_model_path)
        landmark_model.to(device)
        detection_models.append((f"landmark_{index}", landmark_model))

    print(f"[YOLO] Running on device: {device}")

    prev_time = time.time()
    last_auto_announce_time = 0.0
    last_exit_check_time = 0.0
    last_exit_announce_time = 0.0
    persistent_exit_signs = {}

    display_enabled = os.getenv("VICKY_ENABLE_DISPLAY", "1").strip().lower() not in {"0", "false", "no"}
    if display_enabled:
        try:
            cv2.namedWindow("ZED Spatial Live Vision Assistant")
        except cv2.error as e:
            print(f"[VISION] OpenCV display unavailable; continuing headless: {e}")
            display_enabled = False

    while running:
        # Check if map reset is requested
        if reset_map_flag:
            with frame_lock:
                processor.occupancy_grid.fill(0)
                occupancy_grid = processor.occupancy_grid.tolist()
                reset_map_flag = False
            print("[VISION] Persistent occupancy grid reset completed in processor.")

        frame_start_time = time.time()
        # Grab frame from ZED
        if not processor.grab_frame():
            time.sleep(0.01)
            continue

        rgb_frame = processor.get_rgb_frame()
        depth_frame = processor.get_depth_frame() # in mm

        if rgb_frame is None or depth_frame is None:
            continue

        # 1. Navigation spatial zones analysis
        nav_data = processor.process_depth_for_navigation(depth_frame)
        if nav_data is not None:
            zones = nav_data['zones']
            left_dist = zones['left']['median']
            center_dist = zones['center']['median']
            right_dist = zones['right']['median']
            
            if not telemetry_source_active:
                zones_data = {
                    'left': left_dist,
                    'center': center_dist,
                    'right': right_dist
                }

                # Safety Threshold Calculation
                CRITICAL_STOP_DIST = 500.0  # mm
                if center_dist < CRITICAL_STOP_DIST or (left_dist < CRITICAL_STOP_DIST and right_dist < CRITICAL_STOP_DIST):
                    guidance_cmd = "STOP! DANGER"
                    guidance_color = (0, 0, 255)
                elif center_dist >= safe_distance_threshold:
                    guidance_cmd = "GO FORWARD"
                    guidance_color = (0, 255, 0)
                else:
                    if left_dist > right_dist:
                        guidance_cmd = "TURN LEFT"
                        guidance_color = (0, 255, 255)
                    else:
                        guidance_cmd = "TURN RIGHT"
                        guidance_color = (255, 128, 0)

        # 2. YOLO Object Detection
        annotated = rgb_frame.copy()
        
        h, w, _ = rgb_frame.shape
        frame_area = max(w * h, 1)
        detections = []
        temp_semantic_objects = []

        tx_m = processor.tx / 1000.0
        tz_m = processor.tz / 1000.0
        yaw_rad = np.radians(processor.yaw)
        fov_rad = np.radians(90.0)

        for model_source, model in detection_models:
            is_landmark_model = model_source.startswith("landmark")
            confidence = LANDMARK_CONFIDENCE if is_landmark_model else COCO_CONFIDENCE
            image_size = LANDMARK_IMAGE_SIZE if is_landmark_model else COCO_IMAGE_SIZE
            results = model(
                rgb_frame,
                conf=confidence,
                imgsz=image_size,
                device=device,
                verbose=False
            )
            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    class_name = normalize_detection_label(model.names[cls_id])
                    if class_name not in ALLOWED_CLASSES:
                        continue

                    box_area = max(0, x2 - x1) * max(0, y2 - y1)
                    area_ratio = box_area / frame_area

                    min_area_ratio = 0.01 if is_landmark_model else 0.03
                    if area_ratio < min_area_ratio:
                        continue
                    if class_name == "person" and area_ratio < 0.10:
                        continue
                    if class_name == "exit sign":
                        if conf < EXIT_SIGN_CONFIDENCE:
                            continue
                        if area_ratio > EXIT_SIGN_MAX_AREA_RATIO:
                            continue
                        if not looks_like_exit_sign(rgb_frame, (x1, y1, x2, y2)):
                            continue

                    # Compute depth distance of bounding box center
                    center_x = max(0, min(int((x1 + x2) / 2), w - 1))
                    center_y = max(0, min(int((y1 + y2) / 2), h - 1))

                    depth_val_mm = depth_frame[center_y, center_x]
                    depth_distance = None

                    if not (np.isnan(depth_val_mm) or np.isinf(depth_val_mm) or depth_val_mm <= 0):
                        depth_distance = float(depth_val_mm) / 1000.0  # mm to meters

                    if depth_distance is not None:
                        # Project object onto 2D grid coordinates (invert sign to fix mirroring)
                        angle_rad = -fov_rad/2.0 + center_x * (fov_rad / w)
                        x_c = -depth_distance * np.sin(angle_rad)
                        z_c = depth_distance * np.cos(angle_rad)
                        x_w = tx_m + x_c * np.cos(yaw_rad) + z_c * np.sin(yaw_rad)
                        z_w = tz_m - x_c * np.sin(yaw_rad) + z_c * np.cos(yaw_rad)
                        grid_x = max(0, min(int(x_w / 0.1) + 50, 99))
                        grid_z = max(0, min(int(z_w / 0.1) + 50, 99))
                        semantic_label = STATIC_NAVIGATION_LABELS.get(class_name, class_name)
                        temp_semantic_objects.append({
                            "label": semantic_label,
                            "detected_label": class_name,
                            "source_model": model_source,
                            "x": grid_x,
                            "z": grid_z,
                            "distance": depth_distance,
                            "confidence": conf,
                        })

                        if depth_distance < 0.8:
                            distance_lbl = "very close"
                        elif depth_distance < 1.5:
                            distance_lbl = "close"
                        elif depth_distance < 3.0:
                            distance_lbl = "medium"
                        else:
                            distance_lbl = "far"
                    else:
                        # Fallback to area-ratio based distance label
                        if area_ratio > 0.18:
                            distance_lbl = "very close"
                        elif area_ratio > 0.10:
                            distance_lbl = "close"
                        elif area_ratio > 0.05:
                            distance_lbl = "medium"
                        else:
                            distance_lbl = "far"

                    position = get_position_label(center_x, w)

                    detections.append({
                        "class_name": class_name,
                        "confidence": conf,
                        "position": position,
                        "distance": distance_lbl,
                        "area_ratio": area_ratio,
                        "box": (x1, y1, x2, y2),
                        "depth_meters": depth_distance,
                        "source_model": model_source,
                    })

                    # Draw YOLO bounding box & distance label
                    box_color = (0, 180, 255) if is_landmark_model else (0, 0, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    label = f"{class_name} | {position} | {distance_lbl}"
                    if depth_distance is not None:
                        label += f" | {depth_distance:.1f}m"

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

        # 3. Exit Sign Detection (OCR + Green Color Segmentation)
        now = time.time()
        
        # Check for Frontier Peak Depth Maximum
        gdm = zones.get('global_depth_max') if (nav_data is not None and 'zones' in nav_data) else None
        has_depth_peak = False
        segment_x1, segment_x2 = 0, w
        if gdm and gdm.get('value', 0.0) > 1800.0:
            has_depth_peak = True
            peak_col = gdm.get('col', w // 2)
            segment_x1 = max(0, peak_col - 120)
            segment_x2 = min(w, peak_col + 120)
            
        if now - last_exit_check_time >= 1.5:
            last_exit_check_time = now
            
            # Pass the corresponding RGB frame segment bounding coordinates to OCR or color segmentation
            if has_depth_peak:
                upper_roi = rgb_frame[0:int(h * 0.75), segment_x1:segment_x2]
                roi_offset_x = segment_x1
            else:
                upper_roi = rgb_frame[0:int(h * 0.75), :]
                roi_offset_x = 0
            
            # A. EasyOCR text check
            if ocr_reader is not None:
                try:
                    ocr_results = ocr_reader.readtext(upper_roi)
                    for bbox, text, conf in ocr_results:
                        text_clean = text.strip().upper()
                        if "EXIT" in text_clean and conf >= 0.35:
                            cx = roi_offset_x + int((bbox[0][0] + bbox[2][0]) / 2)
                            cy = int((bbox[0][1] + bbox[2][1]) / 2)
                            
                            depth_val_mm = depth_frame[cy, cx]
                            if not (np.isnan(depth_val_mm) or np.isinf(depth_val_mm) or depth_val_mm <= 0):
                                depth_m = float(depth_val_mm) / 1000.0
                                
                                # Invert sign to fix mirroring
                                angle_rad = -fov_rad/2.0 + cx * (fov_rad / w)
                                x_c = -depth_m * np.sin(angle_rad)
                                z_c = depth_m * np.cos(angle_rad)
                                x_w = tx_m + x_c * np.cos(yaw_rad) + z_c * np.sin(yaw_rad)
                                z_w = tz_m - x_c * np.sin(yaw_rad) + z_c * np.cos(yaw_rad)
                                
                                grid_x_exit = max(0, min(int(x_w / 0.1) + 50, 99))
                                grid_z_exit = max(0, min(int(z_w / 0.1) + 50, 99))
                                
                                persistent_exit_signs[(grid_x_exit, grid_z_exit)] = (now, depth_m)
                                
                                if now - last_exit_announce_time >= 10.0:
                                    announce_msg = f"Exit sign detected {depth_m:.1f} meters ahead."
                                    print(f"[zva OCR] Exit announcement: {announce_msg}")
                                    if should_announce(announce_msg, now):
                                        remember_response(announce_msg)
                                        enqueue_speech(announce_msg)
                                    last_exit_announce_time = now
                except Exception as e:
                    print(f"[zva OCR ERROR] Exit sign scan failed: {e}")
            
            # B. Green Color & Aspect Ratio segmentation (for running-man signs without text)
            try:
                hsv = cv2.cvtColor(upper_roi, cv2.COLOR_BGR2HSV)
                # Emerald green bounds
                lower_green = np.array([35, 75, 75])
                upper_green = np.array([85, 255, 255])
                mask = cv2.inRange(hsv, lower_green, upper_green)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 120 <= area <= 15000:
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        aspect_ratio = float(bw) / bh
                        fill_ratio = area / (bw * bh)
                        
                        # Green emergency exit signs are rectangular with high fill ratio
                        if 1.2 <= aspect_ratio <= 2.8 and fill_ratio >= 0.7:
                            cx = roi_offset_x + bx + bw // 2
                            cy = by + bh // 2
                            
                            depth_val_mm = depth_frame[cy, cx]
                            if not (np.isnan(depth_val_mm) or np.isinf(depth_val_mm) or depth_val_mm <= 0):
                                depth_m = float(depth_val_mm) / 1000.0
                                
                                # Invert sign to fix mirroring
                                angle_rad = -fov_rad/2.0 + cx * (fov_rad / w)
                                x_c = -depth_m * np.sin(angle_rad)
                                z_c = depth_m * np.cos(angle_rad)
                                x_w = tx_m + x_c * np.cos(yaw_rad) + z_c * np.sin(yaw_rad)
                                z_w = tz_m - x_c * np.sin(yaw_rad) + z_c * np.cos(yaw_rad)
                                
                                grid_x_exit = max(0, min(int(x_w / 0.1) + 50, 99))
                                grid_z_exit = max(0, min(int(z_w / 0.1) + 50, 99))
                                
                                persistent_exit_signs[(grid_x_exit, grid_z_exit)] = (now, depth_m)
                                
                                if now - last_exit_announce_time >= 10.0:
                                    announce_msg = f"Emergency exit sign detected {depth_m:.1f} meters ahead."
                                    print(f"[zva Color] Green sign announcement: {announce_msg}")
                                    if should_announce(announce_msg, now):
                                        remember_response(announce_msg)
                                        enqueue_speech(announce_msg)
                                    last_exit_announce_time = now
            except Exception as e:
                print(f"[zva Color ERROR] Green sign segmentation failed: {e}")

        # Keep exit signs detected within the last 5.0 seconds
        persistent_exit_signs = {k: v for k, v in persistent_exit_signs.items() if now - v[0] < 5.0}
        
        # Populate persistent exit signs in semantic objects list
        for (gx, gz), (ts, dist) in persistent_exit_signs.items():
            temp_semantic_objects.append({
                "label": "exit sign",
                "x": gx,
                "z": gz,
                "distance": dist
            })

        # Flag Active Semantic Path
        active_semantic_path = False
        if has_depth_peak and len(persistent_exit_signs) > 0:
            active_semantic_path = True
            
        # Override safety guidelines/guidance command
        if active_semantic_path and not telemetry_source_active:
            guidance_cmd = "ENTER DOORWAY"
            guidance_color = (0, 255, 0)

        # Thread-safe global update
        with frame_lock:
            live_grid = processor.occupancy_grid.copy()
            for obj in temp_semantic_objects:
                label = str(obj.get("label") or obj.get("detected_label") or "").lower()
                if label == "person":
                    continue
                gx = obj.get("x")
                gz = obj.get("z")
                if gx is None or gz is None:
                    continue
                gx = max(0, min(int(gx), 99))
                gz = max(0, min(int(gz), 99))
                marker = 2 if ("door" in label or "exit" in label) else 1
                for dz in range(-1, 2):
                    for dx in range(-1, 2):
                        nz = max(0, min(gz + dz, 99))
                        nx = max(0, min(gx + dx, 99))
                        live_grid[nz, nx] = marker

            latest_detections = detections
            latest_frame = rgb_frame.copy()
            latest_depth_map = depth_frame.copy()
            if not telemetry_source_active:
                pose_data = {
                    "x": float(processor.tx),
                    "y": float(processor.ty),
                    "z": float(processor.tz),
                    "roll": float(processor.roll),
                    "pitch": float(processor.pitch),
                    "yaw": float(processor.yaw)
                }
                occupancy_grid = live_grid.tolist()
                semantic_objects = temp_semantic_objects

        # 3. Proactive Auto Announcement
        now = time.time()
        if now - last_auto_announce_time >= 2.0:
            auto_announce(detections)
            last_auto_announce_time = now

        # Calculate Frame Rate (FPS)
        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        total_delay_ms = (current_time - frame_start_time) * 1000.0
        push_telemetry(annotated, zones_data, detections, total_delay_ms, total_delay_ms, active_semantic_path)

        # 4. Render Dashboard Visual Panels
        display_w = 640
        display_h = 360

        rgb_small = cv2.resize(annotated, (display_w, display_h))
        depth_colored = processor.visualize_depth(depth_frame)
        depth_small = cv2.resize(depth_colored, (display_w, display_h))

        # Overlay transparent safety zone colored rectangles
        col_w = display_w // 3
        overlay = rgb_small.copy()
        
        left_color = (0, 0, 255) if left_dist < safe_distance_threshold else (0, 255, 0)
        cv2.rectangle(overlay, (0, 0), (col_w, display_h), left_color, -1)
        center_color = (0, 0, 255) if center_dist < safe_distance_threshold else (0, 255, 0)
        cv2.rectangle(overlay, (col_w, 0), (col_w * 2, display_h), center_color, -1)
        right_color = (0, 0, 255) if right_dist < safe_distance_threshold else (0, 255, 0)
        cv2.rectangle(overlay, (col_w * 2, 0), (display_w, display_h), right_color, -1)

        cv2.addWeighted(overlay, 0.15, rgb_small, 0.85, 0, rgb_small)

        # Draw safety zone line grids
        for col in [col_w, col_w * 2]:
            cv2.line(rgb_small, (col, 0), (col, display_h), (255, 255, 255), 1)
            cv2.line(depth_small, (col, 0), (col, display_h), (255, 255, 255), 1)

        cv2.putText(rgb_small, f"L: {left_dist:.0f}mm", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(rgb_small, f"C: {center_dist:.0f}mm", (col_w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(rgb_small, f"R: {right_dist:.0f}mm", (col_w * 2 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        main_panels = np.hstack((rgb_small, depth_small))

        # Bottom HUD Footer Panel
        hud_h = 120
        hud_w = display_w * 2
        hud = np.zeros((hud_h, hud_w, 3), dtype=np.uint8)
        hud[:] = [24, 20, 18] # slate dark

        cv2.rectangle(hud, (0, 0), (hud_w - 1, hud_h - 1), (50, 50, 50), 1)

        # Draw Guidance instructions box in the center
        cmd_box_x1 = hud_w // 2 - 180
        cmd_box_x2 = hud_w // 2 + 180
        cv2.rectangle(hud, (cmd_box_x1, 15), (cmd_box_x2, 105), guidance_color, -1)
        
        text_size = cv2.getTextSize(guidance_cmd, cv2.FONT_HERSHEY_DUPLEX, 1.1, 3)[0]
        text_x = hud_w // 2 - text_size[0] // 2
        text_y = hud_h // 2 + text_size[1] // 2
        cv2.putText(hud, guidance_cmd, (text_x + 2, text_y + 2), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 3)
        cv2.putText(hud, guidance_cmd, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 3)

        # Left Column: Telemetry
        cv2.putText(hud, "SYSTEM TELEMETRY", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(hud, f"Model: ZED 1 + YOLOv8n", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, f"FPS:   {fps:.1f} Hz", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if fps > 10 else (0, 150, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, f"Voice Assistant: Jarvis", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Right Column: Settings
        cv2.putText(hud, "THRESHOLD SETTINGS", (hud_w - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(hud, f"Safe Distance: {safe_distance_threshold:.0f} mm", (hud_w - 280, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, "Adjust: [+] / [-] keys", (hud_w - 280, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

        if display_enabled:
            dashboard = np.vstack((main_panels, hud))
            cv2.imshow("ZED Spatial Live Vision Assistant", dashboard)

            # Keyboard Interventions
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                running = False
                break
            elif key == ord('+') or key == ord('='):
                safe_distance_threshold = min(4000.0, safe_distance_threshold + 100)
                print(f"[SETTING] Increased Safe Distance Threshold to {safe_distance_threshold:.0f}mm")
            elif key == ord('-') or key == ord('_'):
                safe_distance_threshold = max(500.0, safe_distance_threshold - 100)
                print(f"[SETTING] Decreased Safe Distance Threshold to {safe_distance_threshold:.0f}mm")

    processor.stop()
    if display_enabled:
        cv2.destroyAllWindows()
    print("[VISION] Vision pipeline loop ended cleanly.")

# ==========================================
# WAKE-WORD & TRANSCRIBER SPEECH LOOP
# ==========================================
def is_repeat_command(command: str) -> bool:
    repeat_keywords = [
        "repeat", "say again", "one more time", "repeat it",
        "repeat that", "can you repeat", "say it again"
    ]
    return any(keyword in command for keyword in repeat_keywords)

def handle_vision_query(command: str, detections: list[dict], frame) -> str:
    global last_response_text
    command = command.lower().strip()

    if command in {"stop", "exit", "quit"}:
        return "__EXIT__"

    if "stop vision mode" in command or "exit vision mode" in command:
        return "__EXIT__"

    if is_repeat_command(command):
        if last_response_text:
            return last_response_text
        return "There is nothing recent to repeat."

    # Format detections for LLM prompt payload
    llm_detections = []
    for detection in detections:
        llm_detections.append({
            "class": detection["class_name"],
            "position": detection["position"],
            "distance": detection["distance"],
            "confidence": round(detection["confidence"], 2),
        })

    # Pull OCR text if the user explicitly asks about reading
    ocr_text = ""
    if any(keyword in command for keyword in ["read", "text", "write", "sign", "ocr"]):
        ocr_text = extract_text_from_frame(frame)

    # Compile safety direction metadata
    direction_summary = {
        "left_distance_mm": float(zones_data["left"]),
        "center_distance_mm": float(zones_data["center"]),
        "right_distance_mm": float(zones_data["right"]),
        "best_direction": guidance_cmd
    }

    try:
        answer = ask_llm(
            question=command,
            detections=llm_detections,
            direction_summary=direction_summary,
            ocr_text=ocr_text
        )
        remember_response(answer)
        return answer
    except Exception as e:
        print("[LLM ERROR] Query failure:", e)
        answer = "I could not process that request right now."
        remember_response(answer)
        return answer

def record_until_silence(
    device_index: Optional[int],
    sample_rate: int = 16000,
    chunk_duration: float = 0.1,
    silence_threshold: int = 1800,
    silence_duration: float = 0.8,
) -> io.BytesIO:
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
        device=device_index
    ) as stream:
        wait_start_time = time.time()
        start_time = None

        while running:
            audio, _ = stream.read(chunk_size)
            pcm = np.squeeze(audio).astype(np.int16)
            volume = np.abs(pcm).mean()

            if not speech_detected:
                if volume > 2500:  # START_SPEECH_THRESHOLD
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
                if start_time and time.time() - start_time > 6.0:  # MAX_COMMAND_SECONDS
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

def transcribe_audio(whisper_model, wav_buffer: io.BytesIO) -> str:
    audio_array, sample_rate = sf.read(wav_buffer, dtype="float32")
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)

    if sample_rate != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

    if audio_array.size == 0:
        return ""

    audio_array = whisper.pad_or_trim(audio_array)
    mel = whisper.log_mel_spectrogram(audio_array).to(whisper_model.device)
    options = whisper.DecodingOptions(language="en", fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    return result.text.strip()

def voice_loop():
    global running, voice_interaction_active, last_user_interaction_time, last_manual_speech_time

    # Initialize openwakeword Model
    try:
        print("[VOICE] Loading OpenWakeWord model...")
        wakeword_model = Model(
            wakeword_models=[WAKEWORD_NAME],
            inference_framework="onnx"
        )
        wakeword_model.reset()
    except Exception as e:
        print(f"[VOICE ERROR] Failed to load openwakeword model: {e}")
        return

    # Initialize Whisper Model
    try:
        print("[VOICE] Loading Whisper (tiny) model...")
        whisper_model = whisper.load_model("tiny").to(device)
    except Exception as e:
        print(f"[VOICE ERROR] Failed to load Whisper model: {e}")
        return

    pa = pyaudio.PyAudio()
    device_index = get_input_device_index(pa)
    
    # Sounddevice configurations
    sd_device_index = device_index if device_index is not None and device_index >= 0 else None

    print(f"[VOICE] System ready! Listening for '{WAKEWORD_NAME}' wake word (SD Device: {sd_device_index})...")

    try:
        with sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="int16",
            blocksize=640, # FRAME_SIZE
            device=sd_device_index
        ) as stream:
            
            while running:
                if tts_playing:
                    time.sleep(0.1)
                    continue

                audio, _ = stream.read(640)
                pcm = np.squeeze(audio).astype(np.int16)

                prediction = wakeword_model.predict(pcm)
                score = prediction.get(WAKEWORD_NAME, 0.0)

                if score > 0.35: # WAKEWORD_THRESHOLD
                    print(f"\n[WAKE] Hey Jarvis detected (score: {score:.2f})!")

                    voice_interaction_active = True
                    last_user_interaction_time = time.time()
                    last_manual_speech_time = time.time()

                    clear_speech_queue()

                    # Play wake word response
                    with speech_lock:
                        speak("Yes?")

                    time.sleep(0.3)

                    # Capture user command
                    try:
                        raw_audio = record_until_silence(sd_device_index)
                    except Exception as re:
                        print(f"[VOICE ERROR] Capture failed: {re}")
                        voice_interaction_active = False
                        wakeword_model.reset()
                        continue

                    # Validate command audio
                    if raw_audio.getbuffer().nbytes == 0:
                        print("[VOICE] No speech captured.")
                        voice_interaction_active = False
                        wakeword_model.reset()
                        continue

                    # Transcribe
                    try:
                        print("[VOICE] Transcribing with Whisper...")
                        transcript = transcribe_audio(whisper_model, raw_audio)
                        print(f"[HEARD] '{transcript}'")
                    except Exception as te:
                        print(f"[VOICE ERROR] Transcription failed: {te}")
                        voice_interaction_active = False
                        wakeword_model.reset()
                        continue

                    if not transcript.strip():
                        print("[VOICE] Empty transcript.")
                        voice_interaction_active = False
                        wakeword_model.reset()
                        continue

                    # Process query
                    with frame_lock:
                        detections_copy = list(latest_detections)
                        frame_copy = None if latest_frame is None else latest_frame.copy()

                    try:
                        answer = handle_vision_query(transcript, detections_copy, frame_copy)
                    except Exception as qe:
                        print(f"[VOICE ERROR] Query processing failed: {qe}")
                        answer = "I ran into an issue processing that query."

                    print(f"[ASSISTANT] {answer}")

                    if answer == "__EXIT__":
                        with speech_lock:
                            speak("Stopping live vision assistant.")
                        running = False
                        break

                    clear_speech_queue()

                    # Play TTS Answer
                    try:
                        with speech_lock:
                            speak(answer)
                    except Exception as se:
                        print(f"[VOICE ERROR] TTS playback failed: {se}")

                    last_user_interaction_time = time.time()
                    voice_interaction_active = False
                    wakeword_model.reset()
                    print(f"[VOICE] Listening for '{WAKEWORD_NAME}'...")

    except Exception as e:
        print(f"[VOICE LOOP FATAL ERROR] {e}")
    finally:
        pa.terminate()
        print("[VOICE] Voice loop terminated.")

# ==========================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================
def main():
    global running, ocr_reader, SERVER_URI, SESSION_ID
    print("=" * 60)
    print("      ZED SPATIAL LIVE VISION ASSISTANT (JARVIS)")
    print("=" * 60)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ZED Live Vision Assistant")
    parser.add_argument("--server", type=str, default="127.0.0.1:8000",
                        help="IP and Port of remote FastAPI Cloud Server")
    parser.add_argument("--session", type=str, default="session_assistant_prod",
                        help="Session identifier string")
    args, _ = parser.parse_known_args()
    SERVER_URI = args.server
    SESSION_ID = args.session

    # Initialize Pygame Mixer for sound playing
    pygame.mixer.init()
    pygame.mixer.music.set_volume(1.0)

    # Load OCR Reader
    try:
        print("[OCR] Loading EasyOCR reader...")
        ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        print("[OCR] EasyOCR loaded successfully.")
    except Exception as e:
        print(f"[OCR WARNING] Failed to load EasyOCR: {e}. OCR features disabled.")
        ocr_reader = None

    enqueue_speech(
        "Proactive live vision assistant started. "
        "Automatic guidance is active. "
        "Say Hey Jarvis before optional voice commands."
    )

    # Start multi-threaded loops
    speaker_thread = threading.Thread(target=speech_worker, daemon=True)
    vision_thread = threading.Thread(target=vision_loop, daemon=True)
    voice_thread = threading.Thread(target=voice_loop, daemon=True)
    telemetry_thread = threading.Thread(target=start_telemetry_thread, daemon=True)

    speaker_thread.start()
    vision_thread.start()
    voice_thread.start()
    telemetry_thread.start()

    try:
        while running:
            time.sleep(0.1)
    finally:
        running = False
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("System closed cleanly.")

if __name__ == "__main__":
    main()
