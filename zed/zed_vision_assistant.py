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
import queue
import tempfile
import asyncio
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
MODEL_PATH = "yolov8n.pt"
MIC_INDEX = 0             # -1 for auto-detect microphone index
SD_WAKE_DEVICE_INDEX = -1    # -1 for default sounddevice device index
WAKEWORD_NAME = "jarvis"
TTS_VOICE = "en-US-GuyNeural"
device = "cuda" if torch.cuda.is_available() else "cpu"

ALLOWED_CLASSES = {
    "person", "chair", "couch", "bench", "dining table",
    "bottle", "backpack", "potted plant", "cell phone", "cup",
    "laptop", "book", "handbag", "suitcase", "bed", "tv",
    "keyboard", "mouse", "remote"
}

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
    global last_user_interaction_time, voice_interaction_active, last_manual_speech_time
    if time.time() - last_manual_speech_time < 3.0:
        return
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

    # Initialize YOLO Model
    print("[YOLO] Loading model YOLOv8n...")
    model = YOLO(MODEL_PATH)
    model.to(device)
    print(f"[YOLO] Running on device: {device}")

    prev_time = time.time()
    last_auto_announce_time = 0.0

    cv2.namedWindow("ZED Spatial Live Vision Assistant")

    while running:
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
        results = model(
            rgb_frame,
            conf=0.20,
            imgsz=320,
            device=device,
            verbose=False
        )
        result = results[0]
        annotated = rgb_frame.copy()
        
        h, w, _ = rgb_frame.shape
        frame_area = max(w * h, 1)
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

                if area_ratio < 0.03:
                    continue
                if class_name == "person" and area_ratio < 0.10:
                    continue

                # Compute depth distance of bounding box center
                center_x = max(0, min(int((x1 + x2) / 2), w - 1))
                center_y = max(0, min(int((y1 + y2) / 2), h - 1))

                depth_val_mm = depth_frame[center_y, center_x]
                depth_distance = None

                if not (np.isnan(depth_val_mm) or np.isinf(depth_val_mm) or depth_val_mm <= 0):
                    depth_distance = float(depth_val_mm) / 1000.0  # mm to meters

                if depth_distance is not None:
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
                })

                # Draw YOLO bounding box & distance label
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
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

        # Thread-safe global update
        with frame_lock:
            latest_detections = detections
            latest_frame = rgb_frame.copy()
            latest_depth_map = depth_frame.copy()

        # 3. Proactive Auto Announcement
        now = time.time()
        if now - last_auto_announce_time >= 2.0:
            auto_announce(detections)
            last_auto_announce_time = now

        # Calculate Frame Rate (FPS)
        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

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
    global running, ocr_reader
    print("=" * 60)
    print("      ZED SPATIAL LIVE VISION ASSISTANT (JARVIS)")
    print("=" * 60)

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
        except:
            pass
        print("System closed cleanly.")

if __name__ == "__main__":
    main()
