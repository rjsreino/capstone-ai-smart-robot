import time

import shared.state as state

from config.settings import *

from vision.ocr import (
    extract_text_from_frame,
)

from audio.tts import (
    enqueue_speech,
)

from utils.text_utils import (
    remember_response,
)


def should_announce(message: str, now: float) -> bool:
    global last_global_announce_time

    if now - last_global_announce_time < GLOBAL_COOLDOWN:
        return False

    last_time = state.last_spoken_times.get(message, 0.0)
    if now - last_time < MESSAGE_COOLDOWN:
        return False

    state.last_spoken_times[message] = now
    last_global_announce_time = now
    return True


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
    global last_manual_speech_time
    
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

    maybe_announce_text(frame, now)
    
    
def evaluate_scene_events(detections: list[dict]) -> list[dict]:
    events = []
    center_objects = [d for d in detections if d["position"] == "center"]

    for d in detections:
        class_name = d["class_name"]
        position = d["position"]
        distance = d["distance"]
        depth_meters = d.get("depth_meters")

        if (
            position == "center"
            and depth_meters is not None
            and depth_meters < DEPTH_COLLISION_THRESHOLD
        ):
            events.append({
                "priority": 200,
                "message": "Immediate obstacle ahead."
            })
            continue
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