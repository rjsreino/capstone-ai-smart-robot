from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
import uvicorn
import threading
import zed_vision_assistant as zva
from fastapi.responses import HTMLResponse
import tempfile
import whisper
import time
import os
import socket
from llm_reasoner import ask_llm
from vicky_db import db_logger, AsyncSessionLocal, OccupancyMap
from sqlalchemy import select
import heapq
import numpy as np
from semantic_navigation import SemanticNavigator
from spatial_memory import SpatialMemoryNavigationSystem, classify_object_mobility, object_color_for_label
from map_coordinates import (
    base_yaw_to_display_yaw,
    base_yaw_to_projection_yaw,
    camera_point_to_grid,
    normalize_degrees,
    pose_mm_to_grid,
)

# Global Navigation Goal: grid row Z, grid col X (default 3m forward)
current_goal = (80, 50)

def astar_pathfind(grid: list, start: tuple, goal: tuple) -> list:
    """Runs A* pathfinding on a 100x100 grid. Returns list of (row, col) tuples representing grid path."""
    rows, cols = 100, 100
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return []
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return []
    if start == goal:
        return [start]
        
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    def is_valid(r, c):
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        # Cell == 1 indicates an occupied obstacle
        return grid[r][c] != 1

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    
    def heuristic(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
            
        for dr, dc in neighbors:
            neighbor = (current[0] + dr, current[1] + dc)
            if not is_valid(neighbor[0], neighbor[1]):
                continue
                
            move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
            tentative_g = g_score[current] + move_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))
                
    return []

WHISPER_MODEL_NAME = os.getenv("VICKY_WHISPER_MODEL", "tiny")
WHISPER_DEVICE = os.getenv("VICKY_WHISPER_DEVICE", "cpu")
whisper_model = None
whisper_model_lock = threading.Lock()
app = FastAPI()
navigator = SemanticNavigator()
spatial_memory = SpatialMemoryNavigationSystem()

AUTO_MAPPING_ENABLED = os.getenv("VICKY_AUTO_MAPPING", "1").strip().lower() not in {"0", "false", "no"}
AUTO_MAPPING_NAME = os.getenv("VICKY_AUTO_MAP_NAME", "Auto Room")
AUTO_MAPPING_SAVE_INTERVAL = float(os.getenv("VICKY_AUTO_MAP_SAVE_INTERVAL", "5.0"))
AUTO_MAPPING_REFRESH_INTERVAL = float(os.getenv("VICKY_AUTO_MAP_REFRESH_INTERVAL", "2.0"))
AUTO_MAPPING_MIN_LANDMARKS = int(os.getenv("VICKY_AUTO_MAP_MIN_LANDMARKS", "1"))
MAPPING_PROMPT_ENABLED = os.getenv("VICKY_MAPPING_PROMPT", "1").strip().lower() not in {"0", "false", "no"}
DOORWAY_TRANSITION_PROMPT_ENABLED = os.getenv("VICKY_DOORWAY_TRANSITION_PROMPT", "1").strip().lower() not in {"0", "false", "no"}
STATUS_CACHE_TTL = float(os.getenv("VICKY_STATUS_CACHE_TTL", "1.00"))

auto_mapping_thread = None
auto_mapping_running = False
mapping_prompt_pending = MAPPING_PROMPT_ENABLED
mapping_prompt_awaiting_answer = False
doorway_transition_awaiting_answer = False
doorway_transition_context = None
status_cache_time = 0.0
status_cache_data = None


def get_local_ipv4_addresses():
    addresses = []
    seen = set()

    def add_address(address):
        if not address or address.startswith("127.") or address in seen:
            return
        seen.add(address)
        addresses.append(address)

    try:
        hostname = socket.gethostname()
        for address_info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            add_address(address_info[4][0])
    except Exception:
        pass

    try:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.connect(("8.8.8.8", 80))
        add_address(udp_socket.getsockname()[0])
        udp_socket.close()
    except Exception:
        pass

    return addresses


def _object_mobility(label, fallback=None):
    if fallback in {"static", "dynamic"}:
        return fallback
    return classify_object_mobility(label)


def _object_color(label, mobility=None):
    return object_color_for_label(label, mobility)


def _format_llm_detection(d):
    label = d.get("semantic_label") or d.get("class_name") or d.get("object")
    mobility = _object_mobility(label, d.get("mobility") or d.get("classification"))
    return {
        "class": label,
        "position": d.get("position"),
        "distance": d.get("distance"),
        "classification": mobility,
        "mobility": mobility,
        "confidence": round(float(d.get("confidence", 0.0)), 2),
        "depth_meters": d.get("depth_meters"),
    }


def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        with whisper_model_lock:
            if whisper_model is None:
                print(f"[WHISPER] Loading {WHISPER_MODEL_NAME} on {WHISPER_DEVICE} for transcription.")
                whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=WHISPER_DEVICE)
    return whisper_model


def get_live_spatial_snapshot():
    with zva.frame_lock:
        grid = [row[:] for row in zva.occupancy_grid]
        semantic_objects = [obj.copy() for obj in getattr(zva, "semantic_objects", [])]
        pose = dict(zva.pose_data)
        detections = [d.copy() for d in zva.latest_detections]
        zones = dict(zva.zones_data)

    for obj in semantic_objects:
        label = str(obj.get("label") or obj.get("detected_label") or "").lower()
        mobility = _object_mobility(label, obj.get("mobility") or obj.get("classification"))
        obj["classification"] = mobility
        obj["mobility"] = mobility
        obj["color"] = obj.get("color") or _object_color(label, mobility)
        if mobility != "static":
            continue
        gx = obj.get("x")
        gz = obj.get("z")
        if gx is None or gz is None:
            continue
        gx = max(0, min(int(gx), 99))
        gz = max(0, min(int(gz), 99))
        marker = 2 if "door" in label or "exit" in label else 1
        for dz in range(-1, 2):
            for dx in range(-1, 2):
                nz = max(0, min(gz + dz, 99))
                nx = max(0, min(gx + dx, 99))
                grid[nz][nx] = marker

    return {
        "grid": grid,
        "semantic_objects": semantic_objects,
        "pose": pose,
        "detections": detections,
        "zones": zones,
    }


def _landmark_count(map_data):
    if not map_data:
        return 0
    return len(map_data.get("landmarks", []))


def _ensure_map_for_navigation(snapshot, map_name=AUTO_MAPPING_NAME):
    current_map = spatial_memory.state.get("current_map")
    if current_map:
        if spatial_memory.state.get("mode") == "mapping":
            return spatial_memory.save_current_map(
                live_grid=snapshot["grid"],
                semantic_objects=snapshot["semantic_objects"],
            )
        return current_map

    return spatial_memory.load_map()


def _is_exit_navigation_command(command_text):
    command_text = command_text.lower()
    direct_phrases = [
        "leave the room",
        "exit the room",
        "find the exit",
        "take me to the exit",
        "where is the exit",
        "where's the exit",
        "where is the exit door",
        "where's the exit door",
        "where is the doorway",
        "where's the doorway",
        "guide me to the exit",
        "lead me to the exit",
        "bring me to the exit",
        "go to the exit",
        "pintu keluar",
        "cari pintu",
        "keluar ruangan",
    ]
    if any(phrase in command_text for phrase in direct_phrases):
        return True

    asks_for_route = any(
        phrase in command_text
        for phrase in [
            "where is",
            "where's",
            "find",
            "take me",
            "guide me",
            "lead me",
            "bring me",
            "go to",
            "i want to",
        ]
    )
    mentions_exit = "exit" in command_text or "doorway" in command_text
    mentions_door = "door" in command_text and asks_for_route
    return asks_for_route and (mentions_exit or mentions_door)


def _start_exit_navigation_response(command_text, snapshot):
    current_map = _ensure_map_for_navigation(snapshot)
    if not current_map:
        return {
            "transcript": command_text,
            "response": "I do not have a saved room map yet. Start mapping first, scan the room, then save the map."
        }

    best, error = spatial_memory.start_navigation(
        goal_type="exit",
        pose=snapshot["pose"],
        semantic_objects=snapshot["semantic_objects"],
    )
    if error:
        return {
            "transcript": command_text,
            "response": error
        }

    guidance_payload = spatial_memory.navigation_guidance(
        pose=snapshot["pose"],
        semantic_objects=snapshot["semantic_objects"],
        zones=snapshot.get("zones"),
    )
    instruction = guidance_payload.get("instruction") or "Route planned."
    target = best["door"]
    target_type = target.get("type", "exit")
    target_id = target.get("id", "exit")
    map_name = current_map.get("map_name", "saved map")
    return {
        "transcript": command_text,
        "response": f"Using saved map {map_name}. Safest {target_type} selected: {target_id}. {instruction}"
    }


def _is_yes_response(command_text):
    normalized = str(command_text or "").strip().lower()
    tokens = set(normalized.replace(".", " ").replace(",", " ").replace("!", " ").split())
    return bool(tokens & {"yes", "yeah", "yep", "sure", "okay", "ok", "start"}) or "please do" in normalized


def _is_no_response(command_text):
    normalized = str(command_text or "").strip().lower()
    tokens = set(normalized.replace(".", " ").replace(",", " ").replace("!", " ").split())
    return bool(tokens & {"no", "nope", "cancel"}) or any(
        phrase in normalized for phrase in ["not now", "do not", "don't"]
    )


def _next_room_name():
    room_count = 0
    for item in spatial_memory.list_maps():
        name = str(item.get("map_name") or "").strip().lower()
        if name.startswith("room"):
            room_count += 1
    return f"Room {room_count + 1}"


def _doorway_transition_prompt_payload():
    context = doorway_transition_context or {}
    door_type = context.get("door_type", "doorway")
    return {
        "active": True,
        "guidance": (
            f"You reached the {door_type}. Are you entering a new room? "
            "Say yes or press yes to start mapping it."
        ),
        "target": "new_room_mapping",
        "source": "doorway_transition_prompt",
        "requires_answer": True,
        "answer_type": "yes_no",
        "prompt_id": "new_room_mapping",
        "options": ["yes", "no"],
    }


def _mapping_prompt_payload():
    current_map = spatial_memory.state.get("current_map")
    guidance = "Do you want to start mapping this room?"
    if current_map:
        guidance = (
            f"A saved map named {current_map.get('map_name', 'this room')} is loaded. "
            "Do you want to start mapping this room again?"
        )
    return {
        "active": True,
        "guidance": guidance,
        "target": "mapping",
        "source": "mapping_prompt",
        "requires_answer": True,
        "answer_type": "yes_no",
        "prompt_id": "start_mapping",
        "options": ["yes", "no"],
    }


def _start_new_room_mapping_from_transition(snapshot):
    global doorway_transition_awaiting_answer, doorway_transition_context
    context = doorway_transition_context or {}
    doorway_transition_awaiting_answer = False

    room_name = _next_room_name()
    previous_map_id = context.get("from_map_id")
    previous_door_id = context.get("door_id")

    current_map = spatial_memory.start_mapping(
        map_name=room_name,
        live_grid=snapshot["grid"],
        semantic_objects=snapshot["semantic_objects"],
        pose=snapshot["pose"],
    )

    if previous_map_id and previous_door_id:
        try:
            spatial_memory.link_map_door(
                map_id=previous_map_id,
                door_id=previous_door_id,
                target_map_id=current_map["map_id"],
                target_door_id=None,
            )
        except Exception as exc:
            print(f"[SPATIAL MEMORY] Could not link doorway transition: {exc}")

    doorway_transition_context = None
    return {
        "transcript": "yes",
        "response": (
            f"New room mapping started for {current_map['map_name']}. "
            "Please slowly look around this room so I can find doors, exits, and obstacles."
        ),
        "prompt_resolved": True,
        "prompt_id": "new_room_mapping",
    }


def _handle_pending_yes_no_prompt(command_text, snapshot):
    global mapping_prompt_awaiting_answer
    global doorway_transition_awaiting_answer, doorway_transition_context

    if doorway_transition_awaiting_answer:
        if _is_yes_response(command_text):
            return _start_new_room_mapping_from_transition(snapshot)
        if _is_no_response(command_text):
            doorway_transition_awaiting_answer = False
            doorway_transition_context = None
            return {
                "transcript": command_text,
                "response": "Okay. I will keep the current map loaded and will not start a new room map yet.",
                "prompt_resolved": True,
                "prompt_id": "new_room_mapping",
            }

    if mapping_prompt_awaiting_answer:
        if _is_yes_response(command_text):
            mapping_prompt_awaiting_answer = False
            current_map = spatial_memory.start_mapping(
                map_name=AUTO_MAPPING_NAME,
                live_grid=snapshot["grid"],
                semantic_objects=snapshot["semantic_objects"],
                pose=snapshot["pose"],
            )
            return {
                "transcript": command_text,
                "response": (
                    f"Mapping started for {current_map['map_name']}. "
                    "Please slowly look around the room. Turn left and right so I can find doors, exits, and obstacles."
                ),
                "prompt_resolved": True,
                "prompt_id": "start_mapping",
            }
        if _is_no_response(command_text):
            mapping_prompt_awaiting_answer = False
            return {
                "transcript": command_text,
                "response": "Okay. I will not start mapping yet. Say start mapping when you are ready.",
                "prompt_resolved": True,
                "prompt_id": "start_mapping",
            }

    return None


def _answer_prompt_direct(prompt_id, answer_text, snapshot):
    global mapping_prompt_pending, mapping_prompt_awaiting_answer
    global doorway_transition_awaiting_answer, doorway_transition_context

    command_text = str(answer_text or "").strip().lower()
    prompt_id = str(prompt_id or "").strip()

    handled = _handle_pending_yes_no_prompt(command_text, snapshot)
    if handled:
        return handled

    if prompt_id == "start_mapping":
        mapping_prompt_pending = False
        mapping_prompt_awaiting_answer = False
        if _is_yes_response(command_text):
            current_map = spatial_memory.start_mapping(
                map_name=AUTO_MAPPING_NAME,
                live_grid=snapshot["grid"],
                semantic_objects=snapshot["semantic_objects"],
                pose=snapshot["pose"],
            )
            return {
                "transcript": "yes",
                "response": (
                    f"Mapping started for {current_map['map_name']}. "
                    "Please slowly look around the room. Turn left and right so I can find doors, exits, and obstacles."
                ),
                "prompt_resolved": True,
                "prompt_id": "start_mapping",
            }
        if _is_no_response(command_text):
            return {
                "transcript": "no",
                "response": "Okay. I will not start mapping yet. Say start mapping when you are ready.",
                "prompt_resolved": True,
                "prompt_id": "start_mapping",
            }

    if prompt_id == "new_room_mapping":
        if _is_yes_response(command_text):
            if doorway_transition_context:
                return _start_new_room_mapping_from_transition(snapshot)
            doorway_transition_awaiting_answer = False
            current_map = spatial_memory.start_mapping(
                map_name=_next_room_name(),
                live_grid=snapshot["grid"],
                semantic_objects=snapshot["semantic_objects"],
                pose=snapshot["pose"],
            )
            return {
                "transcript": "yes",
                "response": (
                    f"New room mapping started for {current_map['map_name']}. "
                    "Please slowly look around this room so I can find doors, exits, and obstacles."
                ),
                "prompt_resolved": True,
                "prompt_id": "new_room_mapping",
            }
        if _is_no_response(command_text):
            doorway_transition_awaiting_answer = False
            doorway_transition_context = None
            return {
                "transcript": "no",
                "response": "Okay. I will keep the current map loaded and will not start a new room map yet.",
                "prompt_resolved": True,
                "prompt_id": "new_room_mapping",
            }

    return {
        "transcript": command_text,
        "response": "That prompt has expired. Live guidance is back on.",
        "prompt_resolved": True,
        "prompt_id": prompt_id or "prompt",
    }


def auto_mapping_loop():
    last_save_at = 0.0
    while auto_mapping_running:
        try:
            snapshot = get_live_spatial_snapshot()

            if spatial_memory.state.get("mode") == "mapping":
                current_map = spatial_memory.refresh_mapping(
                    live_grid=snapshot["grid"],
                    semantic_objects=snapshot["semantic_objects"],
                )
                now = time.time()
                if (
                    current_map
                    and _landmark_count(current_map) >= AUTO_MAPPING_MIN_LANDMARKS
                    and not current_map.get("metadata", {}).get("awaiting_user_name")
                    and now - last_save_at >= AUTO_MAPPING_SAVE_INTERVAL
                ):
                    spatial_memory.save_current_map(
                        live_grid=snapshot["grid"],
                        semantic_objects=snapshot["semantic_objects"],
                    )
                    last_save_at = now
                    print(
                        f"[AUTO MAP] Saved {current_map['map_name']} "
                        f"with {_landmark_count(current_map)} landmarks."
                    )
        except Exception as exc:
            print(f"[AUTO MAP] Error: {exc}")

        time.sleep(AUTO_MAPPING_REFRESH_INTERVAL)


def start_auto_mapping():
    global auto_mapping_thread, auto_mapping_running
    if not AUTO_MAPPING_ENABLED or auto_mapping_running:
        return
    auto_mapping_running = True
    auto_mapping_thread = threading.Thread(target=auto_mapping_loop, daemon=True)
    auto_mapping_thread.start()
    print("[AUTO MAP] Auto-save enabled. Mapping starts only after user confirmation.")


def stop_auto_mapping():
    global auto_mapping_running
    auto_mapping_running = False


@app.on_event("startup")
async def startup_event():
    global mapping_prompt_pending, mapping_prompt_awaiting_answer
    global doorway_transition_awaiting_answer, doorway_transition_context
    spatial_memory.stop_mapping()
    spatial_memory.stop_navigation()
    navigator.stop_navigation()
    mapping_prompt_pending = MAPPING_PROMPT_ENABLED
    mapping_prompt_awaiting_answer = False
    doorway_transition_awaiting_answer = False
    doorway_transition_context = None
    await db_logger.start()
    start_auto_mapping()

@app.on_event("shutdown")
async def shutdown_event():
    stop_auto_mapping()
    await db_logger.stop()

@app.websocket("/ws/telemetry/stream")
async def telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    print("[SERVER WS] Telemetry socket connected from edge client.")
    zva.telemetry_source_active = True
    try:
        while True:
            data = await websocket.receive_json()
            user_pose = data.get("user_spatial_pose")
            if user_pose:
                pos = user_pose.get("position_meters", {})
                rot = user_pose.get("rotation_degrees", {})
                with zva.frame_lock:
                    # Update ZED pose data in millimeters
                    zva.pose_data["x"] = pos.get("x", 0.0) * 1000.0
                    zva.pose_data["y"] = pos.get("y", 0.0) * 1000.0
                    zva.pose_data["z"] = pos.get("z", 0.0) * 1000.0
                    zva.pose_data["roll"] = rot.get("roll", 0.0)
                    zva.pose_data["pitch"] = rot.get("pitch", 0.0)
                    base_yaw = normalize_degrees(rot.get("yaw", 0.0))
                    projection_yaw = base_yaw_to_projection_yaw(base_yaw)
                    zva.pose_data["yaw"] = projection_yaw
                    zva.pose_data["display_yaw"] = base_yaw_to_display_yaw(base_yaw)
                    zva.pose_data["projection_yaw"] = projection_yaw
            
            zones = data.get("spatial_depth_zones")
            if zones:
                with zva.frame_lock:
                    zva.zones_data["left"] = zones.get("left_clearance_mm", 0.0)
                    zva.zones_data["center"] = zones.get("center_clearance_mm", 0.0)
                    zva.zones_data["right"] = zones.get("right_clearance_mm", 0.0)
                    zva.guidance_cmd = zones.get("escape_vector", "STOP")
            
            # Map detected obstacles if present
            objects = data.get("semantic_objects_in_frustum")
            if objects is not None:
                with zva.frame_lock:
                    # Clear occupancy grid for simulated/telemetry mode
                    for r in range(100):
                        for c in range(100):
                            zva.occupancy_grid[r][c] = 0
                            
                    map_yaw = normalize_degrees(zva.pose_data.get("projection_yaw", zva.pose_data.get("yaw", 0.0)))
                    user_grid_z, user_grid_x = pose_mm_to_grid(
                        zva.pose_data.get("x", 0.0),
                        zva.pose_data.get("z", 0.0),
                    )
                    
                    zva.semantic_objects = []
                    zva.latest_detections = []
                    
                    for idx, obj in enumerate(objects):
                        coords = obj.get("3d_coordinates", {})
                        x_c = coords.get("x", 0.0)
                        z_c = coords.get("z", 0.0)
                        class_name = obj.get("class", "object")
                        mobility = _object_mobility(
                            class_name,
                            obj.get("mobility") or obj.get("classification"),
                        )
                        object_color = obj.get("color") or _object_color(class_name, mobility)
                        
                        grid_x, grid_z = camera_point_to_grid(user_grid_x, user_grid_z, map_yaw, x_c, z_c)
                        
                        # Persist only static objects into the room occupancy layer.
                        if mobility == "static":
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    r_idx = max(0, min(grid_z + dr, 99))
                                    c_idx = max(0, min(grid_x + dc, 99))
                                    zva.occupancy_grid[r_idx][c_idx] = 1
                        
                        zva.semantic_objects.append({
                            "label": class_name,
                            "detected_label": class_name,
                            "classification": mobility,
                            "mobility": mobility,
                            "color": object_color,
                            "x": grid_x,
                            "z": grid_z,
                            "distance": z_c,
                            "confidence": float(obj.get("confidence", 1.0)),
                        })
                        
                        # Determine position (left, center, right)
                        if x_c < -0.3:
                            pos_lbl = "left"
                        elif x_c > 0.3:
                            pos_lbl = "right"
                        else:
                            pos_lbl = "center"
                            
                        zva.latest_detections.append({
                            "class_name": class_name,
                            "semantic_label": class_name,
                            "classification": mobility,
                            "mobility": mobility,
                            "color": object_color,
                            "position": pos_lbl,
                            "distance": obj.get("distance_category", "medium"),
                            "depth_meters": z_c,
                            "confidence": 1.0
                        })
    except WebSocketDisconnect:
        print("[SERVER WS] Telemetry socket disconnected.")
    except Exception as e:
        print(f"[SERVER WS ERROR] Telemetry: {e}")
    finally:
        zva.telemetry_source_active = False

@app.websocket("/ws/video/stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    print("[SERVER WS] Video socket connected from edge client.")
    try:
        while True:
            # Just consume video bytes (used for HUD in production)
            await websocket.receive_bytes()
    except WebSocketDisconnect:
        print("[SERVER WS] Video socket disconnected.")
    except Exception as e:
        pass

@app.get("/")
def root():
    return {
        "status": "AI server running",
        "host": "0.0.0.0",
        "port": 8000,
        "local_urls": [f"http://{address}:8000" for address in get_local_ipv4_addresses()],
    }

@app.get("/status")
def status():
    global status_cache_time, status_cache_data
    now = time.time()
    if status_cache_data is not None and now - status_cache_time < STATUS_CACHE_TTL:
        return status_cache_data

    with zva.frame_lock:
        raw_detections = [d.copy() for d in zva.latest_detections]
        pose_snapshot = dict(zva.pose_data)
        zones_snapshot = dict(zva.zones_data)
        grid_snapshot = [row[:] for row in zva.occupancy_grid]
        semantic_snapshot = [obj.copy() for obj in getattr(zva, "semantic_objects", [])]
        performance_snapshot = dict(getattr(zva, "performance_data", {}))
        guidance_snapshot = zva.guidance_cmd

    navigation_snapshot = None
    if spatial_memory.state.get("mode") == "navigating":
        navigation_snapshot = spatial_memory.navigation_guidance(
            pose=pose_snapshot,
            semantic_objects=semantic_snapshot,
            zones=zones_snapshot,
        )

    detections = []
    for d in raw_detections:
        label = d.get("semantic_label", d.get("class_name", "object"))
        mobility = _object_mobility(label, d.get("mobility") or d.get("classification"))
        detections.append({
            "object": d.get("class_name", label),
            "semantic_label": label,
            "position": d.get("position"),
            "distance": d.get("distance"),
            "depth_meters": d.get("depth_meters"),
            "classification": mobility,
            "mobility": mobility,
            "color": d.get("color") or _object_color(label, mobility),
            "confidence": round(float(d.get("confidence", 0.0)), 2),
            "passable": d.get("passable"),
            "door_state": d.get("door_state"),
        })

    current_map = spatial_memory.state.get("current_map")
    user_grid_z, user_grid_x = spatial_memory.pose_to_grid(pose_snapshot)
    status_grid = current_map.get("static_grid") if current_map else grid_snapshot

    if spatial_memory.state.get("active_path"):
        path = spatial_memory.state.get("active_path", [])
        active_goal = spatial_memory.state.get("active_goal")
        goal_data = {"x": active_goal[1], "z": active_goal[0]} if active_goal else None
    elif current_goal is None:
        path = []
        goal_data = None
    else:
        path = astar_pathfind(status_grid, (user_grid_z, user_grid_x), current_goal)
        goal_data = {"x": current_goal[1], "z": current_goal[0]}

    live_objects = spatial_memory.semantic_objects_to_current_map(semantic_snapshot)
    live_static_count = sum(1 for obj in live_objects if _object_mobility(
        obj.get("label") or obj.get("detected_label"),
        obj.get("mobility") or obj.get("classification"),
    ) == "static")
    live_dynamic_count = max(0, len(live_objects) - live_static_count)
    live_dynamic_obstacles = spatial_memory.get_live_dynamic_obstacles(live_objects)
    if current_map:
        base_cell_type_map = current_map.get("cell_type_grid")
        if not base_cell_type_map:
            base_cell_type_map = spatial_memory._refresh_cell_type_grid(current_map)
        cell_type_map = spatial_memory.overlay_live_dynamic_cell_types(
            base_cell_type_map,
            live_dynamic_obstacles,
        )
    else:
        cell_type_map = grid_snapshot

    response = {
        "guidance": guidance_snapshot,
        "left_distance": zones_snapshot.get("left", 0),
        "center_distance": zones_snapshot.get("center", 0),
        "right_distance": zones_snapshot.get("right", 0),
        "center_open_space_ratio": zones_snapshot.get("center_open_space_ratio", 0),
        "full_open_space_ratio": zones_snapshot.get("full_open_space_ratio", 0),
        "scene_hint": zones_snapshot.get("scene_hint"),
        "detections": detections,
        "pose": pose_snapshot,
        "performance": performance_snapshot,
        "navigation": navigation_snapshot,
        "map": status_grid,
        "cell_type_map": cell_type_map,
        "path": path,
        "goal": goal_data,
        "user_grid": {"x": user_grid_x, "z": user_grid_z},
        "objects": live_objects,
        "spatial_memory": {
            "mode": spatial_memory.state.get("mode"),
            "current_map_id": spatial_memory.state.get("current_map_id"),
            "current_map_name": current_map.get("map_name") if current_map else None,
            "landmark_count": _landmark_count(current_map),
            "static_object_count": len(current_map.get("static_objects", [])) if current_map else 0,
            "live_static_count": live_static_count,
            "live_dynamic_count": live_dynamic_count,
            "tracking_ok": bool(pose_snapshot.get("tracking_ok")),
            "tracking_state": pose_snapshot.get("tracking_state", "UNKNOWN"),
            "mapping_prompt_pending": mapping_prompt_pending,
            "mapping_prompt_awaiting_answer": mapping_prompt_awaiting_answer,
            "doorway_transition_awaiting_answer": doorway_transition_awaiting_answer,
            "active_prompt": (
                "new_room_mapping" if doorway_transition_awaiting_answer
                else "start_mapping" if mapping_prompt_awaiting_answer
                else None
            ),
            "last_instruction": spatial_memory.state.get("last_instruction"),
            "last_navigation_event": spatial_memory.state.get("last_navigation_event"),
        },
    }
    status_cache_data = response
    status_cache_time = now
    return response


@app.get("/pose")
def pose_status():
    with zva.frame_lock:
        pose_snapshot = dict(zva.pose_data)

    user_grid_z, user_grid_x = spatial_memory.pose_to_grid(pose_snapshot)
    return {
        "pose": pose_snapshot,
        "user_grid": {"x": user_grid_x, "z": user_grid_z},
        "tracking_ok": bool(pose_snapshot.get("tracking_ok")),
        "tracking_state": pose_snapshot.get("tracking_state", "UNKNOWN"),
    }


@app.get("/autopilot-guidance")
def autopilot_guidance():
    global mapping_prompt_pending, mapping_prompt_awaiting_answer
    global doorway_transition_awaiting_answer, doorway_transition_context
    snapshot = get_live_spatial_snapshot()
    spatial_mode = spatial_memory.state.get("mode")

    if spatial_mode == "navigating":
        guidance_payload = spatial_memory.navigation_guidance(
            pose=snapshot["pose"],
            semantic_objects=snapshot["semantic_objects"],
            zones=snapshot.get("zones"),
        )
        instruction = guidance_payload.get("instruction")
        if guidance_payload.get("active") and instruction:
            navigation_event = guidance_payload.get("event")
            if (
                DOORWAY_TRANSITION_PROMPT_ENABLED
                and not doorway_transition_awaiting_answer
                and (
                    navigation_event == "exit_passed"
                    or "exited the room" in instruction.lower()
                )
            ):
                current_map = spatial_memory.state.get("current_map") or {}
                target = spatial_memory.state.get("navigation_target") or {}
                doorway_transition_context = {
                    "from_map_id": current_map.get("map_id"),
                    "from_map_name": current_map.get("map_name"),
                    "door_id": target.get("id"),
                    "door_type": target.get("type", "doorway"),
                }
                doorway_transition_awaiting_answer = True
                spatial_memory.stop_navigation()
                return _doorway_transition_prompt_payload()

            return {
                "active": True,
                "guidance": instruction,
                "target": guidance_payload.get("target"),
                "source": "spatial_memory",
                "navigation": guidance_payload,
            }

    if doorway_transition_awaiting_answer and spatial_mode == "idle":
        return _doorway_transition_prompt_payload()

    if mapping_prompt_awaiting_answer and spatial_mode == "idle":
        return _mapping_prompt_payload()

    if mapping_prompt_pending and spatial_mode == "idle":
        mapping_prompt_pending = False
        mapping_prompt_awaiting_answer = True
        return _mapping_prompt_payload()

    with zva.frame_lock:
        detections = list(zva.latest_detections)

    guidance = navigator.update_navigation(detections)

    if guidance:
        return {
            "active": True,
            "guidance": guidance,
            "target": navigator.active_target_class
        }

    return {
        "active": False,
        "guidance": "",
        "target": None
    }
    
@app.get("/map", response_class=HTMLResponse)
def map_view():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VICKY Live Spatial HUD</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(30, 41, 59, 0.45);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-blue: #00d2ff;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-amber: #fbbf24;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            overflow-x: hidden;
            padding: 20px;
        }

        .header-container {
            width: 100%;
            max-width: 1100px;
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-text {
            display: flex;
            flex-direction: column;
        }

        .header-title {
            font-size: 24px;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-blue), #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-subtitle {
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        .hud-container {
            display: flex;
            flex-direction: row;
            gap: 24px;
            max-width: 1100px;
            width: 100%;
        }

        .map-section {
            flex: 1.2;
            display: flex;
            flex-direction: column;
            align-items: center;
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        .map-section h3 {
            margin-bottom: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
            color: var(--text-main);
            font-size: 16px;
            align-self: flex-start;
        }

        .canvas-container {
            position: relative;
            width: 100%;
            max-width: 500px;
            aspect-ratio: 1 / 1;
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid rgba(255, 255, 255, 0.05);
            background: #020617;
            box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.8);
        }

        canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            cursor: crosshair;
        }

        .dashboard-section {
            flex: 0.8;
            width: 100%;
            max-width: 420px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .panel-card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        .panel-card h4 {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 14px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            padding-bottom: 6px;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-red);
        }

        .status-dot.active {
            background: var(--accent-green);
            box-shadow: 0 0 8px var(--accent-green);
        }

        .guidance-badge {
            font-size: 20px;
            font-weight: 800;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
            margin-top: 5px;
        }

        .guidance-stop {
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.1);
        }

        .guidance-go {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.1);
        }

        .guidance-turn {
            background: rgba(251, 191, 36, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(251, 191, 36, 0.3);
            box-shadow: 0 0 10px rgba(251, 191, 36, 0.1);
        }

        .telemetry-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }

        .telemetry-item {
            background: rgba(0, 0, 0, 0.15);
            padding: 8px 12px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.01);
        }

        .telemetry-label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .telemetry-val {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-main);
            margin-top: 2px;
        }

        .clearance-row {
            margin-bottom: 10px;
        }

        .clearance-info {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            margin-bottom: 4px;
        }

        .progress-bar-bg {
            background: rgba(255, 255, 255, 0.04);
            height: 5px;
            border-radius: 3px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            width: 0%;
            border-radius: 3px;
            transition: width 0.3s ease, background-color 0.3s ease;
        }

        .btn {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: none;
            font-family: inherit;
            font-weight: 600;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .btn-primary {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            color: white;
            box-shadow: 0 4px 10px rgba(59, 130, 246, 0.2);
        }

        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 14px rgba(59, 130, 246, 0.3);
        }

        .btn-danger {
            background: rgba(239, 68, 68, 0.12);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.25);
        }

        .btn-danger:hover {
            background: rgba(239, 68, 68, 0.25);
        }

        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(15, 23, 42, 0.95);
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            border: 1px solid var(--accent-blue);
            box-shadow: 0 8px 20px rgba(0,0,0,0.4);
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            z-index: 1000;
            font-size: 13px;
        }

        .toast.show {
            transform: translateY(0);
            opacity: 1;
        }

        .detection-list {
            max-height: 110px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .detection-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(0, 0, 0, 0.15);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 12px;
        }

        .detection-name {
            font-weight: 600;
            color: #60a5fa;
            text-transform: capitalize;
        }

        .detection-meta {
            color: var(--text-muted);
            font-size: 10px;
        }

        @media (max-width: 900px) {
            body {
                padding: 10px;
            }
            .hud-container {
                flex-direction: column;
                align-items: center;
                gap: 16px;
            }
            .dashboard-section {
                max-width: 500px;
            }
            .header-container {
                flex-direction: column;
                gap: 10px;
                text-align: center;
            }
        }

        /* Embedded (mobile WebView) styles */
        body.embed-mode {
            padding: 0 !important;
            margin: 0 !important;
            background-color: transparent !important;
            min-height: 100vh !important;
            overflow: hidden !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
        }

        body.embed-mode .header-container,
        body.embed-mode .dashboard-section,
        body.embed-mode .map-section h3,
        body.embed-mode .map-section > div:last-child {
            display: none !important;
        }

        body.embed-mode .hud-container {
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
        }

        body.embed-mode .map-section {
            background: transparent !important;
            backdrop-filter: none !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
            border-radius: 0 !important;
            width: 100% !important;
            align-items: center !important;
            justify-content: center !important;
        }

        body.embed-mode .canvas-container {
            width: 90vw !important;
            max-width: 320px !important;
            aspect-ratio: 1 / 1 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            margin: 0 auto !important;
        }
    </style>
</head>
<body>
    <div class="header-container">
        <div class="header-text">
            <h1 class="header-title">VICKY Live Spatial HUD</h1>
            <p class="header-subtitle">ZED SLAM Odometry & A* Path Planning</p>
        </div>
        <div class="status-badge">
            <div id="statusDot" class="status-dot"></div>
            <span id="statusText">Connecting...</span>
        </div>
    </div>

    <div class="hud-container">
        <div class="map-section">
            <h3>2D Occupancy Grid (10m x 10m)</h3>
            <div class="canvas-container">
                <canvas id="mapCanvas" width="500" height="500"></canvas>
            </div>
            <div style="display: flex; justify-content: space-between; width: 100%; max-width: 500px; margin-top: 10px; font-size: 11px; color: var(--text-muted);">
                <span>&lt;- -5m X</span>
                <span>Click grid to set destination goal</span>
                <span>+5m X -&gt;</span>
            </div>
        </div>

        <div class="dashboard-section">
            <div class="panel-card">
                <h4>Guidance Command</h4>
                <div id="guidanceBadge" class="guidance-badge guidance-stop">STOP</div>
            </div>

            <div class="panel-card">
                <h4>6-DoF SLAM Pose (Live)</h4>
                <div class="telemetry-grid">
                    <div class="telemetry-item">
                        <div class="telemetry-label">Translation X</div>
                        <div id="poseX" class="telemetry-val">0.00 m</div>
                    </div>
                    <div class="telemetry-item">
                        <div class="telemetry-label">Rotation Roll</div>
                        <div id="poseRoll" class="telemetry-val">0.0 deg</div>
                    </div>
                    <div class="telemetry-item">
                        <div class="telemetry-label">Translation Y</div>
                        <div id="poseY" class="telemetry-val">0.00 m</div>
                    </div>
                    <div class="telemetry-item">
                        <div class="telemetry-label">Rotation Pitch</div>
                        <div id="posePitch" class="telemetry-val">0.0 deg</div>
                    </div>
                    <div class="telemetry-item">
                        <div class="telemetry-label">Translation Z</div>
                        <div id="poseZ" class="telemetry-val">0.00 m</div>
                    </div>
                    <div class="telemetry-item">
                        <div class="telemetry-label">Rotation Yaw</div>
                        <div id="poseYaw" class="telemetry-val">0.0 deg</div>
                    </div>
                </div>
            </div>

            <div class="panel-card">
                <h4>Safety Clearances</h4>
                <div class="clearance-row">
                    <div class="clearance-info">
                        <span>Left Zone</span>
                        <span id="valLeft">0 mm</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div id="clearanceLeft" class="progress-bar-fill"></div>
                    </div>
                </div>
                <div class="clearance-row">
                    <div class="clearance-info">
                        <span>Center Zone</span>
                        <span id="valCenter">0 mm</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div id="clearanceCenter" class="progress-bar-fill"></div>
                    </div>
                </div>
                <div class="clearance-row">
                    <div class="clearance-info">
                        <span>Right Zone</span>
                        <span id="valRight">0 mm</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div id="clearanceRight" class="progress-bar-fill"></div>
                    </div>
                </div>
            </div>

            <div class="panel-card">
                <h4>Detected Obstacles</h4>
                <div id="detectionList" class="detection-list">
                    <div style="color:var(--text-muted);font-size:12px;text-align:center;padding:10px 0;">No active objects in frustum</div>
                </div>
            </div>

            <div class="panel-card" style="display: flex; flex-direction: column; gap: 8px;">
                <h4>Control Settings</h4>
                <button class="btn btn-primary" onclick="clearGoal()">
                    Clear Navigation Goal
                </button>
                <button class="btn btn-danger" onclick="resetMap()">
                    Reset Occupancy Grid
                </button>
            </div>
        </div>
    </div>

    <div id="toast" class="toast">Target set successfully!</div>

    <script>
        // Check for embed parameter in URL to style map specifically for mobile companion app
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.get('embed') === 'true' || urlParams.get('mobile') === 'true') {
            document.body.classList.add('embed-mode');
        }

        const canvas = document.getElementById('mapCanvas');
        const ctx = canvas.getContext('2d');
        let currentGoal = null;
        let isPolling = true;

        function showToast(message) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.classList.add('show');
            setTimeout(() => {
                toast.classList.remove('show');
            }, 3000);
        }

        async function setGoal(row, col) {
            try {
                const response = await fetch('/api/set-goal', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ row, col })
                });
                const result = await response.json();
                if (result.status === 'success') {
                    showToast("Navigation target updated: (" + col + ", " + row + ")");
                    currentGoal = { x: col, z: row };
                    fetchStatus();
                } else {
                    showToast('Failed to set goal');
                }
            } catch (err) {
                console.error(err);
                showToast('Network error setting goal');
            }
        }

        async function resetMap() {
            if (!confirm('Are you sure you want to completely clear the persistent SLAM occupancy grid?')) {
                return;
            }
            try {
                const response = await fetch('/api/reset-map', {
                    method: 'POST'
                });
                const result = await response.json();
                if (result.status === 'success') {
                    showToast('Spatial grid successfully reset.');
                    fetchStatus();
                } else {
                    showToast('Failed to reset map.');
                }
            } catch (err) {
                console.error(err);
                showToast('Network error resetting map');
            }
        }

        async function clearGoal() {
            try {
                const response = await fetch('/api/set-goal', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ row: null, col: null })
                });
                const result = await response.json();
                if (result.status === 'success') {
                    showToast('Navigation goal cleared.');
                    currentGoal = null;
                    fetchStatus();
                }
            } catch (err) {
                console.error(err);
            }
        }

        canvas.addEventListener('click', (e) => {
            const rect = canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;
            
            const col = Math.floor((clickX / rect.width) * 100);
            const row = Math.floor((clickY / rect.height) * 100);
            
            if (col >= 0 && col < 100 && row >= 0 && row < 100) {
                setGoal(row, col);
            }
        });

        async function fetchStatus() {
            if (!isPolling) return;
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                document.getElementById('statusDot').classList.add('active');
                document.getElementById('statusText').textContent = 'Live System Online';
                
                const guidance = data.guidance || 'STOP';
                const gBadge = document.getElementById('guidanceBadge');
                gBadge.textContent = guidance;
                gBadge.className = 'guidance-badge'; 
                if (guidance.includes('STOP') || guidance.includes('DANGER')) {
                    gBadge.classList.add('guidance-stop');
                } else if (guidance.includes('FORWARD')) {
                    gBadge.classList.add('guidance-go');
                } else {
                    gBadge.classList.add('guidance-turn');
                }
                
                const pose = data.pose || {};
                document.getElementById('poseX').textContent = ((pose.x || 0)/1000).toFixed(2) + ' m';
                document.getElementById('poseY').textContent = ((pose.y || 0)/1000).toFixed(2) + ' m';
                document.getElementById('poseZ').textContent = ((pose.z || 0)/1000).toFixed(2) + ' m';
                const displayYaw = pose.display_yaw ?? pose.yaw ?? 0;
                document.getElementById('poseYaw').textContent = displayYaw.toFixed(1) + ' deg';
                document.getElementById('posePitch').textContent = (pose.pitch || 0).toFixed(1) + ' deg';
                document.getElementById('poseRoll').textContent = (pose.roll || 0).toFixed(1) + ' deg';
                
                updateClearanceBar('clearanceLeft', 'valLeft', data.left_distance);
                updateClearanceBar('clearanceCenter', 'valCenter', data.center_distance);
                updateClearanceBar('clearanceRight', 'valRight', data.right_distance);
                
                updateDetections(data.detections || []);
                
                drawMap(data);
                
            } catch (err) {
                console.error(err);
                document.getElementById('statusDot').classList.remove('active');
                document.getElementById('statusText').textContent = 'Connecting...';
            }
        }

        function updateClearanceBar(barId, textId, distance) {
            const fill = document.getElementById(barId);
            const text = document.getElementById(textId);
            text.textContent = distance ? distance.toFixed(0) + ' mm' : '0 mm';
            
            const pct = Math.min(100, Math.max(0, (distance / 3000) * 100));
            fill.style.width = pct + '%';
            
            if (distance < 500) {
                fill.style.backgroundColor = 'var(--accent-red)';
            } else if (distance < 1200) {
                fill.style.backgroundColor = 'var(--accent-amber)';
            } else {
                fill.style.backgroundColor = 'var(--accent-green)';
            }
        }

        function updateDetections(detections) {
            const listEl = document.getElementById('detectionList');
            listEl.innerHTML = '';
            
            if (detections.length === 0) {
                listEl.innerHTML = '<div style="color:var(--text-muted);font-size:12px;text-align:center;padding:10px 0;">No active objects in frustum</div>';
                return;
            }
            
            detections.forEach(d => {
                const item = document.createElement('div');
                item.className = 'detection-item';
                
                let depthStr = d.depth_meters ? d.depth_meters.toFixed(1) + 'm' : d.distance;
                const mobility = (d.mobility || d.classification || 'dynamic').toUpperCase();
                const color = d.color || (mobility === 'STATIC' ? '#38bdf8' : '#f97316');
                item.innerHTML = `
                    <div>
                        <span class="detection-name">${d.object}</span>
                        <span class="detection-meta">(${d.position} | ${mobility})</span>
                    </div>
                    <div style="font-weight:600;color:${color};">${depthStr}</div>
                `;
                listEl.appendChild(item);
            });
        }

        function drawMap(data) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const cellW = canvas.width / 100;
            const cellH = canvas.height / 100;
            
            // Draw grid blueprint lines
            ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
            ctx.lineWidth = 0.5;
            for (let i = 0; i <= 100; i += 10) {
                ctx.beginPath();
                ctx.moveTo(i * cellW, 0);
                ctx.lineTo(i * cellW, canvas.height);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(0, i * cellH);
                ctx.lineTo(canvas.width, i * cellH);
                ctx.stroke();
            }
            
            // Draw Obstacles
            const grid = data.map || [];
            ctx.fillStyle = "rgba(239, 68, 68, 0.75)";
            for (let r = 0; r < grid.length; r++) {
                for (let c = 0; c < grid[r].length; c++) {
                    if (grid[r][c] === 1) {
                        ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
                    }
                }
            }
            
            // Draw A* Path
            const path = data.path || [];
            if (path.length > 0) {
                ctx.beginPath();
                ctx.strokeStyle = "#10b981";
                ctx.lineWidth = 3.5;
                ctx.lineJoin = "round";
                ctx.lineCap = "round";
                ctx.shadowBlur = 6;
                ctx.shadowColor = "#10b981";
                
                ctx.moveTo(path[0][1] * cellW + cellW/2, path[0][0] * cellH + cellH/2);
                for (let i = 1; i < path.length; i++) {
                    ctx.lineTo(path[i][1] * cellW + cellW/2, path[i][0] * cellH + cellH/2);
                }
                ctx.stroke();
                ctx.shadowBlur = 0;
            }
            
            // Draw Goal
            const goal = data.goal;
            if (goal) {
                const goalX = goal.x * cellW + cellW/2;
                const goalY = goal.z * cellH + cellH/2;
                
                ctx.beginPath();
                ctx.strokeStyle = "#fbbf24";
                ctx.lineWidth = 2;
                ctx.arc(goalX, goalY, 9, 0, 2 * Math.PI);
                ctx.stroke();
                
                ctx.beginPath();
                ctx.fillStyle = "#fbbf24";
                ctx.arc(goalX, goalY, 3, 0, 2 * Math.PI);
                ctx.fill();
            }
            
            // Draw Semantic Objects (e.g. YOLO Detections and Exit Signs)
            const objects = data.objects || [];
            objects.forEach(obj => {
                const oX = obj.x * cellW + cellW/2;
                const oY = obj.z * cellH + cellH/2;
                const label = (obj.label || obj.detected_label || 'object').toLowerCase();
                const mobility = obj.mobility || obj.classification || 'dynamic';
                const color = obj.color || (mobility === 'static' ? '#38bdf8' : '#f97316');
                
                if (label === "exit sign") {
                    ctx.fillStyle = "#10b981";
                    ctx.beginPath();
                    if (ctx.roundRect) {
                        ctx.roundRect(oX - 16, oY - 6, 32, 12, 2);
                    } else {
                        ctx.rect(oX - 16, oY - 6, 32, 12);
                    }
                    ctx.fill();
                    
                    ctx.fillStyle = "#ffffff";
                    ctx.font = "bold 7px Outfit, sans-serif";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText("EXIT", oX, oY);
                } else {
                    ctx.beginPath();
                    ctx.fillStyle = color;
                    ctx.arc(oX, oY, mobility === 'static' ? 4 : 5, 0, 2 * Math.PI);
                    ctx.fill();
                    
                    ctx.fillStyle = "#f8fafc";
                    ctx.font = "8px Outfit, sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText(label, oX, oY - 7);
                }
            });
            
            // Draw User
            if (data.pose) {
                const userGrid = data.user_grid || {};
                const gridX = Math.max(0, Math.min(userGrid.x ?? 50, 99));
                const gridZ = Math.max(0, Math.min(userGrid.z ?? 50, 99));
                
                const uX = gridX * cellW + cellW/2;
                const uY = gridZ * cellH + cellH/2;
                
                const displayYaw = data.pose.display_yaw ?? data.pose.yaw;
                if (displayYaw !== undefined) {
                    const yawRad = displayYaw * Math.PI / 180;
                    const headingX = uX + 15 * Math.sin(yawRad);
                    const headingY = uY - 15 * Math.cos(yawRad);
                    
                    ctx.beginPath();
                    ctx.strokeStyle = "#3b82f6";
                    ctx.lineWidth = 2;
                    ctx.moveTo(uX, uY);
                    ctx.lineTo(headingX, headingY);
                    ctx.stroke();
                    
                    ctx.fillStyle = "#3b82f6";
                    ctx.beginPath();
                    ctx.arc(headingX, headingY, 3, 0, 2 * Math.PI);
                    ctx.fill();
                }
                
                ctx.beginPath();
                ctx.fillStyle = "#3b82f6";
                ctx.shadowBlur = 10;
                ctx.shadowColor = "#3b82f6";
                ctx.arc(uX, uY, 6, 0, 2 * Math.PI);
                ctx.fill();
                ctx.shadowBlur = 0;
            }
        }

        setInterval(fetchStatus, 750);
        fetchStatus();
    </script>
</body>
</html>
"""

@app.post("/api/set-goal")
def set_goal(payload: dict):
    global current_goal
    row = payload.get("row")
    col = payload.get("col")
    if row is None or col is None:
        current_goal = None
        return {"status": "success", "goal": None}
    else:
        row = max(0, min(int(row), 99))
        col = max(0, min(int(col), 99))
        current_goal = (row, col)
        return {"status": "success", "goal": {"row": row, "col": col}}

@app.post("/api/reset-map")
def reset_map():
    zva.trigger_map_reset()
    return {"status": "success"}


@app.post("/command")
def manual_command(payload: dict):
    command_text = payload.get("command", "").strip().lower()

    with zva.frame_lock:
        detections = list(zva.latest_detections)
        guidance = zva.guidance_cmd
        zones_data = dict(zva.zones_data)

    llm_detections = [_format_llm_detection(d) for d in detections]

    response = ask_llm(
        question=command_text,
        detections=llm_detections,
        direction_summary={
            "left_distance_mm": float(zones_data.get("left", 0)),
            "center_distance_mm": float(zones_data.get("center", 0)),
            "right_distance_mm": float(zones_data.get("right", 0)),
            "center_open_space_ratio": float(zones_data.get("center_open_space_ratio", 0)),
            "scene_hint": zones_data.get("scene_hint", "unknown"),
            "best_direction": guidance,
        },
        ocr_text=""
    )

    return {
        "response": response
    }


@app.post("/start-mapping")
def start_mapping(payload: dict):
    global mapping_prompt_pending, mapping_prompt_awaiting_answer
    global doorway_transition_awaiting_answer, doorway_transition_context
    mapping_prompt_pending = False
    mapping_prompt_awaiting_answer = False
    doorway_transition_awaiting_answer = False
    doorway_transition_context = None
    snapshot = get_live_spatial_snapshot()
    map_name = str(payload.get("map_name") or "Room").strip() or "Room"
    map_id = payload.get("map_id")
    current_map = spatial_memory.start_mapping(
        map_name=map_name,
        map_id=map_id,
        live_grid=snapshot["grid"],
        semantic_objects=snapshot["semantic_objects"],
        pose=snapshot["pose"],
    )
    if payload.get("awaiting_name") or payload.get("defer_save_name"):
        current_map.setdefault("metadata", {})["awaiting_user_name"] = True
    return {
        "status": "mapping",
        "message": "Slowly turn around and scan the room.",
        "map": current_map,
    }


@app.post("/answer-prompt")
def answer_prompt(payload: dict):
    snapshot = get_live_spatial_snapshot()
    return _answer_prompt_direct(
        prompt_id=payload.get("prompt_id") or payload.get("id"),
        answer_text=payload.get("answer") or payload.get("command"),
        snapshot=snapshot,
    )


@app.post("/save-map")
def save_spatial_map(payload: dict = None):
    payload = payload or {}
    min_observed_ms = max(0, min(int(payload.get("min_observed_ms") or 0), 2000))
    snapshot = get_live_spatial_snapshot()
    if min_observed_ms:
        spatial_memory.refresh_mapping(
            live_grid=snapshot["grid"],
            semantic_objects=snapshot["semantic_objects"],
        )
        time.sleep(min_observed_ms / 1000.0)
        snapshot = get_live_spatial_snapshot()
    current_map = spatial_memory.save_current_map(
        live_grid=snapshot["grid"],
        semantic_objects=snapshot["semantic_objects"],
        map_name=payload.get("map_name"),
    )
    if not current_map:
        raise HTTPException(status_code=400, detail="No active map to save.")
    return {
        "status": "saved",
        "message": "Room saved.",
        "map_id": current_map["map_id"],
        "map_name": current_map["map_name"],
        "file_name": f"{current_map['map_id']}.json",
        "landmark_count": len(current_map.get("landmarks", [])),
        "static_object_count": len(current_map.get("static_objects", [])),
        "grid_counts": current_map.get("metadata", {}).get("grid_counts", {}),
        "coverage_percent": current_map.get("metadata", {}).get("coverage_percent", 0.0),
    }


@app.post("/stop-mapping")
def stop_mapping():
    spatial_memory.stop_mapping()
    return {"status": "idle", "mode": spatial_memory.state["mode"]}


@app.get("/maps")
def list_spatial_maps():
    return {
        "maps": spatial_memory.list_maps(),
        "current_map_id": spatial_memory.state.get("current_map_id"),
        "graph": spatial_memory.map_graph,
    }


@app.post("/load-map")
def load_spatial_map(payload: dict):
    snapshot = get_live_spatial_snapshot()
    map_data = spatial_memory.load_map(
        map_id=payload.get("map_id"),
        map_name=payload.get("map_name"),
        anchor_pose=snapshot["pose"],
    )
    if not map_data:
        raise HTTPException(status_code=404, detail="Map not found.")
    return {"status": "loaded", "map": map_data}


@app.post("/unload-map")
def unload_spatial_map():
    spatial_memory.unload_map()
    return {
        "status": "unloaded",
        "message": "Map unloaded. Live sensing remains active, but no saved room map is selected.",
    }


@app.post("/set-current-map")
def set_current_map(payload: dict):
    return load_spatial_map(payload)


@app.get("/current-map")
def get_current_map():
    current_map = spatial_memory.state.get("current_map")
    if not current_map:
        return {"active": False, "map": None}
    return {"active": True, "map": current_map}


@app.post("/start-navigation")
def start_spatial_navigation(payload: dict):
    snapshot = get_live_spatial_snapshot()
    goal_type = payload.get("goal_type", "exit")
    best, error = spatial_memory.start_navigation(
        goal_type=goal_type,
        target_landmark_id=payload.get("target_landmark_id"),
        pose=snapshot["pose"],
        semantic_objects=snapshot["semantic_objects"],
    )
    if error:
        raise HTTPException(status_code=400, detail=error)
    return {
        "status": "navigating",
        "target": best["door"],
        "score": best["score"],
        "path": best["path"],
    }


@app.get("/navigation-guidance")
def navigation_guidance():
    snapshot = get_live_spatial_snapshot()
    return spatial_memory.navigation_guidance(
        pose=snapshot["pose"],
        semantic_objects=snapshot["semantic_objects"],
        zones=snapshot.get("zones"),
    )


@app.post("/stop-navigation")
def stop_spatial_navigation():
    spatial_memory.stop_navigation()
    navigator.stop_navigation()
    return {"status": "stopped", "mode": spatial_memory.state["mode"]}


@app.post("/link-map-door")
def link_map_door(payload: dict):
    required = ["map_id", "door_id", "target_map_id", "target_door_id"]
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {', '.join(missing)}")
    link = spatial_memory.link_map_door(
        map_id=payload["map_id"],
        door_id=payload["door_id"],
        target_map_id=payload["target_map_id"],
        target_door_id=payload["target_door_id"],
    )
    return {"status": "linked", "link": link, "graph": spatial_memory.map_graph}
@app.post("/voice-command")
async def voice_command(file: UploadFile = File(...)):
    global mapping_prompt_awaiting_answer
    audio_bytes = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".caf") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    result = get_whisper_model().transcribe(temp_audio_path, language="en")
    command_text = result["text"].strip().lower()

    with zva.frame_lock:
        detections = list(zva.latest_detections)
        guidance = zva.guidance_cmd
        zones_data = dict(zva.zones_data)

    llm_detections = [_format_llm_detection(d) for d in detections]

    snapshot = get_live_spatial_snapshot()

    pending_prompt_response = _handle_pending_yes_no_prompt(command_text, snapshot)
    if pending_prompt_response:
        return pending_prompt_response

    if "start mapping" in command_text or "map this room" in command_text:
        mapping_prompt_awaiting_answer = False
        room_name = "Room"
        for phrase in ["this room as", "room as", "map as"]:
            if phrase in command_text:
                room_name = command_text.split(phrase, 1)[1].strip().title() or room_name
                break
        current_map = spatial_memory.start_mapping(
            map_name=room_name,
            live_grid=snapshot["grid"],
            semantic_objects=snapshot["semantic_objects"],
            pose=snapshot["pose"],
        )
        return {
            "transcript": command_text,
            "response": f"Mapping started for {current_map['map_name']}. Slowly turn around and scan the room."
        }

    if "save map" in command_text or "save this map" in command_text:
        current_map = spatial_memory.save_current_map(
            live_grid=snapshot["grid"],
            semantic_objects=snapshot["semantic_objects"],
        )
        if current_map:
            return {
                "transcript": command_text,
                "response": "Room saved."
            }
        return {
            "transcript": command_text,
            "response": "No active map is being created right now."
        }

    if "stop mapping" in command_text:
        spatial_memory.stop_mapping()
        return {
            "transcript": command_text,
            "response": "Mapping stopped."
        }

    if _is_exit_navigation_command(command_text):
        return _start_exit_navigation_response(command_text, snapshot)

    
    switch_answer = navigator.handle_switch_answer(command_text)
    if switch_answer:
        return {
            "transcript": command_text,
            "response": switch_answer
        }

    if "stop navigation" in command_text or "cancel navigation" in command_text:
        spatial_memory.stop_navigation()
        response = navigator.stop_navigation()
        return {
            "transcript": command_text,
            "response": response
        }

    navigation_keywords = [
        "guide me",
        "take me",
        "go to",
        "find",
        "where is",
        "i want to",
        "lead me",
        "bring me",
    ]

    if any(keyword in command_text for keyword in navigation_keywords):
        response = navigator.start_navigation(command_text, detections)
        return {
            "transcript": command_text,
            "response": response
        }

    active_guidance = navigator.update_navigation(detections)
    if active_guidance:
        return {
            "transcript": command_text,
            "response": active_guidance
        }

    response = ask_llm(
        question=command_text,
        detections=llm_detections,
        direction_summary={
            "left_distance_mm": float(zones_data.get("left", 0)),
            "center_distance_mm": float(zones_data.get("center", 0)),
            "right_distance_mm": float(zones_data.get("right", 0)),
            "center_open_space_ratio": float(zones_data.get("center_open_space_ratio", 0)),
            "scene_hint": zones_data.get("scene_hint", "unknown"),
            "best_direction": guidance,
        },
        ocr_text=""
    )

    return {
        "transcript": command_text,
        "response": response
    }


@app.post("/command")
def text_command(payload: dict):
    command_text = str(payload.get("command", "")).strip().lower()
    if not command_text:
        raise HTTPException(status_code=400, detail="Missing command text.")

    with zva.frame_lock:
        detections = list(zva.latest_detections)
        guidance = zva.guidance_cmd
        zones_data = dict(zva.zones_data)

    llm_detections = [_format_llm_detection(d) for d in detections]

    snapshot = get_live_spatial_snapshot()

    pending_prompt_response = _handle_pending_yes_no_prompt(command_text, snapshot)
    if pending_prompt_response:
        return pending_prompt_response

    if "start mapping" in command_text or "map this room" in command_text:
        room_name = "Room"
        for phrase in ["this room as", "room as", "map as"]:
            if phrase in command_text:
                room_name = command_text.split(phrase, 1)[1].strip().title() or room_name
                break
        current_map = spatial_memory.start_mapping(
            map_name=room_name,
            live_grid=snapshot["grid"],
            semantic_objects=snapshot["semantic_objects"],
            pose=snapshot["pose"],
        )
        return {
            "transcript": command_text,
            "response": f"Mapping started for {current_map['map_name']}. Slowly turn around and scan the room."
        }

    if "save map" in command_text or "save this map" in command_text:
        current_map = spatial_memory.save_current_map(
            live_grid=snapshot["grid"],
            semantic_objects=snapshot["semantic_objects"],
        )
        if current_map:
            return {
                "transcript": command_text,
                "response": "Room saved."
            }
        return {
            "transcript": command_text,
            "response": "No active map is being created right now."
        }

    if "stop mapping" in command_text:
        spatial_memory.stop_mapping()
        return {
            "transcript": command_text,
            "response": "Mapping stopped."
        }

    if _is_exit_navigation_command(command_text):
        return _start_exit_navigation_response(command_text, snapshot)

    switch_answer = navigator.handle_switch_answer(command_text)
    if switch_answer:
        return {
            "transcript": command_text,
            "response": switch_answer
        }

    if "stop navigation" in command_text or "cancel navigation" in command_text:
        spatial_memory.stop_navigation()
        response = navigator.stop_navigation()
        return {
            "transcript": command_text,
            "response": response
        }

    navigation_keywords = [
        "guide me",
        "take me",
        "go to",
        "find",
        "where is",
        "i want to",
        "lead me",
        "bring me",
    ]

    if any(keyword in command_text for keyword in navigation_keywords):
        response = navigator.start_navigation(command_text, detections)
        return {
            "transcript": command_text,
            "response": response
        }

    active_guidance = navigator.update_navigation(detections)
    if active_guidance:
        return {
            "transcript": command_text,
            "response": active_guidance
        }

    response = ask_llm(
        question=command_text,
        detections=llm_detections,
        direction_summary={
            "left_distance_mm": float(zones_data.get("left", 0)),
            "center_distance_mm": float(zones_data.get("center", 0)),
            "right_distance_mm": float(zones_data.get("right", 0)),
            "center_open_space_ratio": float(zones_data.get("center_open_space_ratio", 0)),
            "scene_hint": zones_data.get("scene_hint", "unknown"),
            "best_direction": guidance,
        },
        ocr_text=""
    )

    return {
        "transcript": command_text,
        "response": response
    }


@app.post("/api/map")
async def save_map(payload: dict):
    session_id = payload.get("session_id", "default_session")
    pose = payload.get("pose", {})
    grid = payload.get("grid_data", [])
    
    db_map = OccupancyMap(
        session_id=session_id,
        timestamp=time.time(),
        pose_x=float(pose.get("x", 0.0)),
        pose_z=float(pose.get("z", 0.0)),
        yaw=float(pose.get("yaw", 0.0)),
        grid_data=grid
    )
    
    async with AsyncSessionLocal() as session:
        async with session.begin():
            session.add(db_map)
        await session.commit()
        
    return {"status": "success"}

@app.get("/api/map")
async def get_map(session_id: str = "default_session"):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(OccupancyMap)
            .where(OccupancyMap.session_id == session_id)
            .order_by(OccupancyMap.timestamp.desc())
            .limit(1)
        )
        db_map = result.scalars().first()
        if db_map:
            return {
                "session_id": db_map.session_id,
                "timestamp": db_map.timestamp,
                "pose": {"x": db_map.pose_x, "z": db_map.pose_z, "yaw": db_map.yaw},
                "grid_data": db_map.grid_data
            }
        return {"error": "No map found for this session"}

@app.post("/api/transcribe")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".caf") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    result = get_whisper_model().transcribe(temp_audio_path, language="en")
    transcript = result["text"].strip()
    return {"transcript": transcript}



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VICKY FastAPI Server")
    parser.add_argument("--no-camera", action="store_true", help="Do not run local ZED camera vision loop (use edge client)")
    parser.add_argument("--headless", action="store_true", help="Run local ZED camera without the OpenCV preview window")
    args = parser.parse_known_args()[0]

    if not args.no_camera:
        zva.ENABLE_DISPLAY = not args.headless
        if zva.ENABLE_DISPLAY:
            print("[SERVER] Local ZED preview display forced ON. Use --headless to hide it.")
        else:
            print("[SERVER] Local ZED preview display disabled by --headless.")
        vision_thread = threading.Thread(
            target=zva.vision_loop,
            daemon=True
        )
        vision_thread.start()
        print("[SERVER] Started local ZED vision thread.")
    else:
        print("[SERVER] Running in edge-telemetry mode. Local ZED camera thread disabled.")

    local_urls = [f"http://{address}:8000" for address in get_local_ipv4_addresses()]
    if local_urls:
        print("[SERVER] Local network URLs:")
        for url in local_urls:
            print(f"  {url}")

    uvicorn.run(app, host="0.0.0.0", port=8000)
