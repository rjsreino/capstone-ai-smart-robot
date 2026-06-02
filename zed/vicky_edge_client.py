import os
import sys
import json
import time
import base64
import asyncio
import argparse
import random
import threading
from typing import Dict, Any, List, Tuple, Set, Optional
import cv2
import numpy as np
import pygame
import websockets
import torch

# Import our existing ZED Depth Processor (if available)
ZED_AVAILABLE: bool = False
try:
    from zed_depth_processor import ZedDepthProcessor, ZedDepthConfig
    ZED_AVAILABLE = True
except ImportError:
    pass

LLM_AVAILABLE: bool = False
try:
    from llm_reasoner import ask_llm
    LLM_AVAILABLE = True
except ImportError:
    pass

YOLO_AVAILABLE: bool = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

# ==========================================
# CONFIGURATION AND CLI ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(description="VICKY Project Edge Client")
parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=3,
                    help="Compute Architecture: 1=Local, 2=Cloud, 3=Hybrid")
parser.add_argument("--perception", type=str, choices=["A", "B"], default="A",
                    help="Perception Paradigm: A=YOLO+Depth Context, B=Visual VLM")
parser.add_argument("--safety", type=str, choices=["X", "Y"], default="X",
                    help="Safety Alerting: X=Deterministic Clicks, Y=Conversational Speech")
parser.add_argument("--server", type=str, default="127.0.0.1:8000",
                    help="IP and Port of remote FastAPI Cloud Server")
parser.add_argument("--session", type=str, default="session_prod_01",
                    help="Session identifier string")
parser.add_argument("--simulated", action="store_true", default=False,
                    help="Force ZED camera hardware simulation mode")
args, unknown = parser.parse_known_args()

COMPUTE_MODE: int = args.mode       # 1, 2, 3
PERCEPTION_MODE: str = args.perception # A, B
SAFETY_MODE: str = args.safety       # X, Y
SERVER_URI: str = args.server
SESSION_ID: str = args.session
FORCE_SIMULATION: bool = args.simulated

# Setup sound mixer for alert clicks
pygame.mixer.init()

def generate_click_sound() -> pygame.mixer.Sound:
    sample_rate: int = 44100
    duration: float = 0.05  # 50 ms click
    n_samples: int = int(sample_rate * duration)
    buf: np.ndarray = np.zeros((n_samples, 2), dtype=np.int16)
    
    # Simple sine sweep alert click (chirp down)
    for i in range(n_samples):
        t: float = float(i) / sample_rate
        freq: float = 1500.0 - (1000.0 * (t / duration))
        val: int = int(16384 * np.sin(2.5 * np.pi * freq * t) * (1.0 - (t / duration)))
        buf[i][0] = val  # Left Channel
        buf[i][1] = val  # Right Channel
        
    return pygame.sndarray.make_sound(buf)

click_sound: pygame.mixer.Sound = generate_click_sound()

# Initialize Local YOLO if needed
local_yolo: Optional[YOLO] = None
if COMPUTE_MODE in [1, 3] and YOLO_AVAILABLE:
    try:
        print("[CLIENT] Initializing local YOLOv8n model...")
        local_yolo = YOLO("yolov8n.pt")
        device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
        local_yolo.to(device_str)
        print(f"[CLIENT] Local YOLOv8 loaded on: {device_str}")
    except Exception as e:
        print(f"[CLIENT WARNING] Failed to load local YOLO model: {e}")

# ==========================================
# HARDWARE AND SIMULATOR LAYER
# ==========================================
class EdgeSensorPipeline:
    def __init__(self, use_simulator: bool = False) -> None:
        self.use_simulator: bool = use_simulator or (not ZED_AVAILABLE)
        self.processor: Optional[ZedDepthProcessor] = None
        self.sim_t: float = 0.0
        self.sim_pose_x: float = 0.0
        self.sim_pose_z: float = 0.0
        self.sim_yaw: float = 0.0
        
        if not self.use_simulator:
            try:
                config = ZedDepthConfig(
                    resolution="vga",
                    fps=15,
                    depth_mode="PERFORMANCE",
                    min_depth=400,
                    max_depth=5000
                )
                self.processor = ZedDepthProcessor(config)
                self.processor.start()
                print("[CLIENT] Hardware ZED camera initialized successfully.")
            except Exception as e:
                print(f"[CLIENT WARNING] ZED Camera initialization failed: {e}. Falling back to Simulator.")
                self.use_simulator = True
                
        if self.use_simulator:
            print("[CLIENT] Sensor Pipeline running in SIMULATION mode.")

    def grab_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]], bool]:
        """
        Grabs active frame. Returns:
        - RGB Frame (np.ndarray)
        - Depth Frame (np.ndarray in mm)
        - SLAM Pose dictionary
        - Tracking OK status (bool)
        """
        if not self.use_simulator and self.processor is not None:
            if self.processor.grab_frame():
                rgb = self.processor.get_rgb_frame()
                depth = self.processor.get_depth_frame()
                tracking_ok = getattr(self.processor, "is_tracking_ok", False)
                pose = {
                    "position_meters": {
                        "x": float(self.processor.tx) if hasattr(self.processor, "tx") else 0.0,
                        "y": float(self.processor.ty) if hasattr(self.processor, "ty") else 0.0,
                        "z": float(self.processor.tz) if hasattr(self.processor, "tz") else 0.0
                    },
                    "rotation_degrees": {
                        "roll": float(self.processor.roll) if hasattr(self.processor, "roll") else 0.0,
                        "pitch": float(self.processor.pitch) if hasattr(self.processor, "pitch") else 0.0,
                        "yaw": float(self.processor.yaw) if hasattr(self.processor, "yaw") else 0.0
                    }
                }
                return rgb, depth, pose, tracking_ok
            else:
                time.sleep(0.01)
                return None, None, None, False
        else:
            # Generate simulated walk data and obstacle mapping
            time.sleep(0.067)  # simulate 15 FPS camera loop delay
            self.sim_t += 0.067
            
            # Simulate forward path walking with slight periodic drift
            self.sim_pose_z += 0.05
            self.sim_pose_x = 0.5 * np.sin(self.sim_t * 0.5)
            self.sim_yaw = 5.0 * np.cos(self.sim_t * 0.5)
            
            pose = {
                "position_meters": {"x": self.sim_pose_x, "y": 0.0, "z": self.sim_pose_z},
                "rotation_degrees": {"roll": 0.0, "pitch": 0.0, "yaw": self.sim_yaw}
            }
            
            # Draw synthetic corridor
            rgb = np.zeros((376, 672, 3), dtype=np.uint8)
            rgb[:] = [15, 12, 10]
            
            cv2.line(rgb, (0, 376), (336, 188), (50, 50, 50), 2)
            cv2.line(rgb, (672, 376), (336, 188), (50, 50, 50), 2)
            
            depth = np.zeros((376, 672), dtype=np.float32)
            for r in range(376):
                depth[r, :] = max(400.0, 5000.0 - (r * 12.0))
                
            obs_z: int = 2500 - (100 * (int(self.sim_t) % 20))
            if obs_z > 400:
                c_x: int = int(336 + 100 * np.sin(self.sim_t))
                c_y: int = 220
                radius: int = int(50000 / obs_z)
                cv2.circle(rgb, (c_x, c_y), radius, (0, 0, 255), -1)
                cv2.putText(rgb, "Chair", (c_x - 15, c_y - radius - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
                
                y1, y2 = max(0, c_y - radius), min(376, c_y + radius)
                x1, x2 = max(0, c_x - radius), min(672, c_x + radius)
                depth[y1:y2, x1:x2] = float(obs_z)
                
            # Simulate visual tracking drops periodically (drops for 5s every 20s)
            tracking_ok = (int(self.sim_t) % 20 < 15)
            
            return rgb, depth, pose, tracking_ok

    def stop(self) -> None:
        if self.processor is not None:
            self.processor.stop()

# ==========================================
# PERCEPTION PROCESSING LOGIC
# ==========================================
def process_safety_corridors(depth_frame: np.ndarray) -> Dict[str, Any]:
    """Slices depth matrix into left, center, right vertical corridors."""
    h, w = depth_frame.shape
    col_w: int = w // 3
    
    h_start: int = int(h * 0.3)
    h_end: int = int(h * 0.8)
    
    left_slice = depth_frame[h_start:h_end, 0:col_w]
    center_slice = depth_frame[h_start:h_end, col_w:col_w*2]
    right_slice = depth_frame[h_start:h_end, col_w*2:w]
    
    def get_valid_median(matrix_slice: np.ndarray) -> float:
        valid = matrix_slice[(matrix_slice >= 300) & (matrix_slice <= 6000)]
        return float(np.median(valid)) if len(valid) > 0 else 5000.0

    l_mm = get_valid_median(left_slice)
    c_mm = get_valid_median(center_slice)
    r_mm = get_valid_median(right_slice)
    
    escape_vector = "GO FORWARD"
    if c_mm < 1200:
        if l_mm > r_mm:
            escape_vector = "TURN LEFT"
        else:
            escape_vector = "TURN RIGHT"
        if l_mm < 700 and r_mm < 700:
            escape_vector = "STOP! BLOCKED"
            
    return {
        "left_clearance_mm": l_mm,
        "center_clearance_mm": c_mm,
        "right_clearance_mm": r_mm,
        "escape_vector": escape_vector
    }

def run_local_bounding_box_pipeline(rgb: np.ndarray, depth: np.ndarray) -> List[Dict[str, Any]]:
    """Runs local YOLO object detection and maps depth centers."""
    if local_yolo is None:
        return []
    
    h, w, _ = rgb.shape
    results = local_yolo(rgb, conf=0.25, imgsz=320, verbose=False)
    result = results[0]
    objects: List[Dict[str, Any]] = []
    
    if result.boxes is not None:
        for idx, box in enumerate(result.boxes):
            cls_id = int(box.cls[0].item())
            class_name = local_yolo.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
            cx, cy = max(0, min(cx, w-1)), max(0, min(cy, h-1))
            val_mm = depth[cy, cx]
            
            depth_meters: float = 0.0
            dist_cat: str = "unknown"
            
            if not (np.isnan(val_mm) or np.isinf(val_mm) or val_mm <= 0):
                depth_meters = float(val_mm) / 1000.0
                if depth_meters < 0.8:
                    dist_cat = "very close"
                elif depth_meters < 1.5:
                    dist_cat = "close"
                else:
                    dist_cat = "medium"
                    
            lateral_offset_meters: float = ((cx - (w / 2.0)) / (w / 2.0)) * depth_meters * 0.5
            
            objects.append({
                "tracking_id": idx,
                "class": class_name,
                "3d_coordinates": {"x": lateral_offset_meters, "y": 0.0, "z": depth_meters},
                "distance_category": dist_cat
            })
            
    return objects

# ==========================================
# NETWORK COGNITION ENGINE (CLIENT LOOP)
# ==========================================
class VickyEdgeApp:
    def __init__(self) -> None:
        self.sensor = EdgeSensorPipeline(use_simulator=FORCE_SIMULATION)
        self.websocket_telemetry: Optional[websockets.WebSocketClientProtocol] = None
        self.websocket_video: Optional[websockets.WebSocketClientProtocol] = None
        self.running: bool = True
        self.loop_count: int = 0
        
        # Audio click warning worker
        self.alert_thread = threading.Thread(target=self._safety_click_alert_worker, daemon=True)
        self.current_center_clearance: float = 5000.0
        
        # Smartphone IMU state variables for thread-safety and dead reckoning
        self.latest_imu_data: Optional[Dict[str, Any]] = None
        self.imu_lock = threading.Lock()
        
        self.fallback_active: bool = False
        self.dr_x: float = 0.0
        self.dr_y: float = 0.0
        self.dr_z: float = 0.0
        self.dr_vx: float = 0.0
        self.dr_vz: float = 0.0
        self.last_imu_time: float = time.time()
        self.last_valid_pose: Dict[str, Any] = {
            "position_meters": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation_degrees": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        }
        
        # Local Wi-Fi Smartphone Loop Server
        self.smartphone_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.phone_server_thread = threading.Thread(target=self._start_smartphone_server, daemon=True)
        self.phone_server_thread.start()

    def _start_smartphone_server(self) -> None:
        """Runs a localized WebSocket server to feed commands to the smartphone transceiver."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def handler(websocket: websockets.WebSocketServerProtocol, path: str) -> None:
            print("[EDGE SERVER] Smartphone audio transceiver linked.")
            self.smartphone_clients.add(websocket)
            try:
                async for message in websocket:
                    try:
                        data_str = message.decode("utf-8") if isinstance(message, bytes) else message
                        data = json.loads(data_str)
                        if data.get("packet_type") == "SMARTPHONE_IMU_STREAM":
                            with self.imu_lock:
                                self.latest_imu_data = {
                                    "timestamp": float(data.get("timestamp", time.time())),
                                    "linear_accel": {
                                        "x": float(data.get("linear_accel", {}).get("x", 0.0)),
                                        "y": float(data.get("linear_accel", {}).get("y", 0.0)),
                                        "z": float(data.get("linear_accel", {}).get("z", 0.0))
                                    },
                                    "rotation_rpy": {
                                        "roll": float(data.get("rotation_rpy", {}).get("roll", 0.0)),
                                        "pitch": float(data.get("rotation_rpy", {}).get("pitch", 0.0)),
                                        "yaw": float(data.get("rotation_rpy", {}).get("yaw", 0.0))
                                    }
                                }
                    except Exception as e:
                        pass
            except Exception:
                pass
            finally:
                if websocket in self.smartphone_clients:
                    self.smartphone_clients.remove(websocket)
                print("[EDGE SERVER] Smartphone audio transceiver unlinked.")
                
        async def main_server():
            async with websockets.serve(handler, "0.0.0.0", 8005):
                print("[EDGE SERVER] Local smartphone WebSocket server listening on port 8005.")
                await asyncio.Future() # keep serving forever
                
        try:
            loop.run_until_complete(main_server())
        except Exception as e:
            print(f"[EDGE SERVER WARNING] Failed to start smartphone WebSocket server on port 8005: {e}")

    async def broadcast_to_smartphone(self, message_text: str) -> None:
        """Broadcasts navigation guidance strings to connected smartphone web app clients."""
        if not self.smartphone_clients:
            return
        for client in list(self.smartphone_clients):
            try:
                await client.send(json.dumps({"instruction": message_text}))
            except Exception:
                if client in self.smartphone_clients:
                    self.smartphone_clients.remove(client)

    def _safety_click_alert_worker(self) -> None:
        """Local geometric safety alerting thread (Approach X - alarm clicks)."""
        while self.running:
            dist_meters: float = self.current_center_clearance / 1000.0
            if SAFETY_MODE == "X" and dist_meters < 1.2:
                # 1.2m -> 600ms pause, 0.4m -> 80ms pause
                pause: float = max(0.08, (dist_meters - 0.4) * 0.65)
                try:
                    click_sound.play()
                except Exception:
                    pass
                time.sleep(pause)
            else:
                time.sleep(0.1)

    async def trigger_async_vlm_query(self, rgb: np.ndarray, query_text: str) -> None:
        """Simulates Smartphone cellular uplink posting VLM request to Cloud Server asynchronously."""
        _, jpeg_buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_b64: str = base64.b64encode(jpeg_buf.tobytes()).decode("utf-8")
        
        loop = asyncio.get_running_loop()
        
        def make_http_call() -> Dict[str, Any]:
            import urllib.request
            import urllib.error
            url: str = f"http://{SERVER_URI}/api/infer/vlm"
            data: bytes = json.dumps({
                "image_bytes_base64": img_b64,
                "user_query": query_text
            }).encode("utf-8")
            
            req = urllib.request.Request(
                url, 
                data=data, 
                headers={'Content-Type': 'application/json'}
            )
            try:
                with urllib.request.urlopen(req, timeout=5.0) as response:
                    return json.loads(response.read().decode("utf-8"))
            except Exception as e:
                return {"error": str(e)}
                
        print("[CLIENT HYBRID] Dispatching VLM query asynchronously over cellular link...")
        res = await loop.run_in_executor(None, make_http_call)
        if "error" not in res:
            response_text = res.get("response", "")
            print(f"[CLIENT HYBRID ANSWER] Server VLM Response: '{response_text}' (Latency: {res.get('inference_latency_ms', 0):.0f}ms)")
            await self.broadcast_to_smartphone(f"Cloud update: {response_text}")
        else:
            print(f"[CLIENT HYBRID ERROR] Server VLM query failed: {res['error']}")

    async def connect_sockets(self) -> None:
        """Asynchronously connect video and telemetry sockets to FastAPI."""
        try:
            uri_tele: str = f"ws://{SERVER_URI}/ws/telemetry/stream"
            uri_video: str = f"ws://{SERVER_URI}/ws/video/stream"
            print(f"[CLIENT] Connecting telemetry WS: {uri_tele}")
            self.websocket_telemetry = await websockets.connect(uri_tele)
            print(f"[CLIENT] Connecting video WS: {uri_video}")
            self.websocket_video = await websockets.connect(uri_video)
            print("[CLIENT] Sockets connected successfully to Cloud Backend.")
        except Exception as e:
            print(f"[CLIENT WARNING] Failed to connect to server: {e}. Running local fallbacks.")

    async def run(self) -> None:
        await self.connect_sockets()
        self.alert_thread.start()
        
        print(f"[CLIENT] Starting Edge Main Pipeline Loop...")
        print(f"  Compute Mode:    Approach {COMPUTE_MODE}")
        print(f"  Perception Mode: Approach {PERCEPTION_MODE}")
        print(f"  Safety Mode:     Approach {SAFETY_MODE}")
        print("="*60)
        
        while self.running:
            self.loop_count += 1
            start_time: float = time.time()
            rgb, depth, pose, tracking_ok = self.sensor.grab_frame()
            
            if rgb is None or depth is None or pose is None:
                await asyncio.sleep(0.01)
                continue
                
            current_time = time.time()
            if tracking_ok:
                # Primary spatial map baseline
                self.fallback_active = False
                self.last_valid_pose = pose
                self.last_imu_time = current_time
            else:
                # Initialize fallback tracking protocol when tracking drops
                if not self.fallback_active:
                    self.fallback_active = True
                    self.dr_x = self.last_valid_pose["position_meters"]["x"]
                    self.dr_y = self.last_valid_pose["position_meters"]["y"]
                    self.dr_z = self.last_valid_pose["position_meters"]["z"]
                    self.dr_vx = 0.0
                    self.dr_vz = 0.0
                    self.last_imu_time = current_time
                    
                # Integrate the incoming smartphone linear acceleration metrics over time
                dt = current_time - self.last_imu_time
                self.last_imu_time = current_time
                
                # Fetch latest smartphone IMU
                with self.imu_lock:
                    imu = self.latest_imu_data
                    
                ax_world = 0.0
                az_world = 0.0
                roll_deg = self.last_valid_pose["rotation_degrees"]["roll"]
                pitch_deg = self.last_valid_pose["rotation_degrees"]["pitch"]
                yaw_deg = self.last_valid_pose["rotation_degrees"]["yaw"]
                
                if imu is not None:
                    accel = imu.get("linear_accel", {})
                    rot = imu.get("rotation_rpy", {})
                    
                    ax_local = float(accel.get("x", 0.0))
                    ay_local = float(accel.get("y", 0.0))
                    az_local = float(accel.get("z", 0.0))
                    
                    roll_rad = float(rot.get("roll", 0.0))
                    pitch_rad = float(rot.get("pitch", 0.0))
                    yaw_rad = float(rot.get("yaw", 0.0))
                    
                    # Convert to degrees for payload
                    roll_deg = float(np.degrees(roll_rad))
                    pitch_deg = float(np.degrees(pitch_rad))
                    yaw_deg = float(np.degrees(yaw_rad))
                    
                    # 2D projection along the yaw heading with pitch/roll correction
                    a_forward_h = ay_local * np.cos(pitch_rad) - az_local * np.sin(pitch_rad)
                    a_lateral_h = ax_local * np.cos(roll_rad) + az_local * np.sin(roll_rad)
                    
                    # Rotate horizontally to world coordinates
                    ax_world = a_lateral_h * np.cos(yaw_rad) + a_forward_h * np.sin(yaw_rad)
                    az_world = -a_lateral_h * np.sin(yaw_rad) + a_forward_h * np.cos(yaw_rad)
                
                # Double integration with damping (0.95 velocity dampening factor)
                self.dr_vx = (self.dr_vx + ax_world * dt) * 0.95
                self.dr_vz = (self.dr_vz + az_world * dt) * 0.95
                
                self.dr_x += self.dr_vx * dt
                self.dr_z += self.dr_vz * dt
                
                # Override the SLAM baseline
                pose = {
                    "position_meters": {"x": self.dr_x, "y": self.dr_y, "z": self.dr_z},
                    "rotation_degrees": {"roll": roll_deg, "pitch": pitch_deg, "yaw": yaw_deg}
                }
                
            zones: Dict[str, Any] = process_safety_corridors(depth)
            
            inference_ms: float = 0.0
            objects: List[Dict[str, Any]] = []
            audio_text_guidance: str = ""
            
            # Get latest IMU state under lock to avoid race condition during payload creation
            with self.imu_lock:
                current_imu = self.latest_imu_data
            
            # ----------------------------------------------------
            # COMPUTE MODE 1: LOCAL ALL-IN-ONE
            # ----------------------------------------------------
            if COMPUTE_MODE == 1:
                inference_start: float = time.time()
                if PERCEPTION_MODE == "A":
                    objects = run_local_bounding_box_pipeline(rgb, depth)
                    inference_ms = (time.time() - inference_start) * 1000.0
                    
                    if zones["center_clearance_mm"] < 1200:
                        audio_text_guidance = f"Obstacle ahead. Please {zones['escape_vector'].lower()}."
                    else:
                        audio_text_guidance = "Path ahead looks clear."
                else:
                    await asyncio.sleep(0.2) # simulate VLM local CPU latency
                    inference_ms = 200.0
                    audio_text_guidance = "Local VLM analysis: Obstacle blocking center pathway."
                
                self.current_center_clearance = zones["center_clearance_mm"]
                
                # Broadcast local text guidance to Wi-Fi loop Smartphone clients
                await self.broadcast_to_smartphone(audio_text_guidance)
                
                total_delay_ms: float = (time.time() - start_time) * 1000.0
                print(f"[HUD LOCAL] SRT: {total_delay_ms:.1f}ms | Latency: {inference_ms:.1f}ms | Zones (C): {zones['center_clearance_mm']:.0f}mm | Guidance: {audio_text_guidance}")
                
                # Send telemetry / video to server if connected
                if self.websocket_video and self.websocket_telemetry:
                    try:
                        _, jpeg_buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        await self.websocket_video.send(jpeg_buf.tobytes())
                        
                        payload = {
                            "packet_metadata": {
                                "session_id": SESSION_ID,
                                "timestamp": start_time,
                                "compute_node": "local",
                                "mode_flag": PERCEPTION_MODE
                            },
                            "user_spatial_pose": pose,
                            "spatial_depth_zones": zones,
                            "semantic_objects_in_frustum": objects,
                            "performance_metrics": {
                                "inference_latency_ms": inference_ms,
                                "network_rtt_ms": 0.0,
                                "total_srt_ms": total_delay_ms,
                                "hallucination_flag": False
                            },
                            "smartphone_imu": current_imu
                        }
                        await self.websocket_telemetry.send(json.dumps(payload))
                    except Exception:
                        pass
            
            # ----------------------------------------------------
            # COMPUTE MODE 2: DISTRIBUTED PURE CLOUD (CLOSED-LOOP)
            # ----------------------------------------------------
            elif COMPUTE_MODE == 2:
                if self.websocket_video and self.websocket_telemetry:
                    try:
                        # 1. Compress image to JPEG and encode to Base64
                        _, jpeg_buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frame_jpg_base64: str = base64.b64encode(jpeg_buf.tobytes()).decode("utf-8")
                        
                        # 2. Downsample depth matrix to 32x18 shape to save bandwidth
                        depth_downsampled: List[List[float]] = cv2.resize(
                            depth, (32, 18), interpolation=cv2.INTER_AREA
                        ).tolist()
                        
                        # 3. Form payload meeting exact schema
                        payload = {
                            "packet_metadata": {
                                "session_id": SESSION_ID,
                                "timestamp": start_time,
                                "compute_node": "cloud",
                                "mode_flag": PERCEPTION_MODE
                            },
                            "user_spatial_pose": pose,
                            "depth_matrix_downsampled": depth_downsampled,
                            "frame_jpg_base64": frame_jpg_base64,
                            "performance_metrics": {
                                "inference_latency_ms": 0.0,
                                "network_rtt_ms": 15.0, # assumed default
                                "total_srt_ms": 0.0,
                                "hallucination_flag": False
                            },
                            "smartphone_imu": current_imu
                        }
                        
                        # Send binary frame to HUD video stream endpoint
                        await self.websocket_video.send(jpeg_buf.tobytes())
                        
                        # Send telemetry / image base64 packet
                        await self.websocket_telemetry.send(json.dumps(payload))
                        
                        # Await closed-loop server response
                        response_str: str = await self.websocket_telemetry.recv()
                        response: Dict[str, Any] = json.loads(response_str)
                        
                        server_zones: Dict[str, Any] = response.get("spatial_depth_zones", {})
                        self.current_center_clearance = server_zones.get("center_clearance_mm", 5000.0)
                        
                        inference_ms = response.get("performance_metrics", {}).get("inference_latency_ms", 0.0)
                        total_delay_ms = (time.time() - start_time) * 1000.0
                        
                        print(f"[HUD CLOUD] SRT: {total_delay_ms:.1f}ms | Server Latency: {inference_ms:.1f}ms | Zones (C): {self.current_center_clearance:.0f}mm | Vector: {server_zones.get('escape_vector')}")
                        
                        if self.current_center_clearance < 1200:
                            audio_text_guidance = f"Obstacle ahead. Please {server_zones.get('escape_vector', 'STOP! BLOCKED').lower()}."
                        else:
                            audio_text_guidance = "Path ahead looks clear."
                        await self.broadcast_to_smartphone(audio_text_guidance)
                    
                    except Exception as se:
                        print(f"[CLIENT SERVER ERROR] Mode 2 closed-loop streaming failed: {se}")
                else:
                    await asyncio.sleep(0.067)
            
            # ----------------------------------------------------
            # COMPUTE MODE 3: EDGE-CLOUD HYBRID
            # ----------------------------------------------------
            elif COMPUTE_MODE == 3:
                inference_start = time.time()
                objects = run_local_bounding_box_pipeline(rgb, depth)
                inference_ms = (time.time() - inference_start) * 1000.0
                
                self.current_center_clearance = zones["center_clearance_mm"]
                
                if zones["center_clearance_mm"] < 1200:
                    audio_text_guidance = f"Obstacle ahead. Please {zones['escape_vector'].lower()}."
                else:
                    audio_text_guidance = "Path ahead looks clear."
                await self.broadcast_to_smartphone(audio_text_guidance)
                
                if self.websocket_video and self.websocket_telemetry:
                    try:
                        # Send compressed JPEG frames to video WS
                        _, jpeg_buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        await self.websocket_video.send(jpeg_buf.tobytes())
                        
                        payload = {
                            "packet_metadata": {
                                "session_id": SESSION_ID,
                                "timestamp": start_time,
                                "compute_node": "hybrid",
                                "mode_flag": PERCEPTION_MODE
                            },
                            "user_spatial_pose": pose,
                            "spatial_depth_zones": zones,
                            "semantic_objects_in_frustum": objects,
                            "performance_metrics": {
                                "inference_latency_ms": inference_ms,
                                "network_rtt_ms": 15.0,
                                "total_srt_ms": (time.time() - start_time) * 1000.0,
                                "hallucination_flag": False
                            },
                            "smartphone_imu": current_imu
                        }
                        await self.websocket_telemetry.send(json.dumps(payload))
                    except Exception:
                        pass
                
                # Perform VLM cellular simulation loop asynchronously every ~3 seconds
                if self.loop_count % 45 == 0:
                    asyncio.create_task(self.trigger_async_vlm_query(rgb, "Check path exit paths and warnings."))
                    
                await asyncio.sleep(0.01)
                
            await asyncio.sleep(0.01)

    def stop_pipeline(self) -> None:
        self.running = False
        self.sensor.stop()
        pygame.mixer.quit()

def main() -> None:
    app = VickyEdgeApp()
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\nStopping Edge Client...")
        app.stop_pipeline()

if __name__ == "__main__":
    main()
