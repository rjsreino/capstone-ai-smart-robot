import os
import json
import time
import base64
import asyncio
from typing import Set, Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Import our async DB logging module
from vicky_db import db_logger

app = FastAPI(title="VICKY Project Cloud Server Hub")

# Track connected WebSocket clients for HUD panel
hud_clients: Set[WebSocket] = set()
video_clients: Set[WebSocket] = set()

# Set up templates directory
TEMPLATES_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Load YOLOv8 model on the server
yolo_model: Optional[YOLO] = None
try:
    print("[SERVER] Loading server-side YOLOv8n model...")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.to(device)
    print(f"[SERVER] YOLOv8n successfully loaded on device: {device}")
except Exception as e:
    print(f"[SERVER WARNING] Failed to load YOLOv8n model: {e}")

# VLM Query Model
class VlmQuery(BaseModel):
    image_bytes_base64: str
    user_query: str

def run_vlm_inference(image_rgb: np.ndarray, query: str) -> str:
    """
    Runs mock Vision-Language reasoning to represent server-side VLM.
    Decodes query constraints and returns descriptive navigational prompts.
    """
    print(f"[SERVER VLM] Received query: '{query}'")
    # Simulate VLM inference latency
    time.sleep(0.35)
    
    q_lower = query.lower()
    if "door" in q_lower or "exit" in q_lower:
        return "I see an exit door at 3.5 meters ahead. The pathway is clear."
    elif "path" in q_lower or "safe" in q_lower:
        return "The path straight ahead contains a chair. Please steer right to find a clear corridor."
    else:
        return f"VLM Response: Based on your question '{query}', I see clear passage to your right."

def process_safety_corridors_from_matrix(depth_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Processes safety corridors directly from the downsampled depth matrix (32x18 shape).
    Divided into Left, Center, Right vertical columns.
    """
    h, w = depth_matrix.shape
    col_w = w // 3
    
    # Vertically focus on row slices representing chest/waist level (rows 5 to 14 of 18)
    h_start = int(h * 0.3)
    h_end = int(h * 0.8)
    
    left_slice = depth_matrix[h_start:h_end, 0:col_w]
    center_slice = depth_matrix[h_start:h_end, col_w:col_w*2]
    right_slice = depth_matrix[h_start:h_end, col_w*2:w]
    
    def get_valid_median(matrix_slice: np.ndarray) -> float:
        valid = matrix_slice[(matrix_slice >= 300) & (matrix_slice <= 6000)]
        return float(np.median(valid)) if len(valid) > 0 else 5000.0

    l_mm = get_valid_median(left_slice)
    c_mm = get_valid_median(center_slice)
    r_mm = get_valid_median(right_slice)
    
    escape_vector = "GO FORWARD"
    if c_mm < 1200: # Blocked center
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

# HTTP Endpoint: Serve the visual supervisor HUD dashboard
@app.get("/", response_class=HTMLResponse)
async def get_hud() -> HTMLResponse:
    hud_file_path = os.path.join(TEMPLATES_DIR, "hud.html")
    if os.path.exists(hud_file_path):
        with open(hud_file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>Templates directory or hud.html not found!</h1>", status_code=404)

# HTTP Endpoint: VLM Cellular Inference Endpoint (Mode 3 Hybrid support)
@app.post("/api/infer/vlm")
async def cloud_infer_vlm(query_payload: VlmQuery) -> Dict[str, Any]:
    try:
        img_data = base64.b64decode(query_payload.image_bytes_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Invalid image base64"}
            
        start_time = time.time()
        vlm_response = run_vlm_inference(img, query_payload.user_query)
        latency = (time.time() - start_time) * 1000.0
        
        return {
            "response": vlm_response,
            "inference_latency_ms": latency
        }
    except Exception as e:
        return {"error": f"VLM processing failed: {str(e)}"}

# WebSocket Endpoint: Stream telemetry/closed-loop server-side inference
@app.websocket("/ws/telemetry/stream")
async def telemetry_stream_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    print("[SERVER] Edge client connected to telemetry / closed-loop WebSocket.")
    try:
        while True:
            data_str = await websocket.receive_text()
            payload = json.loads(data_str)
            
            metadata = payload.get("packet_metadata", {})
            compute_node = metadata.get("compute_node", "unknown")
            
            # --- MODE 2: DISTRIBUTED PURE CLOUD INFERENCE ---
            if compute_node == "cloud" and "frame_jpg_base64" in payload:
                inference_start = time.time()
                
                # 1. Decode Frame and run YOLOv8
                jpg_bytes = base64.b64decode(payload["frame_jpg_base64"])
                nparr = np.frombuffer(jpg_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                detections = []
                if img is not None and yolo_model is not None:
                    yolo_res = yolo_model(img, conf=0.25, imgsz=320, verbose=False)[0]
                    if yolo_res.boxes is not None:
                        for idx, box in enumerate(yolo_res.boxes):
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            class_name = yolo_model.names[cls_id]
                            
                            detections.append({
                                "tracking_id": idx,
                                "class": class_name,
                                "3d_coordinates": {"x": 0.0, "y": 0.0, "z": 0.0}, # Fill 3D below using depth matrix
                                "distance_category": "unknown",
                                "bbox": [x1, y1, x2, y2]
                            })
                            
                # 2. Parse downsampled depth matrix to obtain safety clearance zones
                depth_list = payload.get("depth_matrix_downsampled", [])
                zones = {"left_clearance_mm": 5000.0, "center_clearance_mm": 5000.0, "right_clearance_mm": 5000.0, "escape_vector": "GO FORWARD"}
                
                if len(depth_list) > 0:
                    depth_matrix = np.array(depth_list, dtype=np.float32)
                    zones = process_safety_corridors_from_matrix(depth_matrix)
                    
                    # Refine detections 3D coordinates based on downsampled depth matrix scaling
                    if img is not None:
                        h, w, _ = img.shape
                        dh, dw = depth_matrix.shape
                        for det in detections:
                            # Map bounding box center from image to downsampled depth matrix index
                            bbox = det.pop("bbox", [0, 0, 0, 0])
                            x1_det, y1_det, x2_det, y2_det = bbox
                            box_cx = (x1_det + x2_det) // 2
                            box_cy = (y1_det + y2_det) // 2
                            dcx = int((box_cx / w) * dw)
                            dcy = int((box_cy / h) * dh)
                            dcx = max(0, min(dcx, dw - 1))
                            dcy = max(0, min(dcy, dh - 1))
                            
                            val_mm = depth_matrix[dcy, dcx]
                            if not (np.isnan(val_mm) or np.isinf(val_mm) or val_mm <= 0):
                                depth_m = float(val_mm) / 1000.0
                                lateral_offset_m = ((box_cx - (w / 2.0)) / (w / 2.0)) * depth_m * 0.5
                                det["3d_coordinates"] = {"x": lateral_offset_m, "y": 0.0, "z": depth_m}
                                
                                if depth_m < 0.8:
                                    det["distance_category"] = "very close"
                                elif depth_m < 1.5:
                                    det["distance_category"] = "close"
                                else:
                                    det["distance_category"] = "medium"

                inference_latency = (time.time() - inference_start) * 1000.0
                
                # 3. Rebuild processed payload according to schema rules
                processed_payload = {
                    "packet_metadata": {
                        "session_id": metadata.get("session_id", "default"),
                        "timestamp": metadata.get("timestamp", time.time()),
                        "compute_node": "cloud",
                        "mode_flag": metadata.get("mode_flag", "A")
                    },
                    "user_spatial_pose": payload.get("user_spatial_pose", {}),
                    "spatial_depth_zones": zones,
                    "semantic_objects_in_frustum": detections,
                    "performance_metrics": {
                        "inference_latency_ms": inference_latency,
                        "network_rtt_ms": float(payload.get("performance_metrics", {}).get("network_rtt_ms", 0.0)),
                        "total_srt_ms": (time.time() - metadata.get("timestamp", time.time())) * 1000.0,
                        "hallucination_flag": False
                    }
                }
                
                # Send server inference result directly back to the edge client over the same socket
                await websocket.send_text(json.dumps(processed_payload))
                
                # Asynchronously log to Database
                await db_logger.log_frame_data(processed_payload)
                
                # Broadcast payload to HUD dashboard listeners
                hud_data_str = json.dumps(processed_payload)
                for client in list(hud_clients):
                    try:
                        await client.send_text(hud_data_str)
                    except Exception:
                        hud_clients.remove(client)
            
            else:
                # --- MODE 1 & 3: EDGE PROCESSES SENDING TELEMETRY TO DB & HUD ---
                await db_logger.log_frame_data(payload)
                
                # Broadcast incoming telemetry to HUD client list
                for client in list(hud_clients):
                    try:
                        await client.send_text(data_str)
                    except Exception:
                        hud_clients.remove(client)
                        
    except WebSocketDisconnect:
        print("[SERVER] Edge client telemetry WebSocket disconnected.")
    except Exception as e:
        print(f"[SERVER ERROR] Telemetry handler exception: {e}")

# WebSocket Endpoint: Stream Video Frame bytes to HUD Web Page
@app.websocket("/ws/video/stream")
async def video_stream_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            # Receive binary JPEG bytes from edge client
            frame_bytes = await websocket.receive_bytes()
            # Broadcast JPEG bytes to all connected HUD video panels
            for client in list(video_clients):
                try:
                    await client.send_bytes(frame_bytes)
                except Exception:
                    video_clients.remove(client)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[SERVER ERROR] Video stream handler exception: {e}")

# WebSocket Endpoint: HUD Telemetry Listener registration
@app.websocket("/ws/hud")
async def hud_registration_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    hud_clients.add(websocket)
    print(f"[SERVER] HUD Telemetry listener registered. Total: {len(hud_clients)}")
    try:
        while True:
            await websocket.receive_text() # Keep-alive loop
    except WebSocketDisconnect:
        hud_clients.remove(websocket)
        print(f"[SERVER] HUD Telemetry listener disconnected. Total: {len(hud_clients)}")

# WebSocket Endpoint: HUD Video Listener registration
@app.websocket("/ws/video")
async def video_registration_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    video_clients.add(websocket)
    print(f"[SERVER] HUD Video listener registered. Total: {len(video_clients)}")
    try:
        while True:
            await websocket.receive_text() # Keep-alive loop
    except WebSocketDisconnect:
        video_clients.remove(websocket)
        print(f"[SERVER] HUD Video listener disconnected. Total: {len(video_clients)}")

@app.on_event("startup")
async def startup_event() -> None:
    await db_logger.start()

@app.on_event("shutdown")
async def shutdown_event() -> None:
    await db_logger.stop()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
