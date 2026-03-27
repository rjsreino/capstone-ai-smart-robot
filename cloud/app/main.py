from __future__ import annotations

import json
import asyncio
import base64
import hashlib
import hmac
import importlib
import logging
import os
import re
import secrets
import subprocess
import socket
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any, AsyncIterator, Optional
from starlette.middleware.trustedhost import TrustedHostMiddleware

LOGGER = logging.getLogger("uvicorn.error").getChild(__name__)

# Global lock to prevent USB camera contention between OAK-D and webcam
_USB_CAMERA_LOCK = asyncio.Lock()

# Ensure project root is in Python path for imports when running as service
# This MUST happen before any app.* imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
    
from ble import WifiManager

import anyio
import httpx
from fastapi import FastAPI, HTTPException, Header, Query, Request, Response, WebSocket, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Import Rover controller - try multiple import styles so deployment layouts work
Rover = None
serial = None
_rover_import_error = None


def _attempt_import(module_name: str, package: str | None = None) -> tuple[bool, str | None]:
    """Attempt to import the rover_controller module variant."""

    global Rover, serial, _rover_import_error

    try:
        if package is None:
            module = importlib.import_module(module_name)
        else:
            module = importlib.import_module(module_name, package)
    except ImportError as exc:
        LOGGER.debug("Failed to import %s (package=%s): %s", module_name, package, exc, exc_info=True)
        return False, f"{module_name} ({package or '-'}) import error: {exc}"

    rover_cls = getattr(module, "Rover", None)

    if rover_cls is None:
        message = f"{module_name} missing Rover class (Rover={rover_cls})"
        LOGGER.debug(message)
        return False, message

    serial_module = getattr(module, "serial", None)
    if serial_module is None:
        try:
            serial_module = importlib.import_module("serial")
        except ImportError as exc:
            message = (
                f"{module_name} missing serial module and pyserial import failed: {exc}"
            )
            LOGGER.debug(message, exc_info=True)
            return False, message

    Rover = rover_cls
    serial = serial_module
    _rover_import_error = None
    LOGGER.info("rover_controller module imported via %s", module_name)
    return True, None


_import_failures: list[str] = []
_candidates: list[tuple[str, str | None]] = [("rover_controller", None)]
if __package__:
    _candidates.append((".rover_controller", __package__))
_candidates.append(("app.rover_controller", None))

for candidate, package in _candidates:
    success, error_msg = _attempt_import(candidate, package)
    if success:
        break
    if error_msg:
        _import_failures.append(error_msg)
else:
    _rover_import_error = " ; ".join(_import_failures)

from .camera import (
    CameraError,
    CameraService,
    DepthAICameraSource,
    OpenCVCameraSource,
    PlaceholderCameraSource,
)

# Import cv2 for USB camera testing (optional dependency)
try:
    import cv2
except ImportError:
    cv2 = None
from .oak_stream import get_snapshot as oak_snapshot
from .oak_stream import get_video_response as oak_video_response
from .oak_stream import shutdown as oak_shutdown
from .oak_stream import ensure_runtime as oak_ensure_runtime
from .oak_stream import frame_to_jpeg as oak_frame_to_jpeg

# Face recognition service
try:
    from .face_recognition_service import FaceRecognitionService, FaceRecognitionError
    FACE_RECOGNITION_AVAILABLE = True
except ImportError as exc:
    LOGGER.warning("Face recognition service not available: %s", exc)
    FACE_RECOGNITION_AVAILABLE = False
    FaceRecognitionService = None
    FaceRecognitionError = None

# Gesture detection service
try:
    from .gesture_detection import detect_gesture_from_image
    GESTURE_DETECTION_AVAILABLE = True
except ImportError as exc:
    LOGGER.warning("Gesture detection service not available: %s", exc)
    GESTURE_DETECTION_AVAILABLE = False
    detect_gesture_from_image = None

# Meeting service
try:
    from .meeting_service import MeetingService
    MEETING_SERVICE_AVAILABLE = True
except ImportError as exc:
    LOGGER.warning("Meeting service not available: %s", exc)
    MEETING_SERVICE_AVAILABLE = False
    MeetingService = None
from .models import (
    AddFaceRequest,
    AddFaceResponse,
    CaptureRequest,
    CaptureResponse,
    CaptureType,
    ClaimConfirmRequest,
    ClaimConfirmResponse,
    ClaimControlResponse,
    ClaimRequestResponse,
    FaceRecognitionResponse,
    HeadCommand,
    HealthResponse,
    KnownFacesResponse,
    LightCommand,
    MeetingSummary,
    MeetingSummaryListResponse,
    MeetingType,
    MeetingUploadResponse,
    MeetingRecordingStatus,
    Mode,
    ModeResponse,
    MoveCommand,
    NodCommand,
    NetworkInfoResponse,
    StatusResponse,
    StopResponse,
    WiFiConnectRequest,
    WiFiConnectResponse,
    WiFiNetwork,
    WiFiScanResponse,
    WiFiStatusResponse,
)

APP_NAME = "rovy-api"
APP_VERSION = "0.1.0"
ROBOT_NAME = "rover-01"
ROBOT_SERIAL = "rovy-01"
BOUNDARY = "frame"

# Log import status for Rover
if Rover is None:
    if _rover_import_error:
        LOGGER.error("IMPORT ERROR DETAILS: rover_controller module not available: %s; OLED display will be disabled", _rover_import_error)
    else:
        LOGGER.warning("rover_controller module not available; OLED display will be disabled")
else:
    LOGGER.info("rover_controller module imported successfully")

# Claim system state
STATE = {
    "claimed": False,
    "control_token_hash": None,
    "pin": None,
    "pin_exp": 0,
    "controller": {"sid": None, "last": 0, "ttl": 30},
}

_PIN_RESET_TASK: asyncio.Task[None] | None = None

_PLACEHOLDER_JPEG = base64.b64decode(
    """
/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDwwMDw8NDhERExUTGBonHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8f/2wBDARESEhgVGBoZGB4dHy8fLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8v/3QAEAA3/2gAIAQEAAD8A/wD/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAsf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/AR//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/AR//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Ar//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/IX//2Q==
""".strip()
)
app = FastAPI(title="Capstone Robot API", version=APP_VERSION)

# Add before CORS middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]
)

@app.middleware("http")
async def handle_proxy_headers(request: Request, call_next):
    # Check if this is coming through Tailscale Funnel
    if request.headers.get("tailscale-funnel-request"):
        # Tailscale Funnel sometimes doesn't preserve POST method
        # Get the intended method from a custom header if available
        pass
    
    response = await call_next(request)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def hash_token(t: str) -> str:
    """Hash a token using SHA-256."""
    return hashlib.sha256(t.encode()).hexdigest()


def verify_token(token: str) -> bool:
    """Verify a control token using constant-time comparison."""
    if not (STATE["claimed"] and STATE["control_token_hash"]):
        return False
    return hmac.compare_digest(hash_token(token), STATE["control_token_hash"])


def verify_session(session_id: str) -> bool:
    """Verify a controller session ID and update last access time."""
    if not STATE["controller"]["sid"]:
        return False
    # Use constant-time comparison for session ID
    if not hmac.compare_digest(session_id, STATE["controller"]["sid"]):
        return False
    now = time.time()
    if now - STATE["controller"]["last"] > STATE["controller"]["ttl"]:
        # Session expired
        STATE["controller"]["sid"] = None
        STATE["controller"]["last"] = 0
        return False
    # Update last access time
    STATE["controller"]["last"] = now
    return True


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to verify x-control-token and session_id for protected endpoints."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Check if endpoint requires control token
        protected = (
            path.startswith("/control")
            or path.startswith("/claim/release")
            or path.startswith("/settings")
            or path == "/claim-control"
        )

        if protected:
            token = request.headers.get("x-control-token")
            if not token or not verify_token(token):
                raise HTTPException(status_code=401, detail="unauthorized")

            # Check if endpoint requires controller session (all /control/* except /claim-control)
            if path.startswith("/control/"):
                session_id = request.headers.get("session-id")
                if not session_id or not verify_session(session_id):
                    raise HTTPException(status_code=403, detail="invalid_or_expired_session")

        return await call_next(request)


app.add_middleware(AuthMiddleware)

def _get_base_controller() -> Optional[Any]:
    cached_controller: Optional[Any] = getattr(app.state, "base_controller", None)

    if cached_controller is not None:
        return cached_controller

    if Rover is None:
        LOGGER.debug("Rover class not available (import failed)")
        return None

    LOGGER.debug("Attempting to initialize base_controller for PIN display")
    device, _ = _find_serial_device()
    if not device:
        LOGGER.debug("No serial device available for Rover initialization")
        return None

    try:
        base_controller = Rover(device)
        LOGGER.info("Rover initialized on %s", device)
    except Exception as exc:
        LOGGER.warning("Failed to initialize Rover on %s: %s", device, exc, exc_info=True)
        return None

    app.state.base_controller = base_controller
    return base_controller


def _cancel_pin_reset_task() -> None:
    """Cancel any scheduled OLED reset task."""

    global _PIN_RESET_TASK

    if _PIN_RESET_TASK is not None and not _PIN_RESET_TASK.done():
        _PIN_RESET_TASK.cancel()

    _PIN_RESET_TASK = None


def _schedule_pin_reset(pin_value: str, expiration: float) -> None:
    """Schedule OLED reset once the current PIN expires."""

    if expiration <= time.time():
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        LOGGER.debug("No running event loop available to schedule PIN reset task")
        return

    _cancel_pin_reset_task()

    global _PIN_RESET_TASK
    _PIN_RESET_TASK = loop.create_task(
        _reset_display_after_expiration(pin_value, expiration)
    )


async def _reset_display_after_expiration(pin_value: str, expiration: float) -> None:
    """Reset the OLED display once the active PIN has expired."""

    global _PIN_RESET_TASK
    current_task = asyncio.current_task()
    delay = max(expiration - time.time(), 0)
    cancelled_exc = anyio.get_cancelled_exc_class()

    try:
        await anyio.sleep(delay)
    except cancelled_exc:  # pragma: no cover - cancellation is timing dependent
        LOGGER.debug("PIN expiration reset task cancelled before completion")
        return

    if STATE["pin"] != pin_value or time.time() < expiration:
        if _PIN_RESET_TASK is current_task:
            _PIN_RESET_TASK = None
        return

    base_controller = _get_base_controller()

    if base_controller and hasattr(base_controller, "display_reset"):
        try:
            base_controller.display_reset()
            LOGGER.info("OLED display reset after PIN expiration")
        except Exception as exc:  # pragma: no cover - hardware dependent
            LOGGER.error(
                "Failed to reset OLED display after PIN expiration: %s", exc, exc_info=True
            )
    else:
        LOGGER.debug(
            "Skipping OLED reset after PIN expiration; base controller unavailable or missing display_reset"
        )

    if STATE["pin"] == pin_value:
        STATE["pin"] = None
        STATE["pin_exp"] = 0

    if _PIN_RESET_TASK is current_task:
        _PIN_RESET_TASK = None

def _get_env_flag(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False

    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


_FORCE_WEBCAM = _get_env_flag("CAMERA_FORCE_WEBCAM")
_WEBCAM_DEVICE = os.getenv("CAMERA_WEBCAM_DEVICE")


def _iter_webcam_candidates() -> list[int | str]:
    """Return preferred webcam device identifiers.

    The order favours explicit configuration, then any USB cameras,
    and finally generic `/dev/video*` indices. OAK-D devices are explicitly
    skipped since they're reserved for AI vision via the persistent pipeline.
    """

    candidates: list[int | str] = []

    # Log all available video devices for debugging
    by_id_dir = Path("/dev/v4l/by-id")
    if by_id_dir.is_dir():
        all_devices = list(by_id_dir.iterdir())
        LOGGER.info(f"📹 Found {len(all_devices)} video device(s) in /dev/v4l/by-id/")
        for entry in all_devices:
            device_type = "OAK-D" if any(x in entry.name.lower() for x in ["oak", "depthai", "luxonis"]) else "USB"
            LOGGER.info(f"   - {device_type}: {entry.name}")

    if _WEBCAM_DEVICE is not None:
        try:
            candidates.append(int(_WEBCAM_DEVICE))
        except ValueError:
            candidates.append(_WEBCAM_DEVICE)
        LOGGER.info(f"Using explicit camera device from env: {_WEBCAM_DEVICE}")

    # Prefer stable /dev/v4l/by-id/ paths over numeric indices
    # Use the by-id path directly (not resolved) so it stays stable even if device re-enumerates
    if by_id_dir.is_dir():
        for entry in sorted(by_id_dir.iterdir()):
            name = entry.name.lower()
            # Skip OAK-D devices (reserved for AI vision via persistent pipeline)
            if "oak" in name or "depthai" in name or "luxonis" in name:
                LOGGER.debug(f"Skipping OAK-D device for streaming: {entry.name}")
                continue
            # Use any USB camera with video-index0 (main video stream)
            if "usb" in name and "video-index0" in name:
                try:
                    # Use the stable by-id path directly, not the resolved /dev/videoX path
                    # This way if the device re-enumerates, the path still works
                    stable_path = str(entry)
                    candidates.append(stable_path)
                    LOGGER.info(f"Added USB camera candidate: {entry.name}")
                except OSError:
                    continue

    # Fall back to common numeric indices if nothing more specific was found.
    # These entries are appended after any explicit USB camera devices
    # so that external USB webcams are preferred over built-in cameras.
    generic_indices = range(0, 4)
    for index in generic_indices:
        if index not in candidates:
            candidates.append(index)

    LOGGER.info(f"USB camera candidates (in priority order): {candidates}")
    return candidates


def check_cameras_for_microphone() -> list[dict[str, Any]]:
    """Check all connected cameras to see if they have microphone capabilities.
    
    Returns a list of dictionaries with camera info and microphone status.
    Each dict contains:
    - device: device path or index
    - name: device name
    - has_microphone: boolean indicating if microphone is present
    - device_path: resolved /dev/videoX path if available
    """
    cameras_with_mic = []
    
    # Get all video devices
    video_devices = []
    
    # Check /dev/v4l/by-id/ for stable device paths
    by_id_dir = Path("/dev/v4l/by-id")
    if by_id_dir.is_dir():
        for entry in sorted(by_id_dir.iterdir()):
            try:
                resolved_path = entry.resolve()
                if resolved_path.exists() and str(resolved_path).startswith("/dev/video"):
                    video_devices.append({
                        "device": str(entry),
                        "name": entry.name,
                        "resolved_path": str(resolved_path)
                    })
            except (OSError, RuntimeError):
                continue
    
    # Also check /dev/video* directly for any devices not in by-id
    for video_path in Path("/dev").glob("video*"):
        if video_path.is_char_device():
            device_num = video_path.name.replace("video", "")
            try:
                int(device_num)  # Make sure it's a number
                # Check if we already have this device
                if not any(v.get("resolved_path") == str(video_path) for v in video_devices):
                    video_devices.append({
                        "device": str(video_path),
                        "name": f"video{device_num}",
                        "resolved_path": str(video_path)
                    })
            except ValueError:
                continue
    
    # Check each device for microphone capabilities
    for device_info in video_devices:
        device_path = device_info["resolved_path"]
        has_mic = False
        
        try:
            # Use v4l2-ctl to query device capabilities
            # Check if device has audio input (V4L2_CAP_AUDIO indicates audio support)
            result = subprocess.run(
                ["v4l2-ctl", "--device", device_path, "--info"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                output = result.stdout.lower()
                # Check for audio-related capabilities
                # V4L2 devices with microphones typically show audio input capabilities
                if "audio" in output or "mic" in output:
                    has_mic = True
                else:
                    # Also check device capabilities directly
                    caps_result = subprocess.run(
                        ["v4l2-ctl", "--device", device_path, "--all"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if caps_result.returncode == 0:
                        caps_output = caps_result.stdout.lower()
                        # Look for audio input indicators
                        if any(keyword in caps_output for keyword in ["audio", "mic", "input", "capture"]):
                            # More specific check - look for audio device type
                            if "audio" in caps_output:
                                has_mic = True
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            LOGGER.debug(f"Could not check microphone for {device_path}: {e}")
            continue
        
        # Also check if there's a corresponding ALSA device (some USB cameras expose audio via ALSA)
        try:
            # Check for ALSA devices that might correspond to this camera
            alsa_result = subprocess.run(
                ["arecord", "-l"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if alsa_result.returncode == 0:
                # Some USB cameras show up in ALSA as USB audio devices
                # This is a heuristic - if we find USB audio devices, some cameras might have mics
                if "usb" in alsa_result.stdout.lower():
                    # This is not definitive, but worth noting
                    pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        cameras_with_mic.append({
            "device": device_info["device"],
            "name": device_info["name"],
            "has_microphone": has_mic,
            "device_path": device_path
        })
    
    return cameras_with_mic


def _create_camera_service() -> CameraService:
    """Create camera service for mobile app streaming using USB camera."""
    primary_source = None

    # Use USB camera for streaming (OAK-D is reserved for AI vision via /shot endpoint)
    if OpenCVCameraSource.is_available():
        LOGGER.info("🎥 Initializing USB camera for streaming (separate from OAK-D)...")
        
        for candidate in _iter_webcam_candidates():
            try:
                LOGGER.info(
                    "Attempting webcam device %s for primary stream source",
                    candidate,
                )
                # Test if device is accessible before committing
                test_cap = cv2.VideoCapture(candidate)
                if not test_cap.isOpened():
                    LOGGER.warning("Device %s not accessible, skipping", candidate)
                    test_cap.release()
                    continue
                test_cap.release()
                
                # Now create the actual source
                primary_source = OpenCVCameraSource(device=candidate)
                LOGGER.info("✅ Using OpenCV camera source for streaming: %s", candidate)
                break
            except CameraError as exc:
                LOGGER.warning(
                    "OpenCV camera source unavailable on %s: %s",
                    candidate,
                    exc,
                )
                primary_source = None
                continue
            except Exception as exc:
                LOGGER.warning(
                    "Unexpected error with device %s: %s",
                    candidate,
                    exc,
                )
                primary_source = None
                continue
        else:
            LOGGER.warning("⚠️  Unable to open any USB webcam device for streaming")
    else:
        LOGGER.warning(
            "OpenCV package not installed; skipping USB camera stream. Install the 'opencv-python' package to enable it."
        )

    # Do NOT use OAK-D for streaming - it's reserved for AI vision via /shot endpoint
    # if primary_source is None and DepthAICameraSource.is_available():
    #     try:
    #         primary_source = DepthAICameraSource()
    #         LOGGER.info("Using DepthAI camera source for streaming (USB camera unavailable)")
    #     except CameraError as exc:
    #         LOGGER.warning("DepthAI camera source unavailable: %s", exc)

    fallback_source = None
    if _PLACEHOLDER_JPEG:
        fallback_source = PlaceholderCameraSource(_PLACEHOLDER_JPEG)
        LOGGER.info("Using placeholder camera source for fallback frames")

    if primary_source is None and fallback_source is None:
        raise RuntimeError("No camera source available for streaming")

    return CameraService(primary_source, fallback=fallback_source, boundary=BOUNDARY, frame_rate=10.0)


app.state.camera_service = _create_camera_service()


async def _get_webcam_frame_safe() -> bytes:
    """Get frame from webcam with USB lock to prevent interference with OAK-D."""
    async with _USB_CAMERA_LOCK:
        return await app.state.camera_service.get_frame()


# Give USB camera time to fully initialize before starting OAK-D
# This prevents USB bandwidth conflicts during initialization
import time as time_module
time_module.sleep(0.5)

# Initialize OAK-D availability check (but don't start pipeline yet to save USB bandwidth)
# We'll start the pipeline on-demand when /shot is called
if DepthAICameraSource.is_available():
    app.state.oakd_available = True
    app.state.oakd_device = None
    app.state.oakd_queue = None
    app.state.oakd_last_used = 0  # Timestamp of last use
    LOGGER.info("✅ OAK-D camera available (will start on-demand to minimize USB interference)")
else:
    LOGGER.warning("⚠️  DepthAI not available - /shot endpoint will not work")
    LOGGER.info("   USB camera will still work for streaming")
    app.state.oakd_available = False
    app.state.oakd_device = None
    app.state.oakd_queue = None

# Initialize face recognition service
if FACE_RECOGNITION_AVAILABLE:
    try:
        known_faces_path = _project_root / "known-faces"
        app.state.face_recognition = FaceRecognitionService(
            known_faces_dir=known_faces_path,
            model_name="arcface_r100_v1",  # Best accuracy model
            threshold=0.6,  # Similarity threshold (lower = more strict)
        )
        LOGGER.info("Face recognition service initialized successfully")
    except Exception as exc:
        LOGGER.error("Failed to initialize face recognition service: %s", exc, exc_info=True)
        app.state.face_recognition = None
else:
    app.state.face_recognition = None

# Meeting service will be initialized after helper functions are defined
app.state.meeting_service = None


def _find_serial_device() -> tuple[Optional[str], list[str]]:
    """Find available serial device for rover controller."""
    if serial is None:
        return None, []

    # Allow explicit override via environment variable.
    env_device = os.getenv("ROVER_SERIAL_DEVICE")
    attempted: list[str] = []
    if env_device:
        if os.path.exists(env_device):
            try:
                test_ser = serial.Serial(env_device, 115200, timeout=0.2)
                test_ser.close()
            except (serial.SerialException, PermissionError, OSError) as exc:
                LOGGER.warning(
                    "Configured rover serial device %s unavailable: %s", env_device, exc
                )
            else:
                LOGGER.info("Using rover serial device from environment: %s", env_device)
                return env_device, [env_device]
        else:
            LOGGER.warning("Configured rover serial device %s does not exist", env_device)
        attempted.append(env_device)

    candidates: list[str] = []

    # Probe through pyserial's port listing when available for dynamic detection.
    try:
        from serial.tools import list_ports  # type: ignore

        for port in list_ports.comports():
            if port.device:
                candidates.append(port.device)
    except Exception as exc:  # pragma: no cover - defensive; list_ports may be missing
        LOGGER.debug("Failed to enumerate serial ports via pyserial: %s", exc, exc_info=True)

    # Ensure we also try a sensible default set for Jetson-style deployments.
    candidates.extend(
        device
        for device in ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0", "/dev/ttyUSB1"]
        if device not in candidates
    )

    for device in candidates:
        if device not in attempted:
            attempted.append(device)
        if os.path.exists(device):
            try:
                # Try to open it to verify it's accessible
                test_ser = serial.Serial(device, 115200, timeout=0.2)
                test_ser.close()
                LOGGER.info("Detected rover serial device: %s", device)
                return device, attempted
            except (serial.SerialException, PermissionError, OSError) as exc:
                LOGGER.debug("Serial device %s unavailable: %s", device, exc)
                continue
    return None, attempted


@app.get("/")
async def root() -> dict[str, object]:
    """Simple index listing the most commonly used endpoints."""

    return {
        "status": "ok",
        "endpoints": [
            "/video",
            "/shot",
            "/camera/stream",
            "/camera/snapshot",
        ],
    }


@app.get("/video")
async def video_stream() -> StreamingResponse:
    """Expose the main MJPEG stream at the top level for convenience."""
    # Use camera service (USB webcam) instead of oak_stream to free up OAK-D for /shot endpoint
    async def stream_generator() -> AsyncIterator[bytes]:
        async for chunk in _camera_stream(app.state.camera_service, None):
            yield chunk
    
    return StreamingResponse(stream_generator(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")


@app.websocket("/camera/ws")
async def camera_websocket(websocket: WebSocket):
    """WebSocket endpoint for streaming camera frames as base64-encoded JPEG."""
    await websocket.accept()
    LOGGER.info("WebSocket client connected")
    
    try:
        # Use the existing camera service instead of oak_stream
        camera_service = app.state.camera_service
        if not camera_service:
            await websocket.send_text(json.dumps({"error": "Camera not available"}))
            await websocket.close()
            return
            
        LOGGER.info("Using main camera service for WebSocket stream")
        
        while True:
            try:
                # Get frame from the camera service (with USB lock protection)
                frame_bytes = await _get_webcam_frame_safe()
                if frame_bytes:
                    # Send as JSON with base64 frame
                    b64_frame = base64.b64encode(frame_bytes).decode('utf-8')
                    await websocket.send_text(json.dumps({"frame": b64_frame}))
                else:
                    LOGGER.debug("No frame available from camera")
                    await asyncio.sleep(0.1)
                    continue
                    
            except RuntimeError as e:
                # WebSocket closed by client
                if "close message has been sent" in str(e).lower():
                    LOGGER.info("WebSocket closed by client")
                    break
                raise
            except Exception as frame_error:
                LOGGER.warning(f"Frame capture error: {frame_error}")
                await asyncio.sleep(0.1)
                continue
            
            # Control frame rate (e.g., 10 FPS)
            await asyncio.sleep(0.1)
            
    except Exception as exc:
        LOGGER.error("WebSocket error: %s", exc, exc_info=True)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        LOGGER.info("WebSocket disconnected")


@app.websocket("/json")
async def json_control_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time motor/lights control via JSON commands.
    
    Command format:
    - Movement: {"T": 1, "L": left_speed, "R": right_speed}
    - Lights: {"T": 132, "IO4": pwm_value, "IO5": pwm_value}
    """
    await websocket.accept()
    LOGGER.info("JSON control WebSocket connected")
    
    base_controller = _get_base_controller()
    
    try:
        while True:
            data = await websocket.receive_json()
            cmd_type = data.get("T", 0)
            
            if cmd_type == 1:  # Movement command
                left_speed = float(data.get("L", 0))
                right_speed = float(data.get("R", 0))
                
                # Invert controls (forward/backward and left/right are reversed on hardware)
                left_speed = -left_speed
                right_speed = -right_speed
                
                # Try set_motor first (Rover adapter), then base_speed_ctrl (direct BaseController)
                if base_controller:
                    try:
                        if hasattr(base_controller, "set_motor"):
                            await anyio.to_thread.run_sync(
                                base_controller.set_motor, left_speed, right_speed
                            )
                        elif hasattr(base_controller, "base_speed_ctrl"):
                            await anyio.to_thread.run_sync(
                                base_controller.base_speed_ctrl, left_speed, right_speed
                            )
                    except Exception as exc:
                        LOGGER.warning("Motor control failed: %s", exc)
                        
            elif cmd_type == 132:  # Lights command
                io4 = int(data.get("IO4", 0))
                io5 = int(data.get("IO5", 0))
                
                if base_controller and hasattr(base_controller, "lights_ctrl"):
                    try:
                        await anyio.to_thread.run_sync(
                            base_controller.lights_ctrl, io4, io5
                        )
                    except Exception as exc:
                        LOGGER.warning("Lights control failed: %s", exc)
                        
            elif cmd_type == 133:  # Gimbal command
                pan = float(data.get("X", 0))
                tilt = float(data.get("Y", 0))
                speed = int(data.get("SPD", 10))
                
                if base_controller and hasattr(base_controller, "gimbal_ctrl"):
                    try:
                        await anyio.to_thread.run_sync(
                            base_controller.gimbal_ctrl, pan, tilt, speed, 0
                        )
                    except Exception as exc:
                        LOGGER.warning("Gimbal control failed: %s", exc)
                        
    except Exception as exc:
        LOGGER.debug("JSON WebSocket closed: %s", exc)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# Global AI instances (lazy loaded)
_assistant = None
_speech = None

def _get_assistant():
    """Get or create CloudAssistant instance."""
    global _assistant
    if _assistant is None:
        try:
            from ai import CloudAssistant
            _assistant = CloudAssistant()
            LOGGER.info("CloudAssistant loaded for voice endpoint")
        except Exception as e:
            LOGGER.error(f"Failed to load CloudAssistant: {e}")
    return _assistant

def _get_speech():
    """Get or create SpeechProcessor instance."""
    global _speech
    if _speech is None:
        try:
            from speech import SpeechProcessor
            import sys
            from pathlib import Path
            # Import config
            sys.path.insert(0, str(Path(__file__).parent.parent))
            import config
            _speech = SpeechProcessor(
                whisper_model=config.WHISPER_MODEL,
                tts_engine=config.TTS_ENGINE,
                piper_voices=config.PIPER_VOICES,
            )
            LOGGER.info("SpeechProcessor loaded for voice endpoint")
        except Exception as e:
            LOGGER.error(f"Failed to load SpeechProcessor: {e}")
    return _speech


# Initialize meeting service (must be after helper functions are defined)
if MEETING_SERVICE_AVAILABLE:
    try:
        # Get speech processor and assistant for transcription and summarization
        speech_processor = _get_speech()
        assistant = _get_assistant()
        
        app.state.meeting_service = MeetingService(
            speech_processor=speech_processor,
            assistant=assistant
        )
        LOGGER.info("Meeting service initialized successfully")
    except Exception as exc:
        LOGGER.error("Failed to initialize meeting service: %s", exc, exc_info=True)
        app.state.meeting_service = None
else:
    app.state.meeting_service = None


@app.websocket("/voice")
async def voice_websocket(websocket: WebSocket):
    """WebSocket endpoint for voice interaction from mobile app.
    
    Receives: {"type": "audio_chunk", "encoding": "base64", "data": "..."}
              {"type": "audio_end", "encoding": "base64", "sampleRate": 16000}
              {"type": "text", "text": "..."}  # Direct text query
    
    Sends:    {"type": "status", "message": "..."}
              {"type": "chunk_received"}
              {"type": "audio_complete", "total_chunks": N}
              {"type": "transcript", "text": "..."}
              {"type": "response", "text": "..."}
    """
    await websocket.accept()
    LOGGER.info("Voice WebSocket connected from mobile app")
    
    audio_chunks = []
    
    try:
        await websocket.send_json({"type": "status", "message": "Rovy ready"})
        
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")
            
            if msg_type == "audio_chunk":
                # Collect audio chunks
                chunk_data = data.get("data", "")
                audio_chunks.append(chunk_data)
                await websocket.send_json({"type": "chunk_received"})
                
            elif msg_type == "audio_end":
                # Process complete audio
                total_chunks = len(audio_chunks)
                await websocket.send_json({
                    "type": "audio_complete",
                    "total_chunks": total_chunks
                })
                
                if total_chunks > 0:
                    # Combine all chunks
                    full_audio_b64 = "".join(audio_chunks)
                    audio_chunks = []  # Reset for next recording
                    
                    sample_rate = data.get("sampleRate", 16000)
                    
                    # Transcribe audio (English only)
                    speech = _get_speech()
                    if speech:
                        try:
                            audio_bytes = base64.b64decode(full_audio_b64)
                            audio_duration = len(audio_bytes) / (sample_rate * 2)  # 2 bytes per sample (16-bit)
                            LOGGER.info(f"🎤 Transcribing {len(audio_bytes)} bytes ({audio_duration:.2f}s) at {sample_rate}Hz")
                            
                            transcript = await anyio.to_thread.run_sync(
                                speech.transcribe, audio_bytes, sample_rate
                            )
                            
                            if transcript:
                                LOGGER.info(f"✅ Transcription successful: '{transcript}'")
                                await websocket.send_json({
                                    "type": "transcript",
                                    "text": transcript
                                })
                                
                                # Check for navigation commands first
                                transcript_lower = transcript.lower().strip()
                                
                                # Start navigation commands
                                if ('start' in transcript_lower or 'begin' in transcript_lower) and \
                                   ('auto' in transcript_lower or 'autonomous' in transcript_lower) and \
                                   'navigation' in transcript_lower:
                                    LOGGER.info(f"🤖 Auto navigation command detected: '{transcript}'")
                                    try:
                                        pi_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
                                        nav_url = f"http://{pi_ip}:8000/navigation"
                                        async with httpx.AsyncClient(timeout=5.0) as client:
                                            nav_response = await client.post(
                                                nav_url,
                                                json={"action": "start_explore", "duration": None}
                                            )
                                            if nav_response.status_code == 200:
                                                response_text = "Starting autonomous navigation. I will explore and avoid obstacles."
                                                await websocket.send_json({
                                                    "type": "response",
                                                    "text": response_text
                                                })
                                                # Send TTS
                                                pi_url = f"http://{pi_ip}:8000/speak"
                                                async with httpx.AsyncClient(timeout=10.0) as tts_client:
                                                    await tts_client.post(pi_url, json={"text": response_text})
                                                continue
                                    except Exception as nav_error:
                                        LOGGER.error(f"Navigation command failed: {nav_error}")
                                
                                # Start explore commands
                                if ('start' in transcript_lower or 'begin' in transcript_lower) and 'explor' in transcript_lower:
                                    LOGGER.info(f"🤖 Explore command detected: '{transcript}'")
                                    try:
                                        pi_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
                                        nav_url = f"http://{pi_ip}:8000/navigation"
                                        async with httpx.AsyncClient(timeout=5.0) as client:
                                            nav_response = await client.post(
                                                nav_url,
                                                json={"action": "start_explore", "duration": None}
                                            )
                                            if nav_response.status_code == 200:
                                                response_text = "Starting exploration mode."
                                                await websocket.send_json({
                                                    "type": "response",
                                                    "text": response_text
                                                })
                                                # Send TTS
                                                pi_url = f"http://{pi_ip}:8000/speak"
                                                async with httpx.AsyncClient(timeout=10.0) as tts_client:
                                                    await tts_client.post(pi_url, json={"text": response_text})
                                                continue
                                    except Exception as nav_error:
                                        LOGGER.error(f"Exploration command failed: {nav_error}")
                                
                                # Stop navigation commands
                                if ('stop' in transcript_lower or 'end' in transcript_lower) and \
                                   ('navigation' in transcript_lower or 'explor' in transcript_lower):
                                    LOGGER.info(f"🛑 Stop navigation command detected: '{transcript}'")
                                    try:
                                        pi_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
                                        nav_url = f"http://{pi_ip}:8000/navigation"
                                        async with httpx.AsyncClient(timeout=5.0) as client:
                                            nav_response = await client.post(
                                                nav_url,
                                                json={"action": "stop"}
                                            )
                                            if nav_response.status_code == 200:
                                                response_text = "Stopping navigation."
                                                await websocket.send_json({
                                                    "type": "response",
                                                    "text": response_text
                                                })
                                                # Send TTS
                                                pi_url = f"http://{pi_ip}:8000/speak"
                                                async with httpx.AsyncClient(timeout=10.0) as tts_client:
                                                    await tts_client.post(pi_url, json={"text": response_text})
                                                continue
                                    except Exception as nav_error:
                                        LOGGER.error(f"Stop navigation command failed: {nav_error}")
                                
                                # Dance commands
                                if 'dance' in transcript_lower or 'bust a move' in transcript_lower or 'show me your moves' in transcript_lower:
                                    LOGGER.info(f"💃 Dance command detected: '{transcript}'")
                                    try:
                                        # Extract dance style
                                        style = 'party'
                                        if 'wiggle' in transcript_lower:
                                            style = 'wiggle'
                                        elif 'spin' in transcript_lower:
                                            style = 'spin'
                                        
                                        # Extract music genre (default: dance)
                                        music_genre = 'dance'
                                        if 'classical' in transcript_lower:
                                            music_genre = 'classical'
                                        elif 'jazz' in transcript_lower:
                                            music_genre = 'jazz'
                                        elif 'rock' in transcript_lower:
                                            music_genre = 'rock'
                                        elif 'electronic' in transcript_lower or 'edm' in transcript_lower:
                                            music_genre = 'electronic'
                                        
                                        # Use the music endpoint to start music, then dance endpoint
                                        pi_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
                                        
                                        # Start music first using the working music endpoint (same as AI tool uses)
                                        music_url = f"http://{pi_ip}:8000/music/play"
                                        async with httpx.AsyncClient(timeout=10.0) as client:  # Increased timeout for music search
                                            music_response = await client.post(
                                                music_url,
                                                json={"query": music_genre}  # Old endpoint format
                                            )
                                            if music_response.status_code == 200:
                                                LOGGER.info(f"🎵 Started {music_genre} music on robot")
                                                # Wait for music to buffer and start playing before dancing
                                                await anyio.sleep(2.0)
                                            else:
                                                LOGGER.warning(f"Music start failed: {music_response.status_code}, dancing without music")
                                        
                                        # Now send dance command (without music flag since music is already playing)
                                        # Use very long duration (1 hour) - will be stopped by "stop dancing" command
                                        dance_url = f"http://{pi_ip}:8000/dance"
                                        async with httpx.AsyncClient(timeout=5.0) as client:
                                            dance_response = await client.post(
                                                dance_url,
                                                json={
                                                    "style": style, 
                                                    "duration": 3600,  # 1 hour - continues until "stop dancing" command
                                                    "with_music": False,  # Music already started separately
                                                    "music_genre": music_genre
                                                }
                                            )
                                            if dance_response.status_code == 200:
                                                response_text = f"Let me show you my {style} dance moves with {music_genre} music!"
                                                await websocket.send_json({
                                                    "type": "response",
                                                    "text": response_text
                                                })
                                                # Skip TTS for dance to avoid audio conflict with music
                                                LOGGER.info("Skipping TTS for dance command to allow music to play")
                                                
                                                continue
                                            else:
                                                LOGGER.warning(f"Dance request failed: {dance_response.status_code}")
                                    except Exception as dance_error:
                                        LOGGER.error(f"Dance command failed: {dance_error}")
                                
                                # Stop dancing command
                                if ('stop' in transcript_lower or 'end' in transcript_lower) and 'danc' in transcript_lower:
                                    LOGGER.info(f"🛑 Stop dancing command detected: '{transcript}'")
                                    try:
                                        pi_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
                                        
                                        # Stop dance
                                        dance_stop_url = f"http://{pi_ip}:8000/dance/stop"
                                        async with httpx.AsyncClient(timeout=5.0) as client:
                                            dance_stop_response = await client.post(dance_stop_url, json={})
                                            if dance_stop_response.status_code == 200:
                                                LOGGER.info("✅ Dance stopped")
                                            else:
                                                LOGGER.warning(f"Dance stop failed: {dance_stop_response.status_code}")
                                        
                                        # Stop music
                                        music_url = f"http://{pi_ip}:8000/music/stop"
                                        async with httpx.AsyncClient(timeout=5.0) as client:
                                            music_response = await client.post(music_url, json={})
                                            if music_response.status_code == 200:
                                                LOGGER.info("✅ Music stopped")
                                            else:
                                                LOGGER.warning(f"Music stop failed: {music_response.status_code}")
                                        
                                        response_text = "Stopping dance and music."
                                        await websocket.send_json({
                                            "type": "response",
                                            "text": response_text
                                        })
                                        continue
                                    except Exception as stop_error:
                                        LOGGER.error(f"Stop dancing command failed: {stop_error}")
                                
                                # Music commands
                                if 'play music' in transcript_lower or 'play some music' in transcript_lower:
                                    LOGGER.info(f"🎵 Music command detected: '{transcript}'")
                                    try:
                                        # Extract genre
                                        genre = 'fun'
                                        if 'classical' in transcript_lower:
                                            genre = 'classical'
                                        elif 'jazz' in transcript_lower:
                                            genre = 'jazz'
                                        elif 'rock' in transcript_lower:
                                            genre = 'rock'
                                        elif 'pop' in transcript_lower:
                                            genre = 'pop'
                                        elif 'dance' in transcript_lower or 'party' in transcript_lower:
                                            genre = 'dance'
                                        elif 'chill' in transcript_lower or 'relax' in transcript_lower:
                                            genre = 'chill'
                                        elif 'electronic' in transcript_lower or 'edm' in transcript_lower:
                                            genre = 'electronic'
                                        
                                        pi_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
                                        music_url = f"http://{pi_ip}:8000/music/play"
                                        async with httpx.AsyncClient(timeout=5.0) as client:
                                            music_response = await client.post(
                                                music_url,
                                                json={"query": genre}  # Old endpoint format
                                            )
                                            if music_response.status_code == 200:
                                                response_text = f"Playing {genre} music for you!"
                                                await websocket.send_json({
                                                    "type": "response",
                                                    "text": response_text
                                                })
                                                # Send TTS
                                                pi_url = f"http://{pi_ip}:8000/speak"
                                                async with httpx.AsyncClient(timeout=10.0) as tts_client:
                                                    await tts_client.post(pi_url, json={"text": response_text})
                                                continue
                                    except Exception as music_error:
                                        LOGGER.error(f"Music command failed: {music_error}")
                                
                                # Stop music commands
                                if ('stop' in transcript_lower or 'pause' in transcript_lower) and 'music' in transcript_lower:
                                    LOGGER.info(f"⏹️ Stop music command detected: '{transcript}'")
                                    try:
                                        pi_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
                                        music_url = f"http://{pi_ip}:8000/music/stop"
                                        async with httpx.AsyncClient(timeout=5.0) as client:
                                            music_response = await client.post(
                                                music_url,
                                                json={}  # Old endpoint format (no body needed)
                                            )
                                            if music_response.status_code == 200:
                                                response_text = "Stopping music."
                                                await websocket.send_json({
                                                    "type": "response",
                                                    "text": response_text
                                                })
                                                # Send TTS
                                                pi_url = f"http://{pi_ip}:8000/speak"
                                                async with httpx.AsyncClient(timeout=10.0) as tts_client:
                                                    await tts_client.post(pi_url, json={"text": response_text})
                                                continue
                                    except Exception as music_error:
                                        LOGGER.error(f"Stop music command failed: {music_error}")
                                
                                # Photo commands - take picture (uses existing /shot endpoint)
                                if ('take' in transcript_lower or 'capture' in transcript_lower) and ('picture' in transcript_lower or 'photo' in transcript_lower):
                                    LOGGER.info(f"📸 Take picture command detected: '{transcript}'")
                                    try:
                                        pi_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
                                        photo_url = f"http://{pi_ip}:8000/shot"
                                        
                                        # Inform user we're taking photo
                                        await websocket.send_json({
                                            "type": "response",
                                            "text": "Say cheese! Taking your photo now."
                                        })
                                        
                                        async with httpx.AsyncClient(timeout=10.0) as client:
                                            photo_response = await client.get(photo_url)
                                            if photo_response.status_code == 200:
                                                response_text = "Got it! Your photo is ready. You can view it in the Photo Time tab."
                                                await websocket.send_json({
                                                    "type": "response",
                                                    "text": response_text
                                                })
                                                # Send TTS
                                                pi_url = f"http://{pi_ip}:8000/speak"
                                                async with httpx.AsyncClient(timeout=10.0) as tts_client:
                                                    await tts_client.post(pi_url, json={"text": response_text})
                                                continue
                                            else:
                                                response_text = "Sorry, I couldn't capture the photo."
                                                await websocket.send_json({
                                                    "type": "response",
                                                    "text": response_text
                                                })
                                                continue
                                    except Exception as photo_error:
                                        LOGGER.error(f"Photo capture command failed: {photo_error}")
                                        response_text = "Sorry, something went wrong with the camera."
                                        await websocket.send_json({
                                            "type": "response",
                                            "text": response_text
                                        })
                                        continue
                                
                                # Vision commands - solve problems, read text, etc.
                                vision_solve_patterns = [
                                    'solve this', 'solve it', 'solve the problem', 'solve that',
                                    'what is the answer', 'help me solve'
                                ]
                                vision_read_patterns = [
                                    'read this', 'read it', 'read the text', 'read that',
                                    'what does it say', 'what does this say'
                                ]
                                
                                if any(pattern in transcript_lower for pattern in vision_solve_patterns):
                                    LOGGER.info(f"🔍 Vision solve command detected: '{transcript}'")
                                    try:
                                        # Import vision module
                                        import sys
                                        sys.path.insert(0, str(Path(__file__).parent.parent))
                                        from vision import VisionProcessor
                                        
                                        # Inform user we're capturing
                                        capture_msg = "Let me take a look and solve that for you."
                                        await websocket.send_json({
                                            "type": "response",
                                            "text": capture_msg
                                        })
                                        
                                        pi_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
                                        pi_url = f"http://{pi_ip}:8000/speak"
                                        async with httpx.AsyncClient(timeout=10.0) as tts_client:
                                            await tts_client.post(pi_url, json={"text": capture_msg})
                                        
                                        # Fetch image from robot's camera (OAK-D or USB on robot)
                                        LOGGER.info(f"📸 Fetching image from robot camera: {pi_ip}")
                                        shot_url = f"http://{pi_ip}:8000/shot"
                                        
                                        async with httpx.AsyncClient(timeout=15.0) as client:
                                            shot_response = await client.get(shot_url)
                                            
                                            if shot_response.status_code != 200:
                                                raise Exception(f"Failed to get image from robot: {shot_response.status_code}")
                                            
                                            image_bytes = shot_response.content
                                            LOGGER.info(f"✅ Got image from robot ({len(image_bytes)} bytes)")
                                        
                                        # Save captured image for debugging
                                        try:
                                            import datetime
                                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                            debug_image_path = f"vision_debug_solve_{timestamp}.jpg"
                                            with open(debug_image_path, "wb") as f:
                                                f.write(image_bytes)
                                            LOGGER.info(f"💾 Saved debug image: {debug_image_path}")
                                        except Exception as save_err:
                                            LOGGER.warning(f"Could not save debug image: {save_err}")
                                        
                                        # Analyze with Vision API (no camera init needed)
                                        processor = VisionProcessor()
                                        try:
                                            LOGGER.info("🔍 Analyzing with OpenAI Vision API...")
                                            solution = await processor.solve_problem(image_bytes=image_bytes)
                                            
                                            if solution:
                                                LOGGER.info(f"✅ Solution: {solution[:100]}...")
                                                await websocket.send_json({
                                                    "type": "response",
                                                    "text": solution
                                                })
                                                
                                                # Speak the solution
                                                async with httpx.AsyncClient(timeout=15.0) as tts_client:
                                                    await tts_client.post(pi_url, json={"text": solution})
                                            else:
                                                error_msg = "Sorry, I couldn't see the problem clearly. Can you hold it steady?"
                                                await websocket.send_json({
                                                    "type": "response",
                                                    "text": error_msg
                                                })
                                                async with httpx.AsyncClient(timeout=10.0) as tts_client:
                                                    await tts_client.post(pi_url, json={"text": error_msg})
                                        finally:
                                            await processor.close()
                                        
                                        continue
                                        
                                    except Exception as vision_error:
                                        LOGGER.error(f"Vision solve command failed: {vision_error}", exc_info=True)
                                        error_msg = "Sorry, I had trouble analyzing the image."
                                        await websocket.send_json({
                                            "type": "response",
                                            "text": error_msg
                                        })
                                        try:
                                            pi_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
                                            pi_url = f"http://{pi_ip}:8000/speak"
                                            async with httpx.AsyncClient(timeout=10.0) as tts_client:
                                                await tts_client.post(pi_url, json={"text": error_msg})
                                        except:
                                            pass
                                
                                elif any(pattern in transcript_lower for pattern in vision_read_patterns):
                                    LOGGER.info(f"📖 Vision read command detected: '{transcript}'")
                                    try:
                                        # Import vision module
                                        import sys
                                        sys.path.insert(0, str(Path(__file__).parent.parent))
                                        from vision import VisionProcessor
                                        
                                        # Inform user we're capturing
                                        capture_msg = "Let me read that for you."
                                        await websocket.send_json({
                                            "type": "response",
                                            "text": capture_msg
                                        })
                                        
                                        pi_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
                                        pi_url = f"http://{pi_ip}:8000/speak"
                                        async with httpx.AsyncClient(timeout=10.0) as tts_client:
                                            await tts_client.post(pi_url, json={"text": capture_msg})
                                        
                                        # Fetch image from robot's camera (OAK-D or USB on robot)
                                        LOGGER.info(f"📸 Fetching image from robot camera: {pi_ip}")
                                        shot_url = f"http://{pi_ip}:8000/shot"
                                        
                                        async with httpx.AsyncClient(timeout=15.0) as client:
                                            shot_response = await client.get(shot_url)
                                            
                                            if shot_response.status_code != 200:
                                                raise Exception(f"Failed to get image from robot: {shot_response.status_code}")
                                            
                                            image_bytes = shot_response.content
                                            LOGGER.info(f"✅ Got image from robot ({len(image_bytes)} bytes)")
                                        
                                        # Save captured image for debugging
                                        try:
                                            import datetime
                                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                            debug_image_path = f"vision_debug_read_{timestamp}.jpg"
                                            with open(debug_image_path, "wb") as f:
                                                f.write(image_bytes)
                                            LOGGER.info(f"💾 Saved debug image: {debug_image_path}")
                                        except Exception as save_err:
                                            LOGGER.warning(f"Could not save debug image: {save_err}")
                                        
                                        # Analyze with Vision API (no camera init needed)
                                        processor = VisionProcessor()
                                        try:
                                            LOGGER.info("🔍 Reading text with OpenAI Vision API...")
                                            text = await processor.read_text(image_bytes=image_bytes)
                                            
                                            if text:
                                                LOGGER.info(f"✅ Text read: {text[:100]}...")
                                                await websocket.send_json({
                                                    "type": "response",
                                                    "text": text
                                                })
                                                
                                                # Speak the text
                                                async with httpx.AsyncClient(timeout=15.0) as tts_client:
                                                    await tts_client.post(pi_url, json={"text": text})
                                            else:
                                                error_msg = "Sorry, I couldn't read any text. Can you hold it closer?"
                                                await websocket.send_json({
                                                    "type": "response",
                                                    "text": error_msg
                                                })
                                                async with httpx.AsyncClient(timeout=10.0) as tts_client:
                                                    await tts_client.post(pi_url, json={"text": error_msg})
                                        finally:
                                            await processor.close()
                                        
                                        continue
                                        
                                    except Exception as vision_error:
                                        LOGGER.error(f"Vision read command failed: {vision_error}", exc_info=True)
                                        error_msg = "Sorry, I had trouble reading the text."
                                        await websocket.send_json({
                                            "type": "response",
                                            "text": error_msg
                                        })
                                        try:
                                            pi_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
                                            pi_url = f"http://{pi_ip}:8000/speak"
                                            async with httpx.AsyncClient(timeout=10.0) as tts_client:
                                                await tts_client.post(pi_url, json={"text": error_msg})
                                        except:
                                            pass
                                
                                # Get AI response (with vision if asking about camera/image)
                                assistant = _get_assistant()
                                if assistant:
                                    # Use keyword matching to determine if vision is needed
                                    transcript_lower = transcript.lower()
                                    
                                    # Vision keywords - only trigger for explicit visual queries
                                    vision_patterns = [
                                        r'\bwhat\s+(?:do\s+you\s+)?see\b',
                                        r'\bwhat\s+(?:can\s+you\s+)?see\b',
                                        r'\bcan\s+you\s+see\b',
                                        r'\blook\s+at\b',
                                        r'\bdescribe\s+what\b',
                                        r'\bshow\s+me\b'
                                    ]
                                    
                                    use_vision = any(re.search(pattern, transcript_lower) for pattern in vision_patterns)
                                    
                                    if use_vision:
                                        LOGGER.info("Vision query detected via keywords")
                                    
                                    if use_vision:
                                        # Fetch latest frame from Pi camera
                                        try:
                                            pi_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
                                            frame_url = f"http://{pi_ip}:8000/shot"
                                            LOGGER.info(f"📷 Fetching frame from Pi: {frame_url}")
                                            
                                            async with httpx.AsyncClient(timeout=15.0) as client:
                                                frame_response = await client.get(frame_url)
                                                if frame_response.status_code == 200:
                                                    image_bytes = frame_response.content
                                                    response = await anyio.to_thread.run_sync(
                                                        assistant.ask_with_vision, transcript, image_bytes
                                                    )
                                                else:
                                                    LOGGER.warning(f"Failed to get frame: {frame_response.status_code}")
                                                    response = await anyio.to_thread.run_sync(
                                                        assistant.ask, transcript
                                                    )
                                        except Exception as frame_error:
                                            LOGGER.error(f"Error fetching frame: {frame_error}", exc_info=True)
                                            response = await anyio.to_thread.run_sync(
                                                assistant.ask, transcript
                                            )
                                    else:
                                        response = await anyio.to_thread.run_sync(
                                            assistant.ask, transcript
                                        )
                                    
                                    LOGGER.info(f"🤖 AI Response: {response}")
                                    
                                    await websocket.send_json({
                                        "type": "response",
                                        "text": response
                                    })
                                    
                                    # Send TTS command to Pi speakers via HTTP with language support
                                    try:
                                        # Get target language from translation tool if used
                                        response_language = "en"
                                        if hasattr(assistant, 'get_response_language'):
                                            response_language = assistant.get_response_language()
                                        
                                        # Get Pi IP from environment or use default
                                        pi_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
                                        pi_url = f"http://{pi_ip}:8000/speak"
                                        
                                        if response_language != "en":
                                            LOGGER.info(f"🔊 Sending TTS to Pi at {pi_url} (language: {response_language})...")
                                        else:
                                            LOGGER.info(f"🔊 Sending TTS to Pi at {pi_url}...")
                                        
                                        async with httpx.AsyncClient(timeout=10.0) as client:
                                            pi_response = await client.post(
                                                pi_url, 
                                                json={"text": response, "language": response_language}
                                            )
                                            if pi_response.status_code == 200:
                                                LOGGER.info("✅ TTS played on Pi speakers")
                                            else:
                                                LOGGER.warning(f"Pi TTS returned status {pi_response.status_code}")
                                    except Exception as tts_error:
                                        LOGGER.error(f"Failed to send TTS to Pi: {tts_error}")
                                else:
                                    await websocket.send_json({
                                        "type": "response",
                                        "text": "AI assistant not available"
                                    })
                            else:
                                LOGGER.warning(f"⚠️ Transcription returned empty result (audio: {len(audio_bytes)} bytes, {audio_duration:.2f}s)")
                                await websocket.send_json({
                                    "type": "status",
                                    "message": "Could not transcribe audio"
                                })
                        except Exception as e:
                            LOGGER.error(f"Audio processing error: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "message": str(e)
                            })
                    else:
                        await websocket.send_json({
                            "type": "status",
                            "message": "Speech processor not available"
                        })
                else:
                    await websocket.send_json({
                        "type": "status",
                        "message": "No audio data received"
                    })
            
            elif msg_type == "text":
                # Direct text query (no audio)
                text = data.get("text", "")
                if text:
                    assistant = _get_assistant()
                    if assistant:
                        response = await anyio.to_thread.run_sync(
                            assistant.ask, text
                        )
                        await websocket.send_json({
                            "type": "response",
                            "text": response
                        })
                    else:
                        await websocket.send_json({
                            "type": "response",
                            "text": "AI assistant not available"
                        })
                    
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                
    except Exception as exc:
        LOGGER.debug("Voice WebSocket closed: %s", exc)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ============================================================================
# AI Chat/Vision REST Endpoints (for mobile app cloud-api.ts)
# ============================================================================

from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 150
    temperature: float = 0.3

class ChatResponse(BaseModel):
    response: str
    movement: dict | None = None

class VisionRequest(BaseModel):
    question: str
    image_base64: str
    max_tokens: int = 200

class VisionResponse(BaseModel):
    response: str
    movement: dict | None = None


@app.post("/chat", response_model=ChatResponse, tags=["AI"])
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Chat with Rovy AI (text only)."""
    assistant = _get_assistant()
    if not assistant:
        raise HTTPException(status_code=503, detail="AI assistant not available")
    
    try:
        response = await anyio.to_thread.run_sync(
            assistant.ask, request.message, request.max_tokens, request.temperature
        )
        
        # Check for movement commands
        movement = None
        if hasattr(assistant, 'extract_movement'):
            movement = assistant.extract_movement(response, request.message)
        
        # Broadcast response to robot for TTS playback on Pi speakers
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from main import broadcast_to_robot
            await broadcast_to_robot(response)
        except Exception as e:
            LOGGER.warning(f"Could not broadcast to robot: {e}")
        
        return ChatResponse(response=response, movement=movement)
    except Exception as e:
        LOGGER.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vision", response_model=VisionResponse, tags=["AI"])
async def vision_endpoint(request: VisionRequest) -> VisionResponse:
    """Ask Rovy about an image."""
    assistant = _get_assistant()
    if not assistant:
        raise HTTPException(status_code=503, detail="AI assistant not available")
    
    try:
        # Decode image
        image_bytes = base64.b64decode(request.image_base64)
        
        response = await anyio.to_thread.run_sync(
            assistant.ask_with_vision, request.question, image_bytes, request.max_tokens
        )
        
        # Check for movement commands
        movement = None
        if hasattr(assistant, 'extract_movement'):
            movement = assistant.extract_movement(response, request.question)
        
        return VisionResponse(response=response, movement=movement)
    except Exception as e:
        LOGGER.error(f"Vision error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vision/capture-and-analyze", tags=["AI"])
async def capture_and_analyze(
    question: str = Form("What do you see? If there's a problem or question, solve it."),
    save_image: bool = Form(False)
):
    """Capture image from camera and analyze with OpenAI Vision API.
    
    Similar to ChatGPT's camera mode - show a paper with problem and get answer.
    
    Args:
        question: Question to ask about the image
        save_image: Whether to save the captured image
    
    Returns:
        {
            "response": "AI analysis/answer",
            "success": true,
            "image_base64": "..." (if save_image=true)
        }
    
    Example use cases:
        - Show paper with "2+2=?" → Get answer: "4"
        - Show any text → Get it read/transcribed
        - Show diagram → Get it explained
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from vision import VisionProcessor
        
        processor = VisionProcessor()
        
        try:
            # Capture image
            LOGGER.info("Capturing image from camera...")
            image_bytes = await processor.capture_image()
            
            if not image_bytes:
                raise HTTPException(status_code=500, detail="Failed to capture image from camera")
            
            # Analyze with Vision API
            LOGGER.info(f"Analyzing with question: {question}")
            response = await processor.analyze_image(
                image_bytes=image_bytes,
                question=question,
                max_tokens=800
            )
            
            if not response:
                raise HTTPException(status_code=500, detail="Vision API returned no response")
            
            result = {
                "response": response,
                "success": True
            }
            
            # Optionally include the image
            if save_image:
                result["image_base64"] = base64.b64encode(image_bytes).decode('utf-8')
            
            return result
            
        finally:
            await processor.close()
            
    except HTTPException:
        raise
    except ImportError as e:
        LOGGER.error(f"Vision module import failed: {e}")
        raise HTTPException(status_code=503, detail="Vision processor not available")
    except Exception as e:
        LOGGER.error(f"Vision capture/analyze error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vision/solve-problem", tags=["AI"])
async def solve_problem_from_camera(save_image: bool = Form(False)):
    """Capture image and solve any problem shown (optimized for math/text problems).
    
    This is a convenience endpoint optimized for problem-solving.
    Show a paper with a problem and get the solution.
    
    Args:
        save_image: Whether to return the captured image
    
    Returns:
        {
            "solution": "Step-by-step solution",
            "success": true,
            "image_base64": "..." (if save_image=true)
        }
    
    Example:
        - Show "2+2=?" → Get "The answer is 4"
        - Show "5×3=?" → Get "5 multiplied by 3 equals 15"
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from vision import VisionProcessor
        
        processor = VisionProcessor()
        
        try:
            # Capture image
            LOGGER.info("Capturing image for problem solving...")
            image_bytes = await processor.capture_image()
            
            if not image_bytes:
                raise HTTPException(status_code=500, detail="Failed to capture image")
            
            # Solve problem
            LOGGER.info("Solving problem...")
            solution = await processor.solve_problem(image_bytes)
            
            if not solution:
                raise HTTPException(status_code=500, detail="Could not solve problem")
            
            result = {
                "solution": solution,
                "success": True
            }
            
            if save_image:
                result["image_base64"] = base64.b64encode(image_bytes).decode('utf-8')
            
            return result
            
        finally:
            await processor.close()
            
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Problem solving error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vision/read-text", tags=["AI"])
async def read_text_from_camera(save_image: bool = Form(False)):
    """Capture image and read/extract any text (OCR).
    
    Args:
        save_image: Whether to return the captured image
    
    Returns:
        {
            "text": "Extracted text",
            "success": true,
            "image_base64": "..." (if save_image=true)
        }
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from vision import VisionProcessor
        
        processor = VisionProcessor()
        
        try:
            # Capture and read text
            LOGGER.info("Capturing image for text reading...")
            image_bytes = await processor.capture_image()
            
            if not image_bytes:
                raise HTTPException(status_code=500, detail="Failed to capture image")
            
            LOGGER.info("Reading text...")
            text = await processor.read_text(image_bytes)
            
            if not text:
                raise HTTPException(status_code=500, detail="No text found or read failed")
            
            result = {
                "text": text,
                "success": True
            }
            
            if save_image:
                result["image_base64"] = base64.b64encode(image_bytes).decode('utf-8')
            
            return result
            
        finally:
            await processor.close()
            
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Text reading error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stt", tags=["AI"])
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert speech to text using Whisper (English only)."""
    speech = _get_speech()
    if not speech:
        raise HTTPException(status_code=503, detail="Speech processor not available")
    
    try:
        audio_bytes = await audio.read()
        transcript = await anyio.to_thread.run_sync(
            speech.transcribe, audio_bytes, 16000
        )
        
        if transcript:
            return {
                "text": transcript,
                "success": True
            }
        else:
            return {"text": None, "success": False}
    except Exception as e:
        LOGGER.error(f"STT error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts", tags=["AI"])
async def text_to_speech(request: dict):
    """
    Convert text to speech with language support.
    
    Request body:
        text: Text to synthesize (required)
        language: ISO language code (optional, default: 'en')
    """
    speech = _get_speech()
    if not speech:
        raise HTTPException(status_code=503, detail="Speech processor not available")
    
    text = request.get("text", "")
    language = request.get("language", "en")
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        audio_bytes = await anyio.to_thread.run_sync(speech.synthesize, text, language)
        if audio_bytes:
            return Response(content=audio_bytes, media_type="audio/wav")
        else:
            raise HTTPException(status_code=500, detail="TTS failed")
    except Exception as e:
        LOGGER.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Piper TTS for local speech on Pi
_piper_voice = None
_tts_lock = asyncio.Lock()  # Prevent concurrent audio playback
PIPER_MODEL_PATH = "/home/rovy/rovy_client/models/piper/en_US-hfc_male-medium.onnx"

def _get_audio_device():
    """Auto-detect USB audio device."""
    try:
        result = subprocess.run(['aplay', '-l'], capture_output=True, text=True, timeout=2)
        for line in result.stdout.split('\n'):
            if 'UACDemo' in line or 'USB Audio' in line:
                # Extract card number from line like "card 3: UACDemoV10..."
                parts = line.split(':')
                if len(parts) >= 1 and 'card' in parts[0]:
                    card_num = parts[0].split('card')[1].strip()
                    device = f"plughw:{card_num},0"
                    LOGGER.info(f"🔊 Auto-detected USB audio device: {device}")
                    return device
    except Exception as e:
        LOGGER.warning(f"Failed to auto-detect audio device: {e}")
    
    # Fallback to default
    return "plughw:2,0"

AUDIO_DEVICE = _get_audio_device()


def _get_piper_voice():
    """Lazy load Piper voice model."""
    global _piper_voice
    if _piper_voice is None:
        try:
            from piper import PiperVoice
            if os.path.exists(PIPER_MODEL_PATH):
                _piper_voice = PiperVoice.load(PIPER_MODEL_PATH)
                LOGGER.info("Piper TTS loaded successfully")
            else:
                LOGGER.warning(f"Piper model not found: {PIPER_MODEL_PATH}")
        except ImportError:
            LOGGER.warning("Piper not installed")
        except Exception as e:
            LOGGER.error(f"Failed to load Piper: {e}")
    return _piper_voice


def _speak_with_piper(text: str) -> bool:
    """Synthesize and play speech through speakers with dynamic device detection."""
    import wave
    import subprocess
    import tempfile
    
    voice = _get_piper_voice()
    if not voice:
        return False
    
    try:
        # Clean text for TTS (remove special characters that might break Piper)
        clean_text = text.strip()
        # Remove markdown formatting, asterisks, etc
        clean_text = clean_text.replace('*', '').replace('#', '').replace('_', '')
        # Remove parentheses and brackets
        clean_text = clean_text.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        
        LOGGER.debug(f"TTS cleaned text (first 100 chars): {clean_text[:100]}")
        
        # Synthesize audio
        audio_bytes = b''
        sample_rate = 22050
        for chunk in voice.synthesize(clean_text):
            audio_bytes += chunk.audio_int16_bytes
            sample_rate = chunk.sample_rate
        
        # Verify audio was generated
        if not audio_bytes or len(audio_bytes) < 100:
            LOGGER.error(f"Piper generated insufficient audio: {len(audio_bytes)} bytes")
            return False
        
        LOGGER.info(f"Generated {len(audio_bytes)} bytes of audio at {sample_rate}Hz")
        
        # Save to temp WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_path = f.name
            with wave.open(f.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)
        
        # Dynamically detect audio device and try multiple fallback options
        audio_devices = []
        
        # Try to detect USB audio device dynamically
        try:
            result = subprocess.run(['aplay', '-l'], capture_output=True, text=True, timeout=2)
            for line in result.stdout.split('\n'):
                if 'UACDemo' in line or 'USB Audio' in line:
                    parts = line.split(':')
                    if len(parts) >= 1 and 'card' in parts[0]:
                        card_num = parts[0].split('card')[1].strip()
                        audio_devices.append(f"plughw:{card_num},0")
        except Exception as e:
            LOGGER.debug(f"Failed to detect audio device: {e}")
        
        # Add fallback devices
        audio_devices.extend([
            "plughw:3,0",  # Common USB audio card
            "plughw:2,0",
            "default",     # System default
            "plughw:0,0",  # Built-in audio
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        audio_devices = [x for x in audio_devices if not (x in seen or seen.add(x))]
        
        # Try each device until one works
        for device in audio_devices:
            try:
                result = subprocess.run(
                    ['aplay', '-D', device, wav_path],
                    capture_output=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    LOGGER.info(f"✅ Audio played successfully on {device}")
                    os.unlink(wav_path)
                    return True
                else:
                    LOGGER.debug(f"Device {device} failed: {result.stderr.decode()}")
            except Exception as e:
                LOGGER.debug(f"Device {device} error: {e}")
                continue
        
        # All devices failed
        LOGGER.error(f"All audio devices failed. Tried: {audio_devices}")
        os.unlink(wav_path)
        return False
        
    except Exception as e:
        LOGGER.error(f"Piper speak error: {e}")
        return False


@app.post("/speak", tags=["Speech"])
async def speak_text(request: dict):
    """Speak text through the robot's speakers using Piper TTS.
    
    This endpoint is for the Pi to speak responses from the cloud AI.
    Uses a lock to prevent concurrent audio playback (ALSA limitation).
    """
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    LOGGER.info(f"Speaking: {text[:50]}...")
    
    # Use lock to prevent concurrent audio playback
    async with _tts_lock:
        success = await anyio.to_thread.run_sync(_speak_with_piper, text)
    
    if success:
        return {"status": "ok", "message": "Speech played"}
    else:
        raise HTTPException(status_code=503, detail="TTS not available")


@app.post("/dance", tags=["Control"])
async def trigger_dance(request: dict):
    """Trigger a dance routine on the robot.
    
    Request body:
        style: Dance style - 'party', 'wiggle', or 'spin' (optional, default: 'party')
        duration: Duration in seconds (optional, default: 10)
        with_music: Play music during dance (optional, default: False)
        music_genre: Music genre if with_music is True (optional, default: 'dance')
    
    Can be called from:
    - Mobile app directly
    - Cloud server via voice commands
    """
    style = request.get("style", "party")
    duration = request.get("duration", 10)
    with_music = request.get("with_music", False)
    music_genre = request.get("music_genre", "dance")
    
    # Validate style
    valid_styles = ['party', 'wiggle', 'spin']
    if style not in valid_styles:
        raise HTTPException(status_code=400, detail=f"Invalid style. Must be one of: {valid_styles}")
    
    # Validate duration
    if not isinstance(duration, (int, float)) or duration <= 0 or duration > 60:
        raise HTTPException(status_code=400, detail="Duration must be between 0 and 60 seconds")
    
    LOGGER.info(f"💃 Dance triggered: {style} for {duration}s" + (f" with {music_genre} music" if with_music else ""))
    
    # Get the rover controller instance
    base_controller = _get_base_controller()
    
    if not base_controller:
        raise HTTPException(status_code=503, detail="Rover controller not available")
    
    if not hasattr(base_controller, "dance"):
        raise HTTPException(status_code=501, detail="Dance function not supported by this rover")
    
    try:
        # If music requested, try to play it locally
        music_player = None
        if with_music:
            try:
                from robot.music_player import get_music_player
                music_player = get_music_player()
                if music_player and music_player.yt_music:
                    LOGGER.info(f"🎵 Starting {music_genre} music")
                    await anyio.to_thread.run_sync(music_player.play_random, music_genre)
                    await anyio.sleep(2)  # Let music start
                else:
                    LOGGER.info("YouTube Music not configured, dancing without music")
            except ImportError as import_error:
                LOGGER.info(f"Music player module not available: {import_error}, dancing without music")
            except Exception as music_error:
                LOGGER.warning(f"Music failed, dancing without: {music_error}")
        
        # Run dance in background thread to not block the response
        await anyio.to_thread.run_sync(base_controller.dance, style, duration)
        
        # Stop music after dance if it was started
        if with_music and music_player and music_player.is_playing:
            try:
                music_player.stop()
                LOGGER.info("🎵 Music stopped after dance")
            except Exception as stop_error:
                LOGGER.warning(f"Failed to stop music: {stop_error}")
        
        return {
            "status": "ok",
            "message": f"Dancing {style} style for {duration} seconds" + (" with music" if with_music else ""),
            "style": style,
            "duration": duration,
            "with_music": with_music,
            "music_genre": music_genre if with_music else None
        }
    except Exception as exc:
        LOGGER.error(f"Dance execution failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dance failed: {str(exc)}")


@app.post("/music", tags=["Control"])
async def control_music(request: dict):
    """Control music playback on the robot.
    
    Request body:
        action: 'play' or 'stop' (required)
        genre: Music genre for 'play' action (optional, default: 'dance')
                Options: 'dance', 'party', 'classical', 'jazz', 'rock', 'pop', 'chill', 'electronic', 'fun'
    
    Returns:
        Status and currently playing song info (if applicable)
    """
    action = request.get("action", "play")
    genre = request.get("genre", "dance")
    
    # Validate action
    if action not in ['play', 'stop', 'status']:
        raise HTTPException(status_code=400, detail="Action must be 'play', 'stop', or 'status'")
    
    # Validate genre
    valid_genres = ['dance', 'party', 'classical', 'jazz', 'rock', 'pop', 'chill', 'electronic', 'fun']
    if action == 'play' and genre not in valid_genres:
        raise HTTPException(status_code=400, detail=f"Invalid genre. Must be one of: {valid_genres}")
    
    try:
        # Try to play music on cloud server (where YouTube Music is authenticated)
        from robot.music_player import get_music_player
        music_player = get_music_player()
        
        if not music_player or not music_player.yt_music:
            # If cloud doesn't have YouTube Music, forward to robot
            LOGGER.info("Cloud music not available, forwarding to robot")
            import httpx
            pi_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
            music_url = f"http://{pi_ip}:8000/music"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                music_response = await client.post(
                    music_url,
                    json={"action": action, "genre": genre}
                )
                
                if music_response.status_code == 200:
                    result = music_response.json()
                    if action == 'play':
                        LOGGER.info(f"🎵 Playing {genre} music via robot")
                    elif action == 'stop':
                        LOGGER.info("⏹️ Stopping music via robot")
                    return result
                else:
                    error_detail = music_response.text
                    raise HTTPException(
                        status_code=music_response.status_code,
                        detail=f"Music control failed: {error_detail}"
                    )
        
        # Play music on cloud server
        if action == 'play':
            LOGGER.info(f"🎵 Playing {genre} music on cloud server")
            success = await anyio.to_thread.run_sync(music_player.play_random, genre)
            
            if success:
                return {
                    "status": "ok",
                    "action": "playing",
                    "genre": genre,
                    "current_song": music_player.current_song,
                    "location": "cloud"
                }
            else:
                raise HTTPException(status_code=404, detail=f"No {genre} songs found")
        
        elif action == 'stop':
            LOGGER.info("⏹️ Stopping music on cloud server")
            music_player.stop()
            return {
                "status": "ok",
                "action": "stopped",
                "location": "cloud"
            }
        
        elif action == 'status':
            status = music_player.get_status()
            return {
                "status": "ok",
                "is_playing": status['is_playing'],
                "current_song": status['current_song'],
                "location": "cloud"
            }
    
    except ImportError:
        raise HTTPException(status_code=503, detail="Music player module not available")
    except httpx.RequestError as exc:
        LOGGER.error(f"Failed to connect to robot: {exc}", exc_info=True)
        raise HTTPException(status_code=503, detail="Cannot connect to robot")
    except Exception as exc:
        LOGGER.error(f"Music control failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Music control failed: {str(exc)}")


@app.post("/gesture/detect", tags=["AI"])
async def detect_gesture(file: UploadFile = File(...)):
    """Detect hand gesture from image (like, heart, etc.)"""
    if not GESTURE_DETECTION_AVAILABLE or not detect_gesture_from_image:
        raise HTTPException(status_code=503, detail="Gesture detection not available")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Detect gesture
        gesture, confidence = await anyio.to_thread.run_sync(
            detect_gesture_from_image, image_bytes
        )
        
        return {
            "gesture": gesture,
            "confidence": float(confidence),
            "status": "ok"
        }
    except Exception as e:
        LOGGER.error(f"Gesture detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Gesture detection failed: {str(e)}")


def _start_oakd_pipeline():
    """Start OAK-D pipeline on-demand."""
    import depthai as dai
    import time
    
    # Close existing device if any
    if app.state.oakd_device is not None:
        try:
            app.state.oakd_device.close()
            time.sleep(0.5)  # Wait for device to be fully released
        except:
            pass
        app.state.oakd_device = None
        app.state.oakd_queue = None
    
    # Force release any stale devices
    DepthAICameraSource.force_release_devices()
    time.sleep(0.8)  # Increased wait time to ensure device is fully released and ready
    
    # Create pipeline - use MJPEG encoding for less USB bandwidth
    pipeline = dai.Pipeline()
    
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setIspScale(1, 3)  # Downscale to 640x360
    cam_rgb.setFps(10)  # Low FPS to minimize USB bandwidth
    
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("preview")
    xout_rgb.setMetadataOnly(False)
    cam_rgb.preview.link(xout_rgb.input)
    
    # Try to create device with retry logic in case device is not fully ready
    max_retries = 3
    for retry in range(max_retries):
        try:
            app.state.oakd_device = dai.Device(pipeline)
            app.state.oakd_queue = app.state.oakd_device.getOutputQueue(name="preview", maxSize=1, blocking=False)
            app.state.oakd_last_used = time.time()
            LOGGER.info("✅ OAK-D pipeline started on-demand")
            return
        except Exception as e:
            if retry < max_retries - 1:
                LOGGER.warning(f"Failed to start OAK-D pipeline (attempt {retry + 1}/{max_retries}): {e}")
                time.sleep(0.5)  # Wait before retry
                # Force release again before retry
                DepthAICameraSource.force_release_devices()
                time.sleep(0.3)
            else:
                LOGGER.error(f"Failed to start OAK-D pipeline after {max_retries} attempts: {e}")
                raise


def _stop_oakd_pipeline():
    """Stop OAK-D pipeline to free USB bandwidth."""
    if app.state.oakd_device is not None:
        try:
            app.state.oakd_device.close()
            LOGGER.info("🛑 OAK-D pipeline stopped (freeing USB bandwidth)")
        except:
            pass
        app.state.oakd_device = None
        app.state.oakd_queue = None
        # Wait a bit to ensure device is fully released before next use
        import time
        time.sleep(0.5)


def _ensure_oakd_pipeline():
    """Ensure OAK-D pipeline is running, start if needed."""
    import time
    
    # Check if pipeline exists and is healthy
    if app.state.oakd_queue is not None and app.state.oakd_device is not None:
        try:
            # Pipeline already running - just update timestamp
            app.state.oakd_last_used = time.time()
            return
        except:
            # Device might be in bad state, reset it
            LOGGER.warning("OAK-D device appears to be in bad state, reinitializing...")
            app.state.oakd_device = None
            app.state.oakd_queue = None
    
    # Start pipeline on-demand
    LOGGER.info("📷 Starting OAK-D pipeline on-demand...")
    _start_oakd_pipeline()


async def _capture_oakd_snapshot() -> bytes:
    """Capture a single snapshot from OAK-D camera (on-demand) with auto-recovery."""
    import cv2
    import time as time_module
    
    if not app.state.oakd_available:
        raise CameraError("OAK-D camera not available")
    
    # Start/stop pipeline WITHOUT holding lock to avoid freezing webcam
    # Only hold lock for brief periods during actual frame capture
    def get_frame_sync():
        # Start pipeline without lock (this takes several seconds)
        LOGGER.debug("📷 Starting OAK-D (webcam continues in background)...")
        _ensure_oakd_pipeline()
        
        if app.state.oakd_queue is None:
            raise CameraError("Failed to start OAK-D pipeline")
        
        # Wait for pipeline to produce first frames (without holding lock)
        LOGGER.debug("⏳ Waiting for OAK-D to warm up (webcam continues)...")
        time_module.sleep(0.8)
        
        # Try to get the latest frame
        max_attempts = 15
        frame_data = None
        
        for attempt in range(max_attempts):
            try:
                # Use tryGet() which is non-blocking
                frame_data = app.state.oakd_queue.tryGet()
                if frame_data is not None:
                    # Convert to OpenCV frame
                    frame = frame_data.getCvFrame()
                    if frame is None or frame.size == 0:
                        LOGGER.warning(f"Empty frame from OAK-D (attempt {attempt+1}/{max_attempts})")
                        frame_data = None
                        time_module.sleep(0.05)
                        continue
                    
                    # Encode to JPEG
                    success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not success:
                        raise CameraError("Failed to encode frame as JPEG")
                    
                    LOGGER.info(f"✅ Captured OAK-D snapshot ({len(encoded.tobytes())} bytes) on attempt {attempt+1}")
                    
                    # Stop pipeline after successful capture to free USB bandwidth for webcam
                    LOGGER.debug("🛑 Stopping OAK-D (webcam resumes full bandwidth)...")
                    _stop_oakd_pipeline()
                    
                    return encoded.tobytes()
                else:
                    # No frame available yet, wait a short time
                    LOGGER.debug(f"No frame available from OAK-D queue (attempt {attempt+1}/{max_attempts})")
                    time_module.sleep(0.1)
                    
            except Exception as e:
                error_msg = str(e)
                # Check if it's an X_LINK error - pipeline might be corrupted
                if "X_LINK_ERROR" in error_msg or "Communication exception" in error_msg or "XLinkError" in error_msg:
                    LOGGER.warning(f"OAK-D pipeline corrupted: {error_msg}")
                    try:
                        _start_oakd_pipeline()
                        # Give new pipeline time to start producing frames
                        time_module.sleep(1.0)
                        # Retry with new pipeline
                        continue
                    except Exception as reinit_error:
                        raise CameraError(f"Failed to restart OAK-D: {reinit_error}")
                
                # For other errors, log and retry
                LOGGER.warning(f"Error getting OAK-D frame (attempt {attempt+1}/{max_attempts}): {error_msg}")
                if attempt == max_attempts - 1:
                    # Stop pipeline before raising error
                    _stop_oakd_pipeline()
                    raise CameraError(f"Failed to get frame from OAK-D after {max_attempts} attempts: {error_msg}")
                
                time_module.sleep(0.05)
        
        # Stop pipeline before raising error
        _stop_oakd_pipeline()
        raise CameraError(f"No frame available from OAK-D after {max_attempts} attempts. Device not producing frames.")
    
    # Run without USB lock - let both cameras operate simultaneously
    # They share USB bandwidth but don't block each other
    return await anyio.to_thread.run_sync(get_frame_sync)



@app.get("/shot")
async def single_frame() -> Response:
    """Serve a single JPEG frame from OAK-D camera for AI vision."""
    try:
        # Capture frame using temporary OAK-D instance
        frame = await _capture_oakd_snapshot()
        return Response(content=frame, media_type="image/jpeg")
    except CameraError as e:
        LOGGER.error(f"Failed to get OAK-D frame: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        LOGGER.error(f"Unexpected error getting OAK-D frame: {e}")
        raise HTTPException(status_code=503, detail=str(e))


async def _camera_stream(service: CameraService, frames: int | None) -> AsyncIterator[bytes]:
    emitted = 0
    while frames is None or emitted < frames:
        frame = await service.get_frame()
        header = (
            f"--{service.boundary}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(frame)}\r\n\r\n"
        ).encode()
        yield header + frame + b"\r\n"
        emitted += 1
        if service.frame_delay:
            await anyio.sleep(service.frame_delay)


@app.on_event("shutdown")
async def shutdown_camera() -> None:
    await app.state.camera_service.close()
    oak_shutdown()


@app.get("/health", response_model=HealthResponse, tags=["Discovery"])
async def get_health() -> HealthResponse:
    # Try to get battery information
    battery_percent = None
    base_controller = _get_base_controller()
    
    if base_controller and hasattr(base_controller, "get_status"):
        try:
            rover_status = await anyio.to_thread.run_sync(base_controller.get_status)
            battery_percent = _voltage_to_percentage(rover_status.get("voltage"))
        except Exception as exc:
            LOGGER.debug("Failed to get battery for health check: %s", exc)
    
    # Check cloud service availability
    assistant_available = _get_assistant() is not None
    speech_available = _get_speech() is not None
    meeting_service_available = (
        MEETING_SERVICE_AVAILABLE and 
        hasattr(app.state, 'meeting_service') and 
        app.state.meeting_service is not None
    )
    
    return HealthResponse(
        ok=True,
        name=ROBOT_NAME,
        serial=ROBOT_SERIAL,
        claimed=STATE["claimed"],
        mode=Mode.ACCESS_POINT,
        version=APP_VERSION,
        battery=battery_percent,
        assistant_loaded=assistant_available,
        speech_loaded=speech_available,
        meeting_service_available=meeting_service_available,
    )


@app.get("/network-info", response_model=NetworkInfoResponse, tags=["Discovery"])
async def get_network_info() -> NetworkInfoResponse:
    return NetworkInfoResponse(ip="192.168.4.1", ssid=None, hostname=ROBOT_NAME)


@app.get("/camera/snapshot", tags=["Camera"])
async def get_camera_snapshot() -> Response:
    try:
        frame = await _get_webcam_frame_safe()
    except CameraError as exc:
        raise HTTPException(status_code=503, detail="Snapshot unavailable") from exc

    headers = {"Content-Disposition": "inline; filename=snapshot.jpg"}
    return Response(content=frame, media_type="image/jpeg", headers=headers)


@app.get("/camera/stream", tags=["Camera"])
async def get_camera_stream(frames: int | None = Query(default=None, ge=1)) -> StreamingResponse:
    # Always use camera service (USB webcam) to keep OAK-D free for /shot endpoint
    async def stream_generator() -> AsyncIterator[bytes]:
        LOGGER.info("Starting camera stream", extra={"frames": frames})
        frame_count = 0
        try:
            async for chunk in _camera_stream(app.state.camera_service, frames):
                frame_count += 1
                LOGGER.debug("Emitting camera frame chunk (%d bytes)", len(chunk))
                yield chunk
        except CameraError as exc:
            LOGGER.error("Camera stream interrupted: %s", exc)
            raise HTTPException(status_code=503, detail="Camera stream unavailable") from exc
        finally:
            LOGGER.info(
                "Camera stream finished",
                extra={"frames": frames, "frames_sent": frame_count},
            )

    return StreamingResponse(stream_generator(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")


@app.post("/camera/capture", response_model=CaptureResponse, tags=["Camera"])
async def capture_photo(request: CaptureRequest) -> CaptureResponse:
    if request.type != CaptureType.PHOTO:
        raise HTTPException(status_code=400, detail="Only photo capture is supported")

    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    path = f"/media/{timestamp}.jpg"
    url = f"http://192.168.4.1:8000{path}"
    return CaptureResponse(saved=True, path=path, url=url)


def _voltage_to_percentage(voltage: float | None) -> int:
    """Convert a battery voltage reading to a percentage."""

    if voltage is None:
        return 0

    # Voltage range for the rover's 3S LiPo pack with conservative charging
    # Full charge: 12.25V (4.08V per cell) - safer for battery longevity
    # Empty: 9.0V (3.0V per cell) - safe discharge limit
    empty_voltage = 9.0
    full_voltage = 12.25

    percent = (voltage - empty_voltage) / (full_voltage - empty_voltage)
    percent = max(0.0, min(1.0, percent))
    return int(round(percent * 100))


def _default_status() -> StatusResponse:
    """Return a fallback status response when rover data is unavailable."""

    return StatusResponse(battery=82, cpu=37, temp=46.3, ai_state="idle")


@app.get("/status", response_model=StatusResponse, tags=["Status"])
async def get_status() -> StatusResponse:
    LOGGER.info("Status endpoint called")
    base_controller = _get_base_controller()

    if not base_controller:
        LOGGER.info("No base controller available; returning default status")
        return _default_status()
    
    if not hasattr(base_controller, "get_status"):
        LOGGER.warning("Base controller missing get_status method; returning default status")
        return _default_status()

    try:
        LOGGER.debug("Calling base_controller.get_status()")
        rover_status = await anyio.to_thread.run_sync(base_controller.get_status)
        LOGGER.info("Rover status received: %s", rover_status)
    except Exception as exc:  # pragma: no cover - hardware dependent
        LOGGER.error("Failed to obtain rover status: %s", exc, exc_info=True)
        return _default_status()

    battery_percent = _voltage_to_percentage(rover_status["voltage"])
    temperature = rover_status.get("temperature", 0.0) or 0.0

    return StatusResponse(
        battery=battery_percent,
        cpu=int(rover_status.get("cpu", 0)),
        temp=float(temperature),
        ai_state=str(rover_status.get("ai_state", "idle")),
    )


@app.post("/control/move", response_model=MoveCommand, tags=["Control"])
async def move_robot(
    command: MoveCommand,
    x_control_token: str = Header(..., alias="x-control-token"),
    session_id: str = Header(..., alias="session-id"),
) -> MoveCommand:
    """Move the robot. Requires both control token and active session."""
    # Token and session verification handled by middleware
    # This endpoint is currently disabled (returns command without executing)
    return command


@app.post("/control/stop", response_model=StopResponse, tags=["Control"])
async def stop_robot(
    x_control_token: str = Header(..., alias="x-control-token"),
    session_id: str = Header(..., alias="session-id"),
) -> StopResponse:
    """Stop the robot. Requires both control token and active session."""
    # Token and session verification handled by middleware
    # This endpoint is currently disabled (returns success without executing)
    return StopResponse()


@app.post("/control/head", response_model=HeadCommand, tags=["Control"])
async def move_head(command: HeadCommand) -> HeadCommand:
    return command


@app.post("/control/lights", response_model=LightCommand, tags=["Control"])
async def control_lights(
    command: LightCommand,
    x_control_token: str = Header(..., alias="x-control-token"),
    session_id: str = Header(..., alias="session-id"),
) -> LightCommand:
    base_controller = _get_base_controller()

    if not base_controller:
        raise HTTPException(status_code=503, detail="controller_unavailable")

    if not hasattr(base_controller, "lights_ctrl"):
        raise HTTPException(status_code=501, detail="lights_control_not_supported")

    try:
        await anyio.to_thread.run_sync(
            base_controller.lights_ctrl, command.pwmA, command.pwmB
        )
    except Exception as exc:  # pragma: no cover - hardware dependent
        LOGGER.error("Failed to control lights: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="lights_control_failed")

    return command


@app.post("/control/nod", response_model=NodCommand, tags=["Control"])
async def nod(
    command: NodCommand,
    x_control_token: str = Header(..., alias="x-control-token"),
    session_id: str = Header(..., alias="session-id"),
) -> NodCommand:
    base_controller = _get_base_controller()

    if not base_controller:
        raise HTTPException(status_code=503, detail="controller_unavailable")

    if not hasattr(base_controller, "nod"):
        raise HTTPException(status_code=501, detail="nod_not_supported")

    try:
        await anyio.to_thread.run_sync(
            base_controller.nod,
            command.times,
            command.center_tilt,
            command.delta,
            command.pan,
            command.delay,
        )
    except Exception as exc:  # pragma: no cover - hardware dependent
        LOGGER.error("Failed to execute nod command: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="nod_failed")

    return command


@app.get("/mode", response_model=ModeResponse, tags=["Connectivity"])
async def get_mode() -> ModeResponse:
    return ModeResponse(mode=Mode.ACCESS_POINT)


@app.get("/wifi/status", response_model=WiFiStatusResponse, tags=["Connectivity"])
async def get_wifi_status() -> WiFiStatusResponse:
    """Get WiFi connection status including connection state, network name, and IP address."""
    w = WifiManager()
    status, network_name, ip_address = await anyio.to_thread.run_sync(w.current_connection)

    # Normalize to proper bool
    if isinstance(status, bool):
        connected = status
    elif isinstance(status, str):
        connected = status.strip().lower() in {"connected", "online", "yes", "up"}
    else:
        connected = False  # safe fallback

    return WiFiStatusResponse(
        connected=connected,
        network_name=network_name or None,
        ip=ip_address or None,
    )



@app.get("/wifi/scan", response_model=WiFiScanResponse, tags=["Connectivity"])
async def scan_wifi_networks() -> WiFiScanResponse:
    w = WifiManager()
    """Scan for available WiFi networks and return a list of discovered networks."""
    networks = await anyio.to_thread.run_sync(w.scan_networks)
    return WiFiScanResponse(networks=networks)


@app.post("/wifi/connect", response_model=WiFiConnectResponse, tags=["Connectivity"])
async def connect_wifi(request: WiFiConnectRequest) -> WiFiConnectResponse:
    if not request.password:
        raise HTTPException(status_code=400, detail="Password must not be empty")

    LOGGER.info(f"Attempting to connect to {request.ssid}")
    w = WifiManager()
    res = await anyio.to_thread.run_sync(
    lambda: w.connect(ssid=request.ssid, password=request.password)
)

    return WiFiConnectResponse(connecting=res.success, message=res.message)


@app.post("/claim/request", response_model=ClaimRequestResponse, tags=["Claim"])
async def claim_request() -> ClaimRequestResponse:
    """Generate a PIN code for claiming the robot. PIN is valid for ~120 seconds."""
    STATE["pin"] = f"{secrets.randbelow(10**6):06d}"
    STATE["pin_exp"] = time.time() + 120
    
    # Display PIN on OLED screen (try lazy initialization if not already available)
    base_controller = _get_base_controller()
    
    if base_controller:
        LOGGER.debug("base_controller found, attempting to display PIN")
        try:
            # Rover.display_text uses line numbers 0-3 (0=top, 3=bottom)
            # Using lines 2 and 3 (third and fourth lines) to match original request
            LOGGER.debug("Calling display_text(2, 'PIN Code:')")
            base_controller.display_text(2, "PIN Code:")
            LOGGER.debug("Calling display_text(3, '%s')", STATE["pin"])
            base_controller.display_text(3, STATE["pin"])
            LOGGER.info("PIN displayed on OLED: %s", STATE["pin"])
        except AttributeError as exc:
            LOGGER.error("base_controller missing display_text method: %s", exc, exc_info=True)
            app.state.base_controller = None
        except Exception as exc:
            LOGGER.error("Failed to display PIN on OLED: %s", exc, exc_info=True)
            # Mark as failed so we don't keep trying
            app.state.base_controller = None
    else:
        LOGGER.warning("OLED display not available; PIN generated but not displayed. Rover controller is None or not initialized.")

    _schedule_pin_reset(STATE["pin"], STATE["pin_exp"])

    LOGGER.info("Generated claim PIN (expires in 120s)")
    return ClaimRequestResponse(expiresIn=120)


@app.post("/claim/confirm", response_model=ClaimConfirmResponse, tags=["Claim"])
async def claim_confirm(request: ClaimConfirmRequest) -> ClaimConfirmResponse:
    """Confirm PIN and generate control token. Returns control token and robot ID."""
    if request.pin != STATE["pin"] or time.time() > STATE["pin_exp"] or STATE["claimed"]:
        raise HTTPException(status_code=400, detail="invalid_or_expired_pin")

    token = secrets.token_urlsafe(32)
    STATE["control_token_hash"] = hash_token(token)
    STATE["claimed"] = True
    
    # Reset OLED display when PIN is successfully used
    base_controller = _get_base_controller()
    if base_controller and hasattr(base_controller, "display_reset"):
        try:
            base_controller.display_reset()
            LOGGER.info("OLED display reset after successful PIN claim")
        except Exception as exc:
            LOGGER.error("Failed to reset OLED display after claim: %s", exc, exc_info=True)
    
    STATE["pin"] = None
    STATE["pin_exp"] = 0
    _cancel_pin_reset_task()
    
    LOGGER.info("Robot claimed successfully")
    return ClaimConfirmResponse(controlToken=token, robotId=ROBOT_SERIAL)


@app.post("/claim/release", tags=["Claim"])
async def claim_release() -> dict[str, bool]:
    """Release the claim and rotate the control token."""
    if not STATE["claimed"]:
        raise HTTPException(status_code=400, detail="not_claimed")

    # Rotate token
    new_token = secrets.token_urlsafe(32)
    STATE["control_token_hash"] = hash_token(new_token)
    STATE["claimed"] = False
    STATE["controller"]["sid"] = None
    STATE["controller"]["last"] = 0
    LOGGER.info("Robot claim released")
    return {"released": True}


@app.post("/claim-control", response_model=ClaimControlResponse, tags=["Claim"])
async def claim_control() -> ClaimControlResponse:
    """Claim a controller session. Returns session_id that must be used with control endpoints.
    
    Requires x-control-token header (verified by middleware).
    """
    # Generate new session ID
    session_id = secrets.token_urlsafe(16)
    STATE["controller"]["sid"] = session_id
    STATE["controller"]["last"] = time.time()
    LOGGER.info("Controller session claimed")
    return ClaimControlResponse(sessionId=session_id)


# ============================================================================
# Face Recognition Endpoints
# ============================================================================

@app.post("/face-recognition/recognize", response_model=FaceRecognitionResponse, tags=["Face Recognition"])
async def recognize_faces() -> FaceRecognitionResponse:
    """Recognize faces in the current camera frame."""
    if not FACE_RECOGNITION_AVAILABLE or app.state.face_recognition is None:
        raise HTTPException(
            status_code=503,
            detail="Face recognition service not available. Install insightface and onnxruntime."
        )
    
    try:
        # Get frame from camera (with USB lock protection)
        frame_bytes = await _get_webcam_frame_safe()
        
        # Decode JPEG to numpy array
        import cv2
        import numpy as np
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode camera frame")
        
        # Recognize faces
        recognitions = await anyio.to_thread.run_sync(
            app.state.face_recognition.recognize_faces,
            frame,
            True,  # return_locations
        )
        
        # Convert to response format
        from .models import FaceRecognitionResult
        face_results = [
            FaceRecognitionResult(
                name=rec["name"],
                confidence=rec["confidence"],
                bbox=rec.get("bbox"),
            )
            for rec in recognitions
        ]
        
        return FaceRecognitionResponse(
            faces=face_results,
            frame_count=len(face_results),
        )
        
    except FaceRecognitionError as exc:
        raise HTTPException(status_code=500, detail=f"Face recognition error: {exc}") from exc
    except Exception as exc:
        LOGGER.error("Face recognition failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recognition failed: {exc}") from exc


@app.get("/face-recognition/known", response_model=KnownFacesResponse, tags=["Face Recognition"])
async def get_known_faces() -> KnownFacesResponse:
    """Get list of all known faces."""
    if not FACE_RECOGNITION_AVAILABLE or app.state.face_recognition is None:
        raise HTTPException(
            status_code=503,
            detail="Face recognition service not available"
        )
    
    faces = app.state.face_recognition.get_known_faces()
    return KnownFacesResponse(faces=faces, count=len(faces))


@app.post("/face-recognition/add", response_model=AddFaceResponse, tags=["Face Recognition"])
async def add_face(
    name: str = Form(...),
    image: UploadFile = File(None),
    image_base64: Optional[str] = Form(None),
) -> AddFaceResponse:
    """Add a new face to the known faces database.
    
    Can accept either:
    - Form data with 'image' file upload and 'name' field
    - Form data with 'image_base64' (base64-encoded image) and 'name' field
    """
    if not FACE_RECOGNITION_AVAILABLE or app.state.face_recognition is None:
        raise HTTPException(
            status_code=503,
            detail="Face recognition service not available"
        )
    
    try:
        import cv2
        import numpy as np
        
        # Decode image
        if image and image.filename:
            # From file upload
            image_bytes = await image.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif image_base64:
            # From base64
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            raise HTTPException(status_code=400, detail="No image provided. Use file upload or image_base64 field.")
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Add face
        success = await anyio.to_thread.run_sync(
            app.state.face_recognition.add_known_face,
            name,
            frame,
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        return AddFaceResponse(
            success=True,
            message=f"Face for '{name}' added successfully",
            name=name,
        )
        
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error("Failed to add face: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add face: {exc}") from exc


@app.post("/face-recognition/reload", tags=["Face Recognition"])
async def reload_known_faces() -> dict[str, str]:
    """Reload known faces from disk."""
    if not FACE_RECOGNITION_AVAILABLE or app.state.face_recognition is None:
        raise HTTPException(
            status_code=503,
            detail="Face recognition service not available"
        )
    
    try:
        await anyio.to_thread.run_sync(app.state.face_recognition.reload_known_faces)
        return {"status": "success", "message": "Known faces reloaded"}
    except Exception as exc:
        LOGGER.error("Failed to reload faces: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reload: {exc}") from exc


@app.get("/face-recognition/stream", tags=["Face Recognition"])
async def face_recognition_stream() -> StreamingResponse:
    """Stream camera feed with face recognition overlay."""
    if not FACE_RECOGNITION_AVAILABLE or app.state.face_recognition is None:
        raise HTTPException(
            status_code=503,
            detail="Face recognition service not available"
        )
    
    async def stream_generator() -> AsyncIterator[bytes]:
        import cv2
        import numpy as np
        
        try:
            while True:
                # Get frame from camera (with USB lock protection)
                frame_bytes = await _get_webcam_frame_safe()
                
                # Decode JPEG
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Recognize faces
                    recognitions = await anyio.to_thread.run_sync(
                        app.state.face_recognition.recognize_faces,
                        frame,
                        True,
                    )
                    
                    # Draw recognitions on frame
                    annotated_frame = await anyio.to_thread.run_sync(
                        app.state.face_recognition.draw_recognitions,
                        frame,
                        recognitions,
                    )
                    
                    # Encode back to JPEG
                    _, encoded = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    payload = encoded.tobytes()
                else:
                    payload = frame_bytes  # Fallback to original
                
                # Send frame
                header = (
                    f"--{BOUNDARY}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(payload)}\r\n\r\n"
                ).encode()
                yield header + payload + b"\r\n"
                
                await anyio.sleep(0.1)  # ~10 FPS
                
        except Exception as exc:
            LOGGER.error("Face recognition stream error: %s", exc, exc_info=True)
    
    return StreamingResponse(
        stream_generator(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
    )


# ==================== Meeting Summarization Endpoints ====================

@app.post("/meetings/upload", response_model=MeetingUploadResponse, tags=["Meetings"])
async def upload_meeting_audio(
    audio: UploadFile = File(...),
    title: Optional[str] = Form(None),
    meeting_type: str = Form("meeting"),
) -> MeetingUploadResponse:
    """
    Upload an audio file for meeting transcription and summarization.
    
    Args:
        audio: Audio file (WAV, MP3, etc.)
        title: Optional meeting title
        meeting_type: Type of meeting (meeting, lecture, conversation, note)
    
    Returns:
        MeetingUploadResponse with meeting ID and status
    """
    if not MEETING_SERVICE_AVAILABLE or app.state.meeting_service is None:
        raise HTTPException(
            status_code=503,
            detail="Meeting service not available"
        )
    
    try:
        # Read audio data
        audio_bytes = await audio.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Process audio (transcribe and summarize)
        result = await app.state.meeting_service.process_audio(
            audio_bytes=audio_bytes,
            filename=audio.filename or "recording.wav",
            title=title,
            meeting_type=meeting_type
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Failed to process audio")
            )
        
        return MeetingUploadResponse(
            success=True,
            meeting_id=result["meeting_id"],
            message="Meeting processed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error("Meeting upload failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process meeting: {exc}"
        ) from exc


@app.get("/meetings", response_model=MeetingSummaryListResponse, tags=["Meetings"])
async def get_all_meetings() -> MeetingSummaryListResponse:
    """
    Get all meeting summaries.
    
    Returns:
        MeetingSummaryListResponse with list of all meetings
    """
    if not MEETING_SERVICE_AVAILABLE or app.state.meeting_service is None:
        raise HTTPException(
            status_code=503,
            detail="Meeting service not available"
        )
    
    try:
        meetings = app.state.meeting_service.get_all_meetings()
        
        # Convert to response model
        meeting_summaries = [
            MeetingSummary(**meeting) for meeting in meetings
        ]
        
        return MeetingSummaryListResponse(
            summaries=meeting_summaries,
            count=len(meeting_summaries)
        )
        
    except Exception as exc:
        LOGGER.error("Failed to get meetings: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve meetings: {exc}"
        ) from exc


@app.get("/meetings/{meeting_id}", response_model=MeetingSummary, tags=["Meetings"])
async def get_meeting(meeting_id: str) -> MeetingSummary:
    """
    Get a specific meeting by ID.
    
    Args:
        meeting_id: Meeting UUID
    
    Returns:
        MeetingSummary with full meeting details
    """
    if not MEETING_SERVICE_AVAILABLE or app.state.meeting_service is None:
        raise HTTPException(
            status_code=503,
            detail="Meeting service not available"
        )
    
    try:
        meeting = app.state.meeting_service.get_meeting(meeting_id)
        
        if not meeting:
            raise HTTPException(
                status_code=404,
                detail=f"Meeting {meeting_id} not found"
            )
        
        return MeetingSummary(**meeting)
        
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error("Failed to get meeting: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve meeting: {exc}"
        ) from exc


@app.delete("/meetings/{meeting_id}", tags=["Meetings"])
async def delete_meeting(meeting_id: str) -> dict[str, str]:
    """
    Delete a meeting by ID.
    
    Args:
        meeting_id: Meeting UUID
    
    Returns:
        Status message
    """
    if not MEETING_SERVICE_AVAILABLE or app.state.meeting_service is None:
        raise HTTPException(
            status_code=503,
            detail="Meeting service not available"
        )
    
    try:
        success = app.state.meeting_service.delete_meeting(meeting_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Meeting {meeting_id} not found"
            )
        
        return {
            "status": "success",
            "message": f"Meeting {meeting_id} deleted"
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error("Failed to delete meeting: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete meeting: {exc}"
        ) from exc
