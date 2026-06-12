# Role Summary: Dataset, YOLO Training, and Local Server Auto-Discovery

## 1. Role Overview

My role in the VICKY project focused on two technical areas: improving the object detection dataset/model pipeline and improving how the mobile app connects to the local AI server. These parts are important because VICKY needs reliable indoor landmark detection for navigation, and the smartphone app must connect to the laptop server without requiring users to manually type or hardcode an IP address.

The dataset and YOLO work supports the computer vision pipeline, especially for detecting objects that are useful for visually impaired navigation, such as doors, doorways, exit signs, chairs, people, and other indoor obstacles. The local IP server work improves usability during real-world testing because the laptop's IP address can change depending on the Wi-Fi network.

## 2. Dataset, Training, and Fine-Tuning Summary

For the object detection system, the project uses YOLOv8 as the main detection framework. A general COCO-trained YOLO model is used to detect common indoor objects such as people, chairs, tables, laptops, bottles, and bags. However, navigation-specific objects such as exit signs, doors, and doorways are not always detected reliably by the default COCO model, so the system also uses custom-trained YOLO models.

My work included preparing and organizing the custom datasets for these navigation landmarks. The datasets were structured in YOLO format, with image folders, label folders, and YAML configuration files that define the class names and train/validation paths. The exit-sign dataset is used to fine-tune a model specifically for emergency exit sign detection, while the local-door dataset is used to detect doors and doorway-like structures in the indoor test environment.

After dataset preparation, the models were fine-tuned from YOLOv8 base weights. The trained model weights are stored under the `runs/detect/` directory and loaded by the live vision system during runtime. This allows VICKY to combine general object detection with custom landmark detection in the same perception pipeline.

## 3. YOLO Model Testing in the Live System

The YOLO models were tested inside the live ZED vision pipeline, not only as standalone object detectors. This is important because VICKY does not only need a bounding box; it needs to know where the detected object is in the user's surrounding space.

When YOLO detects an object, the system checks the object's center point, reads the depth value from the ZED depth frame, converts the object into camera-space coordinates, and then projects it onto the 2D navigation map. This allows objects such as doors, chairs, exit signs, and people to appear on the map and be used by the spatial memory and path planning system.

The testing process also involved checking detection confidence, filtering noisy detections, separating static objects from dynamic objects, and making sure important landmarks could be stored as part of the room map. Door detections are handled with extra logic because a closed door and an open doorway have different meanings for navigation.

## 4. Dataset and Training Files

These are the main files and folders related to the dataset and training role:

| Path | Purpose |
| --- | --- |
| `zed/training/README_EXIT_SIGN_ONLY.md` | Notes and commands for training the exit-sign and local-door models. |
| `zed/training/exit_sign_only.yaml` | YOLO dataset configuration for exit sign training. |
| `zed/training/door_local.yaml` | YOLO dataset configuration for local door training. |
| `datasets/exit_sign_only/` | Exit sign image and label dataset. |
| `datasets/YOLODataset_door_local/` | Local indoor door dataset. |
| `runs/detect/exit_sign_only_v2/weights/best.pt` | Fine-tuned exit sign model weight. |
| `runs/detect/door_combined_v1/weights/best.pt` | Fine-tuned door / doorway model weight. |
| `zed/zed_vision_assistant.py` | Runtime file that loads and tests the YOLO models in the live vision pipeline. |

## 5. Code: Loading YOLO and Custom Landmark Models

This code defines the general COCO model and the custom landmark models used by VICKY.

```python
# zed/zed_vision_assistant.py

COCO_MODEL_PATH = os.getenv("VICKY_COCO_MODEL", "yolov8m.pt")
DEFAULT_LANDMARK_MODEL_PATHS = (
    "runs/detect/exit_sign_only_v2/weights/best.pt;"
    "runs/detect/door_combined_v1/weights/best.pt"
)

LANDMARK_MODEL_PATHS = [
    path.strip()
    for path in os.getenv("VICKY_YOLO_MODEL", DEFAULT_LANDMARK_MODEL_PATHS).replace(",", ";").split(";")
    if path.strip()
]
```

This means the system can load default custom models, but it can also be reconfigured through the `VICKY_YOLO_MODEL` environment variable without changing the source code.

## 6. Code: Runtime YOLO Model Initialization

This section loads the standard YOLO model and the custom landmark models.

```python
# zed/zed_vision_assistant.py

print(f"[YOLO] Loading COCO safety model: {COCO_MODEL_PATH}")
coco_model = YOLO(COCO_MODEL_PATH)

for index, landmark_model_path in enumerate(LANDMARK_MODEL_PATHS):
    resolved_landmark_model_path = resolve_model_path(landmark_model_path)
    print(f"[YOLO] Loading landmark model {index}: {resolved_landmark_model_path}")
    landmark_model = YOLO(resolved_landmark_model_path)
```

The COCO model handles general obstacles, while the landmark models focus on navigation-specific objects such as exit signs and doors.

## 7. Code: Running YOLO Detection

This is the part of the live pipeline where each YOLO model runs inference on the current RGB frame.

```python
# zed/zed_vision_assistant.py

results = model.predict(
    rgb_frame,
    imgsz=image_size,
    conf=confidence,
    verbose=False,
    device=device,
    max_det=YOLO_MAX_DETECTIONS,
)
```

Important parameters:

| Parameter | Meaning |
| --- | --- |
| `imgsz` | Input image size for YOLO inference. |
| `conf` | Confidence threshold for filtering low-confidence detections. |
| `device` | Runs on CUDA GPU when available, otherwise CPU. |
| `max_det` | Maximum number of detections per frame. |

## 8. Code: Converting YOLO Detection into Map Object

This code is important because it turns a YOLO bounding box into a spatial object that can be used for mapping and navigation.

```python
# zed/zed_vision_assistant.py

if depth_distance is not None:
    x_c, z_c = image_x_to_camera_point(depth_distance, center_x, w, fov_rad)
    grid_x, grid_z = project_camera_point_to_grid(x_c, z_c)

    semantic_label = semantic_label_for_detection(class_name, door_passable)
    mobility = classify_object_mobility(semantic_label)
    object_color = object_color_for_label(semantic_label, mobility)
```

This step combines:

- YOLO object label,
- bounding box center,
- ZED depth value,
- camera-to-map projection,
- semantic label classification,
- static/dynamic mobility classification.

## 9. Code: Door and Doorway Classification

Door detection needs extra validation because a closed door is an obstacle, while an open doorway can be a route.

```python
# zed/zed_vision_assistant.py

if is_landmark_model and is_door_landmark(class_name):
    if conf < DOOR_CONFIDENCE:
        continue
    if not looks_like_door_panel((x1, y1, x2, y2), (h, w)):
        continue

    door_passable = looks_like_passable_door(
        depth_frame,
        (x1, y1, x2, y2),
        (h, w)
    )
```

This helps the system decide whether the detection should become a `door` or a `doorway` in the navigation map.

## 10. Local IP Server Auto-Discovery Summary

The second part of my role was improving the connection between the smartphone app and the local VICKY server. The server runs on the laptop, usually on port `8000`, but the laptop's IP address can change depending on the Wi-Fi network. If the app depends on hardcoded IP addresses, the user would need to manually edit or re-enter the server address often.

To solve this, the mobile app includes an auto-discovery system. It collects possible server addresses from the Expo runtime, the device network IP, the preferred user input, Android emulator fallback, localhost fallback, and local subnet scanning. The app then probes each candidate and checks whether the FastAPI root endpoint returns a valid VICKY server response.

This makes the app easier to use during demos and real testing because the smartphone can find the local server automatically instead of relying on a manually typed or hardcoded IP address.

## 11. Code: Server Port and Scan Configuration

```javascript
// app54/App.js

const SERVER_HTTP_PORT = 8000;
const AUTO_SERVER_PROBE_TIMEOUT_MS = 900;
const AUTO_SERVER_SUBNET_TIMEOUT_MS = 250;
const AUTO_SERVER_SCAN_BATCH_SIZE = 8;
```

These constants define the FastAPI server port and the timeout behavior for auto-discovery.

## 12. Code: Normalizing Server URL

```javascript
// app54/App.js

const normalizeServerUrl = (value) => {
  const trimmed = String(value || "").trim().replace(/\/+$/, "");
  if (!trimmed) return "";
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  return `http://${trimmed}`;
};
```

This allows the app to accept flexible input such as:

```text
192.168.1.20:8000
http://192.168.1.20:8000
```

and normalize it into a valid HTTP URL.

## 13. Code: Resolving Initial Auto Server URL

```javascript
// app54/App.js

const resolveAutoServerUrl = () => {
  const scriptUrl = NativeModules?.SourceCode?.scriptURL;
  const browserHost = Platform.OS === "web" ? globalThis?.location?.hostname : "";
  const detectedHost = getHostFromUrl(scriptUrl) || browserHost;

  if (detectedHost && detectedHost !== "localhost") {
    return `http://${detectedHost}:${SERVER_HTTP_PORT}`;
  }
  if (Platform.OS === "android") {
    return `http://10.0.2.2:${SERVER_HTTP_PORT}`;
  }
  return `http://127.0.0.1:${SERVER_HTTP_PORT}`;
};
```

This function tries to infer the likely server address from the development environment. On Android emulator, it uses `10.0.2.2`, which maps to the host machine.

## 14. Code: Collecting Expo Host Candidates

```javascript
// app54/App.js

const collectExpoHostCandidates = () => {
  const hosts = [];
  const seen = new Set();
  const addHost = (value) => {
    const host = getHostFromUrl(value);
    if (!host || seen.has(host)) return;
    seen.add(host);
    hosts.push(host);
  };

  addHost(NativeModules?.SourceCode?.scriptURL);

  const expoConstants = NativeModules?.ExponentConstants
    || NativeModules?.ExpoConstants
    || NativeModules?.Constants;

  ...

  return hosts.filter(isUsableLanHost);
};
```

This function collects possible LAN hosts from Expo and React Native runtime metadata.

## 15. Code: Probing a Server Candidate

```javascript
// app54/App.js

const probeServerUrl = async (url) => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), AUTO_SERVER_PROBE_TIMEOUT_MS);
  try {
    const response = await fetch(`${url}/`, { signal: controller.signal });
    if (!response.ok) return false;
    const json = await response.json();
    return String(json?.status || "").toLowerCase().includes("server");
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
};
```

This checks whether a candidate URL is actually running the VICKY FastAPI server.

## 16. Code: Discovering Reachable Local Server

```javascript
// app54/App.js

const discoverReachableServerUrl = async (preferredUrl = "", options = {}) => {
  const includeFallbackBases = Boolean(options.includeFallbackBases);
  const candidates = getAutoServerCandidates(preferredUrl);
  for (const candidate of candidates) {
    if (await probeServerUrl(candidate)) {
      return candidate;
    }
  }

  ...

  for (const base of await getSubnetScanBases(preferredUrl, includeFallbackBases)) {
    ...
    const url = `http://${base}.${octet}:${SERVER_HTTP_PORT}`;
    ...
  }

  return "";
};
```

This function first checks likely candidates. If that fails, it scans the local subnet in batches to find the server.

## 17. Backend Support for Auto-Discovery

The backend supports auto-discovery by returning server metadata and local network URLs from the root endpoint.

```python
# zed/server.py

@app.get("/")
def root():
    return {
        "status": "AI server running",
        "host": "0.0.0.0",
        "port": 8000,
        "local_urls": [f"http://{address}:8000" for address in get_local_ipv4_addresses()],
    }
```

The mobile app probes this endpoint and confirms that the response contains a server status.

## 18. Backend Code: Getting Local IPv4 Addresses

```python
# zed/server.py

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
```

This helps expose usable local IP addresses from the laptop server side.

## 19. Short Report-Ready Paragraph

My contribution focused on the dataset/model pipeline and local server connectivity. For the dataset part, I helped prepare and organize YOLO-format datasets for navigation-specific landmarks such as exit signs and local doors. These datasets were used to fine-tune YOLOv8 models so VICKY could detect objects that are important for visually impaired navigation but not always reliable in the default COCO model. I also tested the trained YOLO models inside the live ZED vision pipeline, where each detection is combined with depth data, projected onto the map, classified as static or dynamic, and used by spatial memory and navigation.

For the server connectivity part, I helped make the mobile app connect to the VICKY server through automatic local IP discovery instead of relying on hardcoded or manually typed IP addresses. The app collects possible server addresses from the Expo runtime, device network information, emulator fallbacks, localhost, and local subnet scanning. It then probes each candidate by calling the FastAPI root endpoint and confirms whether the response belongs to the VICKY server. This makes the system easier to run during demos and real testing because the phone can dynamically find the laptop server on the local network.

## 20. Main Files Connected to My Role

| Area | Files |
| --- | --- |
| Dataset configuration | `zed/training/exit_sign_only.yaml`, `zed/training/door_local.yaml` |
| Training notes | `zed/training/README_EXIT_SIGN_ONLY.md` |
| Dataset folders | `datasets/exit_sign_only/`, `datasets/YOLODataset_door_local/` |
| Trained weights | `runs/detect/exit_sign_only_v2/weights/best.pt`, `runs/detect/door_combined_v1/weights/best.pt` |
| YOLO runtime testing | `zed/zed_vision_assistant.py` |
| Mobile local IP discovery | `app54/App.js` |
| Server discovery endpoint | `zed/server.py` |
