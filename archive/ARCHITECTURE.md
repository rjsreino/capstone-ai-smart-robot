# ROVY - Complete System Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Details](#component-details)
4. [Communication Flow](#communication-flow)
5. [Technology Stack](#technology-stack)
6. [Data Flow](#data-flow)
7. [Hardware Architecture](#hardware-architecture)
8. [Software Architecture](#software-architecture)
9. [API Reference](#api-reference)
10. [Deployment Architecture](#deployment-architecture)
11. [Security & Authentication](#security--authentication)
12. [Performance & Scalability](#performance--scalability)

---

## 1. System Overview

**ROVY** is an advanced autonomous robot system featuring:
- **AI-Powered Assistant**: GPT-4 based conversational AI with vision, face recognition, and tool calling
- **Autonomous Navigation**: OAK-D Lite stereo camera with real-time obstacle avoidance
- **Multi-Platform Control**: React Native mobile app for iOS/Android
- **Cloud Intelligence**: Centralized AI server running on PC with GPU acceleration
- **Hardware Integration**: ESP32-based rover with gimbal camera, OLED display, and LED lights

### System Purpose
ROVY is a capstone design project for creating an intelligent mobile robot that can:
- Navigate autonomously using stereo depth vision
- Interact naturally through voice and conversation
- Recognize people and estimate ages
- Perform tasks like playing music, providing weather updates, and web searches
- Be controlled via mobile app or voice commands
- Stream live video and audio

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ROVY SYSTEM                                     │
│                        (Distributed Architecture)                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐         ┌──────────────────────┐
│   MOBILE APP         │         │   CLOUD SERVER       │         │   ROBOT (Pi)         │
│   (React Native)     │◄────────┤   (PC with GPU)      │◄────────┤   (Raspberry Pi 5)   │
│                      │  WiFi/  │                      │WebSocket│                      │
│  - Home Dashboard    │  4G     │  - AI Assistant      │         │  - Camera Stream     │
│  - Manual Control    │         │  - Face Recognition  │         │  - Audio Input       │
│  - Voice Commands    │         │  - Vision API        │         │  - Wake Word Detect  │
│  - Photo Gallery     │         │  - Tool Execution    │         │  - Local API         │
│  - Settings          │         │  - Meeting Recorder  │         │  - Rover Controller  │
│  - Robot Status      │         │  - Gesture Detection │         │  - BLE Provisioning  │
└──────────────────────┘         └──────────────────────┘         └──────────────────────┘
         │                                 │                                 │
         │                                 │                                 │
         └─────────────────────────────────┼─────────────────────────────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │   HARDWARE LAYER       │
                              │                        │
                              │  - ESP32 UGV Rover     │──► Motors (L/R wheels)
                              │  - Gimbal Controller   │──► Camera Pan/Tilt
                              │  - OLED Display        │──► Status Display
                              │  - LED Lights          │──► Front/Back Lights
                              │  - OAK-D Lite Camera   │──► Stereo Depth Vision
                              │  - ReSpeaker Mic       │──► Audio Input
                              │  - Battery Monitor     │──► Power Status
                              └────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          NAVIGATION SUBSYSTEM                                │
│                       (OAK-D Autonomous Navigation)                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌────────────────────────┐
                              │ Navigation Controller  │
                              │  - Mode Management     │
                              │  - Control Loop (10Hz) │
                              │  - State Tracking      │
                              └────────┬───────────────┘
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 │                     │                     │
         ┌───────▼────────┐   ┌───────▼────────┐   ┌───────▼────────┐
         │ Depth Processor│   │   Obstacle     │   │  Path Planner  │
         │  - Stereo 30Hz │   │   Avoidance    │   │   - A* Search  │
         │  - Grid Map    │   │  - Potential   │   │   - Dijkstra   │
         │  - Safe Zones  │   │    Field       │   │   - RRT        │
         └────────────────┘   │  - VFH         │   └────────────────┘
                              │  - Reactive    │
                              └────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          AI INTELLIGENCE LAYER                               │
│                        (Cloud Server Processing)                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  OpenAI GPT-4o   │    │   Whisper STT    │    │   Piper TTS      │
│  - Conversation  │    │  - Voice to Text │    │  - Text to Voice │
│  - Tool Calling  │    └──────────────────┘    └──────────────────┘
│  - Vision        │
└──────────────────┘    ┌──────────────────┐    ┌──────────────────┐
                        │  InsightFace     │    │  Hugging Face    │
                        │  - Face Recog    │    │  - Age Estimate  │
                        │  - ArcFace       │    │  - DEX Model     │
                        └──────────────────┘    └──────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL SERVICES                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Spotify    │  │  YouTube     │  │  Open-Meteo  │  │  DuckDuckGo  │
│   Music API  │  │  Music API   │  │  Weather API │  │  Web Search  │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

---

## 3. Component Details

### 3.1 Mobile Application (React Native + Expo)

**Location**: `/mobile/`

**Technology**: React Native, Expo, TypeScript

**Key Features**:
- **Home Dashboard**: Quick access to all robot functions
- **Manual Control**: Joystick-based robot movement
- **Voice Commands**: Real-time speech-to-text interaction
- **Photo Gallery**: View and manage captured photos
- **Settings**: Configure robot behavior and preferences
- **Robot Status**: Real-time telemetry display
- **WiFi Provisioning**: Configure robot network settings
- **BLE Communication**: Low-level robot control

**Screens** (`/mobile/app/`):
```
├── index.tsx                 # Landing/connection screen
├── (tabs)/
│   ├── home.tsx             # Main dashboard
│   ├── status.tsx           # Robot telemetry
│   ├── photo-time.tsx       # Photo gallery
│   └── settings.tsx         # Configuration
├── agentic.tsx              # AI chat interface
├── connection.tsx           # Robot discovery
├── manual.tsx               # Manual control
├── wifi.tsx                 # WiFi setup
└── robot-status.tsx         # Detailed status
```

**Services** (`/mobile/services/`):
- `robot-api.ts`: REST API client for local robot
- `cloud-api.ts`: WebSocket client for cloud server
- `rovy-ble.ts`: Bluetooth Low Energy communication
- `network-discovery.ts`: mDNS robot discovery
- `gesture-detection.ts`: Hand gesture recognition
- `kodak-printer.ts`: Instant photo printing

**Communication Protocols**:
1. **Local API** (FastAPI on Robot): HTTP/REST for control commands
2. **Cloud WebSocket**: Real-time bidirectional communication
3. **BLE**: Low-level hardware control (provisioning, direct commands)

---

### 3.2 Cloud Server (Python + FastAPI)

**Location**: `/cloud/`

**Technology**: Python 3.10+, FastAPI, OpenAI API, WebSockets

**Purpose**: Centralized AI intelligence running on powerful PC with GPU

**Architecture** (`/cloud/app/`):
```
cloud/
├── main.py                   # FastAPI application (3330 lines!)
│                             # - WebSocket handlers
│                             # - HTTP endpoints
│                             # - Stream management
│
├── ai.py                     # CloudAssistant (GPT-4o)
│   ├── ask()                # Text-only queries
│   ├── ask_with_vision()    # Vision + text queries
│   ├── classify_intent()    # Intent classification
│   └── Tool calling system
│
├── vision.py                 # VisionProcessor
│   ├── capture_and_analyze()
│   ├── analyze_image()
│   └── Camera integration (OAK-D/USB)
│
├── tools.py                  # ToolExecutor
│   ├── get_weather()
│   ├── play_music()
│   ├── calculate()
│   ├── web_search()
│   ├── translate()
│   ├── guess_age()
│   ├── recognize_face()
│   ├── move_robot()
│   └── explore_robot()
│
├── face_recognition_service.py
│   ├── InsightFace/ArcFace
│   ├── Face detection
│   ├── Face recognition
│   └── Face database management
│
├── gesture_detection.py
│   ├── MediaPipe hand detection
│   ├── Gesture classification
│   └── Real-time hand tracking
│
├── meeting_service.py
│   ├── Audio recording
│   ├── Whisper transcription
│   └── Meeting summaries
│
├── camera.py                 # CameraService
│   ├── DepthAICameraSource  # OAK-D
│   ├── OpenCVCameraSource   # USB Camera
│   └── MJPEG streaming
│
├── oak_stream.py             # OAK-D streaming
├── rover_controller.py       # Serial communication
└── models.py                 # Data models
```

**Key Features**:

#### 3.2.1 AI Assistant (`ai.py`)
- **Model**: OpenAI GPT-4o (latest multimodal model)
- **Temperature**: 0.7 for personality
- **System Prompt**: Friendly, encouraging personality with flattery
- **Tool Calling**: Automatic detection and execution
- **Vision Support**: Analyze images from camera
- **Caching**: LRU cache for common queries

**Personality Traits**:
- Warm, friendly, enthusiastic
- Gives genuine compliments
- Encouraging and supportive
- Concise responses (under 30 words)
- Natural conversation, not robotic

#### 3.2.2 Vision Processing (`vision.py`)
- **Model**: GPT-4o Vision
- **Camera Sources**: 
  - OAK-D Lite (preferred) - 640x480 @ 15fps
  - USB webcam (fallback)
- **Capabilities**:
  - Scene understanding
  - Object detection
  - Text reading (OCR)
  - Math problem solving
  - Visual Q&A

#### 3.2.3 Tool Execution (`tools.py`)
Provides external API capabilities:

**Available Tools**:
1. **Weather** (`get_weather`): Open-Meteo API, free, no API key
   - Current weather for any location
   - Temperature, conditions, wind

2. **Time** (`get_time`): System clock
   - Current date and time
   - Formatted output

3. **Calculator** (`calculate`): Python eval (safe)
   - Mathematical expressions
   - Basic arithmetic

4. **Music** (`play_music`): Spotify or YouTube Music
   - Play/pause/skip
   - Volume control
   - Requires authentication

5. **Web Search** (`web_search`): DuckDuckGo API
   - Web search results
   - Free, no API key

6. **Reminders** (`set_reminder`): In-memory storage
   - Time-based reminders
   - Notification system

7. **Translation** (`translate`): GPT-4o
   - Multi-language translation
   - Natural language input

8. **Age Estimation** (`guess_age`): Hugging Face DEX model
   - Face detection
   - Age prediction from image
   - Apparent age estimation

9. **Face Recognition** (`recognize_face`): InsightFace
   - Face detection and embedding
   - Identity matching
   - Known faces database

10. **Robot Movement** (`move_robot`): Serial commands
    - Forward/backward/left/right
    - Configurable distance and speed

11. **Autonomous Exploration** (`explore_robot`): OAK-D navigation
    - Obstacle avoidance
    - Path planning
    - Autonomous roaming

**Tool Detection**:
- Keyword-based fast detection
- AI-based intent classification (fallback)
- Parameter extraction
- Automatic execution

#### 3.2.4 Face Recognition (`face_recognition_service.py`)
- **Model**: InsightFace with ArcFace embeddings
- **Accuracy**: High-quality facial recognition
- **Database**: Known faces stored in `/cloud/known-faces/`
- **Features**:
  - Face detection
  - Face embedding generation
  - Similarity matching
  - Identity management

**How it works**:
1. Detect faces in image using InsightFace
2. Generate 512-dimensional embedding vector
3. Compare with known face database (cosine similarity)
4. Return match if similarity > threshold (0.4)

#### 3.2.5 Meeting Service (`meeting_service.py`)
- **Recording**: Audio capture with timestamps
- **Transcription**: OpenAI Whisper (speech-to-text)
- **Storage**: `/cloud/meetings/` with JSON metadata
- **Features**:
  - Real-time audio recording
  - Meeting transcription
  - Audio playback
  - Meeting history

---

### 3.3 Robot (Raspberry Pi 5)

**Location**: `/robot/`

**Hardware**: Raspberry Pi 5 (4GB+ RAM)

**Purpose**: On-robot control, sensor input, actuator control

**Architecture** (`/robot/`):
```
robot/
├── main.py                   # RovyClient
│   ├── WebSocket client (connects to cloud)
│   ├── Camera streaming
│   ├── Audio streaming
│   ├── Command handling
│   └── Wake word detection
│
├── rover.py                  # Rover hardware controller
│   ├── Serial communication (ESP32)
│   ├── Movement control
│   ├── Gimbal control
│   ├── Display management
│   ├── LED control
│   └── Status reading
│
├── wake_word_detector.py     # Wake word detection
│   ├── Silero VAD
│   ├── Faster-Whisper
│   └── "Hey Rovy" detection
│
└── main_api.py               # Local FastAPI server
    ├── Discovery endpoints
    ├── Camera endpoints
    ├── Control endpoints
    └── Status endpoints
```

**Key Components**:

#### 3.3.1 Robot Client (`main.py`)
- **WebSocket Client**: Connects to cloud server via Tailscale
- **Camera Streaming**: Captures frames, encodes to JPEG, sends to cloud
- **Audio Streaming**: Captures microphone input, sends to cloud
- **Command Handler**: Receives and executes commands from cloud
- **Wake Word**: Local wake word detection ("Hey Rovy")

**Communication Flow**:
```
Robot → WebSocket → Cloud Server
  - Audio chunks (16kHz, 16-bit PCM)
  - Video frames (JPEG, 640x480)
  - Status updates (battery, IMU)
  - Wake word events

Cloud Server → WebSocket → Robot
  - Movement commands
  - Audio responses (TTS)
  - Display messages
  - Configuration updates
```

#### 3.3.2 Rover Controller (`rover.py`)
Controls ESP32-based rover hardware via UART (115200 baud).

**Command Protocol** (JSON over serial):
```json
{
  "T": 1,           // Type: 1=move, 133=gimbal, 132=lights, 3=display
  "L": 0.5,         // Left wheel speed (-1.0 to 1.0)
  "R": 0.5          // Right wheel speed (-1.0 to 1.0)
}
```

**Features**:
- **Movement**: `move(direction, distance, speed)`
- **Gimbal**: `gimbal_ctrl(x, y, speed, acceleration)`
- **Gestures**: `nod_yes()`, `shake_no()`, `dance(style)`
- **Display**: `display_text(line, text)` - 4 lines OLED
- **Lights**: `lights_ctrl(front, back)` - 0-255 brightness
- **Status**: `get_status()` - battery, temperature, IMU

**Dance Modes**:
- **Party**: Spinning, lights, head movements
- **Wiggle**: Side-to-side with head nods
- **Spin**: 360° rotations with lights

#### 3.3.3 Local API (`main_api.py`)
FastAPI server running on robot for direct mobile app control.

**Endpoints**:
- `GET /health` - Robot status
- `GET /network-info` - Network configuration
- `GET /camera/snapshot` - Single frame capture
- `GET /camera/stream` - MJPEG video stream
- `POST /control/move` - Movement commands
- `POST /control/head` - Gimbal control
- `GET /status` - Telemetry data

---

### 3.4 OAK-D Navigation System

**Location**: `/oakd_navigation/`

**Hardware**: OAK-D Lite stereo depth camera

**Purpose**: Autonomous navigation with obstacle avoidance

**Architecture**:
```
oakd_navigation/
├── navigation_controller.py  # Main controller
├── depth_processor.py        # Stereo depth processing
├── obstacle_avoidance.py     # Avoidance algorithms
├── path_planner.py           # Path planning algorithms
├── spatial_ai.py             # Object detection (optional)
└── config.yaml               # Configuration
```

**Navigation Modes**:
1. **Manual**: Direct user control, no autonomy
2. **Assisted**: Obstacle detection + warnings
3. **Autonomous**: Full self-driving to waypoint
4. **Waypoint**: Multi-point navigation
5. **Explore**: Random exploration with obstacle avoidance

#### 3.4.1 Depth Processor (`depth_processor.py`)
- **Input**: Stereo camera pair (OAK-D CAM_B + CAM_C)
- **Output**: Depth map (400p/720p/800p)
- **Frame Rate**: 30 FPS
- **Features**:
  - Grid-based depth analysis
  - Zone detection (left, center, right)
  - Safe direction calculation
  - Occupancy grid generation

**Depth Processing Pipeline**:
```
Stereo Cameras → DepthAI → Depth Map (16-bit) → Grid Analysis
                                                     ↓
                                              Zone Detection
                                                     ↓
                                          Safe Direction Calculation
                                                     ↓
                                            Occupancy Grid Update
```

#### 3.4.2 Obstacle Avoidance (`obstacle_avoidance.py`)
Four avoidance strategies:

**1. Potential Field Method**:
- Attractive force toward goal
- Repulsive force from obstacles
- Force combination for velocity
- Smooth, natural movement

**2. Vector Field Histogram (VFH)**:
- Polar histogram of obstacles
- Gap detection in histogram
- Best direction selection
- Angular velocity computation

**3. Simple Reactive**:
- Fast, direct response
- Turn away from obstacles
- Proportional control
- Low computational cost

**4. Wall Following**:
- Maintain distance from walls
- Follow obstacle boundaries
- Useful for indoor navigation

#### 3.4.3 Path Planner (`path_planner.py`)
Three planning algorithms:

**1. A* (A-star)**:
- Optimal path finding
- Heuristic: Euclidean distance
- Fast for known environments
- Used for waypoint navigation

**2. Dijkstra's Algorithm**:
- Guaranteed shortest path
- No heuristic (slower than A*)
- Useful for complex environments

**3. RRT (Rapidly-exploring Random Tree)**:
- Probabilistic sampling
- Good for dynamic obstacles
- Fast replanning capability

**Path Planning Pipeline**:
```
Start + Goal → Occupancy Grid → Path Search → Path Found?
                                                   ↓ Yes
                                         Obstacle Inflation
                                                   ↓
                                          Path Simplification
                                                   ↓
                                            Waypoint List
```

#### 3.4.4 Navigation Controller (`navigation_controller.py`)
Integrates all navigation components.

**Control Loop** (10 Hz):
```
while running:
    1. Get depth frame from OAK-D
    2. Process depth → navigation data
    3. Update occupancy grid
    4. Check for obstacles
    5. Compute avoidance command (if needed)
    6. Execute path planning (if waypoint mode)
    7. Send velocity command to robot
    8. Update state
    9. Sleep (100ms)
```

**State Management**:
- Current position (x, y, heading)
- Current mode
- Target waypoint
- Waypoint queue
- Emergency stop flag
- Movement state

---

## 4. Communication Flow

### 4.1 Voice Interaction Flow

```
┌──────────────┐
│ User speaks  │
│ "Hey Rovy"   │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────┐
│ Robot (Raspberry Pi)                │
│  1. ReSpeaker captures audio        │
│  2. Silero VAD detects speech       │
│  3. Faster-Whisper detects wake word│
└──────┬──────────────────────────────┘
       │
       ▼ WebSocket
┌─────────────────────────────────────┐
│ Cloud Server (PC)                   │
│  4. Whisper STT → text              │
│  5. GPT-4o processes query          │
│  6. Tool detection & execution      │
│  7. Generate response               │
│  8. Piper TTS → audio               │
└──────┬──────────────────────────────┘
       │
       ▼ WebSocket
┌─────────────────────────────────────┐
│ Robot (Raspberry Pi)                │
│  9. Receive audio response          │
│  10. Play through speaker           │
│  11. Update OLED display            │
└─────────────────────────────────────┘
```

### 4.2 Vision Query Flow

```
┌──────────────────┐
│ User: "What do   │
│ you see?"        │
└────────┬─────────┘
         │
         ▼
┌────────────────────────────────────┐
│ Cloud Server                       │
│  1. Vision tool detected           │
│  2. Capture frame from OAK-D       │
│  3. Encode JPEG                    │
│  4. Send to GPT-4o Vision          │
└────────┬───────────────────────────┘
         │
         ▼ OpenAI API
┌────────────────────────────────────┐
│ OpenAI GPT-4o Vision               │
│  5. Analyze image                  │
│  6. Generate description           │
└────────┬───────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ Cloud Server                       │
│  7. Receive description            │
│  8. TTS synthesis                  │
│  9. Send audio to robot            │
└────────┬───────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ Robot                              │
│  10. Play response                 │
└────────────────────────────────────┘
```

### 4.3 Face Recognition Flow

```
┌──────────────────┐
│ User: "Who am I?"│
└────────┬─────────┘
         │
         ▼
┌────────────────────────────────────┐
│ Cloud Server                       │
│  1. Tool: recognize_face           │
│  2. Capture frame                  │
│  3. InsightFace face detection     │
│  4. Generate face embedding        │
└────────┬───────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ Face Database (/known-faces/)      │
│  5. Load known face embeddings     │
│  6. Compute cosine similarity      │
│  7. Find best match (>0.4)         │
└────────┬───────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ Cloud Server                       │
│  8. Return identity                │
│  9. GPT-4o generates greeting      │
│  10. TTS + send to robot           │
└────────────────────────────────────┘
```

### 4.4 Autonomous Navigation Flow

```
┌──────────────────┐
│ User: "Explore!" │
└────────┬─────────┘
         │
         ▼
┌────────────────────────────────────┐
│ Cloud Server                       │
│  1. Tool: explore_robot            │
│  2. Send explore command to robot  │
└────────┬───────────────────────────┘
         │
         ▼ WebSocket
┌────────────────────────────────────┐
│ Robot - Navigation Controller      │
│  3. Set mode = EXPLORE             │
│  4. Start control loop (10Hz)      │
└────────┬───────────────────────────┘
         │
         ▼ (Loop)
┌────────────────────────────────────┐
│ OAK-D Depth Processing             │
│  5. Capture stereo depth (30fps)   │
│  6. Generate depth map             │
│  7. Analyze zones (left/center/right)│
│  8. Calculate safe directions      │
└────────┬───────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ Obstacle Avoidance                 │
│  9. Potential field computation    │
│  10. Generate velocity command     │
│  11. Check emergency stop          │
└────────┬───────────────────────────┘
         │
         ▼ Serial (UART)
┌────────────────────────────────────┐
│ ESP32 Rover Controller             │
│  12. Receive movement command      │
│  13. Set motor speeds              │
│  14. Move robot                    │
└────────────────────────────────────┘
```

### 4.5 Mobile App Control Flow

```
┌──────────────────┐
│ Mobile App       │
│ User taps        │
│ "Move Forward"   │
└────────┬─────────┘
         │
         ▼ HTTP POST
┌────────────────────────────────────┐
│ Robot Local API (FastAPI)          │
│  1. Receive /control/move          │
│  2. Parse parameters               │
└────────┬───────────────────────────┘
         │
         ▼ Serial
┌────────────────────────────────────┐
│ Rover Controller (rover.py)        │
│  3. Call move(forward, 0.5, medium)│
│  4. Generate JSON command          │
└────────┬───────────────────────────┘
         │
         ▼ UART (115200)
┌────────────────────────────────────┐
│ ESP32                              │
│  5. Parse JSON                     │
│  6. Set motor PWM                  │
│  7. Execute movement               │
│  8. Send status back               │
└────────┬───────────────────────────┘
         │
         ▼ Serial
┌────────────────────────────────────┐
│ Robot Local API                    │
│  9. Return response                │
└────────┬───────────────────────────┘
         │
         ▼ HTTP Response
┌────────────────────────────────────┐
│ Mobile App                         │
│  10. Update UI                     │
│  11. Show status                   │
└────────────────────────────────────┘
```

---

## 5. Technology Stack

### 5.1 Mobile Application

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | React Native 0.81.5 | Cross-platform mobile development |
| **Meta-Framework** | Expo 54 | Build tooling, native modules |
| **Language** | TypeScript 5.9 | Type-safe JavaScript |
| **Navigation** | Expo Router 6 | File-based routing |
| **State Management** | React Context + Hooks | Local state management |
| **Storage** | AsyncStorage | Persistent key-value storage |
| **Networking** | Axios, WebSocket | HTTP + real-time communication |
| **Camera** | expo-camera | Photo capture |
| **Audio** | expo-av, @react-native-voice/voice | Audio playback, speech input |
| **BLE** | react-native-ble-plx | Bluetooth Low Energy |
| **Animations** | Lottie, Reanimated | Smooth UI animations |

### 5.2 Cloud Server

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | FastAPI 0.100+ | Modern async Python web framework |
| **Runtime** | Python 3.10+ | Application runtime |
| **AI Models** | OpenAI GPT-4o | Language model, vision |
| **STT** | OpenAI Whisper | Speech-to-text |
| **TTS** | Piper TTS | Text-to-speech (local) |
| **Face Recognition** | InsightFace (ArcFace) | High-accuracy face recognition |
| **Age Estimation** | Hugging Face Transformers | DEX age estimation model |
| **Gesture Detection** | MediaPipe | Hand tracking and gestures |
| **Camera SDK** | DepthAI 2.30 | OAK-D camera interface |
| **Computer Vision** | OpenCV 4.8+ | Image processing |
| **Audio Processing** | PyDub, SoundDevice | Audio manipulation |
| **HTTP Client** | httpx 0.28 | Async HTTP requests |
| **Music** | Spotipy, YouTube Music API | Music playback control |

### 5.3 Robot (Raspberry Pi)

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | FastAPI | Local API server |
| **Runtime** | Python 3.10+ | Application runtime |
| **WebSocket** | websockets 11+ | Cloud server connection |
| **Serial** | PySerial 3.5 | UART communication |
| **Camera** | OpenCV, DepthAI | Video capture |
| **Audio** | PyAudio, SoundDevice | Audio I/O |
| **Wake Word** | Silero VAD, Faster-Whisper | Local wake word detection |
| **TTS** | Piper TTS | Local text-to-speech |
| **System** | systemd | Service management |

### 5.4 Navigation System

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Camera** | OAK-D Lite | Stereo depth camera |
| **SDK** | DepthAI 2.30 | Camera interface |
| **Vision** | OpenCV, NumPy | Depth processing |
| **Algorithms** | A*, Dijkstra, RRT | Path planning |
| **Control** | Potential Field, VFH | Obstacle avoidance |

### 5.5 Hardware

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **Robot Controller** | ESP32 | Motor control, sensors |
| **Compute** | Raspberry Pi 5 (4GB) | On-robot processing |
| **Cloud PC** | GPU-enabled PC | AI inference |
| **Camera** | OAK-D Lite | Stereo depth vision |
| **Microphone** | ReSpeaker 4-Mic Array | Voice input |
| **Display** | OLED 128x64 | Status display |
| **Motors** | DC motors + H-bridge | Wheel drive |
| **Gimbal** | 2-axis servo gimbal | Camera pan/tilt |
| **Lights** | LED strips | Front/back illumination |
| **Power** | LiPo battery (3S) | 9.0V - 12.6V |

---

## 6. Data Flow

### 6.1 Audio Data Flow

```
Microphone → PyAudio → RovyClient → WebSocket → Cloud Server
(ReSpeaker)   (16kHz)   (chunks)    (binary)     (processing)
                                                      ↓
                                               Whisper STT
                                                      ↓
                                                  Text Query
                                                      ↓
                                                  GPT-4o
                                                      ↓
                                                Response Text
                                                      ↓
                                                  Piper TTS
                                                      ↓
                                             Audio Response
                                                      ↓
                                     WebSocket ← Cloud Server
                                         ↓
                                    Robot Client
                                         ↓
                                   SoundDevice
                                         ↓
                                      Speaker
```

### 6.2 Video Data Flow

```
OAK-D Camera → DepthAI → RovyClient → WebSocket → Cloud Server
(RGB Stream)   (frames)   (JPEG)      (base64)     (storage)
                                                       ↓
                                                Vision Requests
                                                       ↓
                                                  GPT-4o Vision
                                                       ↓
                                              Scene Description
```

### 6.3 Control Data Flow

```
Mobile App → HTTP POST → Robot API → RoverController → Serial → ESP32
(UI Input)   (JSON)      (FastAPI)   (rover.py)       (UART)   (MCU)
                                                                  ↓
                                                            Motor PWM
                                                                  ↓
                                                            Robot Movement
                                                                  ↓
                                                           Status Update
                                                                  ↓
                                                         Serial → Robot
                                                                  ↓
                                                         HTTP Response
                                                                  ↓
                                                           Mobile App
```

### 6.4 Navigation Data Flow

```
OAK-D Stereo → DepthAI → Depth Map → DepthProcessor → Navigation Data
(CAM_B+C)      (VPU)     (16-bit)     (grid)           (zones, safe dirs)
                                                              ↓
                                                    ObstacleAvoidance
                                                              ↓
                                                  NavigationCommand
                                                              ↓
                                                 NavigationController
                                                              ↓
                                                   VelocityCallback
                                                              ↓
                                                     RoverController
                                                              ↓
                                                         Movement
```

---

## 7. Hardware Architecture

### 7.1 Physical Components

```
┌─────────────────────────────────────────────────────────────────┐
│                       ROVY ROBOT CHASSIS                         │
│                                                                  │
│  ┌────────────┐                              ┌────────────┐     │
│  │ Front LEDs │                              │ Back LEDs  │     │
│  └─────┬──────┘                              └──────┬─────┘     │
│        │                                            │           │
│  ┌─────▼──────────────────────────────────────────▼──────┐     │
│  │                 ESP32 Controller                      │     │
│  │  - Motor driver (H-bridge)                            │     │
│  │  - IMU (roll, pitch, yaw)                            │     │
│  │  - Battery monitor                                    │     │
│  │  - OLED display (128x64)                             │     │
│  │  - Gimbal servo control                              │     │
│  └──┬────────────────────────────────────────────────┬──┘     │
│     │                                                  │        │
│  ┌──▼──┐                                           ┌──▼──┐     │
│  │Left │                                           │Right│     │
│  │Motor│                                           │Motor│     │
│  └─────┘                                           └─────┘     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              Raspberry Pi 5 (4GB RAM)                    │ │
│  │  - Python runtime                                        │ │
│  │  - FastAPI server                                        │ │
│  │  - WebSocket client                                      │ │
│  │  - Camera capture                                        │ │
│  │  - Audio I/O                                            │ │
│  └──┬────────────────────────────────────────────┬────────┘ │
│     │                                             │          │
│  ┌──▼─────────────┐                         ┌────▼──────┐   │
│  │  OAK-D Lite    │                         │ ReSpeaker │   │
│  │  Stereo Camera │                         │  4-Mic    │   │
│  │  - CAM_B (Left)│                         │  Array    │   │
│  │  - CAM_C (Right)│                        └───────────┘   │
│  │  - RGB Camera  │                                         │
│  │  - USB 3.0     │                                         │
│  └────────────────┘                                         │
│                                                              │
│  On 2-axis Gimbal:                                          │
│  - Pan: ±180°                                               │
│  - Tilt: ±90°                                               │
│                                                              │
│  ┌────────────────────────────────────────┐                 │
│  │  LiPo Battery (3S)                     │                 │
│  │  - Voltage: 9.0V - 12.6V               │                 │
│  │  - Capacity: ~5000mAh                  │                 │
│  │  - Runtime: ~2 hours                   │                 │
│  └────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Communication Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                         COMMUNICATION MAP                        │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐                                    ┌──────────────┐
│  Mobile App  │◄───────── WiFi / 4G ───────────────┤ Cloud Server │
│  (Phone)     │                                    │   (PC)       │
└──────┬───────┘                                    └──────┬───────┘
       │                                                   │
       │ HTTP REST API                                     │ WebSocket
       │ (192.168.x.x:8000)                               │ (via Tailscale)
       │                                                   │
       ▼                                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Raspberry Pi 5 (Robot)                         │
│                                                                   │
│  ┌─────────────────┐                    ┌────────────────────┐   │
│  │ Local API       │                    │ WebSocket Client   │   │
│  │ (FastAPI)       │                    │ (Cloud Connection) │   │
│  │ Port 8000       │                    │ Port 8765          │   │
│  └─────────────────┘                    └────────────────────┘   │
│           │                                       │               │
│           └───────────────┬───────────────────────┘               │
│                           │                                       │
│                           ▼                                       │
│                  ┌────────────────┐                               │
│                  │ Rover          │                               │
│                  │ Controller     │                               │
│                  └────────┬───────┘                               │
│                           │                                       │
│                           │ UART (Serial)                         │
│                           │ 115200 baud                           │
└───────────────────────────┼───────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     ESP32     │
                    │   Controller  │
                    └───────────────┘
```

### 7.3 Pin Connections (ESP32)

**Motor Control**:
- Motor Left PWM: GPIO 16
- Motor Left DIR: GPIO 17
- Motor Right PWM: GPIO 18
- Motor Right DIR: GPIO 19

**Gimbal Control**:
- Pan Servo: GPIO 21
- Tilt Servo: GPIO 22

**Display**:
- OLED SDA: GPIO 23
- OLED SCL: GPIO 24
- OLED Reset: GPIO 25

**Lights**:
- Front LED: GPIO 26
- Back LED: GPIO 27

**Sensors**:
- IMU SDA: GPIO 4
- IMU SCL: GPIO 5
- Battery ADC: GPIO 34

**Communication**:
- UART TX: GPIO 1
- UART RX: GPIO 3

---

## 8. Software Architecture

### 8.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                          │
│  - Mobile App UI (React Native)                                 │
│  - Voice Interface (Speech I/O)                                 │
│  - OLED Display (Status)                                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                      APPLICATION LAYER                           │
│  - AI Assistant (GPT-4o)                                        │
│  - Tool Executor (External APIs)                                │
│  - Navigation Controller                                         │
│  - Meeting Service                                              │
│  - Face Recognition                                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                      SERVICES LAYER                              │
│  - Vision Processing (GPT-4o Vision)                            │
│  - Speech Services (Whisper STT, Piper TTS)                     │
│  - Camera Service (OAK-D / USB)                                 │
│  - Audio Service (ReSpeaker)                                    │
│  - Rover Controller (Serial)                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                      INFRASTRUCTURE LAYER                        │
│  - FastAPI (HTTP/WebSocket)                                     │
│  - DepthAI SDK (OAK-D)                                          │
│  - OpenAI API (GPT-4o)                                          │
│  - Hugging Face (Models)                                        │
│  - Serial Communication (UART)                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                      HARDWARE LAYER                              │
│  - ESP32 Controller                                             │
│  - OAK-D Lite Camera                                            │
│  - ReSpeaker Microphone                                         │
│  - DC Motors                                                    │
│  - Gimbal Servos                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Microservices Architecture

The system follows a distributed microservices pattern:

**Service 1: Mobile App** (Client)
- Runs on user's phone
- Provides UI/UX
- Communicates via REST/WebSocket

**Service 2: Robot API** (Edge)
- Runs on Raspberry Pi
- Local control interface
- Direct hardware access

**Service 3: Cloud Server** (Central)
- Runs on GPU-enabled PC
- AI processing hub
- Coordinates all intelligence

**Service 4: Navigation System** (Embedded)
- Runs on Raspberry Pi (parallel to Robot API)
- Autonomous navigation
- Real-time depth processing

**Service 5: Hardware Controller** (Firmware)
- Runs on ESP32
- Low-level motor control
- Sensor reading

**Communication**: REST, WebSocket, Serial (UART)

### 8.3 Design Patterns

#### 8.3.1 Singleton Pattern
- `CameraService`: Single camera instance
- `RoverController`: Single serial connection
- `CloudAssistant`: Single AI client

#### 8.3.2 Factory Pattern
- `ObstacleAvoidance`: Creates strategy objects
- `PathPlanner`: Creates planner instances
- `CameraService`: Creates camera sources

#### 8.3.3 Strategy Pattern
- Obstacle avoidance algorithms (Potential Field, VFH, Reactive)
- Path planning algorithms (A*, Dijkstra, RRT)
- Camera sources (OAK-D, USB, Placeholder)

#### 8.3.4 Observer Pattern
- WebSocket message handlers
- Tool execution callbacks
- Navigation state updates

#### 8.3.5 Command Pattern
- Robot movement commands
- Tool execution commands
- Navigation mode changes

---

## 9. API Reference

### 9.1 Cloud Server API

**Base URL**: `ws://[CLOUD_SERVER_IP]:8765` (WebSocket)

#### WebSocket Messages

**Client → Server**:

```json
// Audio chunk
{
  "type": "audio",
  "data": "base64_encoded_audio",
  "sample_rate": 16000,
  "channels": 1
}

// Video frame
{
  "type": "video",
  "data": "base64_encoded_jpeg"
}

// Text query
{
  "type": "query",
  "text": "What's the weather?"
}

// Status update
{
  "type": "status",
  "battery": 11.5,
  "temperature": 45.2,
  "roll": 0.5,
  "pitch": -1.2,
  "yaw": 180.0
}
```

**Server → Client**:

```json
// Text response
{
  "type": "response",
  "text": "The weather is sunny and 25°C"
}

// Audio response
{
  "type": "audio_response",
  "data": "base64_encoded_audio",
  "format": "wav"
}

// Command
{
  "type": "command",
  "action": "move",
  "direction": "forward",
  "distance": 0.5,
  "speed": "medium"
}

// Display message
{
  "type": "display",
  "lines": ["Hello!", "Rovy here", "Ready", ""]
}
```

### 9.2 Robot Local API

**Base URL**: `http://[ROBOT_IP]:8000`

#### Endpoints

**GET /health**
```json
{
  "status": "ok",
  "robot_id": "rovy-001",
  "firmware_version": "1.0.0",
  "battery": 11.5,
  "uptime": 3600
}
```

**GET /network-info**
```json
{
  "ip": "192.168.1.100",
  "ssid": "MyNetwork",
  "signal_strength": -45
}
```

**GET /camera/snapshot**
- Returns: JPEG image

**GET /camera/stream**
- Returns: MJPEG stream

**POST /control/move**
```json
{
  "linear_velocity": 0.5,
  "angular_velocity": 0.0
}
```

**POST /control/head**
```json
{
  "pan": 0,
  "tilt": 0
}
```

**GET /status**
```json
{
  "battery": {
    "voltage": 11.5,
    "percentage": 75
  },
  "imu": {
    "roll": 0.5,
    "pitch": -1.2,
    "yaw": 180.0
  },
  "temperature": 45.2,
  "is_moving": false
}
```

### 9.3 ESP32 Serial Protocol

**Format**: JSON over UART (115200 baud, 8N1)

#### Commands (Pi → ESP32)

**Movement**:
```json
{
  "T": 1,
  "L": 0.5,  // Left wheel speed (-1.0 to 1.0)
  "R": 0.5   // Right wheel speed (-1.0 to 1.0)
}
```

**Gimbal Control**:
```json
{
  "T": 133,
  "X": 0,    // Pan angle (-180 to 180)
  "Y": 0,    // Tilt angle (-90 to 90)
  "SPD": 200,
  "ACC": 10
}
```

**Lights**:
```json
{
  "T": 132,
  "IO4": 255,  // Front LED (0-255)
  "IO5": 0     // Back LED (0-255)
}
```

**Display**:
```json
{
  "T": 3,
  "lineNum": 0,  // Line 0-3
  "Text": "Hello World"
}
```

**Status Request**:
```json
{
  "T": 130
}
```

#### Responses (ESP32 → Pi)

**Status**:
```json
{
  "T": 1001,
  "v": 11.5,      // Voltage
  "temp": 45.2,   // Temperature (°C)
  "r": 0.5,       // Roll (degrees)
  "p": -1.2,      // Pitch (degrees)
  "y": 180.0      // Yaw (degrees)
}
```

---

## 10. Deployment Architecture

### 10.1 Development Environment

```
Developer Machine (Windows/Mac/Linux)
├── Mobile App Development
│   ├── Node.js 18+
│   ├── Expo CLI
│   ├── Android Studio / Xcode
│   └── Metro bundler
│
├── Cloud Server Development
│   ├── Python 3.10+
│   ├── Virtual environment
│   ├── OpenAI API key
│   └── GPU (optional, for testing)
│
└── Robot Development
    ├── SSH access to Raspberry Pi
    ├── Serial monitor (for ESP32)
    └── Camera testing tools
```

### 10.2 Production Deployment

**Mobile App**:
1. Build Android APK: `expo build:android`
2. Build iOS IPA: `expo build:ios`
3. Distribute via TestFlight or internal distribution

**Cloud Server** (GPU PC):
```bash
cd cloud/
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python main.py
```

**Robot** (Raspberry Pi):
```bash
cd robot/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install as systemd service
sudo cp rovy.service /etc/systemd/system/
sudo systemctl enable rovy
sudo systemctl start rovy
```

**Navigation** (integrated with Robot):
```bash
cd oakd_navigation/
pip install -r requirements.txt
# Tested automatically on robot startup
```

### 10.3 Network Configuration

**Option 1: Home Network** (Development)
- All devices on same WiFi
- Robot: `192.168.1.100:8000`
- Cloud: `192.168.1.10:8765`
- Mobile: Auto-discover via mDNS

**Option 2: Robot Hotspot** (Standalone)
- Robot creates WiFi AP: `ROVY-xxxx`
- Robot IP: `192.168.4.1:8000`
- Cloud via Tailscale: `100.x.x.x:8765`
- Mobile connects to robot hotspot

**Option 3: Cloud Deployment** (Advanced)
- Cloud server on public IP or VPN (Tailscale)
- Robot connects via WebSocket
- Mobile connects via 4G/5G
- End-to-end encryption

---

## 11. Security & Authentication

### 11.1 API Security

**Cloud Server**:
- WebSocket: Tailscale VPN (encrypted tunnel)
- No public exposure by default
- Optional: API key authentication

**Robot Local API**:
- Local network only (no public exposure)
- Optional: Basic authentication
- HTTPS with self-signed cert (optional)

**Mobile App**:
- Robot discovery via mDNS (local network)
- PIN-based robot claiming
- Secure storage for credentials (Expo SecureStore)

### 11.2 Data Privacy

**Audio**:
- Processed in memory, not stored
- Meeting recordings stored locally
- No third-party sharing

**Video**:
- Frames sent to OpenAI Vision API
- Not stored permanently
- User consent required

**Face Database**:
- Stored locally only
- Encrypted embeddings
- No cloud upload

---

## 12. Performance & Scalability

### 12.1 Performance Metrics

**AI Response Time**:
- Text query: 1-3 seconds
- Vision query: 3-5 seconds
- Tool execution: 0.5-10 seconds (depends on tool)

**Navigation**:
- Depth processing: 30 FPS
- Control loop: 10 Hz
- Latency: < 100ms

**Video Streaming**:
- Resolution: 640x480
- Frame rate: 15-30 FPS
- Latency: 100-200ms (local), 200-500ms (remote)

**Audio Streaming**:
- Sample rate: 16kHz
- Latency: 200-500ms (includes processing)

### 12.2 Resource Usage

**Raspberry Pi 5**:
- CPU: 40-60% (with camera + audio)
- Memory: 500MB-1GB
- Disk: < 100MB (logs + temp files)

**Cloud Server (GPU PC)**:
- CPU: 20-40% (mostly I/O)
- GPU: 30-50% (during inference)
- Memory: 4-8GB (models loaded)
- Disk: 10GB (models + dependencies)

**Mobile App**:
- Memory: 100-200MB
- Battery: ~5% per hour (active use)

---

## 13. Conclusion

ROVY is a sophisticated multi-component robotic system that demonstrates:

1. **Distributed Architecture**: Mobile app, cloud server, robot, and navigation working together
2. **AI Integration**: GPT-4o for conversation, vision, and tool calling
3. **Autonomous Navigation**: OAK-D stereo vision with obstacle avoidance
4. **Real-time Communication**: WebSocket streaming, REST APIs, serial communication
5. **Modern Tech Stack**: React Native, FastAPI, DepthAI, OpenAI, Hugging Face

**Key Strengths**:
- Modular design (easy to extend)
- Graceful degradation (fallbacks for failures)
- Comprehensive feature set (voice, vision, navigation, control)
- Production-ready (systemd, logging, error handling)

**Use Cases**:
- Educational robotics platform
- AI assistant research
- Autonomous navigation testing
- Human-robot interaction studies
- Capstone design projects

This architecture provides a solid foundation for further development and research in mobile robotics and AI integration.

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Authors**: Capstone Design Team 2025-2  
**Contact**: See project repository for details

