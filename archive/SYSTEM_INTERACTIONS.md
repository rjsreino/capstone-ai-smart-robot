# ROVY System Interactions - Detailed Sequence Diagrams

## Table of Contents
1. [Complete System Initialization](#complete-system-initialization)
2. [Voice Command End-to-End](#voice-command-end-to-end)
3. [Face Recognition Detailed Flow](#face-recognition-detailed-flow)
4. [Autonomous Navigation Cycle](#autonomous-navigation-cycle)
5. [Mobile App Connection](#mobile-app-connection)
6. [Error Recovery Scenarios](#error-recovery-scenarios)

---

## 1. Complete System Initialization

### System Startup Sequence

```
Time: T=0 (Power On)

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   ESP32     │    │   Pi (Robot)│    │Cloud Server │    │  Mobile App │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                   │                   │
       │ Boot (2s)        │                   │                   │
       ├──────────────►   │                   │                   │
       │                  │                   │                   │
       │ <Ready>          │                   │                   │
       │◄─────────────────┤                   │                   │
       │                  │                   │                   │
       │                  │ systemd starts    │                   │
       │                  │ rovy.service      │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │
       │                  │ Init Serial       │                   │
       │                  │ /dev/ttyACM0      │                   │
       │◄─────────────────┤                   │                   │
       │                  │                   │                   │
       │ Status Request   │                   │                   │
       │ {"T":130}        │                   │                   │
       ├──────────────►   │                   │                   │
       │                  │                   │                   │
       │ Status Response  │                   │                   │
       │ {"T":1001,...}   │                   │                   │
       ├──────────────►   │                   │                   │
       │                  │                   │                   │
       │                  │ Display: "ROVY"   │                   │
       │ Display Cmd      │ "Starting..."     │                   │
       │◄─────────────────┤                   │                   │
       │                  │                   │                   │
       │                  │ Init Camera       │                   │
       │                  │ (OAK-D)           │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │
       │                  │ Init Audio        │                   │
       │                  │ (ReSpeaker)       │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │
       │                  │ Init Wake Word    │                   │
       │                  │ Detector          │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │
       │                  │ Start Local API   │                   │
       │                  │ Port 8000         │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │
       │                  │ WebSocket Connect │                   │
       │                  │ ws://cloud:8765   │                   │
       │                  ├──────────────────►│                   │
       │                  │                   │                   │
       │                  │      Accepted     │                   │
       │                  │◄──────────────────┤                   │
       │                  │                   │                   │
       │                  │ Display: "ROVY"   │                   │
       │ Display Update   │ "Ready"           │                   │
       │◄─────────────────┤ "Cloud OK"        │                   │
       │                  │                   │                   │
       │                  │                   │ FastAPI Server    │
       │                  │                   │ Port 8765 Ready   │
       │                  │                   ├──────────────►    │
       │                  │                   │                   │
       │                  │                   │                   │ User opens app
       │                  │                   │                   ├───────────►
       │                  │                   │                   │
       │                  │                   │                   │ Discover robots
       │                  │                   │                   │ (mDNS scan)
       │                  │◄──────────────────────────────────────┤
       │                  │                   │                   │
       │                  │ HTTP GET /health  │                   │
       │                  │◄──────────────────────────────────────┤
       │                  │                   │                   │
       │                  │ {"status":"ok"}   │                   │
       │                  ├───────────────────────────────────────►│
       │                  │                   │                   │
       │                  │                   │                   │ Display robot
       │                  │                   │                   │ in list
       │                  │                   │                   ├───────────►
       │                  │                   │                   │
       │                  │ Display: "Mobile" │                   │
       │ Display Update   │ "Connected"       │                   │
       │◄─────────────────┤                   │                   │
       │                  │                   │                   │

✅ System Ready - All Components Initialized
```

**Initialization Steps**:
1. **ESP32 Boot** (2s): Firmware loads, peripherals initialize
2. **Robot Boot** (10s): Systemd starts service, initializes hardware
3. **Cloud Boot** (5s): FastAPI server starts, loads AI models
4. **Mobile Discovery** (2s): mDNS scan finds robot on network

**Total Startup Time**: ~20 seconds from power-on to fully operational

---

## 2. Voice Command End-to-End

### Complete Voice Interaction Cycle

```
User: "Hey Rovy, what's the weather in Tokyo?"

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Microphone  │    │   Pi (Robot)│    │Cloud Server │    │ OpenAI API  │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                   │                   │
       │ Audio stream     │                   │                   │
       │ (continuous)     │                   │                   │
       ├──────────────►   │                   │                   │
       │                  │                   │                   │
       │                  │ Silero VAD        │                   │
       │                  │ Speech detected   │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │
       │                  │ Buffer audio      │                   │
       │                  │ (2 seconds)       │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │
       │                  │ Faster-Whisper    │                   │
       │                  │ "hey rovy"        │                   │
       │                  │ detected!         │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │
       │                  │ Beep sound        │                   │
       │◄─────────────────┤ (feedback)        │                   │
       │                  │                   │                   │
       │                  │ Recording...      │                   │
       │                  │ (until silence)   │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │
       │                  │ WebSocket         │                   │
       │                  │ {type:"audio"}    │                   │
       │                  ├──────────────────►│                   │
       │                  │                   │                   │
       │                  │                   │ Whisper STT       │
       │                  │                   │ API Call          │
       │                  │                   ├──────────────────►│
       │                  │                   │                   │
       │                  │                   │ "what's the      │
       │                  │                   │ weather in Tokyo?"│
       │                  │                   │◄──────────────────┤
       │                  │                   │                   │
       │                  │                   │ Tool detection    │
       │                  │                   │ → get_weather     │
       │                  │                   ├──────────────►    │
       │                  │                   │                   │
       │                  │                   │ Open-Meteo API    │
       │                  │                   │ (Tokyo weather)   │
       │                  │                   ├──────────────►    │
       │                  │                   │                   │
       │                  │                   │ {"temp": 18,      │
       │                  │                   │  "condition":...} │
       │                  │                   │◄──────────────────┤
       │                  │                   │                   │
       │                  │                   │ GPT-4o            │
       │                  │                   │ Generate response │
       │                  │                   ├──────────────────►│
       │                  │                   │                   │
       │                  │                   │ "It's 18°C in    │
       │                  │                   │ Tokyo with clear  │
       │                  │                   │ skies!"           │
       │                  │                   │◄──────────────────┤
       │                  │                   │                   │
       │                  │                   │ Piper TTS         │
       │                  │                   │ (local synthesis) │
       │                  │                   ├──────────────►    │
       │                  │                   │                   │
       │                  │ WebSocket         │                   │
       │                  │ {type:"audio"}    │                   │
       │                  │◄──────────────────┤                   │
       │                  │                   │                   │
       │ Play audio       │                   │                   │
       │ (speaker)        │                   │                   │
       │◄─────────────────┤                   │                   │
       │                  │                   │                   │
       │                  │ Display: "18°C"   │                   │
       │◄─────────────────┤ "Tokyo Clear"     │                   │
       │                  │                   │                   │
       │ User hears:      │                   │                   │
       │ "It's 18°C..."   │                   │                   │
       ├──────────────►   │                   │                   │
       │                  │                   │                   │

Total latency: ~3-5 seconds (wake word → response)
```

**Performance Breakdown**:
- Wake word detection: 100-300ms (local, Faster-Whisper)
- Audio recording: 1-3s (until user stops speaking)
- Whisper STT: 500-1500ms (OpenAI API)
- Tool execution: 200-500ms (weather API)
- GPT-4o response: 500-1500ms (OpenAI API)
- Piper TTS: 300-800ms (local synthesis)
- **Total**: 3-7 seconds

---

## 3. Face Recognition Detailed Flow

### Complete Face Recognition Cycle

```
User: "Do you recognize me?"

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Camera    │    │Cloud Server │    │InsightFace  │    │ Face DB     │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                   │                   │
       │                  │ Query detected    │                   │
       │                  │ "recognize me"    │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │
       │                  │ Tool: recognize   │                   │
       │                  │ _face()           │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │
       │                  │ Capture frame     │                   │
       │                  │ from OAK-D        │                   │
       │◄─────────────────┤                   │                   │
       │                  │                   │                   │
       │ Frame (RGB)      │                   │                   │
       │ 640x480          │                   │                   │
       ├──────────────►   │                   │                   │
       │                  │                   │                   │
       │                  │ Load InsightFace  │                   │
       │                  │ model (cached)    │                   │
       │                  ├──────────────────►│                   │
       │                  │                   │                   │
       │                  │                   │ Face detection    │
       │                  │                   │ RetinaFace        │
       │                  │                   ├──────────────►    │
       │                  │                   │                   │
       │                  │                   │ Found 1 face      │
       │                  │                   │ @ (x,y,w,h)       │
       │                  │                   │◄──────────────────┤
       │                  │                   │                   │
       │                  │                   │ Face alignment    │
       │                  │                   │ (5 landmarks)     │
       │                  │                   ├──────────────►    │
       │                  │                   │                   │
       │                  │                   │ ArcFace embedding │
       │                  │                   │ (512-dim vector)  │
       │                  │                   ├──────────────►    │
       │                  │                   │                   │
       │                  │                   │ [0.23, -0.15,...] │
       │                  │                   │◄──────────────────┤
       │                  │                   │                   │
       │                  │ Load known faces  │                   │
       │                  │ database          │                   │
       │                  ├───────────────────────────────────────►│
       │                  │                   │                   │
       │                  │ known_faces.pkl   │                   │
       │                  │ {name: embedding} │                   │
       │                  │◄───────────────────────────────────────┤
       │                  │                   │                   │
       │                  │ Cosine similarity │                   │
       │                  │ for each person   │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │
       │                  │ "professor": 0.65 │                   │
       │                  │ "boymirzo": 0.42  │                   │
       │                  │ "nilufar": 0.31   │                   │
       │                  │◄──────────────────┤                   │
       │                  │                   │                   │
       │                  │ Best match:       │                   │
       │                  │ "professor" (0.65)│                   │
       │                  │ > threshold 0.4   │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │
       │                  │ GPT-4o: Generate  │                   │
       │                  │ personalized      │                   │
       │                  │ greeting          │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │
       │                  │ "Hello Professor! │                   │
       │                  │ Great to see you  │                   │
       │                  │ today!"           │                   │
       │                  │◄──────────────────┤                   │
       │                  │                   │                   │
       │                  │ TTS + Send to     │                   │
       │                  │ robot             │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │

Total time: 2-3 seconds
```

**Face Recognition Pipeline Details**:

1. **Face Detection** (100-200ms):
   - Algorithm: RetinaFace
   - Input: RGB image (640x480)
   - Output: Bounding boxes + confidence

2. **Face Alignment** (50-100ms):
   - Detect 5 facial landmarks (eyes, nose, mouth corners)
   - Normalize face orientation
   - Crop and resize to 112x112

3. **Embedding Generation** (100-200ms):
   - Model: ArcFace (ResNet-50 backbone)
   - Output: 512-dimensional embedding vector
   - Normalized to unit length

4. **Similarity Matching** (10-50ms):
   - Cosine similarity with known faces
   - Threshold: 0.4 (adjustable)
   - Return best match above threshold

**Database Structure** (`known-faces/`):
```
known-faces/
├── professor.jpg
├── boymirzo.jpg
├── nilufar.jpg
└── embeddings.pkl  # Cached embeddings
```

---

## 4. Autonomous Navigation Cycle

### Navigation Control Loop (10 Hz)

```
Exploration Mode Active

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   OAK-D     │    │   Depth     │    │  Obstacle   │    │   Rover     │
│   Camera    │    │  Processor  │    │  Avoidance  │    │  (Motors)   │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                   │                   │
       │                  │                   │                   │
╔══════╪══════════════════╪═══════════════════╪═══════════════════╪══════╗
║ LOOP │ (10 Hz = 100ms)  │                   │                   │      ║
╚══════╪══════════════════╪═══════════════════╪═══════════════════╪══════╝
       │                  │                   │                   │
       │ Stereo capture   │                   │                   │
       │ 30 FPS           │                   │                   │
       ├──────────────►   │                   │                   │
       │                  │                   │                   │
       │ Depth map        │                   │                   │
       │ (400p, 16-bit)   │                   │                   │
       ├──────────────►   │                   │                   │
       │                  │                   │                   │
       │                  │ Grid analysis     │                   │
       │                  │ (divide into      │                   │
       │                  │ 10x10 zones)      │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │
       │                  │ Zone depths:      │                   │
       │                  │ - Left: 1200mm    │                   │
       │                  │ - Center: 2500mm  │                   │
       │                  │ - Right: 800mm    │                   │
       │                  ├──────────────────►│                   │
       │                  │                   │                   │
       │                  │                   │ Potential field   │
       │                  │                   │ calculation       │
       │                  │                   ├──────────────►    │
       │                  │                   │                   │
       │                  │                   │ Attractive force: │
       │                  │                   │ F_att = K_att *   │
       │                  │                   │ (goal - pos)      │
       │                  │                   ├──────────────►    │
       │                  │                   │                   │
       │                  │                   │ Repulsive forces: │
       │                  │                   │ - Right: STRONG   │
       │                  │                   │ - Left: MEDIUM    │
       │                  │                   │ - Center: WEAK    │
       │                  │                   ├──────────────►    │
       │                  │                   │                   │
       │                  │                   │ Sum forces:       │
       │                  │                   │ Total_F = F_att + │
       │                  │                   │ ΣF_rep            │
       │                  │                   ├──────────────►    │
       │                  │                   │                   │
       │                  │                   │ Velocity command: │
       │                  │                   │ - Linear: 0.3 m/s │
       │                  │                   │ - Angular: +0.5   │
       │                  │                   │   rad/s (turn left│
       │                  │                   ├──────────────────►│
       │                  │                   │                   │
       │                  │                   │                   │ JSON command
       │                  │                   │                   │ to ESP32
       │                  │                   │                   ├───────────►
       │                  │                   │                   │
       │                  │                   │                   │ {"T":1,
       │                  │                   │                   │  "L":0.2,
       │                  │                   │                   │  "R":0.4}
       │                  │                   │                   ├───────────►
       │                  │                   │                   │
       │                  │                   │                   │ Motors spin
       │                  │                   │                   │ Robot turns
       │                  │                   │                   │ left
       │                  │                   │                   ├───────────►
       │                  │                   │                   │
       │                  │ Update occupancy  │                   │
       │                  │ grid (SLAM-lite)  │                   │
       │                  ├──────────────►    │                   │
       │                  │                   │                   │
       │                  │                   │ Check emergency   │
       │                  │                   │ stop flag         │
       │                  │                   ├──────────────►    │
       │                  │                   │                   │
       │◄─────────────────┴───────────────────┴───────────────────┤
       │                                                           │
       │ sleep(100ms) → Next iteration                            │
       └───────────────────────────────────────────────────────────┘

Loop continues until:
- User stops exploration
- Emergency stop triggered
- Battery low
- Obstacle too close (< 30cm)
```

**Potential Field Method Details**:

```python
# Attractive force (toward goal)
F_attractive = K_att * (goal_position - current_position)
# K_att = 1.0 (configurable)

# Repulsive force (away from obstacles)
for obstacle in obstacles:
    distance = distance_to(obstacle)
    if distance < influence_distance:
        direction = (current_position - obstacle_position).normalize()
        magnitude = K_rep * (1/distance - 1/influence_distance) / distance^2
        F_repulsive += direction * magnitude
# K_rep = 5.0, influence_distance = 1.5m

# Total force
F_total = F_attractive + F_repulsive

# Convert to velocity command
linear_velocity = min(max_speed, |F_total|)
angular_velocity = atan2(F_total.y, F_total.x) - current_heading
```

**Navigation Performance**:
- Control frequency: 10 Hz
- Depth processing: 30 Hz (3 depth frames per control cycle)
- Obstacle detection range: 0.3m - 5.0m
- Safe distance: 1.5m
- Max speed: 0.5 m/s
- Response time: < 100ms

---

## 5. Mobile App Connection

### Detailed Connection Flow

```
User opens mobile app

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Mobile App │    │  Robot API  │    │   ESP32     │    │    OLED     │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                   │                   │
       │ App launch       │                   │                   │
       ├──────────────►   │                   │                   │
       │                  │                   │                   │
       │ Check saved      │                   │                   │
       │ robot IP         │                   │                   │
       ├──────────────►   │                   │                   │
       │                  │                   │                   │
       │ Not found -      │                   │                   │
       │ Start discovery  │                   │                   │
       ├──────────────►   │                   │                   │
       │                  │                   │                   │
       │ mDNS query       │                   │                   │
       │ _rovy._tcp.local │                   │                   │
       ├──────────────────►                   │                   │
       │                  │                   │                   │
       │                  │ mDNS response     │                   │
       │                  │ 192.168.1.100     │                   │
       │◄──────────────────                   │                   │
       │                  │                   │                   │
       │ HTTP GET /health │                   │                   │
       ├──────────────────►                   │                   │
       │                  │                   │                   │
       │                  │ {"status":"ok"}   │                   │
       │◄──────────────────┤                   │                   │
       │                  │                   │                   │
       │ Display robot    │                   │                   │
       │ "ROVY-001"       │                   │                   │
       │ "192.168.1.100"  │                   │                   │
       ├──────────────►   │                   │                   │
       │                  │                   │                   │
       │ User taps        │                   │                   │
       │ "Connect"        │                   │                   │
       ├──────────────►   │                   │                   │
       │                  │                   │                   │
       │ GET /status      │                   │                   │
       ├──────────────────►                   │                   │
       │                  │                   │                   │
       │                  │ Query ESP32       │                   │
       │                  ├──────────────────►│                   │
       │                  │                   │                   │
       │                  │ {"T":130}         │                   │
       │                  ├──────────────────►│                   │
       │                  │                   │                   │
       │                  │ Status data       │                   │
       │                  │◄──────────────────┤                   │
       │                  │                   │                   │
       │ Full status      │                   │                   │
       │ {battery, IMU}   │                   │                   │
       │◄──────────────────┤                   │                   │
       │                  │                   │                   │
       │ Save robot IP    │                   │                   │
       │ to AsyncStorage  │                   │                   │
       ├──────────────►   │                   │                   │
       │                  │                   │                   │
       │ Navigate to      │                   │                   │
       │ Home screen      │                   │                   │
       ├──────────────►   │                   │                   │
       │                  │                   │                   │
       │                  │                   │                   │ Display:
       │                  │                   │                   │ "Mobile"
       │                  │                   ├──────────────────►│ "Connected"
       │                  │                   │                   │
       │ Start polling    │                   │                   │
       │ /status (1 Hz)   │                   │                   │
       ├──────────────────►                   │                   │
       │                  │                   │                   │
       │                  │                   │                   │
╔══════╪══════════════════╪═══════════════════╪═══════════════════╪══════╗
║ POLL │ Every 1 second   │                   │                   │      ║
╚══════╪══════════════════╪═══════════════════╪═══════════════════╪══════╝
       │                  │                   │                   │
       │ GET /status      │                   │                   │
       ├──────────────────►                   │                   │
       │                  │                   │                   │
       │ Status data      │                   │                   │
       │◄──────────────────┤                   │                   │
       │                  │                   │                   │
       │ Update UI        │                   │                   │
       │ - Battery: 75%   │                   │                   │
       │ - Temp: 45°C     │                   │                   │
       │ - IMU: 0,0,180   │                   │                   │
       ├──────────────►   │                   │                   │
       │                  │                   │                   │

✅ Connected - Ready for control
```

**Connection States**:
1. **Discovering**: Scanning network via mDNS
2. **Found**: Robot detected, showing info
3. **Connecting**: Fetching full status
4. **Connected**: Polling status, ready for commands
5. **Error**: Connection failed, retry available

---

## 6. Error Recovery Scenarios

### Scenario 1: WebSocket Disconnection

```
┌─────────────┐    ┌─────────────┐
│   Robot     │    │Cloud Server │
└──────┬──────┘    └──────┬──────┘
       │                  │
       │ Active WS        │
       │◄────────────────►│
       │                  │
       │                  │ Network issue
       │                  │ Connection lost
       │    ✗✗✗✗✗✗✗✗✗✗   │
       │                  │
       │ Detect disconnect│
       │ (ping timeout)   │
       ├──────────────►   │
       │                  │
       │ Wait 1 second    │
       ├──────────────►   │
       │                  │
       │ Retry #1         │
       │ WebSocket connect│
       ├──────────────────►
       │                  │
       │ Failed (timeout) │
       │◄─────────────────┤
       │                  │
       │ Wait 2 seconds   │
       │ (exponential)    │
       ├──────────────►   │
       │                  │
       │ Retry #2         │
       ├──────────────────►
       │                  │
       │ Failed           │
       │◄─────────────────┤
       │                  │
       │ Display: "Cloud" │
       │ "Disconnected"   │
       │ "Retrying..."    │
       ├──────────────►   │
       │                  │
       │ Wait 4 seconds   │
       ├──────────────►   │
       │                  │
       │ Retry #3         │
       ├──────────────────►
       │                  │
       │ Success!         │
       │◄─────────────────┤
       │                  │
       │ Display: "Cloud" │
       │ "Connected"      │
       ├──────────────►   │
       │                  │

✅ Connection restored
```

**Reconnection Strategy**:
- Retry interval: 1s, 2s, 4s, 8s, 16s (exponential backoff)
- Max retries: 10 (then wait 60s, restart)
- Graceful degradation: Local control still works

### Scenario 2: Camera Failure

```
┌─────────────┐    ┌─────────────┐
│   Robot     │    │  OAK-D Cam  │
└──────┬──────┘    └──────┬──────┘
       │                  │
       │ Init camera      │
       ├──────────────────►
       │                  │
       │ Error: USB not   │
       │ detected         │
       │◄─────────────────┤
       │                  │
       │ Log error        │
       ├──────────────►   │
       │                  │
       │ Try USB camera   │
       │ (fallback)       │
       ├──────────────────►
       │                  │
       │ OpenCV: /dev/    │
       │ video0 OK        │
       │◄─────────────────┤
       │                  │
       │ Display: "Camera"│
       │ "USB (fallback)" │
       ├──────────────►   │
       │                  │

✅ Using fallback camera
```

### Scenario 3: Battery Low

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   ESP32     │    │   Robot     │    │  Mobile App │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                   │
       │ Status: v=9.2V   │                   │
       │ (low!)           │                   │
       ├──────────────►   │                   │
       │                  │                   │
       │                  │ Calculate: 5%     │
       │                  ├──────────────►    │
       │                  │                   │
       │                  │ Alert: "Low      │
       │                  │ Battery"          │
       │                  ├───────────────────►
       │                  │                   │
       │ Display: "⚠"     │                   │
       │ "Battery Low"    │                   │
       │◄─────────────────┤                   │
       │                  │                   │
       │                  │ Reduce max speed  │
       │                  │ to 0.3 m/s        │
       │                  ├──────────────►    │
       │                  │                   │
       │                  │ Disable           │
       │                  │ autonomous mode   │
       │                  ├──────────────►    │
       │                  │                   │

✅ Safe mode active
```

---

## Summary

This document provides detailed sequence diagrams for all major system interactions in ROVY:

1. **System Initialization**: 20-second boot sequence
2. **Voice Commands**: 3-7 second end-to-end latency
3. **Face Recognition**: 2-3 second recognition cycle
4. **Autonomous Navigation**: 10 Hz control loop
5. **Mobile Connection**: mDNS discovery + HTTP polling
6. **Error Recovery**: Automatic reconnection and graceful degradation

All components work together to create a robust, responsive, and intelligent robot system.

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Authors**: Capstone Design Team 2025-2

