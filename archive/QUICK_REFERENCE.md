# ROVY System - Quick Reference Guide

## 🚀 Quick Start

### Starting the System

1. **Power On Robot** → ESP32 boots (2s) → Pi boots (10s)
2. **Start Cloud Server** → `cd cloud && python main.py`
3. **Open Mobile App** → Auto-discovers robot → Connect

### System Status Check

```bash
# Robot health
curl http://192.168.1.100:8000/health

# Cloud server
curl http://localhost:8765/health

# Robot logs
sudo journalctl -u rovy.service -f
```

---

## 📱 Mobile App Features

| Screen | Purpose | Key Actions |
|--------|---------|-------------|
| **Home** | Main dashboard | Voice, manual control, status |
| **Manual** | Direct control | Joystick, gimbal, lights |
| **Agentic** | AI chat | Text/voice with AI assistant |
| **Status** | Telemetry | Battery, IMU, sensors |
| **Photo Time** | Gallery | View captured photos |
| **Settings** | Config | WiFi, preferences |

---

## 🤖 Robot Capabilities

### Voice Commands

| Command | Action | Response Time |
|---------|--------|---------------|
| "Hey Rovy" | Wake up | 100-300ms |
| "What's the weather?" | Weather info | 2-3s |
| "What do you see?" | Vision analysis | 3-5s |
| "Do you know me?" | Face recognition | 2-3s |
| "How old am I?" | Age estimation | 2-3s |
| "Move forward" | Robot movement | 500ms |
| "Explore" | Autonomous nav | Immediate |
| "Play music" | Music control | 1-2s |
| "What time is it?" | Time/date | 1s |
| "Translate to Chinese" | Translation | 2s |

### Movement Commands

```python
# Python API
rover.move('forward', distance=0.5, speed='medium')
rover.move('backward', distance=0.3, speed='slow')
rover.move('left', distance=0.2, speed='fast')
rover.stop()
```

```json
# HTTP API
POST http://robot-ip:8000/control/move
{
  "linear_velocity": 0.5,
  "angular_velocity": 0.0
}
```

```json
# Serial Protocol (ESP32)
{
  "T": 1,
  "L": 0.5,  // Left wheel: -1.0 to 1.0
  "R": 0.5   // Right wheel: -1.0 to 1.0
}
```

### Gimbal Control

```python
# Python
rover.gimbal_ctrl(x=0, y=0, speed=200, acceleration=10)
rover.nod_yes(times=3)
rover.shake_no(times=3)
```

```json
# Serial
{
  "T": 133,
  "X": 0,    // Pan: -180 to 180
  "Y": 0,    // Tilt: -90 to 90
  "SPD": 200,
  "ACC": 10
}
```

### Display & Lights

```python
# Display (4 lines, OLED 128x64)
rover.display_text(line=0, text="Hello World")
rover.display_lines(["Line 1", "Line 2", "Line 3", "Line 4"])

# LED lights
rover.lights_ctrl(front=255, back=0)  # 0-255
```

### Dance Modes

```python
rover.dance(style='party', duration=10)   # Energetic
rover.dance(style='wiggle', duration=10)  # Side-to-side
rover.dance(style='spin', duration=10)    # 360° spins
```

---

## 🧠 AI Features

### Tools Available

| Tool | Trigger Keywords | Example |
|------|-----------------|---------|
| **Weather** | weather, temperature, forecast | "What's the weather in Seoul?" |
| **Time** | time, date, today, clock | "What time is it?" |
| **Calculator** | calculate, plus, minus, times | "Calculate 25 times 17" |
| **Music** | play, pause, music, song | "Play some music" |
| **Web Search** | who is, what is, search | "Who is Albert Einstein?" |
| **Translation** | translate, traduire, 翻译 | "Translate hello to Japanese" |
| **Vision** | what do you see, look at | "What do you see?" |
| **Face Recog** | do you know me, recognize | "Who am I?" |
| **Age Guess** | how old, guess my age | "Guess my age" |
| **Move Robot** | move, go, forward, turn | "Move forward" |
| **Explore** | explore, autonomous, navigate | "Start exploring" |

### AI Models Used

| Purpose | Model | Provider |
|---------|-------|----------|
| **Conversation** | GPT-4o | OpenAI |
| **Vision** | GPT-4o Vision | OpenAI |
| **Speech-to-Text** | Whisper | OpenAI |
| **Text-to-Speech** | Piper TTS | Local |
| **Face Recognition** | InsightFace (ArcFace) | Local |
| **Age Estimation** | DEX (Hugging Face) | Local |
| **Gesture Detection** | MediaPipe | Local |
| **Wake Word** | Faster-Whisper + Silero VAD | Local |

---

## 🎯 Navigation System

### Navigation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Manual** | User control only | Testing, precise control |
| **Assisted** | Manual + warnings | Learning, careful navigation |
| **Autonomous** | Full self-driving | Go to waypoint |
| **Waypoint** | Multi-point path | Complex routes |
| **Explore** | Random roaming | Room exploration |

### Obstacle Avoidance

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| **Potential Field** | Smooth, natural | Local minima | Open spaces |
| **VFH** | Gap detection | Computational | Corridors |
| **Reactive** | Fast, simple | Less optimal | Quick response |
| **Wall Following** | Reliable | Predictable | Indoor navigation |

### Path Planning

| Algorithm | Speed | Optimality | Use Case |
|-----------|-------|------------|----------|
| **A*** | Fast | Optimal | Known maps |
| **Dijkstra** | Slow | Guaranteed optimal | Complex maps |
| **RRT** | Fast | Suboptimal | Dynamic obstacles |

### Configuration

```yaml
# oakd_navigation/config.yaml

depth:
  resolution: "400p"        # 400p, 720p, 800p
  fps: 30
  safe_distance: 1.5        # meters

obstacle_avoidance:
  strategy: "potential_field"
  max_speed: 0.5            # m/s
  safe_distance: 1.5        # m
  repulsive_gain: 5.0

path_planning:
  planner_type: "astar"
  robot_radius: 0.3         # m
  grid_resolution: 0.1      # m

navigation:
  update_rate: 10.0         # Hz
  default_mode: "manual"
```

---

## 🔧 Hardware Specifications

### Components

| Component | Model | Specs |
|-----------|-------|-------|
| **Main Computer** | Raspberry Pi 5 | 4GB RAM, ARM Cortex-A76 |
| **Controller** | ESP32 | Dual-core, 240MHz |
| **Camera** | OAK-D Lite | Stereo depth, RGB, USB 3.0 |
| **Microphone** | ReSpeaker 4-Mic | 16kHz, 4 channels |
| **Display** | OLED | 128x64, I2C |
| **Battery** | LiPo 3S | 9.0-12.6V, ~5000mAh |
| **Motors** | DC with encoder | ~100 RPM |

### Pin Connections (ESP32)

```
Motors:
- Left PWM: GPIO 16,  Left DIR: GPIO 17
- Right PWM: GPIO 18, Right DIR: GPIO 19

Gimbal:
- Pan: GPIO 21, Tilt: GPIO 22

Display (I2C):
- SDA: GPIO 23, SCL: GPIO 24, RST: GPIO 25

Lights:
- Front: GPIO 26, Back: GPIO 27

Sensors (I2C):
- IMU SDA: GPIO 4, IMU SCL: GPIO 5
- Battery: GPIO 34 (ADC)

UART:
- TX: GPIO 1, RX: GPIO 3 (115200 baud)
```

---

## 🌐 Network Configuration

### IP Addresses

| Device | IP | Port | Protocol |
|--------|-----|------|----------|
| Robot | 192.168.1.100 | 8000 | HTTP/REST |
| Cloud Server | 192.168.1.10 | 8765 | WebSocket |
| Cloud Server | 192.168.1.10 | 8000 | HTTP |

### Ports Used

| Port | Service | Purpose |
|------|---------|---------|
| **8000** | Robot FastAPI | Local control API |
| **8765** | Cloud WebSocket | AI streaming |
| **5353** | mDNS | Robot discovery |

### Firewall Rules

```bash
# Robot (Raspberry Pi)
sudo ufw allow 8000/tcp  # FastAPI
sudo ufw allow 5353/udp  # mDNS

# Cloud Server
sudo ufw allow 8765/tcp  # WebSocket
sudo ufw allow 8000/tcp  # HTTP API
```

---

## 📊 Performance Metrics

### Latency

| Operation | Latency | Throughput |
|-----------|---------|------------|
| **Voice → Response** | 3-7s | 1 query/7s |
| **Vision Analysis** | 3-5s | 1 image/5s |
| **Face Recognition** | 2-3s | 1 face/3s |
| **Movement Command** | 50-100ms | 10 commands/s |
| **Navigation Loop** | 100ms | 10 Hz |
| **Depth Processing** | 33ms | 30 FPS |
| **Video Streaming** | 100-200ms | 15-30 FPS |

### Resource Usage

**Raspberry Pi 5:**
- CPU: 40-60%
- Memory: 500MB-1GB
- Temperature: 45-65°C

**Cloud Server (GPU PC):**
- CPU: 20-40%
- GPU: 30-50%
- Memory: 4-8GB

**Mobile App:**
- Memory: 100-200MB
- Battery: ~5% per hour

### Battery Life

| Usage | Runtime |
|-------|---------|
| **Idle** | ~4 hours |
| **Normal Use** | ~2 hours |
| **Heavy Use** (navigation + AI) | ~1.5 hours |

---

## 🔍 Debugging

### Common Issues

#### Robot Not Found

```bash
# Check robot is on
ping 192.168.1.100

# Check mDNS
avahi-browse -r -t _rovy._tcp

# Check firewall
sudo ufw status
```

#### WebSocket Connection Failed

```bash
# Check cloud server
curl http://cloud-ip:8765/health

# Test WebSocket
wscat -c ws://cloud-ip:8765

# Check Tailscale
tailscale status
```

#### Camera Not Detected

```bash
# Check USB connection
lsusb | grep 03e7  # OAK-D vendor ID

# Check permissions
ls -la /dev/bus/usb/*/

# Setup udev rules
sudo ./scripts/setup-oakd-udev.sh
```

#### Serial Port Error

```bash
# List ports
ls /dev/ttyACM*

# Check permissions
sudo chmod 666 /dev/ttyACM0

# Test connection
screen /dev/ttyACM0 115200
```

### Log Files

```bash
# Robot
sudo journalctl -u rovy.service -f
sudo journalctl -u rovy.service --since "1 hour ago"

# Cloud
tail -f cloud/logs/server.log

# Check disk space
df -h

# Check processes
ps aux | grep python
```

### Testing Commands

```bash
# Test robot API
curl http://192.168.1.100:8000/health

# Test cloud tools
cd cloud && python test_tools.py

# Test navigation
cd oakd_navigation && python navigation_controller.py

# Test rover
cd robot && python -c "from rover import Rover; r = Rover(); print(r.get_status())"
```

---

## 🛠️ Development Setup

### Prerequisites

**All Platforms:**
- Python 3.10+
- Node.js 18+
- Git

**Cloud Server:**
- NVIDIA GPU (optional, recommended)
- CUDA 11.8+ (if using GPU)
- OpenAI API key

**Robot:**
- Raspberry Pi 5 (4GB+ RAM)
- Raspberry Pi OS (64-bit)
- SSH access

### Installation

**Mobile App:**
```bash
cd mobile/
npm install
npx expo start
```

**Cloud Server:**
```bash
cd cloud/
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
python main.py
```

**Robot:**
```bash
cd robot/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl enable rovy
sudo systemctl start rovy
```

**Navigation:**
```bash
cd oakd_navigation/
pip install -r requirements.txt
python test_system.py
```

---

## 📚 File Structure

```
server/
├── mobile/                    # React Native mobile app
│   ├── app/                   # Screens (Expo Router)
│   ├── services/              # API clients
│   ├── components/            # Reusable UI
│   └── package.json
│
├── cloud/                     # Cloud AI server
│   ├── app/
│   │   ├── main.py           # FastAPI app (3330 lines)
│   │   ├── ai.py             # GPT-4o assistant
│   │   ├── vision.py         # Vision processing
│   │   ├── tools.py          # External APIs
│   │   ├── face_recognition_service.py
│   │   ├── gesture_detection.py
│   │   ├── meeting_service.py
│   │   ├── camera.py
│   │   └── rover_controller.py
│   ├── requirements.txt
│   └── start_cloud.sh
│
├── robot/                     # Raspberry Pi code
│   ├── main.py               # RovyClient (WebSocket)
│   ├── main_api.py           # Local FastAPI
│   ├── rover.py              # Rover controller
│   ├── wake_word_detector.py
│   ├── requirements.txt
│   └── rovy.service          # Systemd unit
│
├── oakd_navigation/           # Autonomous navigation
│   ├── navigation_controller.py
│   ├── depth_processor.py
│   ├── obstacle_avoidance.py
│   ├── path_planner.py
│   ├── config.yaml
│   └── requirements.txt
│
└── docs/                      # Documentation
    ├── ARCHITECTURE.md
    ├── SYSTEM_INTERACTIONS.md
    └── QUICK_REFERENCE.md
```

---

## 🔐 Security

### API Keys

```bash
# Set environment variables
export OPENAI_API_KEY="sk-..."
export SPOTIFY_CLIENT_ID="..."
export SPOTIFY_CLIENT_SECRET="..."

# Or use .env file
echo "OPENAI_API_KEY=sk-..." > cloud/.env
```

### Network Security

- Robot API: Local network only
- Cloud WebSocket: Tailscale VPN (encrypted)
- Mobile App: Secure storage (Expo SecureStore)
- Face Database: Local only, never uploaded

---

## 📖 API Quick Reference

### Robot HTTP API

```bash
# Health check
GET /health

# Robot status
GET /status

# Camera snapshot
GET /camera/snapshot

# Video stream
GET /camera/stream

# Move robot
POST /control/move
{
  "linear_velocity": 0.5,
  "angular_velocity": 0.0
}

# Gimbal control
POST /control/head
{
  "pan": 0,
  "tilt": 0
}
```

### ESP32 Serial Commands

```json
// Move
{"T": 1, "L": 0.5, "R": 0.5}

// Gimbal
{"T": 133, "X": 0, "Y": 0, "SPD": 200, "ACC": 10}

// Lights
{"T": 132, "IO4": 255, "IO5": 0}

// Display
{"T": 3, "lineNum": 0, "Text": "Hello"}

// Status
{"T": 130}
```

---

## 🎓 Learning Resources

### Key Concepts

1. **Distributed Systems**: Mobile, robot, cloud architecture
2. **WebSocket**: Real-time bidirectional communication
3. **Computer Vision**: Depth mapping, obstacle detection
4. **Path Planning**: A*, Dijkstra, RRT algorithms
5. **AI Integration**: GPT-4o, Whisper, face recognition
6. **Robotics**: Motor control, kinematics, sensors

### External Documentation

- **DepthAI**: https://docs.luxonis.com/
- **OpenAI**: https://platform.openai.com/docs
- **React Native**: https://reactnative.dev/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Expo**: https://docs.expo.dev/

---

## 🚨 Emergency Procedures

### Emergency Stop

**Voice**: "Stop" or "Emergency stop"

**Mobile App**: Red emergency stop button

**Physical**: Press robot power button

**Code**:
```python
rover.stop()
# or
navigation_controller.emergency_stop()
```

### System Reset

```bash
# Soft reset (restart services)
sudo systemctl restart rovy

# Hard reset (reboot)
sudo reboot

# Factory reset (re-run setup)
cd robot && ./setup.sh
```

---

## 💡 Pro Tips

1. **Battery**: Always charge when voltage < 9.5V
2. **Camera**: Clean lens regularly for best vision
3. **Navigation**: Test in open space first
4. **Voice**: Speak clearly, ~1m from microphone
5. **Updates**: Pull latest code regularly (`git pull`)
6. **Logs**: Check logs when troubleshooting
7. **Backups**: Save face database regularly
8. **Network**: Use 5GHz WiFi for better streaming
9. **Temperature**: Keep Pi cool (< 70°C)
10. **Practice**: Try voice commands often to improve

---

## 📞 Support

**Issues**: See project repository issues section  
**Documentation**: Check `ARCHITECTURE.md` for details  
**Community**: Capstone Design Team 2025-2

---

**Version**: 1.0  
**Last Updated**: December 2024  
**Quick Reference Guide** - Keep this handy!

