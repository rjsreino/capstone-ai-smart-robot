# How to Run ROVY Cloud Server

This guide shows all the ways to start the cloud server on your PC/Cloud machine.

---

## Prerequisites

1. **Setup cloud environment** (run once):
```bash
cd /home/rovy/rovy_client/cloud
./scripts/setup.sh
```

This creates the virtual environment at `cloud/.venv/` and installs dependencies.

---

## Running Options

### 🚀 Option 1: Quick Start (Recommended)

**For testing and development:**

```bash
cd /home/rovy/rovy_client/cloud
./start_cloud.sh
```

**What it does:**
- ✅ Starts REST API on port 8000 (for mobile app)
- ✅ Starts WebSocket server on port 8765 (for robot)
- ✅ Loads AI models (Qwen2-VL, Whisper, Piper)
- ✅ Shows logs in terminal
- ✅ Press Ctrl+C to stop

**You'll see:**
```
════════════════════════════════════════════════════════════════
                    ROVY CLOUD SERVER
              Unified AI + API + Robot Hub

Services:
• REST API (port 8000) - Mobile app connection
• WebSocket (port 8765) - Robot connection
• AI: LLM + Vision + Speech (local models)
════════════════════════════════════════════════════════════════

✅ WebSocket server running on ws://0.0.0.0:8765
✅ REST API running on http://0.0.0.0:8000
🤖 Robot connected: 172.30.1.99:xxxxx
```

---

### 🔧 Option 2: Direct Python

**Same as Option 1, just direct:**

```bash
cd /home/rovy/rovy_client/cloud
python3 main.py
```

---

### 🤖 Option 3: Systemd Service (Auto-start on boot)

**For production - runs in background, starts on boot:**

#### Install Service (one-time):
```bash
cd /home/rovy/rovy_client/cloud/scripts
sudo ./install-service.sh
```

#### Service Commands:
```bash
# Start the service
sudo systemctl start rovy-cloud.service

# Stop the service
sudo systemctl stop rovy-cloud.service

# Check status
sudo systemctl status rovy-cloud.service

# View logs (live)
sudo journalctl -u rovy-cloud.service -f

# View last 100 lines
sudo journalctl -u rovy-cloud.service -n 100

# Restart service
sudo systemctl restart rovy-cloud.service

# Disable auto-start
sudo systemctl disable rovy-cloud.service

# Enable auto-start
sudo systemctl enable rovy-cloud.service
```

---

## Port Information

The cloud server uses two ports:

| Port | Purpose | Protocol | Client |
|------|---------|----------|--------|
| **8000** | REST API | HTTP/WebSocket | Mobile App |
| **8765** | Robot Stream | WebSocket | Robot (Pi) |

**Firewall:** Make sure these ports are open if using a firewall.

---

## Checking If It's Running

### Check Ports:
```bash
# Check if ports are listening
sudo netstat -tulpn | grep -E ':(8000|8765)'

# Or with ss
ss -tulpn | grep -E ':(8000|8765)'
```

### Test WebSocket:
```bash
# Test robot WebSocket
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  http://localhost:8765/

# Test REST API
curl http://localhost:8000/health
```

---

## Troubleshooting

### Service won't start:
```bash
# Check logs for errors
sudo journalctl -u rovy-cloud.service -n 50 --no-pager

# Check service status
sudo systemctl status rovy-cloud.service
```

### Port already in use:
```bash
# Find what's using the port
sudo lsof -i :8000
sudo lsof -i :8765

# Kill the process if needed
sudo kill <PID>
```

### Python/dependency errors:
```bash
# Recreate virtual environment
cd /home/rovy/rovy_client/cloud
rm -rf .venv
./scripts/setup.sh
```

---

## Architecture

```
Mobile App ──HTTP/WS:8000──┐
                           │
                           v
                    ┌──────────────┐
                    │ Cloud Server │
                    │              │
                    │ REST API     │ :8000
                    │ WebSocket    │ :8765
                    │ AI Models    │
                    └──────┬───────┘
                           ^
                           │
Robot (Pi) ──WS:8765───────┘
```

---

## Summary

**Quick testing:** Use `./start_cloud.sh`  
**Production/Auto-start:** Install systemd service

The cloud server will:
1. Wait for robot connection on port 8765
2. Receive video/audio streams from robot
3. Process with AI (LLM, Vision, STT, TTS)
4. Send commands back to robot
5. Provide REST API for mobile app on port 8000

Once running, the robot on your Pi will automatically connect! 🚀

