#!/usr/bin/env python3
import sys
import os
import subprocess

def test_imports():
    print("=" * 60)
    print("         ZED LIVE VISION ASSISTANT DIAGNOSTIC SCRIPT")
    print("=" * 60)
    
    dependencies = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "pyaudio": "pyaudio",
        "sounddevice": "sounddevice",
        "soundfile": "soundfile",
        "whisper": "openai-whisper",
        "torch": "torch",
        "openwakeword": "openwakeword",
        "ultralytics": "ultralytics",
        "easyocr": "easyocr",
        "pygame": "pygame",
        "edge_tts": "edge-tts",
        "ollama": "ollama",
        "sentence_transformers": "sentence-transformers",
        "sklearn": "scikit-learn"
    }
    
    missing = []
    print("[1/4] Checking Python packages library imports...")
    for mod_name, pkg_name in dependencies.items():
        try:
            __import__(mod_name)
            print(f"  [SUCCESS] {mod_name} (from package {pkg_name}) is installed.")
        except ImportError as e:
            print(f"  [FAILED]  {mod_name} ({pkg_name}) is NOT installed. Error: {e}")
            missing.append(pkg_name)
            
    print("-" * 60)
    
    # Check PyTorch CUDA availability
    print("[2/4] Checking hardware acceleration (CUDA)...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [SUCCESS] CUDA is available. PyTorch is using device: {torch.cuda.get_device_name(0)}")
        else:
            print("  [INFO]    CUDA is NOT available. Running on CPU mode.")
    except Exception as e:
        print(f"  [WARNING] Could not determine PyTorch CUDA status: {e}")
        
    print("-" * 60)
    
    # Check Audio Input Devices
    print("[3/4] Checking Audio Input Devices...")
    audio_found = False
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        info = None
        try:
            info = pa.get_default_input_device_info()
            print(f"  [SUCCESS] Default Input Device: {info['name']} (Index: {info['index']})")
            audio_found = True
        except Exception:
            print("  [WARNING] No default input audio device found.")
            
        print("  Available recording input devices:")
        for i in range(pa.get_device_count()):
            try:
                device_info = pa.get_device_info_by_index(i)
                if device_info.get('maxInputChannels', 0) > 0:
                    print(f"    - Index {i}: {device_info['name']} (Channels: {device_info['maxInputChannels']})")
                    audio_found = True
            except Exception:
                continue
        pa.terminate()
    except Exception as e:
        print(f"  [FAILED]  Could not initialize PyAudio. Audio features might fail. Error: {e}")
        
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print("  Sounddevice query devices list:")
        for idx, dev in enumerate(devices):
            if dev.get('max_input_channels', 0) > 0:
                print(f"    - Index {idx}: {dev['name']} (Input Channels: {dev['max_input_channels']})")
                audio_found = True
    except Exception as e:
        print(f"  [FAILED]  Could not initialize Sounddevice. Wake-word loop will fail. Error: {e}")

    if not audio_found:
        print("  [ERROR]   No input audio/microphone devices detected! Voice assistant command queries will fail.")
        
    print("-" * 60)
    
    # Check Ollama status
    print("[4/4] Checking Ollama and 'phi3' model...")
    try:
        import urllib.request
        import json
        
        # Test connection to local Ollama server
        req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        if req.getcode() == 200:
            data = json.loads(req.read().decode('utf-8'))
            models = [m['name'] for m in data.get('models', [])]
            print("  [SUCCESS] Ollama service is running on http://localhost:11434")
            print(f"  Available models: {models}")
            
            if any("phi3" in m.lower() for m in models):
                print("  [SUCCESS] 'phi3' model is pulled and ready for query reasoning!")
            else:
                print("  [WARNING] 'phi3' model is NOT pulled. Please run: 'ollama pull phi3'")
        else:
            print("  [FAILED]  Ollama replied with status code:", req.getcode())
    except Exception as e:
        print("  [WARNING] Ollama server is not reachable on localhost:11434.")
        print("            Ensure Ollama is running and has 'phi3' pulled.")
        print(f"            Details: {e}")
        print("            (The system will fallback to rule-based responses if Ollama is offline.)")
        
    print("=" * 60)
    
    if missing:
        print("MISSING DEPENDENCIES FOUND!")
        print("To install all requirements, run:")
        print("pip install -r zed/requirements_assistant.txt")
        print("\nIf you are on Windows and see PyAudio installation errors, you can run:")
        print("pip install pipwin && pipwin install pyaudio")
        print("=" * 60)
        return False
    else:
        print("ALL PYTHON DEPENDENCIES MET! Ready to run the assistant.")
        print("=" * 60)
        return True

if __name__ == "__main__":
    test_imports()
