"""
Smart Assistant for Rover
Integrates ReSpeaker mic array, speech recognition, and dual model system:
- Gemma for text/chat tasks (fast, lightweight)
- Phi-Vision for image understanding
"""
import os
import sys
import queue
import threading
import time
import numpy as np
import cv2
from datetime import datetime
import subprocess
import json
import tempfile
import wave
from contextlib import contextmanager

# Import dual model assistant
from dual_model_assistant import DualModelAssistant

# Suppress ALSA warnings
os.environ['ALSA_CARD'] = 'ArrayUAC10'

@contextmanager
def suppress_alsa_warnings():
    """Temporarily suppress stderr to hide ALSA warnings."""
    stderr_fd = sys.stderr.fileno()
    with os.fdopen(os.dup(stderr_fd), 'w') as old_stderr:
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            try:
                yield
            finally:
                os.dup2(old_stderr.fileno(), stderr_fd)

# Speech recognition
try:
    import speech_recognition as sr
except ImportError:
    print("⚠️  speech_recognition not installed. Install with: pip install SpeechRecognition")
    sr = None

# Whisper for better speech recognition accuracy
WHISPER_AVAILABLE = False
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("ℹ️  Whisper not available. For better accuracy, install with: pip install openai-whisper")
    whisper = None

# Audio playback
try:
    import pyaudio
except ImportError:
    print("⚠️  pyaudio not installed. Install with: pip install pyaudio")
    pyaudio = None

# ReSpeaker LED control
try:
    from pixel_ring import pixel_ring
    PIXEL_RING_AVAILABLE = True
except ImportError:
    print("⚠️  pixel_ring not available. Visual feedback will be limited.")
    PIXEL_RING_AVAILABLE = False

# Wake word detection
WAKE_WORD_AVAILABLE = False
WakeWordModel = None
openwakeword = None
try:
    from openwakeword.model import Model as WakeWordModel
    import openwakeword
    WAKE_WORD_AVAILABLE = True
except (ImportError, ValueError, Exception) as e:
    # Catch ImportError, ValueError (numpy compatibility), and other exceptions
    print(f"⚠️  openwakeword not available ({type(e).__name__}: {e})")
    print("⚠️  Falling back to simple text-based wake word detection")
    WAKE_WORD_AVAILABLE = False
    WakeWordModel = None
    openwakeword = None


class ReSpeakerMicrophone(sr.Microphone):
    """Custom Microphone class for ReSpeaker - reads 6 channels, extracts channel 0 for mono."""
    
    def __init__(self, device_index=None, sample_rate=16000, chunk_size=1024):
        """Initialize with ReSpeaker-specific settings."""
        self.device_index = device_index
        self.format = 8  # paInt16
        self.SAMPLE_WIDTH = 2
        self.SAMPLE_RATE = sample_rate  # 16kHz is optimal for speech recognition
        self.CHUNK = chunk_size
        self.audio = None
        self.stream = None
        # ReSpeaker has 6 channels - read all, use beamforming to combine them
        self.CHANNELS = 6
    
    def __enter__(self):
        """Open the audio stream with 6 channels."""
        assert self.stream is None, "This audio source is already inside a context manager"
        
        with suppress_alsa_warnings():
            self.audio = pyaudio.PyAudio()
        
        try:
            with suppress_alsa_warnings():
                self.stream = self.audio.open(
                    input_device_index=self.device_index,
                    channels=self.CHANNELS,  # Read all 6 channels
                    format=self.format,
                    rate=self.SAMPLE_RATE,
                    frames_per_buffer=self.CHUNK,
                    input=True,
                    # Prevent overflow by allowing stream to drop old frames if buffer fills
                    stream_callback=None,
                    start=True
                )
            # Wrap stream to extract only channel 0
            self.stream.read = self._make_channel_0_reader(self.stream.read)
        except Exception as e:
            self.audio.terminate()
            raise e
        return self
    
    def _make_channel_0_reader(self, original_read):
        """Wrap the read function to use proper beamforming for better speech recognition."""
        def read_mono(size):
            try:
                # Read 6-channel data with overflow protection
                data = original_read(size, exception_on_overflow=False)
            except OSError as e:
                # Handle overflow by returning silence
                if e.errno in [-9981, -9988]:  # Input overflowed or stream closed
                    # Return silence (zeros) for this chunk
                    samples = size // self.SAMPLE_WIDTH
                    return np.zeros(samples, dtype=np.int16).tobytes()
                raise
            
            # Convert to numpy array (int16)
            audio_data = np.frombuffer(data, dtype=np.int16)
            # Reshape to (samples, 6 channels)
            audio_data = audio_data.reshape(-1, 6)
            
            # Use proper beamforming: average all channels for better signal-to-noise ratio
            # The ReSpeaker mic array has 6 mics in a circle - averaging them:
            # 1. Reduces noise (noise is random, speech is coherent)
            # 2. Improves signal quality (combines signals from all directions)
            # 3. Provides more consistent audio quality
            # Convert to float32 for better precision during averaging
            audio_float = audio_data.astype(np.float32)
            # Average all channels
            mono_float = np.mean(audio_float, axis=1)
            # Convert back to int16
            mono_data = mono_float.astype(np.int16)
            
            # Convert back to bytes
            return mono_data.tobytes()
        return read_mono
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Close the audio stream."""
        if self.stream is not None:
            try:
                # Check if stream is still active before stopping
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                # Stream may already be closed, ignore
                pass
            finally:
                self.stream = None
        if self.audio is not None:
            try:
                self.audio.terminate()
            except Exception:
                pass
            finally:
                self.audio = None


class ReSpeakerInterface:
    """Interface for ReSpeaker mic array with voice activity detection."""
    
    def __init__(self, device_index=None, use_whisper=True):
        """
        Initialize ReSpeaker microphone.
        
        Args:
            device_index: Audio device index (None = auto-detect)
            use_whisper: Use Whisper for better accuracy (if available)
        """
        if sr is None:
            raise RuntimeError("speech_recognition not installed")
        
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.device_index = device_index
        self.ambient_noise_adjusted = False  # Track if we've already adjusted for ambient noise
        self.use_whisper = use_whisper and WHISPER_AVAILABLE
        self.whisper_model = None
        
        # Load Whisper model if available and requested
        if self.use_whisper:
            try:
                print("[ReSpeaker] Loading Whisper tiny model for fast recognition...")
                # Use tiny model for fastest processing (4-5x faster than base)
                self.whisper_model = whisper.load_model("tiny")
                print("[ReSpeaker] ✅ Whisper tiny model loaded")
            except Exception as e:
                print(f"[ReSpeaker] ⚠️  Could not load Whisper: {e}, falling back to Google")
                self.use_whisper = False
                self.whisper_model = None
        
        # Try to find ReSpeaker device
        if device_index is None:
            self.device_index = self._find_respeaker()
            # If still not found, try device index 1 as last resort (common ReSpeaker location)
            if self.device_index is None:
                try:
                    with suppress_alsa_warnings():
                        p = pyaudio.PyAudio()
                        if p.get_device_count() > 1:
                            info = p.get_device_info_by_index(1)
                            max_input_channels = info.get('maxInputChannels', 0)
                            if max_input_channels >= 6:  # ReSpeaker has 6 channels
                                print(f"[ReSpeaker] Using device index 1 as last resort: {info.get('name')}")
                                self.device_index = 1
                        p.terminate()
                except Exception as e:
                    print(f"[ReSpeaker] Could not use device index 1: {e}")
        
        # Configure recognizer for optimal accuracy with ReSpeaker mic array
        # The ReSpeaker provides high-quality audio, so we can use more sensitive settings
        self.recognizer.energy_threshold = 200  # Sensitive but not too low
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.2  # Responsive adjustment
        self.recognizer.pause_threshold = 1.5  # Longer pause to capture multi-word phrases like "hey Rovy"
        self.recognizer.operation_timeout = None  # No timeout for operations
        self.recognizer.phrase_threshold = 0.1  # Very low - capture all speech clearly
        self.recognizer.non_speaking_duration = 0.8  # Longer padding to capture complete phrases
        # Store minimum threshold to prevent it from getting too low
        self.min_energy_threshold = 80  # Low minimum for high-quality mic
        
        print(f"[ReSpeaker] Initialized (device index: {self.device_index})")
        
        # Initialize LED ring if available
        self.led_available = False
        if PIXEL_RING_AVAILABLE:
            try:
                pixel_ring.set_brightness(10)
                pixel_ring.wakeup()  # Show default wakeup animation
                self.led_available = True
            except Exception as e:
                print(f"[ReSpeaker] LED ring init failed: {e}")
                print(f"[ReSpeaker] Continuing without LED feedback (run with sudo for LED access)")
        
        # Initialize tuning interface for DOA (Direction of Arrival)
        self.tuning = None
        self.doa_available = False
        try:
            import usb.core
            import usb.util
            # Find ReSpeaker device (VID:PID = 2886:0018 for ReSpeaker Mic Array v2.0)
            dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
            if dev:
                from tuning import Tuning
                self.tuning = Tuning(dev)
                
                # Enable aggressive noise suppression for cafe/noisy environments
                try:
                    # Increase noise suppression
                    self.tuning.write('GAMMA_NS', 3.0)  # More aggressive noise suppression (default: 1.0)
                    self.tuning.write('MIN_NS', 0.3)    # Higher noise floor threshold (default: 0.15)
                    # AGC to focus on closer voices
                    self.tuning.write('AGCDESIREDLEVEL', 0.01)  # Higher gain for closer voices
                    print("[ReSpeaker] ✅ Enhanced noise suppression for noisy environments")
                except Exception as e:
                    print(f"[ReSpeaker] ⚠️ Could not enhance noise suppression: {e}")
                
                self.doa_available = True
                print("[ReSpeaker] ✅ DOA (Direction of Arrival) enabled")
            else:
                print("[ReSpeaker] ⚠️  ReSpeaker USB device not found, DOA disabled")
        except ImportError:
            print("[ReSpeaker] ⚠️  'tuning' module not installed. DOA disabled.")
            print("[ReSpeaker]     Install with: pip install pyusb")
        except Exception as e:
            print(f"[ReSpeaker] ⚠️  Could not initialize DOA: {e}")
    
    def _find_respeaker(self):
        """Auto-detect ReSpeaker device index."""
        if pyaudio is None:
            return None
        
        try:
            with suppress_alsa_warnings():
                p = pyaudio.PyAudio()
                device_count = p.get_device_count()
                print(f"[ReSpeaker] Scanning {device_count} audio devices...")
                for i in range(device_count):
                    try:
                        info = p.get_device_info_by_index(i)
                        name = info.get('name', '').lower()
                        max_input_channels = info.get('maxInputChannels', 0)
                        # Look for ReSpeaker by name and check it has input channels
                        if ('respeaker' in name or 'seeed' in name) and max_input_channels > 0:
                            print(f"[ReSpeaker] Found device: {info.get('name')} at index {i} ({max_input_channels} input channels)")
                            p.terminate()
                            return i
                    except Exception as e:
                        print(f"[ReSpeaker] Error checking device {i}: {e}")
                        continue
                p.terminate()
        except Exception as e:
            print(f"[ReSpeaker] Error during device detection: {e}")
        
        # Fallback: Try device index 1 (common ReSpeaker location)
        try:
            with suppress_alsa_warnings():
                p = pyaudio.PyAudio()
                if p.get_device_count() > 1:
                    info = p.get_device_info_by_index(1)
                    name = info.get('name', '').lower()
                    max_input_channels = info.get('maxInputChannels', 0)
                    if 'respeaker' in name or 'seeed' in name:
                        print(f"[ReSpeaker] Using fallback device index 1: {info.get('name')}")
                        p.terminate()
                        return 1
                p.terminate()
        except Exception as e:
            print(f"[ReSpeaker] Fallback detection failed: {e}")
        
        print("[ReSpeaker] ReSpeaker device not found, using default microphone")
        return None
    
    def listen(self, timeout=None, phrase_time_limit=10):
        """
        Listen for speech and return transcribed text.
        
        Args:
            timeout: Maximum time to wait for speech to start (None = infinite)
            phrase_time_limit: Maximum duration of a phrase
            
        Returns:
            str: Transcribed text, or None if no speech detected
        """
        if self.microphone is None:
            # Use custom ReSpeaker microphone class that forces 1 channel (mono)
            self.microphone = ReSpeakerMicrophone(
                device_index=self.device_index,
                sample_rate=16000,
                chunk_size=1024
            )
        
        try:
            # LED: Listening - Blue color
            self.set_led('listen')
            
            print("[ReSpeaker] 🎤 Listening...")
            
            try:
                with self.microphone as source:
                    # Adjust for ambient noise - use shorter duration and cap threshold
                    # Re-calibrate periodically to adapt to changing noise conditions
                    if not self.ambient_noise_adjusted or (time.time() - getattr(self, '_last_noise_adjust', 0)) > 30:
                        print(f"[ReSpeaker] Adjusting for ambient noise (device index: {self.device_index})...")
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Shorter to avoid over-calibration
                        # Cap threshold to prevent it from being too high (which causes missed speech)
                        if hasattr(self, 'min_energy_threshold'):
                            self.recognizer.energy_threshold = min(self.recognizer.energy_threshold, 400)  # Cap at 400
                            self.recognizer.energy_threshold = max(self.recognizer.energy_threshold, self.min_energy_threshold)
                        print(f"[ReSpeaker] Energy threshold after adjustment: {self.recognizer.energy_threshold}")
                        self.ambient_noise_adjusted = True
                        self._last_noise_adjust = time.time()
                    else:
                        # Ensure minimum threshold is maintained and not too high
                        if hasattr(self, 'min_energy_threshold'):
                            self.recognizer.energy_threshold = min(self.recognizer.energy_threshold, 400)  # Cap at 400
                            self.recognizer.energy_threshold = max(self.recognizer.energy_threshold, self.min_energy_threshold)
                    
                    # Listen for audio with longer phrase_time_limit to capture complete phrases
                    # This is especially important for wake words like "hey Rovy"
                    # Use longer limit to ensure we capture multi-word phrases
                    effective_phrase_limit = max(phrase_time_limit, 8.0) if phrase_time_limit else 8.0
                    audio = self.recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=effective_phrase_limit
                    )
                    try:
                        audio_size = len(audio.get_raw_data())
                        print(f"[ReSpeaker] ✅ Audio captured successfully ({audio_size} bytes)")
                    except:
                        print(f"[ReSpeaker] ✅ Audio captured successfully")
            except OSError as e:
                # Microphone channel/configuration error
                print(f"[ReSpeaker] ❌ Microphone configuration error: {e}")
                print("[ReSpeaker] 💡 Tip: ReSpeaker may need specific ALSA configuration")
                self.set_led('off')
                return None
            
            # LED: Processing - Orange/Yellow color
            self.set_led('think')
            
            print("[ReSpeaker] 🔄 Processing audio...")
            
            # Try Whisper first for better accuracy, fallback to Google
            text = None
            
            # Use Whisper if available (more accurate)
            if self.use_whisper and self.whisper_model:
                try:
                    # Convert AudioData to numpy array for Whisper
                    # Get raw audio bytes and convert to float32 array normalized to [-1, 1]
                    raw_audio = audio.get_raw_data(convert_rate=16000, convert_width=2)
                    audio_data = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Whisper expects 16kHz sample rate (which we're using)
                    result = self.whisper_model.transcribe(
                        audio_data,
                        language="en",
                        task="transcribe",
                        fp16=False,  # Use FP32 on Jetson for compatibility
                        verbose=False
                    )
                    text = result["text"].strip()
                    print(f"[ReSpeaker] 📝 Heard (Whisper): '{text}'")
                except Exception as e:
                    print(f"[ReSpeaker] ⚠️  Whisper recognition failed: {e}, trying Google...")
                    import traceback
                    traceback.print_exc()
                    text = None
            
            # Fallback to Google Speech Recognition if Whisper failed or not available
            if text is None:
                try:
                    # Use language hint "en-US" for better English recognition
                    # Also specify show_all=False for cleaner results
                    text = self.recognizer.recognize_google(audio, language="en-US", show_all=False)
                    print(f"[ReSpeaker] 📝 Heard (Google): '{text}'")
                except sr.UnknownValueError:
                    print("[ReSpeaker] ❌ Could not understand audio")
                    self.set_led('off')
                    return None
                except sr.RequestError as e:
                    print(f"[ReSpeaker] ❌ Speech recognition service error: {e}")
                    # Try offline recognition as last resort
                    try:
                        text = self.recognizer.recognize_sphinx(audio)
                        print(f"[ReSpeaker] 📝 Heard (offline): '{text}'")
                    except:
                        self.set_led('off')
                        return None
            
            # LED: Turn off after recognition
            self.set_led('off')
            
            return text if text else None
        
        except sr.WaitTimeoutError:
            print("[ReSpeaker] ⏰ Listening timeout")
            self.set_led('off')
            return None
        
        except Exception as e:
            print(f"[ReSpeaker] ❌ Error: {e}")
            self.set_led('off')
            return None
    
    def set_led(self, mode):
        """
        Set LED ring mode using default ReSpeaker animations.
        
        Args:
            mode: 'off', 'listen', 'think', 'speak', or tuple (r, g, b) for custom color
        """
        if not PIXEL_RING_AVAILABLE or not self.led_available:
            return
        
        try:
            if mode == 'off':
                pixel_ring.off()
            elif mode == 'listen':
                # Use default listening animation (blue spinning)
                pixel_ring.listen()
            elif mode == 'think':
                # Use default thinking animation (pulsing)
                pixel_ring.think()
            elif mode == 'speak':
                # Use default speaking animation (green pulsing)
                pixel_ring.speak()
            elif isinstance(mode, tuple) and len(mode) == 3:
                # Custom RGB color
                r, g, b = mode
                pixel_ring.set_color(r=r, g=g, b=b)
            else:
                # Default to off for unknown modes
                pixel_ring.off()
        except Exception as e:
            # Silently fail if LED control doesn't work
            pass
    
    def get_voice_direction(self, listen_duration=2.0):
        """
        Get the direction (angle) of the voice source using DOA.
        Listens for speech and returns the average DOA during speaking.
        
        Args:
            listen_duration: How long to listen for voice (seconds)
        
        Returns:
            int: Angle in degrees (0-359) where sound is coming from, or None if unavailable.
                 0° = front, 90° = left, 180° = back, 270° = right
        """
        if not self.doa_available or not self.tuning:
            return None
        
        try:
            import time
            
            # Listen for audio and collect DOA values while speaking
            doa_samples = []
            start_time = time.time()
            
            print(f"[ReSpeaker] 🎤 Listening for voice ({listen_duration}s)...")
            
            while time.time() - start_time < listen_duration:
                # Get current DOA
                doa = self.tuning.direction
                
                # Collect all DOA samples (person just spoke, so voice is present)
                doa_samples.append(doa)
                
                time.sleep(0.05)  # Sample 20 times per second for better accuracy
            
            if doa_samples:
                # Average the DOA values (handle circular averaging for angles)
                # Convert to x,y coordinates, average, then back to angle
                import math
                x_sum = sum(math.cos(math.radians(d)) for d in doa_samples)
                y_sum = sum(math.sin(math.radians(d)) for d in doa_samples)
                avg_doa = int(math.degrees(math.atan2(y_sum, x_sum))) % 360
                
                print(f"[ReSpeaker] 🎯 Voice direction: {avg_doa}° (from {len(doa_samples)} samples)")
                return avg_doa
            else:
                print("[ReSpeaker] ⚠️  No voice detected during listening period")
                return None
                
        except Exception as e:
            print(f"[ReSpeaker] ⚠️  Could not get DOA: {e}")
            return None
    
    def doa_to_servo_angles(self, doa_angle, current_pan=0, tilt_angle=0):
        """
        Convert DOA angle to camera servo pan/tilt angles.
        
        Args:
            doa_angle: DOA angle from ReSpeaker (0-359°)
                      0° = front, 90° = left, 180° = back, 270° = right
            current_pan: Current pan position (-180 to 180, 0=center)
            tilt_angle: Desired tilt angle (-30 to 90, 0=forward, positive=up)
        
        Returns:
            dict: {'pan': pan_angle, 'tilt': tilt_angle} for gimbal_ctrl
                  pan: -180 to 180 (0=center, negative=left, positive=right)
                  tilt: -30 to 90 (0=forward looking, positive=up)
        """
        if doa_angle is None:
            return None
        
        # ReSpeaker DOA: 0° = front, 90° = left, 180° = back, 270° = right (counterclockwise)
        # Gimbal pan: 0° = center, negative=left, positive=right
        
        # IMPORTANT: Microphone is physically mounted backwards, so we need to invert!
        # Convert DOA to pan angle, then negate to compensate for backwards mounting
        
        if doa_angle <= 180:
            # Left side: DOA 0-180° 
            pan = -doa_angle  # Negate for backwards mic
        else:
            # Right side: DOA 180-360°
            pan = -(doa_angle - 360)  # Maps 270° -> +90°, then negate -> -90°
        
        # Use multiple angle positions for better accuracy
        # Valid pan angles: -180, -135, -90, -45, 0, 45, 90, 135, 180
        
        # Snap to working angle with tighter thresholds for better tracking
        # Tighten CENTER zone to ±10° so small angles still trigger movement
        if pan < -157.5:  # -180° to -157.5°
            pan_angle, pan_desc = -180, "FAR LEFT"
        elif pan < -95:  # -157.5° to -95° (adjusted for better left coverage)
            pan_angle, pan_desc = -135, "BACK LEFT"
        elif pan < -67.5:  # -95° to -67.5°
            pan_angle, pan_desc = -90, "LEFT"
        elif pan < -22.5:  # -67.5° to -22.5°
            pan_angle, pan_desc = -45, "FRONT LEFT"
        elif pan < 10:  # -22.5° to 10° (tighter center threshold)
            pan_angle, pan_desc = 0, "CENTER"
        elif pan < 67.5:  # 10° to 67.5° (now includes 18°)
            pan_angle, pan_desc = 45, "FRONT RIGHT"
        elif pan < 95:  # 67.5° to 95° (symmetric)
            pan_angle, pan_desc = 90, "RIGHT"
        elif pan < 157.5:  # 95° to 157.5°
            pan_angle, pan_desc = 135, "BACK RIGHT"
        else:  # 157.5° to 180°
            pan_angle, pan_desc = 180, "FAR RIGHT"
        
        # Tilt: slightly up (30°) to catch person's face better
        # Valid tilt range: -30 (down) to 90 (up), 0=forward
        tilt_angle = 30  # Look slightly up for better person detection
        
        print(f"[ReSpeaker] DOA {doa_angle}° (calculated pan={pan:.0f}°) -> Gimbal: pan={pan_angle}° ({pan_desc}), tilt={tilt_angle}°")
        
        return {'pan': pan_angle, 'tilt': tilt_angle}


class WakeWordDetector:
    """Wake word detection using openwakeword for continuous listening."""
    
    def __init__(self, device_index=None, sample_rate=16000, wake_words=None):
        """
        Initialize wake word detector.
        
        Args:
            device_index: Audio device index (None = auto-detect)
            sample_rate: Audio sample rate (16000 Hz recommended)
            wake_words: List of wake words to detect. If None, uses default "hey Rovy" and "Rovy"
        """
        if not WAKE_WORD_AVAILABLE:
            raise RuntimeError("openwakeword not available. Install with: pip install openwakeword")
        
        if pyaudio is None:
            raise RuntimeError("pyaudio not available")
        
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * 0.03)  # 30ms chunks for openwakeword
        
        # Default wake words: "Hey Rovy" and "Rovy"
        if wake_words is None:
            wake_words = ["hey Rovy", "Rovy"]
        self.wake_words = [w.lower() for w in wake_words]
        
        # Download models if needed (only if openwakeword is available)
        if openwakeword is not None:
            try:
                openwakeword.utils.download_models()
            except Exception as e:
                print(f"[WakeWord] ⚠️  Could not download models: {e}")
                print("[WakeWord] Models may already be downloaded")
        
        # Initialize openwakeword model
        # Use custom wake words if available, otherwise use built-in models
        # openwakeword has built-in models, but we'll use text-based matching for custom words
        # For now, we'll use a hybrid approach: continuous listening with VAD + text matching
        self.model = None
        self.audio = None
        self.stream = None
        
        # Try to find ReSpeaker device if not specified
        if device_index is None:
            self.device_index = self._find_respeaker()
        
        print(f"[WakeWord] Initialized (device index: {self.device_index}, wake words: {self.wake_words})")
    
    def _find_respeaker(self):
        """Auto-detect ReSpeaker device index."""
        try:
            with suppress_alsa_warnings():
                p = pyaudio.PyAudio()
                device_count = p.get_device_count()
                for i in range(device_count):
                    try:
                        info = p.get_device_info_by_index(i)
                        name = info.get('name', '').lower()
                        max_input_channels = info.get('maxInputChannels', 0)
                        if ('respeaker' in name or 'seeed' in name) and max_input_channels > 0:
                            p.terminate()
                            return i
                    except:
                        continue
                p.terminate()
        except Exception as e:
            print(f"[WakeWord] Error during device detection: {e}")
        return None
    
    def listen_for_wake_word(self, timeout=None, callback=None):
        """
        Continuously listen for wake word.
        
        Args:
            timeout: Maximum time to listen (None = infinite)
            callback: Optional callback function when wake word is detected
        
        Returns:
            bool: True if wake word detected, False if timeout
        """
        if self.audio is None:
            with suppress_alsa_warnings():
                self.audio = pyaudio.PyAudio()
        
        # Use ReSpeaker microphone for audio input
        microphone = ReSpeakerMicrophone(
            device_index=self.device_index,
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size
        )
        
        start_time = time.time()
        
        try:
            with microphone as mic:
                print(f"[WakeWord] 👂 Listening for wake words: {', '.join(self.wake_words)}...")
                
                # Use speech recognition with continuous listening
                # Optimized settings for better accuracy
                recognizer = sr.Recognizer()
                recognizer.energy_threshold = 250  # Lower for better sensitivity
                recognizer.dynamic_energy_threshold = True
                recognizer.dynamic_energy_adjustment_damping = 0.25
                recognizer.pause_threshold = 1.5  # Longer pause to capture multi-word phrases like "hey Rovy"
                recognizer.phrase_threshold = 0.1  # Very low to capture all speech
                recognizer.non_speaking_duration = 0.8  # Longer padding for complete phrases
                min_energy_threshold = 100  # Lower minimum for better sensitivity
                
                # Adjust for ambient noise - use shorter duration to avoid setting threshold too high
                # Shorter duration prevents over-calibration that can miss speech
                recognizer.adjust_for_ambient_noise(mic, duration=0.5)  # Shorter to avoid over-calibration
                # Cap the threshold to prevent it from being too high (which causes missed speech)
                recognizer.energy_threshold = min(recognizer.energy_threshold, 400)  # Cap at 400
                recognizer.energy_threshold = max(recognizer.energy_threshold, min_energy_threshold)
                print(f"[WakeWord] Energy threshold set to: {recognizer.energy_threshold}")
                
                while True:
                    # Check timeout
                    if timeout and (time.time() - start_time) > timeout:
                        return False
                    
                    try:
                        # Listen for audio chunks with longer timeout and phrase limit for complete wake words
                        # Longer phrase_time_limit ensures we capture the full "hey Rovy" phrase
                        audio = recognizer.listen(mic, timeout=2.0, phrase_time_limit=6.0)
                        
                        # Recognize speech with language hint for better accuracy
                        try:
                            text = recognizer.recognize_google(audio, language="en-US", show_all=False).lower()
                            print(f"[WakeWord] Heard: '{text}'")
                            
                            # Check for wake words
                            for wake_word in self.wake_words:
                                if wake_word in text:
                                    print(f"[WakeWord] ✅ Wake word detected: '{wake_word}'")
                                    if callback:
                                        callback(wake_word)
                                    return True
                        except sr.UnknownValueError:
                            # No speech detected, continue listening
                            continue
                        except sr.RequestError as e:
                            print(f"[WakeWord] ⚠️  Recognition error: {e}")
                            continue
                    
                    except sr.WaitTimeoutError:
                        # Timeout waiting for speech, continue listening
                        continue
                    except OSError as e:
                        # Handle audio stream errors (overflow, stream closed, etc.)
                        if e.errno == -9981:  # Input overflowed
                            print(f"[WakeWord] ⚠️  Audio buffer overflowed, recovering...")
                            # Stream is likely broken, exit and let it be recreated
                            return False
                        elif e.errno == -9988:  # Stream closed
                            print(f"[WakeWord] ⚠️  Audio stream closed unexpectedly")
                            return False
                        else:
                            print(f"[WakeWord] ⚠️  Audio error: {e}")
                            return False
                    except Exception as e:
                        print(f"[WakeWord] ⚠️  Error: {e}")
                        time.sleep(0.1)
                        continue
        
        except KeyboardInterrupt:
            print("[WakeWord] Interrupted")
            return False
        finally:
            if self.audio:
                self.audio.terminate()
                self.audio = None


class LLaVaAssistant:
    """Interface to LLaVa vision-language model via llama.cpp."""
    
    def __init__(self, model_path=None, mmproj_path=None):
        """
        Initialize LLaVa assistant.
        
        Args:
            model_path: Path to LLaVa model (GGUF format)
            mmproj_path: Path to multimodal projector
        """
        # Auto-detect common paths if not specified
        if model_path is None:
            possible_paths = [
                "/home/jetson/models/llava-v1.5-7b-Q4_K_M.gguf",
                "/home/jetson/llava/models/llava-v1.5-7b-Q4_K_M.gguf",
                "./models/llava-v1.5-7b-Q4_K_M.gguf",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if mmproj_path is None:
            possible_paths = [
                "/home/jetson/models/mmproj-model-f16.gguf",
                "/home/jetson/llava/models/mmproj-model-f16.gguf",
                "./models/mmproj-model-f16.gguf",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    mmproj_path = path
                    break
        
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.llm = None
        self.chat_handler = None
        self.use_cpp_python = False
        
        # Try to use llama-cpp-python first (better API, like in depth_llava_nav.py)
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            from PIL import Image
            self.Llama = Llama
            self.Llava15ChatHandler = Llava15ChatHandler
            self.PIL_Image = Image
            self.use_cpp_python = True
            print("[LLaVa] Using llama-cpp-python API (recommended)")
        except Exception as e:
            print(f"[LLaVa] ⚠️  llama-cpp-python not available, falling back to CLI")
            self.llava_cli = self._find_llava_cli()
            self.use_cpp_python = False
        
        print(f"[LLaVa] Model: {model_path}")
        print(f"[LLaVa] MMProj: {mmproj_path}")
        
        # Initialize model if using Python API
        if self.use_cpp_python and model_path and mmproj_path:
            self._init_cpp_python()
        elif not self.use_cpp_python:
            if not self.llava_cli:
                print("⚠️  Warning: llama.cpp llava-cli not found!")
                print("    Looking for: llava-cli, llama-llava-cli, or ./llama.cpp/build/bin/llava-cli")
            print(f"[LLaVa] CLI: {self.llava_cli}")
    
    def _init_cpp_python(self):
        """Initialize llama-cpp-python with LLaVa support."""
        try:
            print("[LLaVa] Loading model with llama-cpp-python...")
            
            # Use standard LLaVA 1.5 handler
            print("[LLaVa] Using LLaVA 1.5 model...")
            self.chat_handler = self.Llava15ChatHandler(clip_model_path=self.mmproj_path)
            
            # Load model with minimal GPU usage to avoid OOM with other services running
            # API server, face recognition, and camera already use GPU memory
            try:
                self.llm = self.Llama(
                    model_path=self.model_path,
                    chat_handler=self.chat_handler,
                    n_gpu_layers=8,  # Only 8 layers on GPU to leave room for API server + face recognition + CLIP
                    n_ctx=1024,  # Balanced: enough for vision (608 needed) but not too much for Jetson memory
                    logits_all=True,
                    verbose=False,
                    n_threads=8,  # More CPU threads since mostly CPU inference
                    n_batch=256  # Larger batch for better CPU performance
                )
                print("[LLaVa] ✅ Model loaded (8 GPU layers, rest on CPU)")
            except Exception as load_err:
                # Try CPU-only if GPU still fails
                print(f"[LLaVa] ⚠️  GPU load failed: {load_err}")
                print("[LLaVa] Retrying with CPU-only mode...")
                self.llm = self.Llama(
                    model_path=self.model_path,
                    chat_handler=self.chat_handler,
                    n_gpu_layers=0,  # CPU only
                    n_ctx=1024,
                    logits_all=True,
                    verbose=False,
                    n_threads=8,
                    n_batch=512
                )
                print("[LLaVa] ✅ Model loaded on CPU!")
        except Exception as e:
            print(f"[LLaVa] ❌ Failed to load with llama-cpp-python: {e}")
            import traceback
            traceback.print_exc()
            print("[LLaVa] Falling back to CLI mode")
            self.use_cpp_python = False
            self.llava_cli = self._find_llava_cli()
    
    def _ask_with_cpp_python(self, question, image, max_tokens, temperature, realtime_context):
        """Ask question using llama-cpp-python API (like in llava_cpp_navigator.py)."""
        try:
            # Convert OpenCV image (numpy array) to PIL Image
            if isinstance(image, np.ndarray):
                # Use full camera resolution for maximum detail (640x480)
                # No resizing - let CLIP encoder handle it internally
                h, w = image.shape[:2]
                print(f"[LLaVa] Using full resolution image: {w}x{h}")
                
                # OpenCV uses BGR, PIL uses RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = self.PIL_Image.fromarray(image_rgb)
            else:
                pil_image = image
            
            # Convert to base64 data URI (like in llava_cpp_navigator.py)
            import io
            import base64
            buffered = io.BytesIO()
            # Use lower JPEG quality for faster encoding and less memory (was 75)
            pil_image.save(buffered, format="JPEG", quality=60, optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            data_uri = f"data:image/jpeg;base64,{img_str}"
            
            # Simple, direct prompt
            prompt_text = question
            
            # Query model (like in llava_cpp_navigator.py)
            print(f"[LLaVa] Running inference with llama-cpp-python (mode: vision)...")
            import time
            start_time = time.time()
            
            # Allow longer responses for better descriptions with high-res images
            vision_max_tokens = min(max_tokens, 80)  # 80 tokens for detailed descriptions
            
            # Memory safety: wrap the actual model call in try-except to catch segfaults
            try:
                response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": f"Describe this image in detail: {prompt_text}"}
                        ]
                    }
                ],
                    temperature=0.7,
                    max_tokens=vision_max_tokens,
                    top_p=0.9,
                    repeat_penalty=1.0,
                    stop=None  # Don't stop early
                )
            except (RuntimeError, MemoryError, ValueError) as mem_err:
                print(f"[LLaVa] ❌ Vision inference failed (memory/runtime error): {mem_err}")
                print("[LLaVa] 💡 Try: 1) Restart the program, 2) Close other apps to free memory")
                return "Vision processing failed due to memory constraints. Please try again."
            
            elapsed = time.time() - start_time
            
            # Extract response
            answer = response['choices'][0]['message']['content']
            
            # Clean up response - remove artifacts and weird prefixes
            answer = answer.replace('#', '').strip()
            answer = answer.replace("[end of text]", "").strip()
            answer = answer.replace("[End of text]", "").strip()
            answer = answer.replace("</s>", "").strip()
            
            # Clean up weird prefixes/artifacts
            import re
            answer = re.sub(r'^\d+st image:\s*', '', answer)
            answer = re.sub(r'^\d+nd image:\s*', '', answer)
            answer = re.sub(r'^Förstesicht:\s*', '', answer)
            answer = re.sub(r'^(First|Second|Third|Image|Picture):\s*', '', answer, flags=re.IGNORECASE)
            answer = " ".join(answer.split())
            
            print(f"[LLaVa] ⚡ Inference completed in {elapsed:.1f}s")
            print(f"[LLaVa] Response: {answer[:100]}...")
            
            return answer if answer else "I'm not sure how to answer that."
            
        except Exception as e:
            print(f"[LLaVa] ❌ Error with llama-cpp-python: {e}")
            import traceback
            traceback.print_exc()
            # Provide more helpful error message
            if "segmentation fault" in str(e).lower() or "core dumped" in str(e).lower():
                return "Vision system crashed. Please restart the program."
            return f"Error: {str(e)}"
    
    def _ask_text_only_cpp_python(self, question, max_tokens, temperature, realtime_context):
        """Ask text-only question using llama-cpp-python API."""
        try:
            # Only include real-time context if explicitly asked about time/date
            question_lower = question.lower()
            needs_time_info = any(phrase in question_lower for phrase in [
                'what time', 'what is the time', 'time is it', 'current time',
                'what date', 'what is the date', 'today\'s date', 'current date'
            ])
            
            realtime_section = ""
            if needs_time_info and realtime_context:
                realtime_section = f"\n\nReal-time information: {realtime_context}"
            
            # Simple, direct prompt
            prompt_text = question
            
            print(f"[LLaVa] Running inference with llama-cpp-python (mode: text-only)...")
            import time
            start_time = time.time()
            
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": f"You are Rovy, a helpful robot assistant.{realtime_section}\n\nGive direct, accurate answers. Be concise but complete. Always respond in English."
                    },
                    {
                        "role": "user",
                        "content": prompt_text
                    }
                ],
                temperature=0.7,  # Normal temperature for natural responses
                max_tokens=max_tokens,
                top_p=0.9,  # Higher for better quality
                repeat_penalty=1.0  # No repeat penalty
            )
            
            elapsed = time.time() - start_time
            
            # Extract response
            answer = response['choices'][0]['message']['content']
            
            # Clean up response
            answer = answer.replace('#', '').strip()
            answer = answer.replace("[end of text]", "").strip()
            answer = answer.replace("[End of text]", "").strip()
            answer = answer.replace("</s>", "").strip()
            answer = " ".join(answer.split())
            
            print(f"[LLaVa] ⚡ Inference completed in {elapsed:.1f}s")
            print(f"[LLaVa] Response: {answer[:100]}...")
            
            return answer if answer else "I'm not sure how to answer that."
            
        except Exception as e:
            print(f"[LLaVa] ❌ Error with llama-cpp-python (text-only): {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def _find_llava_cli(self):
        """Find llama.cpp llava executable."""
        possible_names = [
            "/home/jetson/llama.cpp/build/bin/llama-mtmd-cli",  # Multimodal (preferred, but may have mmproj compatibility issues)
            "/home/jetson/llama.cpp/build/bin/llama-llava-cli",  # LLaVa CLI (deprecated but may work better with current mmproj)
            "/home/jetson/llama.cpp/build/bin/llama-cli",  # Text-only mode (fallback)
            "llama-mtmd-cli",
            "llama-llava-cli",
            "llama-cli",
            "/usr/local/bin/llama-cli",
        ]
        
        for name in possible_names:
            try:
                # Check if command exists
                if name.startswith("/"):
                    # Absolute path - check if file exists
                    if os.path.exists(name):
                        return name
                else:
                    # Command name - use which
                    result = subprocess.run(
                        ["which", name],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        return name
            except:
                pass
        
        return None
    
    def ask(self, question, image=None, max_tokens=512, temperature=0.7, realtime_context=None):
        """
        Ask LLaVa a question, optionally with an image.
        
        Args:
            question: Question text
            image: OpenCV image (numpy array), or None for text-only
            max_tokens: Maximum response length
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            realtime_context: Optional string with real-time info (time, weather, etc.)
            
        Returns:
            str: Model's response
        """
        if not self.model_path:
            return "Error: LLaVa not properly configured"
        
        print(f"[LLaVa] Question: {question}")
        
        # Determine if we're using text-only or multimodal mode
        use_vision = image is not None and self.mmproj_path is not None
        
        # Use llama-cpp-python if available (preferred method)
        if self.use_cpp_python and self.llm:
            if use_vision:
                return self._ask_with_cpp_python(question, image, max_tokens, temperature, realtime_context)
            else:
                # Text-only mode with llama-cpp-python
                return self._ask_text_only_cpp_python(question, max_tokens, temperature, realtime_context)
        
        # Fallback to CLI mode
        if not hasattr(self, 'llava_cli') or not self.llava_cli:
            return "Error: LLaVa not properly configured (no CLI available)"
        
        # Check if CLI supports vision (llava-cli or mtmd-cli)
        is_vision_cli = "llava" in self.llava_cli or "mtmd" in self.llava_cli
        
        # Save image to temporary file if provided
        image_path = None
        if use_vision:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                image_path = tmp.name
                cv2.imwrite(image_path, image)
        
        try:
            # Build prompt with real-time context - let LLM understand naturally
            # Add real-time context if provided (includes time, weather, etc.)
            realtime_section = ""
            if realtime_context:
                realtime_section = f"\n\nReal-time information available:\n{realtime_context}"
            
            # Build prompt - different for vision vs text questions
            if use_vision:
                # For vision questions, explicitly ask to describe what's in the image
                system_context = f"""You are Rovy, a robot assistant with a camera.{realtime_section}

When asked what you see, describe what is actually visible in the image. Be specific and concise. You MUST respond ONLY in English language. Never use Chinese or any other language."""
            else:
                # For text questions, keep it concise but complete
                system_context = f"""You are Rovy, a helpful robot assistant.{realtime_section}

Give direct, accurate answers. Keep responses concise but complete. You MUST respond ONLY in English language. Never use Chinese or any other language."""
            
            # For vision, use proper format
            if use_vision and is_vision_cli:
                prompt = f"{system_context}\n\nUSER: <image>\n{question}\nASSISTANT:"
            else:
                prompt = f"{system_context}\n\nUSER: {question}\nASSISTANT:"
            
            # Build command - Optimized for Jetson Orin unified memory
            # Jetson Orin has 16GB unified memory (CPU+GPU shared)
            # Increased context to 384 to fit longer prompts (was 256, but prompt was 265 tokens)
            cmd = [
                self.llava_cli,
                "-m", self.model_path,
                "-p", prompt,
                "-n", str(max_tokens),
                "--temp", str(temperature),
                "-ngl", "20",  # 20 GPU layers (works well, leaves room for other processes)
                "-c", "256" if use_vision else "128",   # More context for vision (needs more for image descriptions)
                "-b", "256",   # Larger batch size for faster inference (works with small context)
                "--n-predict", str(max_tokens),  # Hard limit
            ]
            
            # Add multimodal components for vision-capable CLI
            if is_vision_cli:
                # Check for mtmd first (since llama-mtmd-cli contains "llava" in its name)
                if "mtmd" in self.llava_cli:
                    # llama-mtmd-cli ALWAYS requires --mmproj (even for text-only)
                    cmd.extend([
                        "--mmproj", self.mmproj_path,
                    ])
                    # Add image if provided
                    if use_vision and image_path:
                        cmd.extend([
                            "--image", image_path,
                            "--chat-template", "vicuna",
                        ])
                elif "llava" in self.llava_cli:
                    # llama-llava-cli uses --mmproj and --image flags
                    if use_vision and image_path:
                        cmd.extend([
                            "--mmproj", self.mmproj_path,
                            "--image", image_path,
                        ])
                
                if use_vision and image_path:
                    print(f"[LLaVa] Using image: {image_path}")
                    print(f"[LLaVa] Image size: {os.path.getsize(image_path) if image_path and os.path.exists(image_path) else 'N/A'} bytes")
            
            print(f"[LLaVa] Running inference (mode: {'vision' if use_vision else 'text-only'})...")
            
            # Run inference with timing
            import time
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                # If llama-mtmd-cli failed and we're using vision, try llama-llava-cli as fallback
                if use_vision and "mtmd" in self.llava_cli and "unknown projector type" in result.stderr:
                    print(f"[LLaVa] ⚠️  llama-mtmd-cli failed with mmproj, trying llama-llava-cli as fallback...")
                    llava_cli_fallback = self.llava_cli.replace("mtmd", "llava")
                    if os.path.exists(llava_cli_fallback):
                        # Retry with llama-llava-cli
                        cmd_fallback = cmd.copy()
                        cmd_fallback[0] = llava_cli_fallback
                        # Remove --chat-template for llama-llava-cli
                        if "--chat-template" in cmd_fallback:
                            idx = cmd_fallback.index("--chat-template")
                            cmd_fallback.pop(idx)  # Remove --chat-template
                            cmd_fallback.pop(idx)  # Remove "vicuna"
                        
                        print(f"[LLaVa] Retrying with: {llava_cli_fallback}")
                        result = subprocess.run(
                            cmd_fallback,
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        if result.returncode == 0:
                            # Success with fallback
                            elapsed = time.time() - start_time
                            print(f"[LLaVa] ⚡ Inference completed in {elapsed:.1f}s (using llama-llava-cli fallback)")
                            response = result.stdout.strip()
                            if "ASSISTANT:" in response:
                                response = response.split("ASSISTANT:")[-1].strip()
                            if "USER:" in response:
                                response = response.split("USER:")[0].strip()
                            response = response.replace("[end of text]", "").strip()
                            response = response.replace("[End of text]", "").strip()
                            response = response.replace("</s>", "").strip()
                            response = " ".join(response.split())
                            print(f"[LLaVa] Response: {response[:100]}...")
                            return response if response else "I'm not sure how to answer that."
                
                print(f"[LLaVa] ❌ Command failed with return code {result.returncode}")
                print(f"[LLaVa] Full stderr: {result.stderr}")
                print(f"[LLaVa] Full stdout: {result.stdout}")
                # Try to extract meaningful error message
                error_msg = "I encountered an error processing your request."
                if "error" in result.stderr.lower() or "failed" in result.stderr.lower():
                    # Extract the actual error line
                    error_lines = [line for line in result.stderr.split('\n') if 'error' in line.lower() or 'failed' in line.lower()]
                    if error_lines:
                        error_msg = f"I encountered an error: {error_lines[0][:100]}"
                return error_msg
            
            print(f"[LLaVa] ⚡ Inference completed in {elapsed:.1f}s")
            
            # Parse output
            response = result.stdout.strip()
            
            # Clean up response
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            if "USER:" in response:
                response = response.split("USER:")[0].strip()
            
            # Remove prompt echo if present
            if question in response:
                response = response.replace(question, "").strip()
            
            # Remove common artifacts
            response = response.replace("[end of text]", "").strip()
            response = response.replace("[End of text]", "").strip()
            response = response.replace("</s>", "").strip()
            
            # Clean up extra whitespace
            response = " ".join(response.split())
            
            print(f"[LLaVa] Response: {response[:100]}...")
            return response if response else "I'm not sure how to answer that."
        
        except subprocess.TimeoutExpired:
            print("[LLaVa] ⏰ Inference timeout")
            return "Sorry, that took too long to process."
        
        except Exception as e:
            print(f"[LLaVa] ❌ Error: {e}")
            return f"Error: {str(e)}"
        
        finally:
            # Clean up temporary image
            if image_path and os.path.exists(image_path):
                try:
                    os.unlink(image_path)
                except:
                    pass


class TextToSpeech:
    """Simple text-to-speech using espeak or piper, with print-only fallback."""
    
    def __init__(self, engine='espeak', print_only=False):
        """
        Initialize TTS engine.
        
        Args:
            engine: 'espeak' (fast, robotic), 'piper' (natural, slower), or 'print' (text only)
            print_only: If True, only print text instead of speaking
        """
        self.print_only = print_only
        self.engine = engine
        self.piper_voice = None
        self.audio_device = None  # Will be auto-detected
        
        if print_only or engine == 'print':
            print("[TTS] Using print-only mode (no speakers)")
            self.engine = 'print'
            return
        
        # Auto-detect audio output device
        self._detect_audio_device()
        
        # Check if engine is available
        if engine == 'espeak':
            try:
                subprocess.run(['espeak', '--version'], capture_output=True, timeout=2)
                print("[TTS] Using espeak")
            except:
                print("⚠️  espeak not found or no speakers - using print-only mode")
                self.engine = 'print'
        elif engine == 'piper':
            try:
                from piper import PiperVoice
                # Load HFC MALE MEDIUM - clear, professional male voice for robot assistant
                print("[TTS] Loading professional voice (HFC Male Medium)...")
                
                # Priority order: HFC Male as default, then fallbacks
                voice_options = [
                    ("~/.local/share/piper-voices/en_US-hfc_male-medium.onnx", "HFC Male Medium (professional)"),
                    ("~/.local/share/piper-voices/en_US-danny-low.onnx", "Danny Low (most natural)"),
                    ("~/.local/share/piper-voices/en_US-ryan-high.onnx", "Ryan High"),
                    ("~/.local/share/piper-voices/en_US-libritts_r-medium.onnx", "LibriTTS_r Medium (improved)"),
                ]
                
                voice_path = None
                voice_name = None
                for path, name in voice_options:
                    full_path = os.path.expanduser(path)
                    if os.path.exists(full_path):
                        voice_path = full_path
                        voice_name = name
                        break
                
                if not voice_path:
                    print(f"⚠️  No Piper voices found")
                    print("⚠️  Falling back to espeak")
                    self.engine = 'espeak'
                    return
                
                print(f"[TTS] Using {voice_name}")
                
                self.piper_voice = PiperVoice.load(
                    voice_path,
                    use_cuda=False  # Use CPU on Jetson
                )
                print("[TTS] ✅ Natural human voice ready!")
            except Exception as e:
                print(f"⚠️  Piper not available ({e}). Falling back to espeak")
                self.engine = 'espeak'
    
    def _detect_audio_device(self):
        """Auto-detect USB audio output device."""
        try:
            # List all audio devices
            result = subprocess.run(
                ['aplay', '-l'],
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if result.returncode != 0:
                print("[TTS] ⚠️  Could not list audio devices")
                self.audio_device = None
                return
            
            # Look for USB audio devices (excluding ReSpeaker which is input)
            lines = result.stdout.split('\n')
            for line in lines:
                # Look for USB audio devices that are not ReSpeaker
                if 'card' in line and 'USB Audio' in line:
                    # Extract card number - exclude ReSpeaker/ArrayUAC (those are input mics)
                    if 'ReSpeaker' not in line and 'ArrayUAC' not in line:
                        # Found a USB audio device (likely speakers)
                        parts = line.split()
                        card_num = None
                        device_num = '0'  # Default device number
                        
                        # Extract card number
                        for i, part in enumerate(parts):
                            if part == 'card' and i + 1 < len(parts):
                                card_num = parts[i + 1].rstrip(':')
                                break
                        
                        # Extract device number if specified
                        for i, part in enumerate(parts):
                            if part == 'device' and i + 1 < len(parts):
                                device_num = parts[i + 1].rstrip(':')
                                break
                        
                        if card_num:
                            # Use plughw for format conversion
                            device_str = f'plughw:{card_num},{device_num}'
                            # Just use it - if it doesn't work, we'll fall back to default when playing
                            self.audio_device = device_str
                            print(f"[TTS] ✅ Detected audio output: {device_str} ({line.strip()})")
                            return
            
            # If no USB device found, try default
            print("[TTS] ⚠️  No USB audio device found, will use default")
            self.audio_device = None
            
        except Exception as e:
            print(f"[TTS] ⚠️  Error detecting audio device: {e}")
            import traceback
            print(f"[TTS] Traceback: {traceback.format_exc()}")
            self.audio_device = None
    
    def _preprocess_text(self, text):
        """
        Preprocess text for more natural TTS pronunciation.
        Converts numbers, expands abbreviations, adds natural pauses.
        """
        import re
        
        # Convert numbers to words for better pronunciation
        def number_to_words(num_str):
            """Convert number string to words"""
            try:
                num = int(num_str)
                if num < 20:
                    words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 
                            'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
                            'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
                    return words[num] if num < len(words) else num_str
                elif num < 100:
                    tens = ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
                    ones = num % 10
                    ten = num // 10
                    if ones == 0:
                        return tens[ten - 2]
                    return f"{tens[ten - 2]} {number_to_words(str(ones))}"
                else:
                    return num_str  # Keep large numbers as digits
            except:
                return num_str
        
        # Replace standalone numbers (1-99) with words
        text = re.sub(r'\b(\d{1,2})\b', lambda m: number_to_words(m.group(1)), text)
        
        # Expand common abbreviations
        abbreviations = {
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Missus',
            r'\bMs\.': 'Miss',
            r'\bProf\.': 'Professor',
            r'\bvs\.': 'versus',
            r'\betc\.': 'etcetera',
            r'\bi\.e\.': 'that is',
            r'\be\.g\.': 'for example',
        }
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Add natural pauses after punctuation
        text = re.sub(r'([.!?])\s+', r'\1 ', text)  # Ensure space after sentence enders
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _wait_for_audio_device(self, device=None, max_attempts=2, delay=0.1):
        """Wait for audio device to be available - fast check only."""
        # Use detected device or default
        if device is None:
            device = self.audio_device if self.audio_device else 'default'
        
        # Quick check - don't wait long
        for attempt in range(max_attempts):
            try:
                # Quick test - just check if aplay works
                result = subprocess.run(
                    ['aplay', '-l'],
                    capture_output=True,
                    timeout=0.5
                )
                if result.returncode == 0:
                    return True
            except Exception as e:
                pass
            if attempt < max_attempts - 1:
                time.sleep(delay)
        return False
    
    def speak(self, text, speed=150):
        """
        Speak text aloud (or print if no speakers).
        
        Args:
            text: Text to speak
            speed: Speech speed (words per minute) - used for espeak only
        """
        if not text:
            return
        
        if self.engine == 'print' or self.print_only:
            print(f"\n[ASSISTANT] 🔊 {text}\n")
            return
        
        # Preprocess text for natural pronunciation
        processed_text = self._preprocess_text(text)
        
        print(f"[TTS] 🔊 Speaking: '{processed_text[:50]}...'")
        
        # Quick check for audio device (don't wait long - just try to play)
        # Removed long wait - just try to play immediately
        
        try:
            if self.engine == 'espeak':
                # Write to temp file, then play to USB speakers (plughw:2,0)
                # This ensures proper routing to USB speakers
                subprocess.run(
                    ['espeak', '-w', '/tmp/speech.wav', '-s', str(speed), processed_text],
                    capture_output=True,
                    timeout=30
                )
                # Play to detected audio device or default
                device = self.audio_device if self.audio_device else 'default'
                if device != 'default':
                    aplay_cmd = ['aplay', '-D', device, '/tmp/speech.wav']
                else:
                    aplay_cmd = ['aplay', '/tmp/speech.wav']
                
                aplay_result = subprocess.run(
                    aplay_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if aplay_result.returncode != 0:
                    print(f"[TTS] ⚠️  aplay error ({device}): {aplay_result.stderr}")
                    # Try default device as fallback
                    if device != 'default':
                        print("[TTS] Trying default audio device...")
                        default_result = subprocess.run(
                            ['aplay', '/tmp/speech.wav'],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if default_result.returncode != 0:
                            print(f"[TTS] ⚠️  Default device also failed: {default_result.stderr}")
                else:
                    print(f"[TTS] ✅ Audio played successfully on {device}")
                # Clean up temp file
                try:
                    os.unlink('/tmp/speech.wav')
                except:
                    pass
            elif self.engine == 'piper':
                # Piper produces high-quality, natural-sounding speech
                if self.piper_voice:
                    from piper.config import SynthesisConfig
                    
                    # Configure for naturalness with faster speech
                    config = SynthesisConfig(
                        noise_scale=0.667,      # Default for LibriTTS (more stable than 0.8)
                        length_scale=0.9,       # Slightly faster speech (0.9 = 10% faster)
                        noise_w_scale=0.8,      # Phoneme duration variation (adds natural rhythm)
                        volume=1.2              # Slightly louder for clarity
                    )
                    
                    # Use preprocessed text for better pronunciation
                    # Synthesize speech with natural parameters
                    audio_chunks = []
                    for chunk in self.piper_voice.synthesize(processed_text, syn_config=config):
                        audio_chunks.append(chunk.audio_int16_bytes)
                    
                    audio_data = b''.join(audio_chunks)
                    
                    # Write WAV file
                    with wave.open('/tmp/speech.wav', 'wb') as wav_file:
                        wav_file.setnchannels(1)    # Mono
                        wav_file.setsampwidth(2)    # 16-bit
                        wav_file.setframerate(22050)  # 22.05 kHz
                        wav_file.writeframes(audio_data)
                    
                    # Play with aplay to detected audio device or default
                    device = self.audio_device if self.audio_device else 'default'
                    if device != 'default':
                        aplay_cmd = ['aplay', '-D', device, '/tmp/speech.wav']
                    else:
                        aplay_cmd = ['aplay', '/tmp/speech.wav']
                    
                    aplay_result = subprocess.run(
                        aplay_cmd,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if aplay_result.returncode != 0:
                        print(f"[TTS] ⚠️  aplay error ({device}): {aplay_result.stderr}")
                        # Try default device as fallback
                        if device != 'default':
                            print("[TTS] Trying default audio device...")
                            default_result = subprocess.run(
                                ['aplay', '/tmp/speech.wav'],
                                capture_output=True,
                                text=True,
                                timeout=30
                            )
                            if default_result.returncode != 0:
                                print(f"[TTS] ⚠️  Default device also failed: {default_result.stderr}")
                    else:
                        print(f"[TTS] ✅ Audio played successfully on {device}")
                    # Clean up temp file
                    try:
                        os.unlink('/tmp/speech.wav')
                    except:
                        pass
        
        except Exception as e:
            print(f"[TTS] ❌ Error: {e}")
            print(f"[ASSISTANT] 🔊 {text}")


class RealTimeInfo:
    """Helper class to get real-time information like time, weather, etc."""
    
    def __init__(self, weather_api_key=None):
        """
        Initialize real-time info helper.
        
        Args:
            weather_api_key: OpenWeatherMap API key (optional, for weather)
        """
        self.weather_api_key = weather_api_key
        self.weather_cache = {}
        self.weather_cache_time = 0
        self.weather_cache_duration = 300  # Cache weather for 5 minutes
    
    def get_current_time(self, format_type='full'):
        """Get current time in various formats."""
        now = datetime.now()
        
        if format_type == 'time':
            return now.strftime("%I:%M %p")
        elif format_type == 'date':
            return now.strftime("%A, %B %d, %Y")
        elif format_type == 'full':
            return now.strftime("%A, %B %d, %Y at %I:%M %p")
        elif format_type == 'simple':
            return now.strftime("%I:%M %p on %B %d")
        else:
            return now.strftime("%A, %B %d, %Y at %I:%M %p")
    
    def get_weather(self, city="auto", units="metric"):
        """
        Get current weather information.
        
        Args:
            city: City name or "auto" to try to detect
            units: "metric" (Celsius) or "imperial" (Fahrenheit)
        """
        if not self.weather_api_key:
            return None
        
        # Check cache
        cache_key = f"{city}_{units}"
        if time.time() - self.weather_cache_time < self.weather_cache_duration:
            if cache_key in self.weather_cache:
                return self.weather_cache[cache_key]
        
        try:
            import urllib.request
            import json
            
            # Try to auto-detect city from IP if "auto"
            if city == "auto":
                try:
                    # Get approximate location from IP
                    with urllib.request.urlopen('http://ip-api.com/json/', timeout=3) as response:
                        location_data = json.loads(response.read().decode())
                        city = location_data.get('city', 'London')  # Default fallback
                except:
                    city = "London"  # Default fallback
            
            # Get weather from OpenWeatherMap
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.weather_api_key}&units={units}"
            
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())
                
                temp = data['main']['temp']
                description = data['weather'][0]['description']
                humidity = data['main']['humidity']
                temp_unit = "°C" if units == "metric" else "°F"
                
                weather_info = {
                    'temperature': temp,
                    'description': description,
                    'humidity': humidity,
                    'unit': temp_unit,
                    'city': data['name']
                }
                
                # Cache it
                self.weather_cache[cache_key] = weather_info
                self.weather_cache_time = time.time()
                
                return weather_info
        except Exception as e:
            print(f"[Weather] ⚠️  Error fetching weather: {e}")
            return None
    
    def get_context_string(self):
        """Get formatted context string with all available real-time info."""
        context_parts = []
        
        # Always include time
        current_time = self.get_current_time('full')
        context_parts.append(f"Current date and time: {current_time}")
        
        # Include weather if available
        if self.weather_api_key:
            weather = self.get_weather()
            if weather:
                temp = int(weather['temperature'])
                desc = weather['description']
                city = weather['city']
                unit = weather['unit']
                context_parts.append(f"Current weather in {city}: {desc}, {temp}{unit}")
        
        return ". ".join(context_parts) + "."


class SmartAssistant:
    """
    Smart assistant combining speech recognition, vision, and LLaVa.
    """
    
    def __init__(self, camera=None, motor_controller=None, text_model_path=None, vision_model_path=None, vision_mmproj_path=None, print_only=False, weather_api_key=None, llava_model_path=None, llava_mmproj_path=None):
        """
        Initialize smart assistant with dual model system.
        
        Args:
            camera: Camera object with capture_frames() method (e.g., OakDDepthCamera)
            motor_controller: Motor controller object for movement commands
            text_model_path: Path to text model (Gemma) for chat/Q&A
            vision_model_path: Path to vision model (Phi-Vision) for image understanding
            vision_mmproj_path: Path to vision projector for Phi
            print_only: If True, print responses instead of speaking (no speakers)
            weather_api_key: API key for weather information
            llava_model_path: (deprecated, for compatibility) Maps to vision_model_path
            llava_mmproj_path: (deprecated, for compatibility) Maps to vision_mmproj_path
        """
        self.motor_controller = motor_controller
        self.print_only = print_only
        
        # Handle legacy parameter names for backward compatibility
        if llava_model_path is not None and vision_model_path is None:
            vision_model_path = llava_model_path
        if llava_mmproj_path is not None and vision_mmproj_path is None:
            vision_mmproj_path = llava_mmproj_path
        
        # Initialize camera if not provided
        if camera is None:
            try:
                from oakd_depth_navigator import OakDDepthCamera
                print("[Assistant] Initializing Oak-D camera...")
                # Use 640x352 to match YOLOv8 model resolution (avoids warnings)
                self.camera = OakDDepthCamera(resolution=(640, 352), enable_person_detection=True)
                self.camera.start()
                print("[Assistant] ✅ Camera initialized")
            except Exception as e:
                print(f"[Assistant] ⚠️  Could not initialize camera: {e}")
                print("[Assistant] Vision features will be disabled")
                self.camera = None
        else:
            self.camera = camera
        
        # Initialize components
        print("[Assistant] Initializing smart assistant with dual model system...")
        print("[Assistant] 💬 Text model: Gemma (fast chat/Q&A)")
        print("[Assistant] 👁️  Vision model: Phi-Vision (image understanding)")
        
        # Disable Whisper for faster recognition - use Google Speech Recognition (cloud-based, much faster)
        self.respeaker = ReSpeakerInterface(use_whisper=False)
        
        # Initialize dual model assistant (Gemma for text, Phi for vision)
        # DualModelAssistant will auto-detect model paths if not provided
        self.llava = DualModelAssistant(
            text_model_path=text_model_path,
            vision_model_path=vision_model_path,
            vision_mmproj_path=vision_mmproj_path,
            lazy_load_vision=True  # Load vision model only when needed
        )
        
        # Use piper for natural human voice, falls back to espeak if piper unavailable
        self.tts = TextToSpeech(engine='print' if print_only else 'piper', print_only=print_only)
        # Real-time information helper
        self.realtime_info = RealTimeInfo(weather_api_key=weather_api_key)
        
        # Wake word detection - focus on "Hey Rovy" and "Rovy"
        self.wake_words = ['hey Rovy', 'Rovy']
        
        # Initialize wake word detector if available
        self.wake_word_detector = None
        if WAKE_WORD_AVAILABLE:
            try:
                self.wake_word_detector = WakeWordDetector(
                    device_index=self.respeaker.device_index,
                    wake_words=self.wake_words
                )
                print("[Assistant] ✅ Wake word detector initialized")
            except Exception as e:
                print(f"[Assistant] ⚠️  Could not initialize wake word detector: {e}")
                print("[Assistant] Falling back to simple text-based wake word detection")
        
        # Conversation state
        self.is_active = False
        self.last_interaction = time.time()
        
        # Conversation history for context-aware follow-ups
        self.conversation_history = []  # List of (question, answer) tuples
        
        # Automatic navigation process tracking
        self.auto_nav_process = None
        self.auto_nav_script_path = os.path.join(os.path.dirname(__file__), 'depth_llava_nav.py')
        
        # Face recognition - use simple FaceRecognizer (ageitgey/face_recognition library)
        self.face_recognizer = None
        self.recognized_person = None  # Store the last recognized person's name
        self._initialize_face_recognition()
        
        print("[Assistant] ✅ Smart assistant ready!")
        if print_only:
            print("[Assistant] 💡 Running in print-only mode (no speakers)")
    
    def _initialize_face_recognition(self):
        """Initialize face recognition using ageitgey/face_recognition library."""
        try:
            from face_recognizer import FaceRecognizer
            
            # Get known faces directory
            known_faces_dir = os.path.join(os.path.dirname(__file__), "known-faces")
            if not os.path.exists(known_faces_dir):
                print(f"[Assistant] ⚠️  Known faces directory not found: {known_faces_dir}")
                return
            
            self.face_recognizer = FaceRecognizer(known_dir=known_faces_dir)
            print("[Assistant] ✅ Face recognition initialized")
            if self.face_recognizer.known_names:
                print(f"[Assistant] 📸 Loaded {len(self.face_recognizer.known_names)} known face(s): {', '.join(self.face_recognizer.known_names)}")
        except Exception as e:
            print(f"[Assistant] ⚠️  Could not initialize face recognition: {e}")
            import traceback
            traceback.print_exc()
            self.face_recognizer = None
    
    def _recognize_user_face(self):
        """
        Recognize the user's face from camera.
        
        Returns:
            str: Recognized person's name or None
        """
        if not self.face_recognizer:
            return None
        
        if not self.camera:
            print("[Assistant] ⚠️  Camera not available for face recognition")
            return None
        
        try:
            print("[Assistant] 📸 Capturing image for face recognition...")
            
            # Capture frame from camera
            # OAK-D outputs BGR (not RGB despite setColorOrder setting)
            bgr_frame, _ = self.camera.capture_frames()
            
            # Recognize faces using ageitgey/face_recognition
            recognitions = self.face_recognizer.recognize_face(bgr_frame, is_rgb=False)
            
            if not recognitions:
                print("[Assistant] 👤 No face detected")
                return None
            
            # Get the best match - format: (name, confidence, bbox)
            name, confidence, bbox = recognitions[0]
            
            if name != "Unknown":
                print(f"[Assistant] 👤 Recognized: {name} (confidence: {confidence:.1%})")
                self.recognized_person = name
                return name
            else:
                print(f"[Assistant] 👤 Unknown person (confidence: {confidence:.1%})")
                self.recognized_person = None
                return None
                
        except Exception as e:
            print(f"[Assistant] ⚠️  Face recognition error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _add_face_to_database(self, name):
        """
        Add the current camera frame to the known faces database.
        
        Args:
            name: Name of the person to add
            
        Returns:
            bool: True if face was successfully added, False otherwise
        """
        if not self.face_recognizer:
            return False
        
        if not self.camera:
            return False
        
        try:
            # Capture current frame
            # OAK-D outputs BGR (not RGB despite setColorOrder setting)
            bgr_frame, _ = self.camera.capture_frames()
            
            # Add face to database
            success = self.face_recognizer.add_face(name, bgr_frame)
            
            if success:
                print(f"[Assistant] ✅ Added face for '{name}' to database")
                return True
            else:
                print(f"[Assistant] ❌ Failed to add face - no face detected in image")
                return False
                
        except Exception as e:
            print(f"[Assistant] ⚠️  Error adding face: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _check_wake_word(self, text):
        """Check if text contains wake word."""
        if not text:
            return False
        
        text_lower = text.lower()
        for wake_word in self.wake_words:
            if wake_word in text_lower:
                return True
        return False
    
    def _remove_wake_word(self, text):
        """Remove wake word from text."""
        if not text:
            return text
        
        text_lower = text.lower()
        for wake_word in self.wake_words:
            if wake_word in text_lower:
                # Remove wake word and clean up
                text = text_lower.replace(wake_word, '').strip()
                # Remove leading question words if isolated
                text = text.lstrip('?,. ')
                return text
        return text
    
    def listen_for_wake_word(self, timeout=None):
        """
        Listen for wake word using improved detection.
        Continuously listens until wake word is detected or timeout.
        
        Args:
            timeout: Timeout in seconds (None = infinite)
            
        Returns:
            bool: True if wake word detected
        """
        # Use dedicated wake word detector if available
        if self.wake_word_detector:
            def on_wake_word_detected(wake_word):
                print(f"[Assistant] 👂 Wake word detected: '{wake_word}'!")
                self.respeaker.set_led('speak')
                self.tts.speak("Yes?")
                self.respeaker.set_led('off')
            
            return self.wake_word_detector.listen_for_wake_word(timeout=timeout, callback=on_wake_word_detected)
        else:
            # Fallback to simple text-based detection - continuously listen
            print(f"[Assistant] 👂 Listening for wake words: {', '.join(self.wake_words)}...")
            start_time = time.time()
            
            while True:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    return False
                
                # Listen for speech with longer timeout and phrase limit for better wake word detection
                self.respeaker.set_led('listen')
                text = self.respeaker.listen(timeout=3, phrase_time_limit=5)  # Longer timeout for better detection
                self.respeaker.set_led('off')
                
                if text:
                    text_lower = text.lower()
                    print(f"[WakeWord] Heard: '{text}'")
                    
                    # Check for wake words (more flexible matching)
                    for wake_word in self.wake_words:
                        # Check if wake word is in the text (allows for variations)
                        if wake_word in text_lower:
                            print(f"[Assistant] 👂 Wake word detected: '{wake_word}'!")
                            self.respeaker.set_led('speak')
                            self.tts.speak("Yes?")
                            self.respeaker.set_led('off')
                            return True
                        
                        # Also check for similar sounds (e.g., "Rovy" variations)
                        # This helps with speech recognition errors
                        similar_sounds = ["Rovy", "jarvez", "jarves"]
                        if wake_word == "Rovy":
                            # Check if any similar sound is in the text
                            if any(sound in text_lower for sound in similar_sounds):
                                print(f"[Assistant] 👂 Wake word detected (similar sound): '{wake_word}'!")
                                self.respeaker.set_led('speak')
                                self.tts.speak("Yes?")
                                self.respeaker.set_led('off')
                                return True
                        elif wake_word == "hey Rovy":
                            # Check if "hey" + any similar sound is in the text
                            if any(f"hey {sound}" in text_lower for sound in similar_sounds):
                                print(f"[Assistant] 👂 Wake word detected (similar sound): '{wake_word}'!")
                                self.respeaker.set_led('speak')
                                self.tts.speak("Yes?")
                                self.respeaker.set_led('off')
                                return True
                
                # Small delay to avoid busy waiting
                time.sleep(0.1)
    
    def _check_camera_command(self, text):
        """
        Check if text contains a camera control command.
        
        Returns:
            dict: {'action': str, 'pan': int, 'tilt': int} or None
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Camera position commands
        camera_commands = {
            'center': ['look center', 'look straight', 'look forward', 'center camera', 'camera center'],
            'up': ['look up', 'tilt up', 'camera up'],
            'down': ['look down', 'tilt down', 'camera down'],
            'left': ['look left', 'pan left', 'camera left'],
            'right': ['look right', 'pan right', 'camera right'],
        }
        
        # Check for commands
        for action, keywords in camera_commands.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Map to pan/tilt values
                    if action == 'center':
                        return {'action': 'center', 'pan': 0, 'tilt': 0}
                    elif action == 'up':
                        return {'action': 'up', 'pan': 0, 'tilt': 40}
                    elif action == 'down':
                        return {'action': 'down', 'pan': 0, 'tilt': -20}
                    elif action == 'left':
                        return {'action': 'left', 'pan': -90, 'tilt': 0}
                    elif action == 'right':
                        return {'action': 'right', 'pan': 90, 'tilt': 0}
        
        return None
    
    def _execute_camera_control(self, command):
        """Execute a camera control command."""
        action = command['action']
        pan = command['pan']
        tilt = command['tilt']
        
        if not self.motor_controller:
            return "I don't have camera control available."
        
        try:
            print(f"[Camera] 📹 Moving camera: {action} (pan={pan}°, tilt={tilt}°)")
            
            # Move camera
            self.motor_controller.gimbal_ctrl_move(pan, tilt, input_speed_x=500, input_speed_y=500)
            
            # Wait for movement
            time.sleep(1.5)
            
            return f"Camera moved {action}"
            
        except Exception as e:
            print(f"[Camera] ❌ Error: {e}")
            return f"Camera control error: {str(e)}"
    
    def _check_movement_command(self, text):
        """
        Check if text contains a movement command.
        
        Returns:
            dict: {'action': str, 'duration': float} or None
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Check for automatic navigation commands first
        auto_nav_keywords = [
            'automatic movement', 'auto movement', 'autonomous movement',
            'show automatic movement', 'start automatic', 'start autonomous',
            'automatic navigation', 'auto navigation', 'autonomous navigation',
            'start navigation', 'begin navigation'
        ]
        for keyword in auto_nav_keywords:
            if keyword in text_lower:
                return {'action': 'auto_nav', 'duration': 0}
        
        # Check for stop command (can stop both manual and automatic movement)
        if any(word in text_lower for word in ['stop', 'halt', 'freeze']):
            return {'action': 'stop', 'duration': 0}
        
        # Movement command patterns
        commands = {
            'forward': ['forward', 'go forward', 'move forward', 'go ahead'],
            'backward': ['backward', 'back', 'go back', 'move back', 'reverse'],
            'left': ['left', 'turn left', 'go left'],
            'right': ['right', 'turn right', 'go right'],
            # Battery/status check
            'status': ['battery', 'power', 'charge', 'status'],
        }
        
        # Check for commands
        for action, keywords in commands.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Try to extract duration (e.g., "forward 2 seconds")
                    duration = 3.0  # Default 3 seconds = ~1 meter movement
                    import re
                    match = re.search(r'(\d+)\s*(second|meter|m)', text_lower)
                    if match:
                        duration = float(match.group(1))
                    
                    return {'action': action, 'duration': duration}
        
        return None
    
    def _execute_movement(self, command):
        """Execute a movement command on the motor controller."""
        action = command['action']
        duration = command['duration']
        
        try:
            # Handle automatic navigation start
            if action == 'auto_nav':
                return self._start_automatic_navigation()
            
            # Handle stop command (can stop both manual movement and automatic navigation)
            if action == 'stop':
                return self._stop_movement()
            
            # Manual movement commands require motor controller
            if not self.motor_controller:
                return "I don't have motor control available."
            
            # Handle status check
            if action == 'status':
                return "Battery monitoring not available. If rover moves weakly, charge the battery or use power adapter."
            
            print(f"[Motors] 🚗 Executing: {action} for {duration}s")
            
            # Use same low-level motor control as depth_llava_nav.py
            # Use stronger speeds for noticeable movement (rover_controller uses 0.4-0.7)
            speed_val = 0.5  # Strong medium speed
            
            # Calculate motor power for each direction (like depth_llava_nav.py)
            if action == 'forward':
                L = speed_val
                R = speed_val
            elif action == 'backward':
                L = -speed_val
                R = -speed_val
            elif action == 'left':
                turn_power = 0.5  # Strong turn
                L = -turn_power
                R = turn_power
            elif action == 'right':
                turn_power = 0.5  # Strong turn
                L = turn_power
                R = -turn_power
            else:
                L = 0.0
                R = 0.0
            
            # Send motor commands for the specified duration
            if hasattr(self.motor_controller, '_send'):
                print(f"[Motors] 🔧 Sending commands: L={L:.2f}, R={R:.2f} for {duration}s")
                start_time = time.time()
                command_count = 0
                
                while (time.time() - start_time) < duration:
                    self.motor_controller._send(L, R)
                    command_count += 1
                    time.sleep(0.1)  # 10Hz control rate
                
                elapsed = time.time() - start_time
                print(f"[Motors] ✅ Sent {command_count} commands over {elapsed:.2f}s")
                
                # Stop after duration
                self.motor_controller.stop()
                return f"Moved {action} for {duration:.1f} seconds"
            else:
                return "Motor control method (_send) not found"
        
        except Exception as e:
            print(f"[Motors] ❌ Error: {e}")
            # Make sure to stop on error
            try:
                self.motor_controller.stop()
            except:
                pass
            return f"Motor error: {str(e)}"
    
    def _start_automatic_navigation(self):
        """Start the automatic navigation script (depth_llava_nav.py)."""
        # Check if already running
        if self.auto_nav_process is not None:
            if self.auto_nav_process.poll() is None:
                return "Automatic navigation is already running."
            else:
                # Process finished, clean up
                self.auto_nav_process = None
        
        # Check if script exists
        if not os.path.exists(self.auto_nav_script_path):
            return f"Automatic navigation script not found at {self.auto_nav_script_path}"
        
        try:
            print(f"[AutoNav] 🚀 Starting automatic navigation...")
            # Start the script as a subprocess
            # Use the same port as the motor controller if available
            port = '/dev/ttyACM0'
            if self.motor_controller and hasattr(self.motor_controller, 'port'):
                port = self.motor_controller.port
            
            # Start the process in the background
            # Use subprocess.DEVNULL for output to prevent blocking, or log to files
            log_dir = os.path.join(os.path.dirname(self.auto_nav_script_path), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            stdout_file = os.path.join(log_dir, 'autonav_stdout.log')
            stderr_file = os.path.join(log_dir, 'autonav_stderr.log')
            
            with open(stdout_file, 'w') as fout, open(stderr_file, 'w') as ferr:
                self.auto_nav_process = subprocess.Popen(
                    [sys.executable, self.auto_nav_script_path, '--port', port],
                    stdout=fout,
                    stderr=ferr,
                    cwd=os.path.dirname(self.auto_nav_script_path)
                )
            
            # Give it a moment to start, then check if it's still running
            time.sleep(0.5)
            if self.auto_nav_process.poll() is not None:
                # Process already exited - read error output
                error_msg = "Process exited immediately"
                try:
                    if os.path.exists(stderr_file):
                        with open(stderr_file, 'r') as f:
                            error_content = f.read()
                            if error_content:
                                error_msg = f"Process exited: {error_content[:200]}"
                except:
                    pass
                self.auto_nav_process = None
                print(f"[AutoNav] ❌ Failed to start: {error_msg}")
                return f"Failed to start automatic navigation: {error_msg}"
            
            print(f"[AutoNav] ✅ Started (PID: {self.auto_nav_process.pid})")
            print(f"[AutoNav] Logs: {stdout_file} and {stderr_file}")
            return "Starting automatic navigation. Say 'stop' to stop it."
        
        except Exception as e:
            print(f"[AutoNav] ❌ Error starting automatic navigation: {e}")
            self.auto_nav_process = None
            return f"Failed to start automatic navigation: {str(e)}"
    
    def _stop_movement(self):
        """Stop both manual movement and automatic navigation."""
        result_messages = []
        
        # Stop manual movement if motor controller is available
        if self.motor_controller:
            try:
                self.motor_controller.stop()
                result_messages.append("Stopped manual movement")
            except Exception as e:
                print(f"[Motors] Error stopping: {e}")
        
        # Stop automatic navigation if running
        if self.auto_nav_process is not None:
            if self.auto_nav_process.poll() is None:
                # Process is still running, terminate it
                try:
                    print(f"[AutoNav] 🛑 Stopping automatic navigation (PID: {self.auto_nav_process.pid})...")
                    self.auto_nav_process.terminate()
                    
                    # Wait a bit for graceful shutdown
                    try:
                        self.auto_nav_process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't stop
                        print("[AutoNav] Force killing process...")
                        self.auto_nav_process.kill()
                        self.auto_nav_process.wait()
                    
                    self.auto_nav_process = None
                    result_messages.append("Stopped automatic navigation")
                    print("[AutoNav] ✅ Stopped")
                except Exception as e:
                    print(f"[AutoNav] ❌ Error stopping: {e}")
                    result_messages.append(f"Error stopping automatic navigation: {str(e)}")
            else:
                # Process already finished
                self.auto_nav_process = None
        
        if result_messages:
            return ". ".join(result_messages) + "."
        else:
            return "Nothing to stop."
    
    def _handle_voice_localization(self):
        """
        Handle voice localization - turn camera to face the speaker.
        Uses the exact logic from voice_localization_demo.py
        
        Returns:
            str: Response message to speak
        """
        print("\n[Voice Localization] 🎯 Starting voice localization...")
        
        # Check if DOA is available
        if not self.respeaker.doa_available:
            return "I don't have voice localization available."
        
        # Check if motor controller is available
        if not self.motor_controller:
            return "I don't have camera control available."
        
        try:
            # Import and use the exact logic from voice_localization_demo.py
            from voice_localization import locate_speaker
            
            # Use the face_recognizer directly (already uses ageitgey/face_recognition)
            result = locate_speaker(
                respeaker=self.respeaker,
                rover=self.motor_controller,
                camera=self.camera,
                face_recognizer=self.face_recognizer,
                tts=None  # We'll speak the message ourselves
            )
            
            return result['message']
            
        except Exception as e:
            print(f"[Voice Localization] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return f"Sorry, I had trouble with voice localization: {str(e)}"
    
    def process_question(self, question, use_vision=True, max_tokens=100, conversation_context=None):
        """
        Process a question using LLaVa with fast path for common questions.
        
        Args:
            question: Question text
            use_vision: Whether to include camera image
            max_tokens: Maximum response length (default 25 for instant answers)
            conversation_context: Previous (question, answer) tuple for context
            
        Returns:
            str: Response
        """
        question_lower = question.lower().strip()
        
        # Check if this is a follow-up question that needs context
        needs_context = any(phrase in question_lower for phrase in [
            'tell me more', 'more about', 'what about', 'and', 'also', 
            'how about', 'what else', 'anything else', 'explain more'
        ])
        
        # Fast path for time questions - instant response without LLM
        if any(phrase in question_lower for phrase in ['what time', 'what is the time', 'time is it', 'current time']):
            from datetime import datetime
            now = datetime.now()
            # Format time (remove leading zero from hour on Linux)
            hour = now.strftime("%I").lstrip('0') or '12'
            time_str = f"{hour}:{now.strftime('%M %p')}"  # e.g., "6:18 PM"
            return time_str
        
        # Fast path for date questions
        if any(phrase in question_lower for phrase in ['what date', 'what is the date', 'today\'s date', 'current date']):
            from datetime import datetime
            now = datetime.now()
            date_str = now.strftime("%B %d, %Y")  # e.g., "November 18, 2025"
            return date_str
        
        image = None
        
        # Capture image if vision is requested and camera available
        if use_vision and self.camera:
            try:
                rgb_frame, _ = self.camera.capture_frames()
                image = rgb_frame
                print("[Assistant] 📸 Using camera image for context")
            except Exception as e:
                print(f"[Assistant] ⚠️  Could not capture image: {e}")
        
        # Get real-time context (time, weather, etc.) - let LLM understand naturally
        realtime_context = self.realtime_info.get_context_string()
        
        # Add conversation context if this is a follow-up question
        if needs_context and conversation_context:
            prev_question, prev_answer = conversation_context
            # Enhance the question with context
            question = f"Previous question: {prev_question}. Previous answer: {prev_answer}. Now asking: {question}"
            # Allow more tokens for follow-up questions that need more detail
            max_tokens = max(max_tokens, 50)
        
        # Increase max_tokens for vision questions to allow complete descriptions
        if use_vision and image is not None:
            max_tokens = max(max_tokens, 80)  # Increased to 80 for complete vision descriptions
        
        # Get response from LLaVa with lower temperature for faster, more deterministic responses
        response = self.llava.ask(question, image=image, max_tokens=max_tokens, temperature=0.3, realtime_context=realtime_context)
        
        return response
    
    def locate_and_detect_person(self, tilt_angle=0, detection_timeout=5.0):
        """
        Use voice direction from ReSpeaker to point camera and detect person with OAK-D.
        
        This workflow:
        1. Gets voice direction from ReSpeaker DOA
        2. Moves camera servos to point towards voice
        3. Uses OAK-D person detection to verify person is there
        
        Args:
            tilt_angle: Camera tilt angle to use (0=forward, positive=up)
            detection_timeout: How long to wait for person detection (seconds)
        
        Returns:
            dict: {
                'voice_direction': DOA angle or None,
                'servo_position': {'pan': angle, 'tilt': angle} or None,
                'person_detected': True/False,
                'person_location': 'center'/'left'/'right' or None,
                'detections': list of person detections
            }
        """
        result = {
            'voice_direction': None,
            'servo_position': None,
            'person_detected': False,
            'person_location': None,
            'detections': []
        }
        
        print("\n[Voice Localization] 🎯 Starting voice-directed person detection...")
        
        # Step 1: Get voice direction
        if not self.respeaker.doa_available:
            print("[Voice Localization] ⚠️  DOA not available - cannot locate voice")
            return result
        
        doa_angle = self.respeaker.get_voice_direction()
        if doa_angle is None:
            print("[Voice Localization] ⚠️  Could not determine voice direction")
            return result
        
        result['voice_direction'] = doa_angle
        
        # Step 2: Convert to servo angles and move camera
        if not self.motor_controller:
            print("[Voice Localization] ⚠️  No motor controller - cannot move camera")
            return result
        
        servo_angles = self.respeaker.doa_to_servo_angles(doa_angle, tilt_angle=tilt_angle)
        if servo_angles:
            print(f"[Voice Localization] 📹 Moving camera to pan={servo_angles['pan']}°, tilt={servo_angles['tilt']}°")
            
            # Move camera servos
            self.motor_controller.gimbal_ctrl(
                servo_angles['pan'], 
                servo_angles['tilt'], 
                input_speed=150,  # Moderate speed
                input_acceleration=10
            )
            
            # Wait for servo to reach position
            time.sleep(1.0)
            result['servo_position'] = servo_angles
        
        # Step 3: Use OAK-D person detection
        if not self.camera:
            print("[Voice Localization] ⚠️  No camera - cannot detect person")
            return result
        
        # Check if camera has person detection enabled
        if not hasattr(self.camera, 'enable_person_detection') or not self.camera.enable_person_detection:
            print("[Voice Localization] ⚠️  Person detection not enabled on camera")
            print("[Voice Localization]     Reinitialize with: OakDDepthCamera(enable_person_detection=True)")
            return result
        
        # Try to detect person
        print(f"[Voice Localization] 👤 Looking for person... (timeout: {detection_timeout}s)")
        
        start_time = time.time()
        detections_found = []
        
        while time.time() - start_time < detection_timeout:
            try:
                # Get detections
                detections = self.camera.detect_person()
                
                if detections:
                    # Person(s) detected!
                    result['person_detected'] = True
                    result['detections'] = detections
                    detections_found = detections
                    
                    # Get location of first (largest/closest) detection
                    person = detections[0]
                    location = self.camera.get_person_direction(person['bbox'])
                    result['person_location'] = location
                    
                    confidence = int(person['confidence'] * 100)
                    print(f"[Voice Localization] ✅ Person detected! Location: {location}, Confidence: {confidence}%")
                    
                    # If person is not centered, suggest adjustment
                    if location != 'center':
                        print(f"[Voice Localization] 💡 Person is to the {location} - consider adjusting camera")
                    
                    break
                
                time.sleep(0.2)  # Check 5 times per second
                
            except Exception as e:
                print(f"[Voice Localization] ⚠️  Detection error: {e}")
                break
        
        if not result['person_detected']:
            print("[Voice Localization] ❌ No person detected in camera view")
        
        return result
    
    def run_interactive_session(self, duration=None, use_wake_word=True, greeting=True):
        """
        Run interactive Q&A session.
        
        Args:
            duration: Session duration in seconds (None = until interrupted)
            use_wake_word: If True, require wake word. If False, listen continuously.
            greeting: If True, say greeting message on startup
        """
        print("\n" + "="*60)
        print("🤖 Smart Assistant Active")
        print("="*60)
        if use_wake_word:
            print(f"Wake words: {', '.join(self.wake_words)}")
            print("Say a wake word, then ask your question!")
        else:
            print("Continuous listening mode - just ask your question!")
        print("Press Ctrl+C to exit")
        print("="*60 + "\n")
        
        # Say greeting on startup
        if greeting:
            # Short greeting for faster startup
            greeting_text = "Hi! I'm Rovy. How can I help you?"
            print(f"[Assistant] 🔊 Greeting: {greeting_text}")
            self.respeaker.set_led('speak')
            # Quick check for audio (speak() will also check, but this is faster)
            self.tts._wait_for_audio_device(max_attempts=1, delay=0.1)
            self.tts.speak(greeting_text)
            self.respeaker.set_led('off')
            print("[Assistant] 👂 Now listening for questions...\n")
        
        start_time = time.time()
        in_conversation = False  # Track if we're in an active conversation
        last_qa = None  # Track last question-answer pair for context
        
        try:
            while True:
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    print("\n[Assistant] Session time limit reached")
                    break
                
                # If not in conversation, listen for wake word (if enabled)
                if not in_conversation:
                    if use_wake_word:
                        # Listen for wake word
                        if self.listen_for_wake_word(timeout=5):
                            # Wake word detected - simple greeting
                            # Now listen for question
                            in_conversation = True
                            last_qa = None  # Clear conversation history for new conversation
                            print("[Assistant] 👂 Listening for your question...")
                            self.respeaker.set_led('listen')
                            question = self.respeaker.listen(timeout=10, phrase_time_limit=15)
                            self.respeaker.set_led('off')
                            
                            # If no question captured, go back to wake word detection
                            if not question:
                                print("[Assistant] ⚠️  No question heard. Going back to wake word detection...")
                                in_conversation = False
                                continue
                            
                            # Question captured - will be processed below
                            print(f"[Assistant] ✅ Question captured: '{question}' - will process now...")
                        else:
                            continue
                    else:
                        # No wake word - listen directly
                        in_conversation = True
                        print("[ReSpeaker] 🎤 Listening for question...")
                        self.respeaker.set_led('listen')
                        question = self.respeaker.listen(timeout=10, phrase_time_limit=15)
                        self.respeaker.set_led('off')
                else:
                    # In conversation - listen for follow-up questions
                    print("[Assistant] 👂 Listening for follow-up...")
                    self.respeaker.set_led('listen')
                    question = self.respeaker.listen(timeout=8, phrase_time_limit=15)  # 8 second timeout for follow-ups
                    self.respeaker.set_led('off')
                    
                    # If no question heard, end conversation and go back to wake word detection
                    if not question:
                        print("[Assistant] 💤 No follow-up. Going back to wake word detection...")
                        in_conversation = False
                        last_qa = None  # Clear conversation history
                        continue
                
                if question:
                    # Start total timer
                    total_start = time.time()
                    
                    # Clean up question
                    question = question.strip()
                    
                    print(f"\n[Assistant] ❓ Question: {question}")
                    
                    # Check for recognition questions FIRST
                    question_lower = question.lower()
                    recognition_keywords = [
                        'do you recognize me', 'recognize me', 'do you know me', 'know me',
                        'who am i', 'who am i?', 'who i am', 'am i recognized',
                        'do you remember me', 'remember me', 'have we met'
                    ]
                    is_recognition_question = any(kw in question_lower for kw in recognition_keywords)
                    
                    if is_recognition_question:
                        # LED: Processing recognition
                        self.respeaker.set_led('think')
                        print("[Assistant] 📸 Performing face recognition...")
                        
                        # Recognize user's face
                        recognized_name = self._recognize_user_face()
                        
                        # Respond based on recognition result
                        if recognized_name:
                            response = f"Yes, I recognize you! You're {recognized_name}."
                            self.recognized_person = recognized_name
                        else:
                            response = "I don't recognize you. Would you like me to remember you?"
                            self.recognized_person = None
                        
                        total_time = time.time() - total_start
                        print(f"[Assistant] 💬 Response: {response}")
                        print(f"[Assistant] ⏱️  Total time: {total_time:.2f}s")
                        
                        # LED: Speaking response
                        self.respeaker.set_led('speak')
                        self.tts.speak(response)
                        self.respeaker.set_led('off')
                        
                        # If not recognized, listen for response about remembering
                        if not recognized_name:
                            print("[Assistant] 👂 Listening for your response...")
                            self.respeaker.set_led('listen')
                            remember_response = self.respeaker.listen(timeout=5, phrase_time_limit=10)
                            self.respeaker.set_led('off')
                            
                            # Check if user wants to be remembered
                            if remember_response:
                                remember_lower = remember_response.lower()
                                if any(word in remember_lower for word in ['yes', 'yeah', 'sure', 'okay', 'ok', 'remember', 'add']):
                                    # Ask for their name
                                    print("[Assistant] 💬 Asking for name...")
                                    self.respeaker.set_led('speak')
                                    self.tts.speak("What's your name?")
                                    self.respeaker.set_led('off')
                                    
                                    # Listen for name
                                    print("[Assistant] 👂 Listening for name...")
                                    self.respeaker.set_led('listen')
                                    name_response = self.respeaker.listen(timeout=5, phrase_time_limit=10)
                                    self.respeaker.set_led('off')
                                    
                                    if name_response:
                                        name = name_response.strip()
                                        # Try to add face to known faces
                                        if self._add_face_to_database(name):
                                            print(f"[Assistant] 💬 Confirming...")
                                            self.respeaker.set_led('speak')
                                            self.tts.speak(f"Nice to meet you {name}! I'll remember you next time.")
                                            self.respeaker.set_led('off')
                                            self.recognized_person = name
                                        else:
                                            print(f"[Assistant] 💬 Error message...")
                                            self.respeaker.set_led('speak')
                                            self.tts.speak("Sorry, I couldn't detect your face clearly. Please try again later.")
                                            self.respeaker.set_led('off')
                        
                        # Stay in conversation after recognition
                        continue
                    
                    # Check for camera control commands first (before movement commands)
                    camera_cmd = self._check_camera_command(question)
                    if camera_cmd:
                        # LED: Thinking/Processing
                        self.respeaker.set_led('think')
                        response = self._execute_camera_control(camera_cmd)
                        total_time = time.time() - total_start
                        
                        print(f"[Assistant] 💬 Response: {response}")
                        print(f"[Assistant] ⏱️  Total time: {total_time:.2f}s")
                        # LED: Speaking response
                        self.respeaker.set_led('speak')
                        self.tts.speak(response)
                        self.respeaker.set_led('off')
                        # Stay in conversation after camera command
                        continue
                    
                    # Check for movement commands (faster than LLM)
                    movement_cmd = self._check_movement_command(question)
                    if movement_cmd:
                        # LED: Thinking/Processing movement
                        self.respeaker.set_led('think')
                        response = self._execute_movement(movement_cmd)
                        total_time = time.time() - total_start
                        
                        print(f"[Assistant] 💬 Response: {response}")
                        print(f"[Assistant] ⏱️  Total time: {total_time:.2f}s")
                        # LED: Speaking response
                        self.respeaker.set_led('speak')
                        self.tts.speak(response)
                        self.respeaker.set_led('off')
                        # Stay in conversation after movement command
                        continue
                    
                    # Check for voice localization commands ("look at me")
                    voice_loc_keywords = ['look at me', 'turn to me', 'face me', 'find me', 
                                         'where am i', 'locate me', 'turn towards me']
                    is_voice_localization = any(kw in question_lower for kw in voice_loc_keywords)
                    
                    if is_voice_localization:
                        # LED: Processing
                        self.respeaker.set_led('think')
                        response = self._handle_voice_localization()
                        total_time = time.time() - total_start
                        
                        print(f"[Assistant] 💬 Response: {response}")
                        print(f"[Assistant] ⏱️  Total time: {total_time:.2f}s")
                        # LED: Speaking response
                        self.respeaker.set_led('speak')
                        self.tts.speak(response)
                        self.respeaker.set_led('off')
                        # Stay in conversation after voice localization
                        continue
                    
                    # Check for vision-related keywords - be more specific to avoid false positives
                    # Only use vision for questions that explicitly ask about what's seen
                    vision_keywords = ['what do you see', 'what can you see', 'what are you seeing', 
                                     'describe what you see', 'what is in front', 'what is there',
                                     'look at', 'show me what', 'describe the', 'what does the camera see']
                    use_vision = any(kw in question.lower() for kw in vision_keywords) and self.camera is not None
                    
                    if use_vision:
                        print("[Assistant] 📸 Vision mode enabled - will use camera")
                    
                    # Process question with LLM (include conversation context for follow-ups)
                    self.respeaker.set_led('think')
                    
                    response = self.process_question(question, use_vision=use_vision, conversation_context=last_qa)
                    
                    # Calculate total time
                    total_time = time.time() - total_start
                    
                    print(f"[Assistant] 💬 Answer: {response}")
                    print(f"[Assistant] ⏱️  Total time: {total_time:.1f}s")
                    
                    # Update conversation history
                    last_qa = (question, response)
                    
                    # Speak response
                    self.respeaker.set_led('speak')
                    self.tts.speak(response)
                    self.respeaker.set_led('off')
                    # Stay in conversation - will listen for follow-up
                else:
                    if not use_wake_word:
                        print("[Assistant] ⚠️  No question heard")
                        in_conversation = False
        
        except KeyboardInterrupt:
            print("\n[Assistant] Session ended by user")
        
        finally:
            self.respeaker.set_led('off')
            # Stop automatic navigation if running
            if self.auto_nav_process is not None:
                try:
                    self._stop_movement()
                except:
                    pass
    
    def run_continuous(self):
        """Run assistant continuously in background."""
        print("[Assistant] Starting continuous listening mode...")
        self.run_interactive_session(duration=None)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Assistant')
    parser.add_argument('--no-speakers', action='store_true', help='Disable audio output (print text only)')
    parser.add_argument('--duration', type=int, help='Session duration in seconds')
    parser.add_argument('--no-wake-word', action='store_true', help='Skip wake word, listen continuously')
    parser.add_argument('--no-greeting', action='store_true', help='Skip greeting message on startup')
    parser.add_argument('--port', default='/dev/ttyACM0', help='Rover serial port')
    parser.add_argument('--no-motors', action='store_true', help='Run without motor control')
    parser.add_argument('--weather-api-key', type=str, help='OpenWeatherMap API key for weather info (get free key at openweathermap.org)')
    
    args = parser.parse_args()
    
    # Initialize rover controller if not disabled
    motor_controller = None
    if not args.no_motors:
        try:
            from rover_controller import Rover
            print("[Assistant] Connecting to rover...")
            motor_controller = Rover(port=args.port)
            print("[Assistant] ✅ Rover connected")
        except Exception as e:
            print(f"[Assistant] ⚠️  Could not connect to rover: {e}")
            print("[Assistant] Running without motor control")
    
    # Test mode - run without camera
    print("\n" + "="*60)
    print("Smart Assistant" + (" - Motors Enabled" if motor_controller else " - No Motors"))
    print("="*60)
    
    assistant = SmartAssistant(
        camera=None,
        motor_controller=motor_controller,
        print_only=args.no_speakers,  # Speakers enabled by default, can disable with --no-speakers
        weather_api_key=args.weather_api_key  # Optional weather API key
    )
    
    try:
        assistant.run_interactive_session(
            duration=args.duration, 
            use_wake_word=not args.no_wake_word,
            greeting=not args.no_greeting
        )
    finally:
        # Cleanup
        if motor_controller:
            try:
                motor_controller.stop()
                motor_controller.cleanup()
            except:
                pass
        # Stop automatic navigation if running
        if assistant.auto_nav_process is not None:
            try:
                assistant._stop_movement()
            except:
                pass

