#!/usr/bin/env python3
"""
Wake Word Detector using Silero VAD + Faster-Whisper
Configured for wake word: "hey tom" / "tom"
Forces preferred microphone selection toward HyperX QuadCast.
"""

import time
import logging
from typing import Optional, Callable
import numpy as np
import difflib

try:
    import pyaudio
    PYAUDIO_OK = True
except ImportError:
    PYAUDIO_OK = False
    print("ERROR: PyAudio not installed. Run: pip install pyaudio")

try:
    import torch
    torch.set_num_threads(4)
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("ERROR: torch not installed. Run: pip install torch")

try:
    from faster_whisper import WhisperModel
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False
    print("ERROR: faster-whisper not installed. Run: pip install faster-whisper")

logger = logging.getLogger(__name__)


class WakeWordDetector:
    def __init__(
        self,
        wake_words: list = None,
        sample_rate: int = 16000,
        device_sample_rate: Optional[int] = None,
        device_index: Optional[int] = None,
        vad_threshold: float = 0.35,
        whisper_model: str = "base",
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8",
        min_speech_duration: float = 0.2,
        min_silence_duration: float = 0.8,
    ):
        if not PYAUDIO_OK:
            raise RuntimeError("PyAudio not available. Install: pip install pyaudio")
        if not TORCH_OK:
            raise RuntimeError("Torch not available. Install: pip install torch")
        if not WHISPER_OK:
            raise RuntimeError("faster-whisper not available. Install: pip install faster-whisper")

        self.wake_words = [w.lower().strip() for w in (wake_words or ["hey tom", "tom", "hey thom", "thom"])]
        self.sample_rate = sample_rate
        self.device_sample_rate = device_sample_rate or sample_rate
        self.needs_resampling = self.device_sample_rate != self.sample_rate
        self.device_index = device_index
        self.vad_threshold = vad_threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration

        logger.info(f"Initializing WakeWordDetector for: {self.wake_words}")

        self.pyaudio = pyaudio.PyAudio()

        if self.device_index is None:
            self.device_index = self._find_microphone()
        else:
            info = self.pyaudio.get_device_info_by_index(self.device_index)
            self.device_sample_rate = int(info.get("defaultSampleRate", 48000))
            self.needs_resampling = self.device_sample_rate != self.sample_rate
            logger.info(
                f"Using manually selected audio device: {info.get('name', 'Unknown')} "
                f"(index {self.device_index}, {self.device_sample_rate}Hz)"
            )

        logger.info(f"Using audio device index: {self.device_index}")
        if self.needs_resampling:
            logger.info(f"Audio: {self.device_sample_rate}Hz (device) -> {self.sample_rate}Hz (processing)")
        else:
            logger.info(f"Audio: {self.sample_rate}Hz")

        logger.info("Loading Silero VAD model...")
        import silero_vad
        self.vad_model = silero_vad.load_silero_vad()
        logger.info("✅ Silero VAD loaded")

        logger.info(f"Loading Whisper {whisper_model} model...")
        self.whisper_model = WhisperModel(
            whisper_model,
            device=whisper_device,
            compute_type=whisper_compute_type,
            cpu_threads=4,
            num_workers=1,
        )
        logger.info(f"✅ Whisper {whisper_model} loaded")

        self.stream = None
        self.running = False

    def _find_microphone(self) -> int:
        """
        Prefer HyperX QuadCast. Never choose camera mics or Sound Mapper if avoidable.
        """
        device_count = self.pyaudio.get_device_count()
        print(f"\nℹ️  Scanning {device_count} audio devices...\n")

        preferred_keywords = [
            "hyperx",
            "quadcast",
            "quad cast",
            "hyper x",
        ]
        banned_keywords = [
            "streamplify",
            "cam mic",
            "camera",
            "sound mapper",
        ]

        candidates = []

        for i in range(device_count):
            try:
                info = self.pyaudio.get_device_info_by_index(i)
                name = info.get("name", "")
                name_lower = name.lower()
                max_input = int(info.get("maxInputChannels", 0))
                native_rate = int(info.get("defaultSampleRate", 44100))

                print(f"   Device {i}: {name} - Input channels: {max_input} - Rate: {native_rate}Hz")

                if max_input <= 0:
                    continue

                candidates.append((i, name, name_lower, native_rate))

            except Exception as e:
                print(f"   Error checking device {i}: {e}")

        print("")

        for i, name, name_lower, native_rate in candidates:
            if any(k in name_lower for k in preferred_keywords):
                print(f"✅ USING QUADCAST CANDIDATE: {name} (index {i})")
                print(f"   Native sample rate: {native_rate}Hz\n")
                self.device_sample_rate = native_rate
                self.needs_resampling = self.device_sample_rate != self.sample_rate
                return i

        for i, name, name_lower, native_rate in candidates:
            if not any(k in name_lower for k in banned_keywords):
                print(f"⚠️ QuadCast name not matched. Using best non-banned mic: {name} (index {i})")
                print(f"   Native sample rate: {native_rate}Hz\n")
                self.device_sample_rate = native_rate
                self.needs_resampling = self.device_sample_rate != self.sample_rate
                return i

        raise RuntimeError("❌ No usable microphone found.")

    def _detect_speech(self, audio_chunk: np.ndarray) -> bool:
        try:
            if self.needs_resampling:
                decimation_factor = max(1, int(self.device_sample_rate / self.sample_rate))
                audio_chunk = audio_chunk[::decimation_factor]

            required_samples = 512 if self.sample_rate == 16000 else 256
            if len(audio_chunk) < required_samples:
                audio_chunk = np.pad(audio_chunk, (0, required_samples - len(audio_chunk)), mode="constant")
            elif len(audio_chunk) > required_samples:
                audio_chunk = audio_chunk[:required_samples]

            audio_float = audio_chunk.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float)
            speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
            return speech_prob >= self.vad_threshold
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False

    def _transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        try:
            if self.needs_resampling:
                decimation_factor = max(1, int(self.device_sample_rate / self.sample_rate))
                audio_data = audio_data[::decimation_factor]

            audio_float = audio_data.astype(np.float32) / 32768.0

            segments, _ = self.whisper_model.transcribe(
                audio_float,
                language="en",
                beam_size=3,
                best_of=3,
                vad_filter=False,
                without_timestamps=True,
            )

            text = " ".join(seg.text for seg in segments).strip()
            return text if text else None
        except Exception as e:
            logger.error(f"Whisper error: {e}")
            return None

    def _normalize_text(self, text: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower())
        return " ".join(cleaned.split())

    def _check_wake_word(self, text: str) -> bool:
        if not text:
            return False

        normalized = self._normalize_text(text)
        words = normalized.split()
        logger.info(f"📝 Heard: '{normalized}'")

        for wake_word in self.wake_words:
            wake_norm = self._normalize_text(wake_word)

            if wake_norm in normalized:
                return True

            wake_parts = wake_norm.split()
            if len(wake_parts) == 1:
                target = wake_parts[0]
                for word in words:
                    ratio = difflib.SequenceMatcher(None, target, word).ratio()
                    if ratio >= 0.8:
                        logger.info(f"✅ Close match: '{word}' ~ '{target}' ({ratio:.2f})")
                        return True

        return False

    def listen_for_wake_word(
        self,
        callback: Optional[Callable[[str], None]] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        logger.info(f"👂 Listening for wake words: {self.wake_words}")

        self.running = True
        start_time = time.time()

        try:
            required_samples_after_resample = 512 if self.sample_rate == 16000 else 256
            if self.needs_resampling:
                decimation_factor = max(1, int(self.device_sample_rate / self.sample_rate))
                chunk_samples = required_samples_after_resample * decimation_factor
            else:
                chunk_samples = required_samples_after_resample

            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.device_sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=chunk_samples,
            )

            speech_buffer = []
            is_speaking = False
            silence_start = None
            speech_start = None

            while self.running:
                if timeout and (time.time() - start_time) > timeout:
                    logger.info("⏱️ Wake word detection timeout")
                    return False

                audio_bytes = self.stream.read(chunk_samples, exception_on_overflow=False)
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)

                has_speech = self._detect_speech(audio_chunk)

                if has_speech:
                    if not is_speaking:
                        is_speaking = True
                        speech_start = time.time()
                        speech_buffer = []
                        logger.info("🗣️ Speech started")

                    speech_buffer.append(audio_chunk)
                    silence_start = None
                else:
                    if is_speaking:
                        if silence_start is None:
                            silence_start = time.time()

                        silence_duration = time.time() - silence_start
                        if silence_duration >= self.min_silence_duration:
                            speech_duration = time.time() - speech_start
                            if speech_duration >= self.min_speech_duration and speech_buffer:
                                logger.info(f"🎤 Processing speech ({speech_duration:.2f}s)")
                                audio_data = np.concatenate(speech_buffer)
                                text = self._transcribe_audio(audio_data)

                                if text and self._check_wake_word(text):
                                    logger.info("✅ Wake word detected!")
                                    if callback:
                                        callback(text)
                                    return True

                            is_speaking = False
                            speech_buffer = []
                            silence_start = None
                            speech_start = None

        except KeyboardInterrupt:
            logger.info("Wake word detection interrupted")
            return False
        finally:
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None

        return False

    def record_query(self, duration: float) -> bytes:
        logger.info(f"🎙️ Recording query for {duration}s...")

        try:
            if not self.stream:
                self.stream = self.pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.device_sample_rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=1024,
                )

            frames = []
            num_chunks = int(self.device_sample_rate / 1024 * duration)

            for _ in range(num_chunks):
                data = self.stream.read(1024, exception_on_overflow=False)
                frames.append(data)

            audio_bytes = b"".join(frames)
            logger.info(f"✅ Recorded {len(audio_bytes)} bytes")
            return audio_bytes
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return b""

    def stop(self):
        self.running = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def cleanup(self):
        self.stop()
        if hasattr(self, "pyaudio") and self.pyaudio:
            try:
                self.pyaudio.terminate()
            except Exception:
                pass
        logger.info("Wake word detector cleaned up")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    detector = WakeWordDetector(
        wake_words=["hey tom", "tom", "hey thom", "thom"],
        sample_rate=16000,
        whisper_model="base",
    )

    try:
        print("\n🎤 Say 'Hey Tom' to test...")
        print("Press Ctrl+C to exit\n")

        while True:
            detected = detector.listen_for_wake_word(timeout=30)

            if detected:
                print("\n✅ Wake word detected! Recording query...")
                audio = detector.record_query(5.0)
                print(f"✅ Recorded {len(audio)} bytes\n")
            else:
                print("No wake word detected, listening again...")

    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        detector.cleanup()