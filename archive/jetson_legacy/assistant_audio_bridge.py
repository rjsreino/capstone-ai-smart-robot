from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np

LOGGER = logging.getLogger("uvicorn.error").getChild(__name__)

# Optional dependencies (same pattern as SmartAssistant)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    whisper = None  # type: ignore
    WHISPER_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    sr = None  # type: ignore
    SR_AVAILABLE = False


@dataclass
class AudioResult:
    ok: bool
    transcript: Optional[str] = None
    answer: Optional[str] = None
    error: Optional[str] = None


class WebSocketAudioSession:
    """
    Session that accumulates audio from WebSocket chunks and
    feeds it into SmartAssistant for STT + LLM answer.
    """

    def __init__(
        self,
        assistant,
        sample_rate: int = 16000,
        sample_width: int = 2,  # bytes per sample (int16)
        use_whisper: bool = True,
    ):
        """
        :param assistant: SmartAssistant instance
        :param sample_rate: audio sample rate (Hz)
        :param sample_width: bytes per sample (2 for int16)
        :param use_whisper: prefer Whisper if available
        """
        self.assistant = assistant
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.buffer = bytearray()

        self.use_whisper = use_whisper and WHISPER_AVAILABLE
        self._whisper_model = None
        self._sr_recognizer = None

        if self.use_whisper:
            try:
                # Try to reuse assistant's whisper model if it exists
                wm = getattr(getattr(assistant, "respeaker", None), "whisper_model", None)
                if wm is not None:
                    self._whisper_model = wm
                    LOGGER.info("[AudioBridge] Using assistant.respeaker Whisper model")
                else:
                    LOGGER.info("[AudioBridge] Loading Whisper tiny model for WebSocket audio")
                    self._whisper_model = whisper.load_model("tiny")
            except Exception as exc:
                LOGGER.error("[AudioBridge] Whisper init failed: %s", exc, exc_info=True)
                self.use_whisper = False
                self._whisper_model = None

        if not self.use_whisper and SR_AVAILABLE:
            self._sr_recognizer = sr.Recognizer()
            LOGGER.info("[AudioBridge] Using Google SpeechRecognition fallback")

        if not self.use_whisper and not SR_AVAILABLE:
            LOGGER.warning("[AudioBridge] No STT backend available (no whisper, no speech_recognition)")

    # ------------------------------------------------------------------
    # Incoming audio handling
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset session buffer for a new utterance."""
        self.buffer.clear()

    def add_chunk_base64(self, b64_data: str) -> int:
        """
        Add one base64-encoded audio chunk (PCM int16 mono).

        Returns: number of raw bytes added.
        """
        try:
            raw = base64.b64decode(b64_data)
        except Exception as exc:
            LOGGER.error("[AudioBridge] Failed to decode base64 audio chunk: %s", exc, exc_info=True)
            return 0

        self.buffer.extend(raw)
        return len(raw)

    # ------------------------------------------------------------------
    # Speech-to-text
    # ------------------------------------------------------------------
    def _stt_whisper(self) -> Tuple[Optional[str], Optional[str]]:
        """Run Whisper STT on current buffer (returns transcript, error)."""
        if not self._whisper_model:
            return None, "Whisper model not initialized"

        if not self.buffer:
            return None, "Empty audio buffer"

        try:
            audio_np = np.frombuffer(bytes(self.buffer), dtype=np.int16).astype(np.float32) / 32768.0
            result = self._whisper_model.transcribe(
                audio_np,
                language="en",
                task="transcribe",
                fp16=False,
                verbose=False,
            )
            text = (result.get("text") or "").strip()
            return (text or None), None
        except Exception as exc:
            LOGGER.error("[AudioBridge] Whisper STT failed: %s", exc, exc_info=True)
            return None, str(exc)

    def _stt_google(self) -> Tuple[Optional[str], Optional[str]]:
        """Run Google SpeechRecognition STT on current buffer."""
        if not SR_AVAILABLE or not self._sr_recognizer:
            return None, "SpeechRecognition not available"

        if not self.buffer:
            return None, "Empty audio buffer"

        try:
            audio_data = sr.AudioData(bytes(self.buffer), self.sample_rate, self.sample_width)
            text = self._sr_recognizer.recognize_google(audio_data, language="en-US", show_all=False)
            text = text.strip()
            return (text or None), None
        except sr.UnknownValueError:
            return None, "Could not understand audio"
        except sr.RequestError as exc:
            LOGGER.error("[AudioBridge] Google STT request error: %s", exc, exc_info=True)
            return None, f"SpeechRecognition service error: {exc}"
        except Exception as exc:
            LOGGER.error("[AudioBridge] Google STT failed: %s", exc, exc_info=True)
            return None, str(exc)

    # ------------------------------------------------------------------
    # Public: STT + assistant answer
    # ------------------------------------------------------------------
    def finalize_and_answer(
        self,
        use_vision: bool = True,
        max_tokens: int = 80,
    ) -> AudioResult:
        """
        Run STT on collected audio, then pass transcript to SmartAssistant
        and get a natural language answer.

        :return: AudioResult(ok, transcript, answer, error)
        """
        if not self.buffer:
            return AudioResult(ok=False, error="No audio received")

        # 1) STT
        transcript: Optional[str] = None
        stt_error: Optional[str] = None

        if self.use_whisper:
            transcript, stt_error = self._stt_whisper()
        else:
            transcript, stt_error = self._stt_google()

        if not transcript:
            return AudioResult(ok=False, transcript=None, answer=None, error=stt_error or "STT failed")

        LOGGER.info("[AudioBridge] Transcript: %s", transcript)

        # 2) Call SmartAssistant (vision + LLM)
        try:
            # SmartAssistant.process_question is sync
            answer = self.assistant.process_question(
                transcript,
                use_vision=use_vision,
                max_tokens=max_tokens,
                conversation_context=None,
            )

            # Optionally speak on-robot (if not print_only)
            try:
                self.assistant.respeaker.set_led("speak")
            except Exception:
                pass

            try:
                self.assistant.tts.speak(answer)
            except Exception:
                LOGGER.warning("[AudioBridge] TTS speak failed", exc_info=True)

            try:
                self.assistant.respeaker.set_led("off")
            except Exception:
                pass

            return AudioResult(ok=True, transcript=transcript, answer=answer, error=None)
        except Exception as exc:
            LOGGER.error("[AudioBridge] Assistant error: %s", exc, exc_info=True)
            return AudioResult(ok=False, transcript=transcript, answer=None, error=str(exc))
