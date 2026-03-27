"""
Speech Processing - Whisper STT and TTS
Uses local Whisper for speech recognition and Piper for TTS.
"""
import os
import re
import io
import wave
import logging
from typing import Optional, Any

import numpy as np

logger = logging.getLogger('Speech')

# Try to import local Whisper
WHISPER_OK = False
try:
    import whisper
    WHISPER_OK = True
except ImportError:
    logger.warning("Local Whisper not available. Install: pip install openai-whisper")

# Try to import Piper
PIPER_OK = False
try:
    from piper import PiperVoice
    PIPER_OK = True
except ImportError:
    pass  # Piper TTS optional


class SpeechProcessor:
    """Speech recognition and synthesis using local models."""
    
    def __init__(
        self, 
        whisper_model: str = "base", 
        tts_engine: str = "piper", 
        piper_voices: dict = None
    ):
        self.whisper_model = None
        self.piper_voices = {}  # Dictionary of language -> PiperVoice
        self.piper_voice_paths = piper_voices or {}  # Dictionary of language -> voice path
        self.tts_engine = tts_engine
        
        # Load local Whisper
        if WHISPER_OK:
            try:
                logger.info(f"Loading local Whisper ({whisper_model})...")
                self.whisper_model = whisper.load_model(whisper_model)
                logger.info("✅ Local Whisper ready")
            except Exception as e:
                logger.error(f"Local Whisper load failed: {e}")
        else:
            logger.warning("Local Whisper not available")
        
        # Setup TTS
        if tts_engine == "piper" and PIPER_OK:
            self._init_piper_voices()
        else:
            self.tts_engine = "none"
            logger.warning("TTS not available (Piper not installed)")
    
    def _init_piper_voices(self):
        """Initialize Piper TTS voices for multiple languages."""
        if not self.piper_voice_paths:
            logger.warning("No Piper voice paths configured")
            self.tts_engine = "none"
            return
        
        # Try to load at least one voice (preferably English)
        loaded_count = 0
        for lang, path in self.piper_voice_paths.items():
            if os.path.exists(path):
                try:
                    self.piper_voices[lang] = PiperVoice.load(path)
                    logger.info(f"✅ Piper voice loaded for {lang}: {os.path.basename(path)}")
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load Piper voice for {lang}: {e}")
        
        if loaded_count > 0:
            logger.info(f"✅ Piper ready with {loaded_count} language(s)")
        else:
            logger.warning("No Piper voices loaded, TTS not available")
            self.tts_engine = "none"
    
    def _load_piper_voice(self, language: str) -> Optional[Any]:
        """Lazy-load a Piper voice for a specific language."""
        # If already loaded, return it
        if language in self.piper_voices:
            return self.piper_voices[language]
        
        # Try to load it
        if language in self.piper_voice_paths:
            path = self.piper_voice_paths[language]
            if os.path.exists(path):
                try:
                    voice = PiperVoice.load(path)
                    self.piper_voices[language] = voice
                    logger.info(f"✅ Loaded Piper voice for {language}: {os.path.basename(path)}")
                    return voice
                except Exception as e:
                    logger.warning(f"Failed to load Piper voice for {language}: {e}")
        
        return None
    
    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000, use_api: bool = False) -> Optional[str]:
        """Transcribe audio to text using local Whisper.
        
        Args:
            audio_bytes: Raw audio data
            sample_rate: Audio sample rate
            use_api: Ignored (kept for compatibility)
        """
        if not self.whisper_model:
            logger.error("Local Whisper not available")
            return None
        
        return self._transcribe_local(audio_bytes, sample_rate)
    
    def _transcribe_local(self, audio_bytes: bytes, sample_rate: int = 16000) -> Optional[str]:
        """Transcribe using local Whisper model."""
        try:
            # Validate buffer size - must be multiple of 2 for int16
            if len(audio_bytes) % 2 != 0:
                logger.warning(f"Buffer size {len(audio_bytes)} is not a multiple of 2, trimming last byte")
                audio_bytes = audio_bytes[:-1]
            
            if len(audio_bytes) == 0:
                logger.error("Empty audio buffer after validation")
                return None
            
            # Convert bytes to float array
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Check audio energy/volume
            audio_energy = np.sqrt(np.mean(audio ** 2))
            audio_max = np.max(np.abs(audio))
            logger.info(f"Audio diagnostics: energy={audio_energy:.4f}, max_amplitude={audio_max:.4f}, samples={len(audio)}")
            
            # Warn if audio is very quiet
            if audio_energy < 0.001:
                logger.warning("⚠️ Audio is very quiet (energy < 0.001), may be silence or poor recording")
            elif audio_max < 0.01:
                logger.warning("⚠️ Audio amplitude is very low (max < 0.01), may be too quiet to transcribe")
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                from scipy import signal
                # Use scipy's resample for better quality and dtype compatibility
                factor = 16000 / sample_rate
                new_len = int(len(audio) * factor)
                audio = signal.resample(audio, new_len).astype(np.float32)
            
            # Transcribe (English only, better accuracy)
            result = self.whisper_model.transcribe(
                audio,
                language="en",  # Force English
                task="transcribe",
                fp16=False,
                verbose=False,
                temperature=0.0,  # Use greedy decoding for better accuracy
                beam_size=5,  # Use beam search for better results
                best_of=5,  # Sample multiple times and pick best
                condition_on_previous_text=False  # Each segment is independent
            )
            
            text = result["text"].strip()
            if text:
                logger.info(f"Transcribed: '{text}'")
            else:
                logger.warning("Whisper returned empty text")
            return text if text else None
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None
    
    def synthesize(self, text: str, language: str = "en") -> Optional[bytes]:
        """
        Convert text to speech audio (WAV bytes).
        
        Args:
            text: Text to synthesize
            language: ISO language code (e.g., 'en', 'es', 'fr', 'de', etc.) - default 'en'
        """
        if not text:
            return None
        
        # Preprocess for TTS
        text = self._preprocess(text)
        logger.info(f"Synthesizing ({language}): '{text[:50]}...'")
        
        # Use Piper if available
        if self.tts_engine == "piper" and PIPER_OK:
            voice = self._load_piper_voice(language)
            if voice:
                return self._synth_piper(text, voice)
            # Fallback to English voice if language not available
            elif language != "en":
                voice = self._load_piper_voice("en")
                if voice:
                    return self._synth_piper(text, voice)
        
        logger.warning("TTS not available - no Piper voices loaded")
        return None
    
    def _synth_piper(self, text: str, voice: Any) -> Optional[bytes]:
        """Synthesize using Piper with the specified voice."""
        try:
            audio_data = []
            for chunk in voice.synthesize_stream_raw(text):
                audio_data.append(chunk)
            
            if not audio_data:
                return None
            
            raw = b''.join(audio_data)
            
            # Convert to WAV
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(22050)
                wav.writeframes(raw)
            
            return buf.getvalue()
            
        except Exception as e:
            logger.error(f"Piper synthesis failed: {e}")
            return None
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text for better TTS pronunciation."""
        # Number to words (simple cases)
        def num_to_word(m):
            n = int(m.group(0))
            words = ['zero', 'one', 'two', 'three', 'four', 'five', 
                    'six', 'seven', 'eight', 'nine', 'ten']
            if n < len(words):
                return words[n]
            return m.group(0)
        
        text = re.sub(r'\b(\d)\b', num_to_word, text)
        
        # Abbreviations
        abbrevs = {
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Missus',
            r'\bi\.e\.': 'that is',
            r'\be\.g\.': 'for example',
        }
        for pat, repl in abbrevs.items():
            text = re.sub(pat, repl, text, flags=re.IGNORECASE)
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text

