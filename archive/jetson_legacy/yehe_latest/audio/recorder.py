import io
import wave
import time

import numpy as np
import sounddevice as sd
import pyaudio

from typing import Optional

import shared.state as state

from config.settings import *

def record_until_silence(
    pa: pyaudio.PyAudio,
    device_index: Optional[int],
    sample_rate: int = SAMPLE_RATE,
    chunk_duration: float = CHUNK_DURATION,
    silence_threshold: int = SILENCE_THRESHOLD,
    silence_duration: float = SILENCE_DURATION,
) -> io.BytesIO:

    sample_rate = SD_SAMPLE_RATE

    chunk_size = int(sample_rate * chunk_duration)

    silence_chunks_needed = max(
        1,
        int(silence_duration / chunk_duration)
    )

    frames = []
    silent_chunks = 0
    speech_detected = False

    print("[REC] Listening for command...")

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=chunk_size,
        device=SD_WAKE_DEVICE_INDEX
    ) as stream:

        wait_start_time = time.time()
        start_time = None

        while state.running:

            audio, _ = stream.read(chunk_size)

            pcm = np.squeeze(audio).astype(np.int16)

            volume = np.abs(pcm).mean()

            print(f"[VOL] {volume}")

            if not speech_detected:

                if volume > START_SPEECH_THRESHOLD:
                    
                    print("[VOICE DETECTED]")
                    
                    speech_detected = True
                    silent_chunks = 0
                    start_time = time.time()

                    frames.append(pcm.tobytes())

                if time.time() - wait_start_time > 5.0:
                    print("[REC] No speech detected.")
                    return io.BytesIO()

            else:
                frames.append(pcm.tobytes())

                if volume > silence_threshold:
                    silent_chunks = 0
                else:
                    silent_chunks += 1

                if silent_chunks >= silence_chunks_needed:
                    break

                if (
                    start_time
                    and time.time() - start_time > MAX_COMMAND_SECONDS
                ):
                    print("[REC] Max command time reached.")
                    break

    if not speech_detected or not frames:
        return io.BytesIO()

    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    wav_buffer.seek(0)

    return wav_buffer