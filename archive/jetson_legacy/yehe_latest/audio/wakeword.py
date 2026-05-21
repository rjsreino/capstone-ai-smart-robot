import time
import numpy as np
import sounddevice as sd
import pyaudio

from typing import Optional

import shared.state as state

from config.settings import *

from models.wakeword_model import wakeword_model

def listen_for_wake_word(
    pa: pyaudio.PyAudio,
    device_index: Optional[int]
) -> bool:

    global last_wakeword_time
    global wake_block_until
    global tts_playing

    wakeword_model.reset()

    print("[WAKE] Listening for Jarvis...")

    try:
        with sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SIZE,
            device=SD_WAKE_DEVICE_INDEX
        ) as stream:

            while state.running:

                if tts_playing:
                    time.sleep(0.05)
                    continue

                audio, overflowed = stream.read(FRAME_SIZE)

                pcm = np.squeeze(audio)

                pcm = pcm.astype(np.int16)

                if pcm.ndim > 1:
                    pcm = pcm[:, 0]

                pcm = pcm.flatten()
                
                print(
    f"[WAKE AUDIO] shape={pcm.shape} "
    f"dtype={pcm.dtype} "
    f"max={np.max(np.abs(pcm))}"
)

                now = time.time()

                if now < wake_block_until:
                    continue

                pcm = pcm.astype(np.float32)

                pcm *= 12.0

                pcm = np.clip(pcm, -32768, 32767)

                pcm = pcm.astype(np.int16)

                prediction = wakeword_model.predict(pcm)
                score = prediction.get(WAKEWORD_NAME, 0)

                print(f"[WAKE SCORE] {score:.2f}")

                if (
                    score > WAKEWORD_THRESHOLD
                    and now - last_wakeword_time > WAKEWORD_COOLDOWN
                ):
                    print(f"[WAKE] Hey Jarvis detected ({score:.2f})")

                    last_wakeword_time = now
                    global last_manual_speech_time
                    last_manual_speech_time = time.time()
                    return True

    except Exception as e:
        print(f"[WAKE ERROR] {e}")
        time.sleep(1.0)

    return False