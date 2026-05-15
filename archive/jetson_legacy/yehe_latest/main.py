import threading
import time
import cv2

import shared.state as state

from audio.tts import (
    enqueue_speech,
    speech_worker,
)

from vision.detector import vision_loop

from audio.voice_loop import voice_loop


def main():

    enqueue_speech(
        "Proactive live vision assistant started. "
        "Automatic guidance is active. "
        "Say Hey Jarvis before optional voice commands."
    )

    speaker_thread = threading.Thread(
        target=speech_worker,
        daemon=True
    )

    vision_thread = threading.Thread(
        target=vision_loop,
        daemon=True
    )

    voice_thread = threading.Thread(
        target=voice_loop,
        daemon=True
    )

    speaker_thread.start()
    vision_thread.start()
    voice_thread.start()

    try:
        while state.running:
            time.sleep(0.1)

    finally:

        state.running = False

        try:
            cv2.destroyAllWindows()

        except Exception:
            pass

        print(
            "[INFO] Proactive live vision assistant terminated."
        )


if __name__ == "__main__":
    main()