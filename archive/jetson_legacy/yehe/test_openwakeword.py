import time
import numpy as np
import sounddevice as sd
from openwakeword.model import Model

SAMPLE_RATE = 16000
FRAME_SIZE = 1280
WAKEWORD_THRESHOLD = 0.5

model = Model(
    wakeword_models=["hey_jarvis"],
    inference_framework="onnx"
)

print("Mock test started.")
print("Say: Hey Jarvis")
print("Press Ctrl + C to stop.")
DEVICE_INDEX = 1
last_detect_time = 0
DETECTION_COOLDOWN = 2.0
try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SIZE,
        device=DEVICE_INDEX
    ) as stream:


        while True:
            audio, _ = stream.read(FRAME_SIZE)
            audio = np.squeeze(audio)

            prediction = model.predict(audio)
            score = prediction.get("hey_jarvis", 0)

            print(f"Score: {score:.2f}")

            now = time.time()

            if score > WAKEWORD_THRESHOLD and now - last_detect_time > DETECTION_COOLDOWN:
                print("WAKE WORD DETECTED: Hey Jarvis")
                last_detect_time = now

except KeyboardInterrupt:
    print("\nStopped.")