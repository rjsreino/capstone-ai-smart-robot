import pyaudio

from typing import Optional

from config.settings import (
    MIC_INDEX,
    FORMAT,
)

def get_input_device_index(pa: pyaudio.PyAudio) -> Optional[int]:
    if MIC_INDEX >= 0:
        print(f"[MIC] Using hardcoded microphone index: {MIC_INDEX}")
        return MIC_INDEX

    blocked_keywords = [
    "voicemod",
    "steelseries",
    "sonar",
    "steam",
    "intelligo",
    "virtual",
    "vad",
    "speaker",
    "output",
    "microphone (realtek hd audio mic input)",
]

    try:
        default_info = pa.get_default_input_device_info()
        name = default_info["name"].lower()

        if not any(word in name for word in blocked_keywords):
            print(
                f"[MIC] Using Windows default microphone: "
                f"{default_info['index']} | {default_info['name']}"
            )
            return int(default_info["index"])

        print(f"[MIC] Default mic is virtual/skipped: {default_info['name']}")

    except Exception as e:
        print("[MIC] No Windows default mic. Using fallback scan.")

    print("\n=== INPUT DEVICES ===")

    fallback_devices = []

    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
            name = info["name"].lower()
            max_channels = int(info.get("maxInputChannels", 0))

            if max_channels <= 0:
                continue

            print(
                f"INDEX {i} | {info['name']} | "
                f"INPUTS={max_channels} | "
                f"RATE={int(info['defaultSampleRate'])}"
            )

            if any(word in name for word in blocked_keywords):
                print(f"[MIC] Skipping virtual/bad device: {i}")
                continue

            fallback_devices.append(i)

        except Exception:
            continue

    print("=====================\n")

    for i in fallback_devices:
        try:
            info = pa.get_device_info_by_index(i)

            pa.is_format_supported(
                rate=int(info["defaultSampleRate"]),
                input_device=i,
                input_channels=1,
                input_format=FORMAT,
            )

            print(f"[MIC] Selected fallback physical mic: {i} | {info['name']}")
            return i

        except Exception as e:
            print(f"[MIC] Skipping unsupported physical mic {i}: {e}")

    print("[MIC ERROR] No usable physical microphone found.")
    return None