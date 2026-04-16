import pyaudio

pa = pyaudio.PyAudio()

print("\n=== AUDIO DEVICES ===")
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    print(f"{i}: {info['name']} | input: {info['maxInputChannels']}")

pa.terminate()