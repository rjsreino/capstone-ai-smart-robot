import io
import wave
import signal
import sys
import time

import numpy as np
import pyaudio
import soundfile as sf
import whisper
import pyttsx3


MIC_INDEX = 1
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAME_SIZE = 1280

CHUNK_DURATION = 0.5
SILENCE_THRESHOLD = 200
SILENCE_DURATION = 2.0
START_SPEECH_THRESHOLD = 300
MIN_COMMAND_SECONDS = 1.0

WAKE_WORDS = ["hey viko", "hey vicko", "vicko", "vico", "viko", "hey viko","hey fico","hey ficko","hello viko","hello vicko","hello ficko","hello fico","hey,fiko!","hey fiko!","hey fiko"]

whisper_model = whisper.load_model("base")
tts_engine = pyttsx3.init()


def handle_interrupt(sig, frame):
    print("\nExiting...")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_interrupt)


def speak(text: str):
    print(f"Assistant: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()


def listen_for_speech_gate(pa: pyaudio.PyAudio):
    stream = pa.open(
        rate=SAMPLE_RATE,
        channels=1,
        format=FORMAT,
        input=True,
        input_device_index=MIC_INDEX,
        frames_per_buffer=FRAME_SIZE
    )

    print("Listening for wake word...")

    try:
        while True:
            data = stream.read(FRAME_SIZE, exception_on_overflow=False)
            pcm = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(pcm).mean()

            if volume > START_SPEECH_THRESHOLD:
                print("Speech started.")
                return
    finally:
        stream.stop_stream()
        stream.close()


def record_until_silence(
    pa: pyaudio.PyAudio,
    sample_rate: int = SAMPLE_RATE,
    chunk_duration: float = CHUNK_DURATION,
    silence_threshold: int = SILENCE_THRESHOLD,
    silence_duration: float = SILENCE_DURATION
) -> io.BytesIO:
    chunk_size = int(sample_rate * chunk_duration)

    stream = pa.open(
        rate=sample_rate,
        channels=1,
        format=FORMAT,
        input=True,
        input_device_index=MIC_INDEX,
        frames_per_buffer=chunk_size
    )

    print("Recording...")
    frames = []
    silence_chunks_needed = int(silence_duration / chunk_duration)
    silent_chunks = 0

    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)

            audio_np = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_np).mean()

            if volume < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks >= silence_chunks_needed:
                break
    finally:
        stream.stop_stream()
        stream.close()

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    wav_buffer.seek(0)
    return wav_buffer


def transcribe_audio(wav_buffer: io.BytesIO) -> str:
    audio_array, sample_rate = sf.read(wav_buffer, dtype="float32")

    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)

    if sample_rate != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

    audio_array = whisper.pad_or_trim(audio_array)
    mel = whisper.log_mel_spectrogram(audio_array).to(whisper_model.device)

    options = whisper.DecodingOptions(language="en", fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    return result.text.strip()


def contains_wake_word(text: str) -> bool:
    normalized = text.lower().strip()
    return any(w in normalized for w in WAKE_WORDS)


def remove_wake_word(text: str) -> str:
    cleaned = text.lower()
    for w in WAKE_WORDS:
        cleaned = cleaned.replace(w, "")
    return cleaned.strip()


def handle_command(command: str):
    if not command:
        speak("Hi, I'm Viko. Ready.")
        return

    command = command.lower()

    if "hello" in command or "hi" in command:
        speak("Hello.")
    
    elif "time" in command:
        import datetime as dt
        now = dt.datetime.now().strftime("%H:%M")
        speak(f"The time is {now}.")
    
    elif "name" in command:
        speak("I'm Viko.")
    
    elif "stop" in command or "exit" in command:
        speak("Stopping.")
        raise SystemExit
    
    else:
        # 🔥 INI YANG DIGANTI
        speak("Hello, Im Vicko.")


def main():
    pa = pyaudio.PyAudio()

    try:
        while True:
            listen_for_speech_gate(pa)

            raw_audio = record_until_silence(pa, SAMPLE_RATE)

            raw_audio.seek(0, io.SEEK_END)
            size_in_bytes = raw_audio.tell()
            raw_audio.seek(0)

            approx_num_samples = size_in_bytes // 2
            approx_seconds = approx_num_samples / SAMPLE_RATE

            if approx_seconds < MIN_COMMAND_SECONDS:
                print("Too short.")
                time.sleep(0.2)
                continue

            try:
                transcript = transcribe_audio(raw_audio)
            except Exception as e:
                print(f"Transcription error: {e}")
                time.sleep(0.2)
                continue

            if not transcript:
                print("No speech recognized.")
                time.sleep(0.2)
                continue

            print(f"You said: {transcript}")

            if contains_wake_word(transcript):
                print("Wake word detected.")

                command = remove_wake_word(transcript)

                if not command:
                    speak("Hi, I'm Viko. Ready.")
                else:
                    handle_command(command)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()