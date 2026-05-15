import time
import pyaudio

import shared.state as state

from config.settings import *

from models.wakeword_model import wakeword_model

from audio.wakeword import listen_for_wake_word
from audio.recorder import record_until_silence
from audio.transcriber import transcribe_audio

from utils.text_utils import normalize_text

from shared.state import (
    speech_lock,
    frame_lock,
)

from audio.tts import (
    speak,
    clear_speech_queue,
)

from audio.device_utils import (
    get_input_device_index,
)

from assistant.query_handler import (
    handle_vision_query,
)

def voice_loop():
    global running
    global last_user_interaction_time
    global voice_interaction_active
    global wake_block_until
    global tts_playing

    wakeword_model.reset()
    pa = pyaudio.PyAudio()
    device_index = get_input_device_index(pa)

    print(f"[INFO] Using microphone device index: {device_index}")

    if device_index is None:
        print("[VOICE ERROR] No valid microphone found.")
        return

    try:
        while state.running:
            heard = listen_for_wake_word(pa, device_index)

            if not heard or not state.running:
                continue

            print("[VOICE] Wake word accepted.")

            voice_interaction_active = True
            last_user_interaction_time = time.time()

            clear_speech_queue()

            try:
                wakeword_model.reset()
            except Exception:
                pass

            with speech_lock:
                speak("Yes?")

            time.sleep(0.3)

            try:
                raw_audio = record_until_silence(
                    pa,
                    device_index,
                    SAMPLE_RATE
                )
            except Exception as e:
                print(f"[VOICE ERROR] Audio capture failed: {e}")

                voice_interaction_active = False
                wake_block_until = time.time() + WAKE_RESTART_DELAY

                try:
                    wakeword_model.reset()
                except Exception:
                    pass

                
                continue

            try:
                raw_audio.seek(0)

                if raw_audio.getbuffer().nbytes == 0:
                    print("[VOICE] Empty audio buffer.")

                    voice_interaction_active = False
                    wake_block_until = time.time() + WAKE_RESTART_DELAY

                    try:
                        wakeword_model.reset()
                    except Exception:
                        pass

                    
                    continue

            except Exception as e:
                print(f"[VOICE ERROR] Invalid audio buffer: {e}")

                voice_interaction_active = False
                wake_block_until = time.time() + WAKE_RESTART_DELAY

                try:
                    wakeword_model.reset()
                except Exception:
                    pass

                
                continue

            try:
                transcript = transcribe_audio(raw_audio)
                print(f"[TRANSCRIPT RAW] '{transcript}'")

            except Exception as e:
                print(f"[VOICE ERROR] Transcription failed: {e}")

                voice_interaction_active = False
                wake_block_until = time.time() + WAKE_RESTART_DELAY

                try:
                    wakeword_model.reset()
                except Exception:
                    pass

                
                continue

            if not transcript or not transcript.strip():
                print("[VOICE] No valid speech detected.")

                voice_interaction_active = False
                wake_block_until = time.time() + WAKE_RESTART_DELAY

                try:
                    wakeword_model.reset()
                except Exception:
                    pass

                
                continue

            normalized_transcript = normalize_text(transcript)

            print(f"[HEARD] {normalized_transcript}")

            last_user_interaction_time = time.time()

            with frame_lock:
                detections_copy = list(state.latest_detections)
                frame_copy = None if state.latest_frame is None else state.latest_frame.copy()

            try:
                answer = handle_vision_query(
                    normalized_transcript,
                    detections_copy,
                    frame_copy
                )

            except Exception as e:
                print(f"[VOICE ERROR] Query handling failed: {e}")
                answer = "I could not process that request."

            print(f"[ASSISTANT] {answer}")

            if answer == "__EXIT__":
                with speech_lock:
                    speak("Stopping live vision assistant.")

                running = False
                break

            clear_speech_queue()

            try:
                with speech_lock:
                    speak(answer)

            except Exception as e:
                print(f"[VOICE ERROR] TTS failed: {e}")

            last_user_interaction_time = time.time()
            voice_interaction_active = False

            wake_block_until = time.time() + WAKE_RESTART_DELAY

            try:
                wakeword_model.reset()
            except Exception:
                pass

            print("[VOICE] Wake word ready.")
            

            try:
                wakeword_model.reset()
            except Exception:
                pass

    finally:
        voice_interaction_active = False

        try:
            wakeword_model.reset()
        except Exception:
            pass

        try:
            pa.terminate()
        except Exception:
            pass

        print("[VOICE] Voice loop terminated.")