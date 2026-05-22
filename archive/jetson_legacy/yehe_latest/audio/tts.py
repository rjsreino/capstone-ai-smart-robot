import asyncio
import tempfile
import os
import queue

import edge_tts
import pygame
pygame.mixer.init()

import shared.state as state

from utils.text_utils import normalize_tts_text
from config.settings import TTS_VOICE

def speak(text: str):
    text = normalize_tts_text(str(text).strip())

    if not text:
        return

    try:
        asyncio.run(async_edge_tts(text))

    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(async_edge_tts(text))
        loop.close()
        
        
async def async_edge_tts(text: str):
    global tts_playing

    tts_playing = True

    communicate = edge_tts.Communicate(
        text=text,
        voice=TTS_VOICE,
        rate="+0%"
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        temp_path = f.name

    try:
        await communicate.save(temp_path)

        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.05)

    finally:
        tts_playing = False

        try:
            pygame.mixer.music.unload()
        except:
            pass

        try:
            os.remove(temp_path)
        except:
            pass
        

def speech_worker():
    global running

    while state.running:
        try:
            text = state.speech_queue.get(timeout=0.2)

        except queue.Empty:
            continue

        try:
            with state.speech_lock:
                speak(text)

        except Exception as e:
            print(f"[TTS ERROR] {e}")

        state.speech_queue.task_done()
        

def enqueue_speech(text: str):
    text = str(text).strip()

    if not text:
        return

    state.speech_queue.put(text)
    

def clear_speech_queue():

    while not state.speech_queue.empty():

        try:
            state.speech_queue.get_nowait()
            state.speech_queue.task_done()

        except queue.Empty:
            break