#!/usr/bin/env python3
"""
TTS Voice Comparison Tool
Test different offline TTS voices to find the most natural one
"""
import os
import subprocess
import time

# Test sentences (robot-like responses)
TEST_SENTENCES = [
    "Hello, I am your robot assistant.",
    "The capital of France is Paris.",
    "I am ready to help you.",
    "Calculating: 25 percent of 80 is 20.",
    "All systems operational.",
    "Photosynthesis is how plants use sunlight to create food.",
]


class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def detect_audio_device():
    """Auto-detect USB audio output device."""
    try:
        result = subprocess.run(['aplay', '-l'], capture_output=True, text=True, timeout=3)
        if result.returncode != 0:
            return None
        
        lines = result.stdout.split('\n')
        for line in lines:
            if 'card' in line and 'USB Audio' in line:
                if 'ReSpeaker' not in line and 'ArrayUAC' not in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'card' and i + 1 < len(parts):
                            card_num = parts[i + 1].rstrip(':')
                            return f"plughw:{card_num},0"
        return None
    except:
        return None


def test_espeak(text, audio_device):
    """Test espeak (very robotic, but fast)."""
    print(f"\n{Colors.CYAN}[Testing: espeak]{Colors.END}")
    print(f"Quality: ⭐☆☆☆☆ (Very robotic)")
    print(f"Speed:   ⭐⭐⭐⭐⭐ (Very fast)")
    
    try:
        start = time.time()
        cmd = ['espeak', text]
        if audio_device:
            # espeak doesn't have direct device selection, use aplay
            process = subprocess.run(
                ['espeak', '--stdout', text],
                capture_output=True,
                timeout=10
            )
            subprocess.run(
                ['aplay', '-D', audio_device],
                input=process.stdout,
                timeout=10
            )
        else:
            subprocess.run(cmd, timeout=10)
        elapsed = time.time() - start
        print(f"{Colors.GREEN}✅ Completed in {elapsed:.2f}s{Colors.END}")
        return True
    except Exception as e:
        print(f"{Colors.RED}❌ Failed: {e}{Colors.END}")
        return False


def test_piper_voice(voice_path, voice_name, text, audio_device):
    """Test a Piper voice."""
    print(f"\n{Colors.CYAN}[Testing: Piper - {voice_name}]{Colors.END}")
    
    if not os.path.exists(voice_path):
        print(f"{Colors.RED}❌ Voice file not found: {voice_path}{Colors.END}")
        return False
    
    try:
        from piper import PiperVoice
        import numpy as np
        import wave
        import io
        
        print(f"Loading voice...")
        start_load = time.time()
        voice = PiperVoice.load(voice_path, use_cuda=False)
        load_time = time.time() - start_load
        print(f"Voice loaded in {load_time:.2f}s")
        
        print(f"Synthesizing...")
        start_synth = time.time()
        audio_chunks = []
        for chunk in voice.synthesize(text):
            # Use audio_int16_bytes for proper audio data
            audio_chunks.append(chunk.audio_int16_bytes)
        audio_bytes = b''.join(audio_chunks)
        synth_time = time.time() - start_synth
        
        # Play audio
        if audio_device:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(voice.config.sample_rate)
                wav_file.writeframes(audio_bytes)
            
            wav_buffer.seek(0)
            subprocess.run(
                ['aplay', '-D', audio_device],
                input=wav_buffer.read(),
                timeout=10
            )
        
        total_time = time.time() - start_load
        print(f"{Colors.GREEN}✅ Synthesis: {synth_time:.2f}s, Total: {total_time:.2f}s{Colors.END}")
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ Failed: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        return False


def test_festival(text, audio_device):
    """Test Festival TTS (better than espeak, still robotic)."""
    print(f"\n{Colors.CYAN}[Testing: Festival]{Colors.END}")
    print(f"Quality: ⭐⭐☆☆☆ (Slightly robotic)")
    print(f"Speed:   ⭐⭐⭐⭐☆ (Fast)")
    
    # Check if festival is installed
    try:
        subprocess.run(['festival', '--version'], capture_output=True, timeout=2)
    except:
        print(f"{Colors.YELLOW}⚠️  Festival not installed. Install with: sudo apt install festival{Colors.END}")
        return False
    
    try:
        start = time.time()
        # Create temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            temp_file = f.name
        
        # Generate speech
        subprocess.run(
            ['text2wave', temp_file, '-o', '/tmp/festival_out.wav'],
            timeout=10
        )
        
        # Play
        if audio_device:
            subprocess.run(['aplay', '-D', audio_device, '/tmp/festival_out.wav'], timeout=10)
        else:
            subprocess.run(['aplay', '/tmp/festival_out.wav'], timeout=10)
        
        os.unlink(temp_file)
        os.unlink('/tmp/festival_out.wav')
        
        elapsed = time.time() - start
        print(f"{Colors.GREEN}✅ Completed in {elapsed:.2f}s{Colors.END}")
        return True
    except Exception as e:
        print(f"{Colors.RED}❌ Failed: {e}{Colors.END}")
        return False


def main():
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}TTS VOICE COMPARISON TOOL{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
    
    # Detect audio device
    audio_device = detect_audio_device()
    if audio_device:
        print(f"{Colors.GREEN}✅ Audio device detected: {audio_device}{Colors.END}")
    else:
        print(f"{Colors.YELLOW}⚠️  No audio device detected - will try default{Colors.END}")
    
    # Test sentence
    test_text = TEST_SENTENCES[0]
    print(f"\n{Colors.BOLD}Test sentence:{Colors.END} \"{test_text}\"\n")
    
    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}AVAILABLE VOICES{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    
    voices_to_test = []
    
    # Festival (optional)
    try:
        subprocess.run(['festival', '--version'], capture_output=True, timeout=2)
        voices_to_test.append(('festival', None, '⭐⭐☆☆☆', 'Slightly robotic, fast'))
    except:
        pass
    
    # Piper voices - BEST NATURAL VOICES ONLY
    piper_voices = [
        # BEST/MOST NATURAL (recommended)
        ("~/.local/share/piper-voices/en_US-danny-low.onnx", "Piper Danny Low", '⭐⭐⭐⭐⭐', 'MOST NATURAL male voice'),
        ("~/.local/share/piper-voices/en_US-ryan-high.onnx", "Piper Ryan High", '⭐⭐⭐⭐⭐', 'Natural male, high quality'),
        ("~/.local/share/piper-voices/en_US-hfc_male-medium.onnx", "Piper HFC Male Medium", '⭐⭐⭐⭐☆', 'Clear male, professional'),
        # Female voices
        ("~/.local/share/piper-voices/en_US-kristin-medium.onnx", "Piper Kristin Medium", '⭐⭐⭐⭐⭐', 'Very natural female'),
        ("~/.local/share/piper-voices/en_US-amy-medium.onnx", "Piper Amy Medium", '⭐⭐⭐⭐☆', 'Natural female'),
        # Improved version
        ("~/.local/share/piper-voices/en_US-libritts_r-medium.onnx", "Piper LibriTTS_r Medium", '⭐⭐⭐⭐☆', 'Improved LibriTTS'),
    ]
    
    for path, name, quality, desc in piper_voices:
        full_path = os.path.expanduser(path)
        if os.path.exists(full_path):
            voices_to_test.append(('piper', (full_path, name), quality, desc))
    
    # Display menu
    print()
    for i, (engine, data, quality, desc) in enumerate(voices_to_test, 1):
        name = data[1] if engine == 'piper' else engine
        print(f"{i}. {Colors.BOLD}{name}{Colors.END}")
        print(f"   Quality: {quality}")
        print(f"   {desc}\n")
    
    print(f"0. Test all voices")
    print(f"q. Quit\n")
    
    # Interactive testing
    while True:
        try:
            choice = input(f"{Colors.CYAN}Select voice to test (0-{len(voices_to_test)}, q to quit): {Colors.END}").strip()
            
            if choice.lower() == 'q':
                break
            
            if choice == '0':
                # Test all
                print(f"\n{Colors.BOLD}Testing all voices...{Colors.END}")
                for i, sentence in enumerate(TEST_SENTENCES, 1):
                    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
                    print(f"{Colors.BOLD}Test {i}/{len(TEST_SENTENCES)}: \"{sentence}\"{Colors.END}")
                    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
                    
                    for engine, data, quality, desc in voices_to_test:
                        if engine == 'espeak':
                            test_espeak(sentence, audio_device)
                        elif engine == 'festival':
                            test_festival(sentence, audio_device)
                        elif engine == 'piper':
                            test_piper_voice(data[0], data[1], sentence, audio_device)
                        
                        time.sleep(0.5)  # Brief pause between voices
                    
                    if i < len(TEST_SENTENCES):
                        input(f"\n{Colors.YELLOW}Press Enter for next test...{Colors.END}")
                break
            
            idx = int(choice) - 1
            if 0 <= idx < len(voices_to_test):
                engine, data, quality, desc = voices_to_test[idx]
                
                # Ask for test sentence
                print(f"\n{Colors.BOLD}Select test sentence:{Colors.END}")
                for i, sentence in enumerate(TEST_SENTENCES, 1):
                    print(f"{i}. {sentence}")
                print(f"0. All sentences")
                
                sentence_choice = input(f"{Colors.CYAN}Choice (0-{len(TEST_SENTENCES)}): {Colors.END}").strip()
                
                if sentence_choice == '0':
                    sentences = TEST_SENTENCES
                else:
                    sidx = int(sentence_choice) - 1
                    if 0 <= sidx < len(TEST_SENTENCES):
                        sentences = [TEST_SENTENCES[sidx]]
                    else:
                        continue
                
                for sentence in sentences:
                    if engine == 'espeak':
                        test_espeak(sentence, audio_device)
                    elif engine == 'festival':
                        test_festival(sentence, audio_device)
                    elif engine == 'piper':
                        test_piper_voice(data[0], data[1], sentence, audio_device)
                    time.sleep(0.5)
            
        except (ValueError, KeyboardInterrupt):
            break
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.GREEN}✅ Testing complete!{Colors.END}")
    print(f"\n{Colors.BOLD}RECOMMENDATIONS:{Colors.END}")
    print(f"  {Colors.CYAN}Most Natural:{Colors.END} Piper Lessac High (if you have it)")
    print(f"  {Colors.CYAN}Best Balance:{Colors.END} Piper Lessac Medium")
    print(f"  {Colors.CYAN}Fastest:{Colors.END} espeak (but very robotic)")
    print(f"\n{Colors.YELLOW}To download more Piper voices:{Colors.END}")
    print(f"  Visit: https://huggingface.co/rhasspy/piper-voices")
    print(f"  Install to: ~/.local/share/piper-voices/")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}[!] Interrupted{Colors.END}\n")

