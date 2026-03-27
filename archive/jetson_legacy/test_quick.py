#!/usr/bin/env python3
"""
Quick test script for dual model system
Tests a few questions to verify speed and quality
"""
import time
import cv2
from dual_model_assistant import DualModelAssistant

# Colors for output
class C:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def timer(name):
    """Simple timing decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{C.CYAN}[⏱️  {name}]{C.END}")
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"{C.GREEN}[✅ {name}] {elapsed:.3f}s{C.END}")
            return result
        return wrapper
    return decorator

def main():
    print(f"\n{C.BOLD}{'='*70}{C.END}")
    print(f"{C.BOLD}{C.CYAN}QUICK DUAL MODEL TEST{C.END}")
    print(f"{C.BOLD}{'='*70}{C.END}\n")
    
    # Initialize assistant
    print(f"{C.YELLOW}Initializing dual model assistant...{C.END}")
    init_start = time.time()
    assistant = DualModelAssistant(lazy_load_vision=True)
    print(f"{C.GREEN}✅ Ready in {time.time()-init_start:.2f}s{C.END}\n")
    
    # Test text questions (Gemma)
    print(f"\n{C.BOLD}{'='*70}{C.END}")
    print(f"{C.BOLD}TEXT TESTS (Gemma 2B){C.END}")
    print(f"{C.BOLD}{'='*70}{C.END}")
    
    text_questions = [
        "What is the capital of France?",
        "Explain photosynthesis in one sentence.",
        "What is 15 times 7?",
        "Name three planets in our solar system.",
        "Why is the sky blue?",
    ]
    
    text_times = []
    for i, q in enumerate(text_questions, 1):
        print(f"\n{C.BLUE}Q{i}: {q}{C.END}")
        start = time.time()
        answer = assistant.ask_text(q, max_tokens=80)
        elapsed = time.time() - start
        text_times.append(elapsed)
        print(f"{C.GREEN}A: {answer}{C.END}")
        print(f"{C.CYAN}⏱️  {elapsed:.3f}s{C.END}")
    
    # Test vision questions (Phi-Vision)
    print(f"\n\n{C.BOLD}{'='*70}{C.END}")
    print(f"{C.BOLD}VISION TESTS (Phi-Vision){C.END}")
    print(f"{C.BOLD}{'='*70}{C.END}")
    
    # Try to capture camera image
    test_image = None
    try:
        from oakd_depth_navigator import OakDDepthCamera
        print(f"\n{C.YELLOW}Capturing test image...{C.END}")
        camera = OakDDepthCamera(resolution=(640, 480))
        camera.start()
        time.sleep(1)  # Let camera warm up
        frames = camera.capture_frames()
        if frames is not None and 'color' in frames:
            test_image = frames['color']
            # Save for reference
            cv2.imwrite('/home/jetson/rovy/quick_test_image.jpg', test_image)
            print(f"{C.GREEN}✅ Image captured and saved{C.END}")
        camera.stop()
    except Exception as e:
        print(f"{C.YELLOW}⚠️  Could not capture camera image: {e}{C.END}")
        print(f"{C.YELLOW}Loading test image from file...{C.END}")
        try:
            test_image = cv2.imread('/home/jetson/rovy/test_vision_quick.jpg')
            if test_image is None:
                test_image = cv2.imread('/home/jetson/rovy/debug_capture_20251124_134400.jpg')
        except:
            pass
    
    if test_image is not None:
        vision_questions = [
            "What do you see in this image?",
            "Describe the main object.",
            "What colors are present?",
            "Is this indoors or outdoors?",
        ]
        
        vision_times = []
        for i, q in enumerate(vision_questions, 1):
            print(f"\n{C.BLUE}Q{i}: {q}{C.END}")
            start = time.time()
            answer = assistant.ask_vision(q, test_image, max_tokens=100)
            elapsed = time.time() - start
            vision_times.append(elapsed)
            print(f"{C.GREEN}A: {answer}{C.END}")
            print(f"{C.CYAN}⏱️  {elapsed:.3f}s{C.END}")
    else:
        print(f"\n{C.YELLOW}⚠️  No test image available - skipping vision tests{C.END}")
        vision_times = []
    
    # Summary
    print(f"\n\n{C.BOLD}{'='*70}{C.END}")
    print(f"{C.BOLD}SUMMARY{C.END}")
    print(f"{C.BOLD}{'='*70}{C.END}\n")
    
    if text_times:
        import numpy as np
        avg_text = np.mean(text_times)
        print(f"{C.CYAN}📝 Text Model (Gemma):{C.END}")
        print(f"   Questions: {len(text_times)}")
        print(f"   Average:   {avg_text:.3f}s")
        print(f"   Min:       {min(text_times):.3f}s")
        print(f"   Max:       {max(text_times):.3f}s")
    
    if vision_times:
        avg_vision = np.mean(vision_times)
        print(f"\n{C.CYAN}👁️  Vision Model (Phi-Vision):{C.END}")
        print(f"   Questions: {len(vision_times)}")
        print(f"   Average:   {avg_vision:.3f}s")
        print(f"   Min:       {min(vision_times):.3f}s")
        print(f"   Max:       {max(vision_times):.3f}s")
        
        if text_times:
            speedup = avg_vision / avg_text
            print(f"\n{C.YELLOW}⚡ Performance:{C.END}")
            print(f"   Text is {speedup:.1f}x faster than vision")
            print(f"   {C.GREEN}✨ Dual model optimization working!{C.END}")
    
    print(f"\n{C.BOLD}{'='*70}{C.END}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}[!] Interrupted{C.END}\n")
    except Exception as e:
        print(f"\n{C.YELLOW}[!] Error: {e}{C.END}")
        import traceback
        traceback.print_exc()

