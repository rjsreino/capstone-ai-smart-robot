#!/usr/bin/env python3
"""
Dual Model Performance Test
Tests the optimized dual-model setup (Gemma-2-2B + LLaVA-Phi-3-Mini)
"""

import os
import sys
import time
import cv2
import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dual_model_assistant import DualModelAssistant

def get_system_memory():
    """Get system RAM usage."""
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024**3)
    total_gb = mem.total / (1024**3)
    percent = mem.percent
    return used_gb, total_gb, percent

def print_system_stats(prefix=""):
    """Print current system statistics."""
    used_gb, total_gb, mem_percent = get_system_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    print(f"{prefix}System Stats:")
    print(f"  RAM: {used_gb:.1f}GB / {total_gb:.1f}GB ({mem_percent:.1f}%)")
    print(f"  CPU: {cpu_percent:.1f}%")

def capture_test_image():
    """Capture test image from camera."""
    print(f"\n📷 Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera unavailable, using test pattern")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "TEST", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        return img, None
    
    # Warm up
    for _ in range(5):
        cap.read()
        time.sleep(0.1)
    
    ret, frame = cap.read()
    if ret:
        print(f"   ✅ Captured {frame.shape[1]}x{frame.shape[0]} image")
        return frame, cap
    
    cap.release()
    return None, None

def test_text_inference(assistant):
    """Test text-only inference speed."""
    print(f"\n{'='*60}")
    print(f"💬 TEXT INFERENCE TEST (Gemma-2-2B)")
    print(f"{'='*60}")
    
    print("\n📊 Before text tests:")
    print_system_stats("  ")
    
    questions = [
        "What is the capital of France?",
        "Explain artificial intelligence in one sentence.",
        "What are the three primary colors?",
        "What is 15 multiplied by 8?",
        "Name three planets in our solar system.",
    ]
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Text Test {i}/{len(questions)} ---")
        print(f"❓ Question: {question}")
        
        start_time = time.time()
        start_mem = psutil.virtual_memory().used / (1024**3)
        
        try:
            answer = assistant.ask_text(question, max_tokens=100, temperature=0.7)
            
            elapsed = time.time() - start_time
            end_mem = psutil.virtual_memory().used / (1024**3)
            mem_delta = end_mem - start_mem
            
            print(f"\n✅ Answer: {answer}")
            print(f"\n⏱️  Time: {elapsed:.2f}s")
            print(f"💾 Memory: {mem_delta:+.2f}GB")
            
            words = len(answer.split())
            tokens_estimate = int(words * 1.3)
            tokens_per_sec = tokens_estimate / elapsed if elapsed > 0 else 0
            
            print(f"📝 Length: {len(answer)} chars, ~{tokens_estimate} tokens")
            print(f"⚡ Speed: ~{tokens_per_sec:.1f} tokens/sec")
            
            results.append({
                'question': question,
                'answer': answer,
                'time': elapsed,
                'tokens': tokens_estimate,
                'tokens_per_sec': tokens_per_sec,
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({'question': question, 'error': str(e)})
        
        print_system_stats("\n  ")
        
        if i < len(questions):
            time.sleep(1)
    
    return results

def test_vision_inference(assistant, image):
    """Test vision inference speed."""
    print(f"\n{'='*60}")
    print(f"🔍 VISION INFERENCE TEST (LLaVA-Phi-3-Mini)")
    print(f"{'='*60}")
    
    print("\n📊 Before vision tests:")
    print_system_stats("  ")
    
    questions = [
        "Describe what you see in this image in detail.",
        "What objects can you identify?",
        "What are the main colors in this image?",
    ]
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Vision Test {i}/{len(questions)} ---")
        print(f"❓ Question: {question}")
        
        start_time = time.time()
        start_mem = psutil.virtual_memory().used / (1024**3)
        
        try:
            answer = assistant.ask_vision(question, image, max_tokens=150, temperature=0.3)
            
            elapsed = time.time() - start_time
            end_mem = psutil.virtual_memory().used / (1024**3)
            mem_delta = end_mem - start_mem
            
            print(f"\n✅ Answer: {answer}")
            print(f"\n⏱️  Time: {elapsed:.2f}s")
            print(f"💾 Memory: {mem_delta:+.2f}GB")
            
            words = len(answer.split())
            tokens_estimate = int(words * 1.3)
            tokens_per_sec = tokens_estimate / elapsed if elapsed > 0 else 0
            
            print(f"📝 Length: {len(answer)} chars, ~{tokens_estimate} tokens")
            print(f"⚡ Speed: ~{tokens_per_sec:.1f} tokens/sec")
            
            results.append({
                'question': question,
                'answer': answer,
                'time': elapsed,
                'tokens': tokens_estimate,
                'tokens_per_sec': tokens_per_sec,
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({'question': question, 'error': str(e)})
        
        print_system_stats("\n  ")
        
        if i < len(questions):
            time.sleep(2)
    
    return results

def print_summary(text_results, vision_results):
    """Print performance summary."""
    print(f"\n{'='*60}")
    print("📊 PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    # Text results
    print("\n💬 TEXT INFERENCE (Gemma-2-2B):")
    text_times = [r['time'] for r in text_results if 'time' in r]
    text_speeds = [r['tokens_per_sec'] for r in text_results if 'tokens_per_sec' in r]
    
    if text_times:
        print(f"  Average time: {np.mean(text_times):.2f}s")
        print(f"  Min time: {np.min(text_times):.2f}s")
        print(f"  Max time: {np.max(text_times):.2f}s")
        print(f"  Average speed: {np.mean(text_speeds):.1f} tokens/sec")
    
    # Vision results
    print("\n🔍 VISION INFERENCE (LLaVA-Phi-3-Mini):")
    vision_times = [r['time'] for r in vision_results if 'time' in r]
    vision_speeds = [r['tokens_per_sec'] for r in vision_results if 'tokens_per_sec' in r]
    
    if vision_times:
        print(f"  Average time: {np.mean(vision_times):.2f}s")
        print(f"  Min time: {np.min(vision_times):.2f}s")
        print(f"  Max time: {np.max(vision_times):.2f}s")
        print(f"  Average speed: {np.mean(vision_speeds):.1f} tokens/sec")
    
    # Comparison with old results
    print("\n📈 COMPARISON WITH OLD SETUP:")
    print("  Old LLaVA 7B (text): ~3.0 tok/s, 4.1s avg")
    print("  Old LLaVA 7B (vision): ~2.6 tok/s, 20.1s avg")
    
    if text_speeds:
        old_text_speed = 3.0
        new_text_speed = np.mean(text_speeds)
        improvement = (new_text_speed / old_text_speed)
        print(f"  Text speedup: {improvement:.1f}x faster! ✨")
    
    if vision_speeds:
        old_vision_time = 20.1
        new_vision_time = np.mean(vision_times)
        improvement = (old_vision_time / new_vision_time)
        print(f"  Vision speedup: {improvement:.1f}x faster! ✨")
    
    print(f"\n{'='*60}")

def main():
    """Main test function."""
    print("="*60)
    print("🧪 DUAL MODEL PERFORMANCE TEST")
    print("="*60)
    print("Testing: Gemma-2-2B (text) + Vision Model (auto-detected)")
    
    # Initial stats
    print("\n📊 Initial System State:")
    print_system_stats("  ")
    
    # Initialize assistant
    print("\n🚀 Initializing Dual Model Assistant...")
    init_start = time.time()
    
    try:
        assistant = DualModelAssistant(lazy_load_vision=True)
        init_time = time.time() - init_start
        print(f"✅ Assistant initialized in {init_time:.2f}s")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Show stats after loading
    print("\n📊 After Text Model Load:")
    print_system_stats("  ")
    
    # Test text inference first
    text_results = test_text_inference(assistant)
    
    # Capture test image
    image, cap = capture_test_image()
    
    if image is not None:
        # Save test image
        cv2.imwrite("/tmp/test_dual_model_image.jpg", image)
        
        # Test vision inference
        vision_results = test_vision_inference(assistant, image)
    else:
        vision_results = []
    
    # Print summary
    print_summary(text_results, vision_results)
    
    # Cleanup
    if cap is not None:
        cap.release()
    
    print(f"\n✅ Testing complete!")

if __name__ == "__main__":
    main()

