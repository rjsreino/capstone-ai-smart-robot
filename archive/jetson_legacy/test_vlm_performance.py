#!/usr/bin/env python3
"""
VLM Performance Testing Script
Tests LLaVA model efficiency, speed, memory usage, and quality
"""

import os
import sys
import time
import cv2
import numpy as np
import psutil
import subprocess
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the LLaVa assistant
from smart_assistant import LLaVaAssistant

def get_gpu_memory():
    """Get GPU memory usage on Jetson."""
    try:
        # Try tegrastats (Jetson-specific)
        result = subprocess.run(
            ["tegrastats", "--interval", "100"],
            capture_output=True,
            text=True,
            timeout=0.2
        )
        output = result.stdout
        # Parse EMC usage as proxy for GPU memory
        if "EMC" in output:
            # Extract EMC percentage
            import re
            match = re.search(r'EMC_FREQ (\d+)%', output)
            if match:
                return f"GPU Load: {match.group(1)}%"
    except:
        pass
    
    # Fallback to general info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            used, total = result.stdout.strip().split(',')
            return f"GPU Memory: {used.strip()}MB / {total.strip()}MB"
    except:
        pass
    
    return "GPU info not available"

def get_system_memory():
    """Get system RAM usage."""
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024**3)
    total_gb = mem.total / (1024**3)
    percent = mem.percent
    return used_gb, total_gb, percent

def get_cpu_usage():
    """Get CPU usage percentage."""
    return psutil.cpu_percent(interval=0.1)

def print_system_stats(prefix=""):
    """Print current system statistics."""
    used_gb, total_gb, mem_percent = get_system_memory()
    cpu_percent = get_cpu_usage()
    gpu_info = get_gpu_memory()
    
    print(f"{prefix}System Stats:")
    print(f"  RAM: {used_gb:.1f}GB / {total_gb:.1f}GB ({mem_percent:.1f}%)")
    print(f"  CPU: {cpu_percent:.1f}%")
    print(f"  {gpu_info}")

def capture_test_image(camera_id=0):
    """Capture an image from camera for testing."""
    print(f"\n📷 Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera {camera_id}")
        print("   Trying camera 1...")
        cap = cv2.VideoCapture(1)
        
    if not cap.isOpened():
        print("❌ No camera available, using test pattern")
        # Create a test pattern
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "TEST PATTERN", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        return img, None
    
    # Let camera warm up
    print("   Warming up camera...")
    for i in range(5):
        cap.read()
        time.sleep(0.1)
    
    # Capture image
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Failed to capture frame")
        cap.release()
        return None, None
    
    print(f"   ✅ Captured {frame.shape[1]}x{frame.shape[0]} image")
    
    return frame, cap

def test_vision_inference(llava, image, test_name="Vision Test"):
    """Test vision inference with detailed metrics."""
    print(f"\n{'='*60}")
    print(f"🔍 {test_name}")
    print(f"{'='*60}")
    
    # Get baseline stats
    print("\n📊 Before inference:")
    print_system_stats("  ")
    
    # Test questions
    questions = [
        "Describe what you see in this image in detail.",
        "What objects can you identify?",
        "What are the main colors in this image?",
    ]
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Test {i}/{len(questions)} ---")
        print(f"❓ Question: {question}")
        
        # Measure inference time
        start_time = time.time()
        start_mem = psutil.virtual_memory().used / (1024**3)
        
        try:
            answer = llava.ask(question, image=image, max_tokens=100, temperature=0.7)
            
            elapsed = time.time() - start_time
            end_mem = psutil.virtual_memory().used / (1024**3)
            mem_delta = end_mem - start_mem
            
            print(f"\n✅ Answer: {answer}")
            print(f"\n⏱️  Inference time: {elapsed:.2f}s")
            print(f"💾 Memory delta: {mem_delta:+.2f}GB")
            
            # Calculate tokens per second (rough estimate)
            words = len(answer.split())
            tokens_estimate = int(words * 1.3)  # Rough token estimate
            tokens_per_sec = tokens_estimate / elapsed if elapsed > 0 else 0
            
            print(f"📝 Response length: {len(answer)} chars, ~{tokens_estimate} tokens")
            print(f"⚡ Speed: ~{tokens_per_sec:.1f} tokens/sec")
            
            results.append({
                'question': question,
                'answer': answer,
                'time': elapsed,
                'tokens': tokens_estimate,
                'tokens_per_sec': tokens_per_sec,
                'mem_delta': mem_delta
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'question': question,
                'error': str(e)
            })
        
        # Show stats after each inference
        print_system_stats("\n  ")
        
        # Small delay between tests
        if i < len(questions):
            print("\n⏳ Waiting 2 seconds...")
            time.sleep(2)
    
    return results

def test_text_only_inference(llava, test_name="Text-Only Test"):
    """Test text-only inference (no vision)."""
    print(f"\n{'='*60}")
    print(f"💬 {test_name}")
    print(f"{'='*60}")
    
    # Get baseline stats
    print("\n📊 Before inference:")
    print_system_stats("  ")
    
    # Test questions
    questions = [
        "What is the capital of France?",
        "Explain what artificial intelligence is in one sentence.",
        "What are the three primary colors?",
    ]
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Test {i}/{len(questions)} ---")
        print(f"❓ Question: {question}")
        
        # Measure inference time
        start_time = time.time()
        start_mem = psutil.virtual_memory().used / (1024**3)
        
        try:
            answer = llava.ask(question, image=None, max_tokens=100, temperature=0.7)
            
            elapsed = time.time() - start_time
            end_mem = psutil.virtual_memory().used / (1024**3)
            mem_delta = end_mem - start_mem
            
            print(f"\n✅ Answer: {answer}")
            print(f"\n⏱️  Inference time: {elapsed:.2f}s")
            print(f"💾 Memory delta: {mem_delta:+.2f}GB")
            
            # Calculate tokens per second
            words = len(answer.split())
            tokens_estimate = int(words * 1.3)
            tokens_per_sec = tokens_estimate / elapsed if elapsed > 0 else 0
            
            print(f"📝 Response length: {len(answer)} chars, ~{tokens_estimate} tokens")
            print(f"⚡ Speed: ~{tokens_per_sec:.1f} tokens/sec")
            
            results.append({
                'question': question,
                'answer': answer,
                'time': elapsed,
                'tokens': tokens_estimate,
                'tokens_per_sec': tokens_per_sec,
                'mem_delta': mem_delta
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'question': question,
                'error': str(e)
            })
        
        # Show stats after each inference
        print_system_stats("\n  ")
        
        # Small delay between tests
        if i < len(questions):
            print("\n⏳ Waiting 2 seconds...")
            time.sleep(2)
    
    return results

def print_summary(vision_results, text_results):
    """Print summary of all tests."""
    print(f"\n{'='*60}")
    print("📊 PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    # Vision results
    print("\n🔍 VISION INFERENCE:")
    vision_times = [r['time'] for r in vision_results if 'time' in r]
    vision_speeds = [r['tokens_per_sec'] for r in vision_results if 'tokens_per_sec' in r]
    
    if vision_times:
        print(f"  Average time: {np.mean(vision_times):.2f}s")
        print(f"  Min time: {np.min(vision_times):.2f}s")
        print(f"  Max time: {np.max(vision_times):.2f}s")
        print(f"  Average speed: {np.mean(vision_speeds):.1f} tokens/sec")
    
    # Text results
    print("\n💬 TEXT-ONLY INFERENCE:")
    text_times = [r['time'] for r in text_results if 'time' in r]
    text_speeds = [r['tokens_per_sec'] for r in text_results if 'tokens_per_sec' in r]
    
    if text_times:
        print(f"  Average time: {np.mean(text_times):.2f}s")
        print(f"  Min time: {np.min(text_times):.2f}s")
        print(f"  Max time: {np.max(text_times):.2f}s")
        print(f"  Average speed: {np.mean(text_speeds):.1f} tokens/sec")
    
    # Comparison
    if vision_times and text_times:
        print("\n📈 COMPARISON:")
        vision_avg = np.mean(vision_times)
        text_avg = np.mean(text_times)
        print(f"  Vision vs Text ratio: {vision_avg/text_avg:.2f}x slower")
        print(f"  Vision overhead: +{vision_avg - text_avg:.2f}s per query")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if text_times and np.mean(text_times) > 5.0:
        print("  ⚠️  Text inference is slow (>5s). Consider:")
        print("      - Using a smaller dedicated LLM for text-only queries")
        print("      - Options: Phi-3-mini (3.8B), Qwen2-1.5B, TinyLlama (1.1B)")
    
    if vision_times and np.mean(vision_times) > 10.0:
        print("  ⚠️  Vision inference is slow (>10s). Consider:")
        print("      - Using a smaller VLM like LLaVA-Phi (3B)")
        print("      - Reducing image resolution")
        print("      - Increasing GPU layers (currently 8)")
    
    if text_speeds and np.mean(text_speeds) < 5.0:
        print("  ⚠️  Token generation is slow (<5 tok/s). Consider:")
        print("      - Using int4 quantization (Q4_K_M)")
        print("      - Reducing context window")
        print("      - More GPU layers if memory allows")
    
    print(f"\n{'='*60}")

def main():
    """Main test function."""
    print("="*60)
    print("🧪 VLM PERFORMANCE TEST")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initial system stats
    print("\n📊 Initial System State:")
    print_system_stats("  ")
    
    # Initialize LLaVa with explicit paths
    print("\n🚀 Initializing LLaVa model...")
    
    # Use the actual model paths found in .cache
    model_path = "/home/jetson/.cache/llava-v1.5-7b-q4.gguf"  # Q4 for speed
    mmproj_path = "/home/jetson/.cache/llava-mmproj-fixed.gguf"  # Use the working mmproj file
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Please check the model path or download LLaVA 1.5 7B Q4")
        return
    
    if not os.path.exists(mmproj_path):
        print(f"❌ MMProj not found: {mmproj_path}")
        return
    
    print(f"✅ Using model: {model_path}")
    print(f"✅ Using mmproj: {mmproj_path}")
    
    init_start = time.time()
    
    try:
        llava = LLaVaAssistant(model_path=model_path, mmproj_path=mmproj_path)
        init_time = time.time() - init_start
        print(f"✅ Model loaded in {init_time:.2f}s")
    except Exception as e:
        print(f"❌ Failed to initialize LLaVa: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Show stats after loading
    print("\n📊 After Model Load:")
    print_system_stats("  ")
    
    # Capture test image
    image, cap = capture_test_image()
    
    if image is not None:
        # Save test image for reference
        test_img_path = "/tmp/test_vlm_image.jpg"
        cv2.imwrite(test_img_path, image)
        print(f"💾 Test image saved to: {test_img_path}")
        
        # Show preview (if display available)
        try:
            cv2.imshow("Test Image", image)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        except:
            pass
    
    # Run tests
    vision_results = []
    text_results = []
    
    if image is not None:
        vision_results = test_vision_inference(llava, image)
    
    text_results = test_text_only_inference(llava)
    
    # Print summary
    print_summary(vision_results, text_results)
    
    # Cleanup
    if cap is not None:
        cap.release()
    
    print(f"\n✅ Testing complete!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save results to file
    results_file = f"/tmp/vlm_test_results_{int(time.time())}.txt"
    try:
        with open(results_file, 'w') as f:
            f.write(f"VLM Performance Test Results\n")
            f.write(f"{'='*60}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Vision Results:\n")
            for r in vision_results:
                if 'time' in r:
                    f.write(f"  Q: {r['question']}\n")
                    f.write(f"  A: {r['answer']}\n")
                    f.write(f"  Time: {r['time']:.2f}s, Speed: {r['tokens_per_sec']:.1f} tok/s\n\n")
            
            f.write("\nText Results:\n")
            for r in text_results:
                if 'time' in r:
                    f.write(f"  Q: {r['question']}\n")
                    f.write(f"  A: {r['answer']}\n")
                    f.write(f"  Time: {r['time']:.2f}s, Speed: {r['tokens_per_sec']:.1f} tok/s\n\n")
        
        print(f"\n📄 Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️  Failed to save results: {e}")

if __name__ == "__main__":
    main()

