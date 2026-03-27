#!/usr/bin/env python3
"""
Debug vision - save captured image and test
"""
import os
import sys
import time
import cv2
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dual_model_assistant import DualModelAssistant

def main():
    print("="*70)
    print("🔍 VISION DEBUG TEST")
    print("="*70)
    
    # Capture image
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera unavailable")
        return
    
    # Warm up camera
    for _ in range(5):
        cap.read()
        time.sleep(0.1)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Failed to capture image")
        return
    
    print(f"✅ Captured {frame.shape[1]}x{frame.shape[0]} image\n")
    
    # Save captured image for inspection
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = f"/home/jetson/rovy/debug_capture_{timestamp}.jpg"
    cv2.imwrite(debug_path, frame)
    print(f"💾 Saved debug image to: {debug_path}")
    print(f"   You can view it with: display {debug_path}\n")
    
    # Show image stats
    print(f"Image info:")
    print(f"  - Shape: {frame.shape}")
    print(f"  - Dtype: {frame.dtype}")
    print(f"  - Min/Max pixel values: {frame.min()}/{frame.max()}")
    print(f"  - Mean pixel value: {frame.mean():.1f}\n")
    
    # Initialize assistant
    print("🚀 Initializing assistant...\n")
    assistant = DualModelAssistant(lazy_load_vision=True)
    
    # Single test with very simple, direct prompt
    question = "What do you see?"
    print(f"Question: {question}\n")
    
    start = time.time()
    answer = assistant.ask_vision(question, frame, max_tokens=50, temperature=0.1)
    elapsed = time.time() - start
    
    print(f"\n{'='*70}")
    print(f"ANSWER: {answer}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Length: {len(answer)} chars")
    print(f"{'='*70}\n")
    
    # Try one more with even more specific prompt
    question2 = "List only the objects you can clearly see."
    print(f"Question 2: {question2}\n")
    
    start = time.time()
    answer2 = assistant.ask_vision(question2, frame, max_tokens=50, temperature=0.1)
    elapsed = time.time() - start
    
    print(f"\n{'='*70}")
    print(f"ANSWER 2: {answer2}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Length: {len(answer2)} chars")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
