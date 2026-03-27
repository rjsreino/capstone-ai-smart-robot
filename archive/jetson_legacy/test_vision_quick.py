#!/usr/bin/env python3
"""
Quick vision test with proper chat_handler configuration
"""
import os
import sys
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dual_model_assistant import DualModelAssistant

def main():
    print("="*70)
    print("🔍 QUICK VISION TEST (with proper chat_handler)")
    print("="*70)
    
    # Capture image
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera unavailable")
        return
    
    for _ in range(5):
        cap.read()
        time.sleep(0.1)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Failed to capture image")
        return
    
    print(f"✅ Captured {frame.shape[1]}x{frame.shape[0]} image\n")
    
    # Initialize assistant
    print("🚀 Initializing assistant...\n")
    assistant = DualModelAssistant(lazy_load_vision=True)
    
    # Test questions
    questions = [
        "Describe everything you see in this image in as much detail as possible.",
        "What is the person doing in this image?",
        "What objects can you identify?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {question}")
        print(f"{'='*70}\n")
        
        start = time.time()
        answer = assistant.ask_vision(question, frame, max_tokens=80, temperature=0.2)
        elapsed = time.time() - start
        
        print(f"\n✅ ANSWER: {answer if answer else '(EMPTY)'}")
        print(f"⏱️  Time: {elapsed:.2f}s")
        print(f"📝 Length: {len(answer)} chars")
        
        if not answer:
            print("\n⚠️  WARNING: EMPTY RESPONSE!")
        
        if i < len(questions):
            print("\n⏳ Waiting 1 second...")
            time.sleep(1)
    
    print(f"\n{'='*70}")
    print("✅ Test complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

