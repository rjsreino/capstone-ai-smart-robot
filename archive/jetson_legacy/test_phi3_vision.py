#!/usr/bin/env python3
"""
Test Phi-3-Vision with clean implementation
"""

import os
import sys
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phi3_vision_assistant import Phi3VisionAssistant


def capture_image():
    """Capture test image from camera."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera unavailable")
        return None
    
    print("📸 Warming up camera...")
    for _ in range(5):
        cap.read()
        time.sleep(0.1)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        print(f"✅ Captured {frame.shape[1]}x{frame.shape[0]} image\n")
        return frame
    return None


def main():
    print("="*70)
    print("🔍 PHI-3-VISION TEST")
    print("="*70)
    print("Using official Microsoft Phi-3-Vision implementation\n")
    
    # Capture image
    image = capture_image()
    if image is None:
        print("❌ Cannot proceed without image")
        return
    
    # Initialize assistant
    print("🚀 Initializing Phi-3-Vision...\n")
    assistant = Phi3VisionAssistant(lazy_load=False)
    
    # Test questions
    test_questions = [
        "Describe what you see in this image.",
        "What are the main colors?",
        "What objects can you identify?",
        "What is happening in this scene?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_questions)}")
        print(f"{'='*70}")
        print(f"Q: {question}\n")
        
        answer = assistant.ask(question, image, max_tokens=150, temperature=0.5)
        
        print(f"\n✅ A: {answer}")
        print(f"📊 Length: {len(answer)} chars, {len(answer.split())} words")
        
        if i < len(test_questions):
            print("\n⏳ Waiting 2 seconds...")
            time.sleep(2)
    
    print(f"\n{'='*70}")
    print("✅ Test complete!")
    print("="*70)


if __name__ == "__main__":
    main()

