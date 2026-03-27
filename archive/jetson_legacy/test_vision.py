#!/usr/bin/env python3
"""Test script to capture image and describe it using LLaVa."""

import sys
import os
import tempfile
import subprocess
import cv2

# Add the rovy directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smart_assistant import LLaVaAssistant
try:
    from oakd_depth_navigator import OakDDepthCamera
except ImportError:
    print("⚠️  Could not import OakDDepthCamera")
    OakDDepthCamera = None

def test_vision():
    """Test capturing an image and describing it."""
    print("="*60)
    print("Testing Vision Capabilities")
    print("="*60)
    
    # Initialize camera
    print("\n[Test] Initializing camera...")
    if OakDDepthCamera is None:
        print("[Test] ❌ OakDDepthCamera not available")
        return
    
    try:
        camera = OakDDepthCamera(resolution=(640, 480))
        camera.start()
        print("[Test] ✅ Camera initialized")
    except Exception as e:
        print(f"[Test] ❌ Camera initialization failed: {e}")
        return
    
    # Capture image
    print("\n[Test] Capturing image...")
    try:
        rgb_frame, _ = camera.capture_frames()
        print(f"[Test] ✅ Image captured (shape: {rgb_frame.shape})")
        
        # Save image for inspection
        test_image_path = "/tmp/test_capture.jpg"
        cv2.imwrite(test_image_path, rgb_frame)
        print(f"[Test] Image saved to: {test_image_path}")
    except Exception as e:
        print(f"[Test] ❌ Image capture failed: {e}")
        camera.stop()
        return
    
    # Initialize LLaVa
    print("\n[Test] Initializing LLaVa...")
    try:
        # Find model paths
        llava_model_path = None
        possible_model_paths = [
            "/home/jetson/.cache/llava-v1.5-7b-q4.gguf",
            "/home/jetson/.cache/llava-v1.5-7b-q5.gguf",
            "/home/jetson/models/llava-v1.5-7b-Q4_K_M.gguf",
        ]
        for path in possible_model_paths:
            if os.path.exists(path):
                llava_model_path = path
                break
        
        llava_mmproj_path = None
        possible_mmproj_paths = [
            "/home/jetson/.cache/llava-mmproj-fixed.gguf",
            "/home/jetson/.cache/llava-mmproj-f16.gguf",
            "/home/jetson/models/mmproj-model-f16.gguf",
        ]
        for path in possible_mmproj_paths:
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                llava_mmproj_path = path
                break
        
        if not llava_model_path:
            print("[Test] ❌ LLaVa model not found")
            camera.stop()
            return
        
        if not llava_mmproj_path:
            print("[Test] ❌ LLaVa mmproj not found")
            camera.stop()
            return
        
        print(f"[Test] Model: {llava_model_path}")
        print(f"[Test] MMProj: {llava_mmproj_path}")
        
        llava = LLaVaAssistant(llava_model_path, llava_mmproj_path)
        print("[Test] ✅ LLaVa initialized")
    except Exception as e:
        print(f"[Test] ❌ LLaVa initialization failed: {e}")
        camera.stop()
        return
    
    # Ask question with image
    print("\n[Test] Asking: 'What do you see?'")
    print("-"*60)
    try:
        response = llava.ask(
            "What do you see? Describe what is actually visible in the image. Be specific and concise.",
            image=rgb_frame,
            max_tokens=80,
            temperature=0.3
        )
        print("\n" + "="*60)
        print("RESPONSE:")
        print("="*60)
        print(response)
        print("="*60)
    except Exception as e:
        print(f"[Test] ❌ Description failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    print("\n[Test] Cleaning up...")
    try:
        if hasattr(camera, 'stop'):
            camera.stop()
        elif hasattr(camera, 'close'):
            camera.close()
    except:
        pass
    print("[Test] ✅ Test complete")

if __name__ == "__main__":
    test_vision()

