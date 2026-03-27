#!/usr/bin/env python3
"""
Test Voice Localization in Smart Assistant
Quick test to verify "look at me" functionality
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smart_assistant import SmartAssistant
from rover_controller import Rover

def main():
    print("\n" + "="*70)
    print("🎯 Testing Voice Localization in Smart Assistant")
    print("="*70)
    print("\nThis test will:")
    print("1. Initialize the Smart Assistant")
    print("2. Listen for wake word + 'look at me' command")
    print("3. Turn camera to face your voice direction")
    print("4. Optionally detect and recognize you with the camera")
    print("\nMake sure you have:")
    print("  - ReSpeaker Mic Array v2.0 connected")
    print("  - Rover controller connected")
    print("  - (Optional) OAK-D camera for person detection")
    print("="*70 + "\n")
    
    # Initialize rover controller
    print("[Setup] Initializing rover controller...")
    try:
        rover = Rover()
        print("[Setup] ✅ Rover connected")
    except Exception as e:
        print(f"[Setup] ❌ Could not connect to rover: {e}")
        print("[Setup] Exiting...")
        return
    
    # Initialize camera (optional - for person detection)
    camera = None
    try:
        print("[Setup] Initializing OAK-D camera (optional)...")
        from oakd_depth_navigator import OakDDepthCamera
        camera = OakDDepthCamera(resolution=(640, 352), enable_person_detection=True)
        camera.start()
        print("[Setup] ✅ Camera initialized")
    except Exception as e:
        print(f"[Setup] ⚠️  Camera not available: {e}")
        print("[Setup] Will proceed without person detection")
    
    # Initialize Smart Assistant
    print("[Setup] Initializing Smart Assistant...")
    try:
        assistant = SmartAssistant(
            camera=camera,
            motor_controller=rover,
            print_only=False  # Enable audio output
        )
        print("[Setup] ✅ Assistant initialized")
    except Exception as e:
        print(f"[Setup] ❌ Could not initialize assistant: {e}")
        if camera:
            camera.close()
        return
    
    # Initialize gimbal
    print("\n[Camera] Initializing gimbal...")
    print("[Camera] Unlocking servos...")
    rover.gimbal_unlock()
    import time
    time.sleep(0.5)
    
    print("[Camera] Centering camera...")
    rover.gimbal_ctrl_move(0, 0, input_speed_x=500, input_speed_y=500)
    time.sleep(2.0)
    
    print("\n" + "="*70)
    print("TEST INSTRUCTIONS:")
    print("="*70)
    print("1. Say 'Hey Rovy' (wake word)")
    print("2. When prompted, say 'look at me'")
    print("3. The camera should turn to face your direction")
    print("4. The assistant will respond 'I see you!' or 'I've turned to face you!'")
    print("\nPress Ctrl+C to exit")
    print("="*70 + "\n")
    
    try:
        # Run interactive session
        assistant.run_interactive_session(
            duration=None,  # Run indefinitely
            use_wake_word=True,
            greeting=True
        )
    except KeyboardInterrupt:
        print("\n\n[Test] Interrupted by user")
    finally:
        # Cleanup
        print("\n[Cleanup] Shutting down...")
        
        # Turn off lights
        print("[Cleanup] Turning off lights...")
        rover.lights_ctrl(0, 0)
        
        # Return camera to center
        print("[Cleanup] Returning camera to center...")
        rover.gimbal_ctrl_move(0, 0, input_speed_x=500, input_speed_y=500)
        time.sleep(0.5)
        
        # Close camera
        if camera:
            camera.close()
        
        print("[Cleanup] Done!")
        print("\n" + "="*70)
        print("👋 Thanks for testing!")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()

