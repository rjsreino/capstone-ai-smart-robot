#!/usr/bin/env python3
"""
Voice Localization Demo (Lightweight)
Demonstrates using ReSpeaker DOA to locate voice and OAK-D to detect person
No LLaVA - for faster testing!

Note: Core logic has been extracted to voice_localization.py module for reuse
"""
import time
from smart_assistant import ReSpeakerInterface, TextToSpeech, WakeWordDetector
from rover_controller import Rover
from oakd_depth_navigator import OakDDepthCamera
from face_recognizer import FaceRecognizer

def main():
    print("\n" + "="*70)
    print("🎯 Voice Localization + Person Detection Demo")
    print("="*70)
    print("\nThis demo will:")
    print("1. Use ReSpeaker's DOA to detect where your voice is coming from")
    print("2. Automatically move the camera to point towards your voice")
    print("3. Use OAK-D's person detection to verify you're there")
    print("\nMake sure you:")
    print("  - Have ReSpeaker Mic Array v2.0 connected")
    print("  - Have OAK-D camera connected")
    print("  - Rover controller is connected")
    print("="*70 + "\n")
    
    # Initialize components
    print("[Setup] Initializing rover controller...")
    rover = Rover()
    
    print("[Setup] Initializing OAK-D camera with YOLOv8 person detection...")
    camera = OakDDepthCamera(resolution=(640, 352), enable_person_detection=True)  # Match YOLOv8 model resolution
    camera.start()
    
    print("[Setup] Initializing ReSpeaker...")
    from smart_assistant import ReSpeakerInterface, TextToSpeech, WakeWordDetector
    
    respeaker = ReSpeakerInterface(use_whisper=False)
    tts = TextToSpeech(engine='piper', print_only=False)  # Use professional voice
    wake_detector = WakeWordDetector(device_index=respeaker.device_index)
    
    print("[Setup] Initializing face recognition...")
    face_recognizer = FaceRecognizer(known_dir="known-faces")
    
    print("\n✅ Setup complete!\n")
    
    # Initialize gimbal - IMPORTANT: Must unlock servos first
    print("[Camera] Initializing gimbal...")
    print("[Camera] Unlocking servos (T:135)...")
    rover.gimbal_unlock()
    time.sleep(0.5)
    
    print("[Camera] Centering camera to forward position...")
    rover.gimbal_ctrl_move(0, 0, input_speed_x=500, input_speed_y=500)
    time.sleep(2.0)
    
    try:
        # Main demo loop
        print("\n" + "="*70)
        print("DEMO INSTRUCTIONS:")
        print("="*70)
        print("1. Stand somewhere around the rover")
        print("2. Say 'Hey Rovy' and keep talking for 1-2 seconds")
        print("   (e.g., 'Hey Rovy, hello there!')")
        print("3. The rover will detect your voice direction")
        print("4. The camera will turn towards you")
        print("5. The rover will say 'Yes, I see you!'")
        print("\nPress Ctrl+C to exit")
        print("="*70 + "\n")
        
        while True:
            print("\n👂 Say 'Hey Rovy' to activate voice localization...")
            
            # Listen for wake word
            wake_result = wake_detector.listen_for_wake_word(timeout=30)
            if wake_result:
                print("✅ Wake word detected! Capturing voice direction...")
                tts.speak("Yes?")
                
                # Use the reusable voice localization module
                from voice_localization import locate_speaker
                
                result = locate_speaker(
                    respeaker=respeaker,
                    rover=rover,
                    camera=camera,
                    face_recognizer=face_recognizer,
                    tts=tts
                )
                
                # Show results
                if result['success']:
                    print("\n" + "-"*70)
                    print("RESULTS:")
                    print("-"*70)
                    print(f"✅ Voice Direction: {result['doa']}° (0°=front, 90°=left, 180°=back, 270°=right)")
                    
                    if result['pan'] is not None:
                        angle_map = {
                            -180: "FAR LEFT", -135: "BACK LEFT", -90: "LEFT", -45: "FRONT LEFT",
                            0: "CENTER", 45: "FRONT RIGHT", 90: "RIGHT", 135: "BACK RIGHT", 180: "FAR RIGHT"
                        }
                        pan_desc = angle_map.get(result['pan'], f"{result['pan']}°")
                        print(f"✅ Camera panned to: {pan_desc} ({result['pan']}°)")
                    
                    if result['person_found']:
                        if result['recognized_name']:
                            print(f"✅ Person detected and recognized as: {result['recognized_name']}")
                        else:
                            print(f"✅ Person detected")
                    else:
                        print(f"⚠️  No person detected - Acknowledged with 'I heard you!'")
                    print("-"*70)
                else:
                    print(f"❌ {result['message']}")
                
                # Return camera to center
                print("\n[Camera] Returning to center position...")
                rover.gimbal_ctrl_move(0, 0, input_speed_x=500, input_speed_y=500)
                time.sleep(1.5)
                
                # Ask if user wants to try again
                choice = input("\n🔄 Try again? (y/n): ").lower()
                if choice != 'y':
                    break
            else:
                print("⏰ No wake word detected, listening again...")
    
    except KeyboardInterrupt:
        print("\n\n[Demo] Interrupted by user")
    
    finally:
        # Cleanup
        print("\n[Cleanup] Shutting down...")
        
        # Turn off flash lights
        print("[Cleanup] Turning off flash lights...")
        rover.lights_ctrl(0, 0)
        
        # Return camera to center
        rover.gimbal_ctrl_move(0, 0, input_speed_x=500, input_speed_y=500)
        time.sleep(0.5)
        
        # Close camera
        if camera:
            camera.close()
        
        print("[Cleanup] Done!")
        print("\n" + "="*70)
        print("👋 Thanks for using the voice localization demo!")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()

