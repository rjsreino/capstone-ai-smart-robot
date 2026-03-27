#!/usr/bin/env python3
"""
Simple Voice Localization Demo (No Camera Required)
Demonstrates ReSpeaker DOA to locate voice and move camera servos
"""
import time
from smart_assistant import ReSpeakerInterface
from rover_controller import Rover

def main():
    print("\n" + "="*70)
    print("🎯 Voice Localization Demo (Servo Control Only)")
    print("="*70)
    print("\nThis demo will:")
    print("1. Use ReSpeaker's DOA to detect where your voice is coming from")
    print("2. Automatically move the camera servos to point towards your voice")
    print("\nMake sure you:")
    print("  - Have ReSpeaker Mic Array v2.0 connected")
    print("  - Rover controller is connected")
    print("="*70 + "\n")
    
    # Initialize components
    print("[Setup] Initializing rover controller...")
    rover = Rover()
    
    print("[Setup] Initializing ReSpeaker...")
    respeaker = ReSpeakerInterface(use_whisper=False)
    
    if not respeaker.doa_available:
        print("\n❌ ERROR: ReSpeaker DOA not available!")
        print("Please check:")
        print("  1. ReSpeaker is plugged in")
        print("  2. USB permissions are configured (unplug/replug after setup)")
        return
    
    print("\n✅ Setup complete!\n")
    
    # Center camera first
    print("[Camera] Centering camera to forward position...")
    rover.gimbal_ctrl(0, 0, input_speed=150, input_acceleration=10)
    time.sleep(1.5)
    
    try:
        print("\n" + "="*70)
        print("DEMO INSTRUCTIONS:")
        print("="*70)
        print("1. Stand somewhere around the rover")
        print("2. Say something (e.g., 'Hello' or talk continuously)")
        print("3. The camera will automatically turn towards your voice")
        print("\nPress Ctrl+C to exit")
        print("="*70 + "\n")
        
        print("🎤 The system will listen for your voice continuously...\n")
        
        last_doa = None
        
        while True:
            print("\n👂 Speak now (2 seconds)...")
            
            # Get current voice direction by listening for 2 seconds
            doa = respeaker.get_voice_direction(listen_duration=2.0)
            
            if doa is not None:
                # Voice detected!
                
                # Only move camera if direction changed significantly (>15 degrees)
                if last_doa is None or abs(doa - last_doa) > 15:
                    # Convert DOA to servo angles
                    servo_angles = respeaker.doa_to_servo_angles(doa, tilt_angle=0)
                    
                    if servo_angles:
                        print(f"🎯 Voice at {doa}° → Moving camera to pan={servo_angles['pan']}°, tilt={servo_angles['tilt']}°")
                        
                        # Move camera
                        rover.gimbal_ctrl(
                            servo_angles['pan'],
                            servo_angles['tilt'],
                            input_speed=150,
                            input_acceleration=10
                        )
                        
                        last_doa = doa
                
            else:
                # No voice detected
                print("❌ No voice detected")
            
            time.sleep(0.5)  # Wait a bit before next listening cycle
    
    except KeyboardInterrupt:
        print("\n\n[Demo] Interrupted by user")
    
    finally:
        # Cleanup
        print("\n[Cleanup] Returning camera to center...")
        rover.gimbal_ctrl(0, 0, input_speed=150, input_acceleration=10)
        time.sleep(0.5)
        
        print("\n" + "="*70)
        print("👋 Thanks for using the voice localization demo!")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()

