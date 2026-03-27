#!/usr/bin/env python3
"""
Microphone-Camera Angle Offset Calibration Tool

This tool helps you find the correct angle offset between your ReSpeaker 
microphone mounting position and your camera servo mounting position.

HOW TO USE:
1. Run this script
2. Stand directly in front of the rover (where you want 0° to be)
3. Say 'Hey Rovy' and speak for a moment
4. The script will show the raw DOA angle and move the camera
5. Check if the camera points at you correctly
6. If not, adjust the offset value and try again
7. Once calibrated, update MICROPHONE_TO_CAMERA_OFFSET in voice_localization_demo.py
"""
import time
import sys
from smart_assistant import ReSpeakerInterface, TextToSpeech, WakeWordDetector
from rover_controller import Rover

# ============================================================================
# TEST OFFSET VALUE - Adjust this until camera points correctly
# ============================================================================
TEST_OFFSET = 0  # Start with 0 and adjust based on results

def main():
    global TEST_OFFSET  # Allow modification of the global variable
    
    print("\n" + "="*80)
    print("🎯 MICROPHONE-CAMERA ANGLE OFFSET CALIBRATION TOOL")
    print("="*80)
    print("\nPURPOSE:")
    print("  Find the correct angle offset between your microphone and camera mounting.")
    print("\nSTEPS:")
    print("  1. Stand directly where you want the camera to point when you speak")
    print("  2. Say 'Hey Rovy' and keep talking for a moment")
    print("  3. Observe where the camera points")
    print("  4. If camera doesn't point at you, adjust TEST_OFFSET value and try again")
    print("\nCURRENT TEST OFFSET: {}°".format(TEST_OFFSET))
    print("="*80 + "\n")
    
    input("Press ENTER to start calibration...")
    
    # Initialize components
    print("\n[Setup] Initializing rover controller...")
    rover = Rover()
    
    print("[Setup] Initializing ReSpeaker...")
    respeaker = ReSpeakerInterface(use_whisper=False)
    tts = TextToSpeech(print_only=False)
    wake_detector = WakeWordDetector(device_index=respeaker.device_index)
    
    print("\n✅ Setup complete!\n")
    
    # Initialize gimbal
    print("[Camera] Initializing gimbal...")
    print("[Camera] Unlocking servos...")
    rover.gimbal_unlock()
    time.sleep(0.5)
    
    print("[Camera] Centering camera to forward position (0°, 0°)...")
    rover.gimbal_ctrl_move(0, 0, input_speed_x=300, input_speed_y=300)
    time.sleep(2.0)
    
    try:
        while True:
            print("\n" + "="*80)
            print("CALIBRATION TEST")
            print("="*80)
            print(f"Current test offset: {TEST_OFFSET}°")
            print("\n1. Stand where you want 0° (front) to be")
            print("2. Say 'Hey Rovy' and keep talking")
            print("3. Check if camera points at you correctly\n")
            
            # Listen for wake word
            print("👂 Listening for 'Hey Rovy'...")
            wake_result = wake_detector.listen_for_wake_word(timeout=30)
            
            if wake_result:
                print("✅ Wake word detected! Measuring voice direction...")
                tts.speak("Measuring your position")
                
                # Get voice direction
                doa = respeaker.get_voice_direction(listen_duration=1.0)
                
                if doa is not None:
                    print("\n" + "-"*80)
                    print("RESULTS:")
                    print("-"*80)
                    print(f"📍 Raw DOA from microphone: {doa}°")
                    print(f"   (0°=front, 90°=left, 180°=back, 270°=right)")
                    
                    # Apply test offset
                    calibrated_doa = (doa + TEST_OFFSET) % 360
                    print(f"\n📐 Applied offset: {TEST_OFFSET}°")
                    print(f"📍 Calibrated DOA: {calibrated_doa}°")
                    
                    # Convert to servo angles
                    servo_angles = respeaker.doa_to_servo_angles(calibrated_doa, tilt_angle=0)
                    
                    if servo_angles:
                        pan = -servo_angles['pan']  # Inverted camera rotation
                        
                        # Describe direction
                        angle_map = {
                            -180: "FAR LEFT", -135: "BACK LEFT", -90: "LEFT", 
                            -45: "FRONT LEFT", 0: "CENTER", 45: "FRONT RIGHT", 
                            90: "RIGHT", 135: "BACK RIGHT", 180: "FAR RIGHT"
                        }
                        
                        # Find closest description
                        closest_angle = min(angle_map.keys(), key=lambda x: abs(x - pan))
                        pan_desc = angle_map[closest_angle]
                        
                        print(f"\n🎥 Moving camera to: {pan_desc} (pan={pan}°)")
                        print("-"*80)
                        
                        # Move camera
                        rover.gimbal_ctrl_move(pan, 0, input_speed_x=300, input_speed_y=300)
                        time.sleep(2.5)
                        
                        tts.speak("Is the camera pointing at you?")
                        
                        # Ask for feedback
                        print("\n" + "="*80)
                        print("CALIBRATION CHECK:")
                        print("="*80)
                        print("❓ Is the camera pointing directly at you?")
                        print("\nOptions:")
                        print("  y = YES! Camera is pointing correctly (calibration done!)")
                        print("  n = NO, camera needs adjustment")
                        print("  q = Quit calibration")
                        
                        response = input("\nYour answer (y/n/q): ").lower().strip()
                        
                        if response == 'y':
                            print("\n" + "="*80)
                            print("✅ CALIBRATION SUCCESSFUL!")
                            print("="*80)
                            print(f"\nYour calibrated offset value is: {TEST_OFFSET}°")
                            print("\nTO APPLY THIS CALIBRATION:")
                            print("1. Open voice_localization_demo.py")
                            print("2. Find the line: MICROPHONE_TO_CAMERA_OFFSET = 0")
                            print(f"3. Change it to: MICROPHONE_TO_CAMERA_OFFSET = {TEST_OFFSET}")
                            print("4. Save the file and run the demo!")
                            print("="*80 + "\n")
                            break
                        
                        elif response == 'q':
                            print("\nExiting calibration...")
                            break
                        
                        else:
                            print("\n" + "="*80)
                            print("ADJUSTMENT GUIDE:")
                            print("="*80)
                            print("Camera pointing too far LEFT?  → INCREASE offset (+10, +20, etc.)")
                            print("Camera pointing too far RIGHT? → DECREASE offset (-10, -20, etc.)")
                            print("\nCurrent offset: {}°".format(TEST_OFFSET))
                            
                            new_offset = input("\nEnter new offset value (or press ENTER to try again): ").strip()
                            if new_offset:
                                try:
                                    TEST_OFFSET = int(new_offset)
                                    print(f"\n✅ Offset updated to {TEST_OFFSET}°")
                                except ValueError:
                                    print("\n⚠️  Invalid number, keeping current offset")
                    else:
                        print("❌ Could not calculate servo position")
                else:
                    print("❌ Could not determine voice direction")
                
                # Return camera to center
                print("\n[Camera] Returning to center...")
                rover.gimbal_ctrl_move(0, 0, input_speed_x=300, input_speed_y=300)
                time.sleep(1.5)
            else:
                print("⏰ No wake word detected, try again...")
    
    except KeyboardInterrupt:
        print("\n\n[Calibration] Interrupted by user")
    
    finally:
        # Cleanup
        print("\n[Cleanup] Shutting down...")
        rover.gimbal_ctrl_move(0, 0, input_speed_x=300, input_speed_y=300)
        time.sleep(0.5)
        print("[Cleanup] Done!")
        print("\n" + "="*80)
        print("👋 Calibration tool closed")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()

