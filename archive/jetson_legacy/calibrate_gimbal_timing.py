#!/usr/bin/env python3
"""
Gimbal Timing Calibration Test
This helps determine how long it takes for the camera to stop moving
"""
import time
from rover_controller import Rover

def test_movement_timing():
    print("\n" + "="*70)
    print("🎯 Gimbal Movement Timing Test")
    print("="*70)
    print("\nThis will help us figure out exactly how long the camera takes to move.")
    print("Watch the camera and tell me when it STOPS moving.\n")
    
    rover = Rover()
    
    # Initialize gimbal
    print("[Setup] Unlocking servos...")
    rover.gimbal_unlock()
    time.sleep(0.5)
    
    print("[Setup] Centering camera...")
    rover.gimbal_ctrl_move(0, 0, input_speed_x=500, input_speed_y=500)
    time.sleep(3.0)
    
    # Test large movement (like voice localization does)
    test_angle = -135  # BACK LEFT - typical voice localization movement
    
    print("\n" + "="*70)
    print(f"TEST: Moving camera from 0° to {test_angle}° (135° distance)")
    print("="*70)
    
    input("\nPress ENTER when ready to start the test...")
    
    print(f"\n🔵 APPLE - Sending command to move to {test_angle}°")
    start_time = time.time()
    rover.gimbal_ctrl_move(test_angle, 0, input_speed_x=500, input_speed_y=500)
    
    # Wait in 1-second increments and ask user
    for i in range(1, 16):
        time.sleep(1.0)
        elapsed = time.time() - start_time
        print(f"⏱️  {i} seconds - Type 'yes' if camera STOPPED, or just ENTER to continue: ", end='', flush=True)
        response = input().strip().lower()
        if response == 'yes':
            print(f"\n✅ Camera stopped after {elapsed:.1f} seconds!")
            print(f"📊 Recommendation: Use {elapsed + 1.5:.1f}s wait time (with 1.5s buffer)")
            break
    else:
        print("\n⚠️  Test ended at 15 seconds")
    
    # Test tilt movement
    print("\n" + "="*70)
    print("TEST: Tilt movement (from 0° to 15°)")
    print("="*70)
    
    input("\nPress ENTER when ready to test tilt movement...")
    
    print("\n🔵 APPLE - Sending command to tilt to 15°")
    start_time = time.time()
    rover.gimbal_ctrl_move(test_angle, 15, input_speed_x=500, input_speed_y=500)
    
    # Wait in 0.5-second increments
    for i in range(1, 11):
        time.sleep(0.5)
        elapsed = time.time() - start_time
        print(f"⏱️  {elapsed:.1f} seconds - Type 'yes' if camera STOPPED, or just ENTER: ", end='', flush=True)
        response = input().strip().lower()
        if response == 'yes':
            print(f"\n✅ Tilt stopped after {elapsed:.1f} seconds!")
            print(f"📊 Recommendation: Use {elapsed + 0.5:.1f}s wait time for tilt (with 0.5s buffer)")
            break
    else:
        print("\n⚠️  Test ended at 5 seconds")
    
    # Return to center
    print("\n[Cleanup] Returning to center...")
    rover.gimbal_ctrl_move(0, 0, input_speed_x=500, input_speed_y=500)
    time.sleep(2.0)
    
    print("\n" + "="*70)
    print("✅ Test complete! Use the recommendations above to update wait times.")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        test_movement_timing()
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted")

