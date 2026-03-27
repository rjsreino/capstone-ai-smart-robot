#!/usr/bin/env python3
"""
Test person detection at a specific angle
Helps debug if camera is pointing correctly and detection is working
"""
import time
from rover_controller import Rover
from oakd_depth_navigator import OakDDepthCamera

def test_detection():
    print("\n" + "="*70)
    print("🧪 Person Detection Test at Specific Angle")
    print("="*70)
    
    rover = Rover()
    camera = OakDDepthCamera(resolution=(640, 352), enable_person_detection=True)
    camera.start()
    
    # Initialize
    print("\n[Setup] Unlocking servos...")
    rover.gimbal_unlock()
    time.sleep(0.5)
    
    print("[Setup] Centering camera...")
    rover.gimbal_ctrl_move(0, 0, input_speed_x=500, input_speed_y=500)
    time.sleep(2.0)
    
    # Test the back-left position where voice was detected
    pan = -135
    tilt = 15
    
    print(f"\n📍 Moving camera to pan={pan}°, tilt={tilt}°")
    print(f"   This is where your voice was detected (BACK LEFT)")
    rover.gimbal_ctrl_move(pan, tilt, input_speed_x=500, input_speed_y=500)
    
    print("\n⏳ Waiting for camera to move and stabilize...")
    time.sleep(3.0)
    
    input("\n👀 Stand in front of the camera and press ENTER when ready to test detection...")
    
    print("\n🔍 Running 10 detection attempts...")
    print("="*70)
    
    detection_count = 0
    for i in range(10):
        time.sleep(0.3)  # Small delay between attempts
        print(f"\nAttempt {i+1}/10:")
        detections = camera.detect_person(debug=True)
        
        if detections:
            print(f"  ✅ Detected {len(detections)} object(s)")
            for j, d in enumerate(detections):
                depth_str = f"{d['depth']}mm" if d['depth'] else "no depth"
                print(f"    #{j+1}: conf={d['confidence']:.2f}, {depth_str}")
                if d['confidence'] >= 0.45:
                    detection_count += 1
        else:
            print(f"  ❌ No detections")
    
    print("\n" + "="*70)
    print(f"📊 Summary: Detected person in {detection_count}/10 attempts")
    print("="*70)
    
    if detection_count == 0:
        print("\n⚠️  ISSUE: Camera is not detecting you!")
        print("Possible reasons:")
        print("  1. Camera not pointing at you (check physical angle)")
        print("  2. Too far/close (optimal: 1-5 meters)")
        print("  3. Lighting issues (too dark/bright)")
        print("  4. Obstacles blocking view")
    elif detection_count < 5:
        print(f"\n⚠️  WARNING: Only {detection_count}/10 detections")
        print("Detection is unreliable - try improving conditions")
    else:
        print(f"\n✅ Good detection rate: {detection_count}/10")
    
    # Cleanup
    print("\n[Cleanup] Returning to center...")
    rover.gimbal_ctrl_move(0, 0, input_speed_x=500, input_speed_y=500)
    time.sleep(1.0)
    camera.close()
    
    print("\n✅ Test complete!\n")

if __name__ == "__main__":
    try:
        test_detection()
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted")

