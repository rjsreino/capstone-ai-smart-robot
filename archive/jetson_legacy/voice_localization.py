#!/usr/bin/env python3
"""
Voice Localization Module
Reusable voice localization logic for turning camera towards speaker
"""
import time
import cv2

# Calibration offset from demo
MICROPHONE_TO_CAMERA_OFFSET = -121

def locate_speaker(respeaker, rover, camera=None, face_recognizer=None, tts=None):
    """
    Locate speaker using voice direction and turn camera to face them.
    This is the exact logic from voice_localization_demo.py
    
    Args:
        respeaker: ReSpeakerInterface instance
        rover: Rover controller instance
        camera: OakDDepthCamera instance (optional, for person detection)
        face_recognizer: FaceRecognizer instance (optional)
        tts: TextToSpeech instance (optional)
    
    Returns:
        dict: {
            'success': bool,
            'doa': int or None,
            'pan': int or None,
            'person_found': bool,
            'recognized_name': str or None,
            'message': str
        }
    """
    result = {
        'success': False,
        'doa': None,
        'pan': None,
        'person_found': False,
        'recognized_name': None,
        'message': ''
    }
    
    # Get voice direction - listen a bit to capture their voice direction
    print("[Voice Localization] 🎤 Detecting voice direction...")
    doa = respeaker.get_voice_direction(listen_duration=0.5)
    
    if doa is None:
        result['message'] = "I couldn't detect where your voice is coming from. Please speak again."
        return result
    
    result['doa'] = doa
    print(f"[Voice Localization] 🎯 Voice detected at {doa}°")
    
    # Apply calibration offset for microphone mounting position
    calibrated_doa = (doa + MICROPHONE_TO_CAMERA_OFFSET) % 360
    if calibrated_doa != doa:
        print(f"[Voice Localization] 📐 Applying {MICROPHONE_TO_CAMERA_OFFSET}° offset: {doa}° → {calibrated_doa}°")
    
    # Get pan angle from voice direction
    servo_angles = respeaker.doa_to_servo_angles(calibrated_doa, tilt_angle=0)
    
    if not servo_angles:
        result['message'] = "I heard you but couldn't calculate where to turn."
        return result
    
    # Turn towards voice
    pan = servo_angles['pan']
    result['pan'] = pan
    print(f"[Voice Localization] 📹 Turning to direction: {pan}°")
    
    # Try multiple tilt angles to find person with YOLOv8 (exact logic from demo)
    person_found = False
    found_tilt = 0
    tilt_angles = [0, 15, 30, 40]  # Start straight, then scan up
    recognized_name = None
    
    for tilt in tilt_angles:
        print(f"[Voice Localization] Scanning at tilt={tilt}°...")
        
        # Send movement command with timestamp
        start_time = time.time()
        print(f"[Voice Localization] 🔵 APPLE - Camera starting to move NOW")
        rover.gimbal_ctrl_move(pan, tilt, input_speed_x=500, input_speed_y=500)
        
        # Calculate wait time based on angle distance (for first move from center)
        if tilt == tilt_angles[0]:
            angle_distance = abs(pan)
            wait_time = 7.0 if angle_distance > 90 else 5.0
            print(f"[Voice Localization] 🟡 BANANA - Waiting {wait_time}s for pan movement ({angle_distance}°)...")
            time.sleep(wait_time)
            print(f"[Voice Localization] 🟢 CHERRY - Camera should be stopped after {wait_time}s wait")
        else:
            print(f"[Voice Localization] 🟡 BANANA - Waiting 1.5s for tilt change...")
            time.sleep(1.5)  # Subsequent tilt changes are smaller
            print(f"[Voice Localization] 🟢 CHERRY - Tilt should be complete")
        
        # Only do detection if camera is available
        if not camera:
            print(f"[Voice Localization] ⚠️  No camera available for person detection")
            result['success'] = True
            result['message'] = "I've turned to face you!"
            return result
        
        # Fast detection with 3 frame checks (camera warm-up)
        print(f"[Voice Localization] 🟣 DRAGON - Stabilizing 1.5s...")
        time.sleep(1.5)  # Longer stabilization
        print(f"[Voice Localization] 🔴 EAGLE - Checking 3 frames...")
        
        # OPTIMIZATION: Try 3 frames but stop early if confident detection found
        detections = None
        best_detections = []
        for check in range(3):
            time.sleep(0.3)  # Quick wait between frames
            current_detections = camera.detect_person(debug=(check == 0))
            
            if current_detections:
                # Collect all detections across frames
                best_detections.extend(current_detections)
                print(f"[Voice Localization]   ✓ Detected in frame {check+1}/3")
                
                # OPTIMIZATION: If we found high-confidence detection, stop early
                if current_detections[0]['confidence'] > 0.7:
                    detections = current_detections
                    print(f"[Voice Localization]   ✓ High confidence detection - stopping early")
                    break
        
        # Use best detection across all frames
        if not detections and best_detections:
            # Sort by confidence and use best one
            best_detections.sort(key=lambda x: x['confidence'], reverse=True)
            detections = [best_detections[0]]
        
        # Debug: Show what we found
        if detections:
            print(f"[Voice Localization] [Debug] Found {len(detections)} detection(s):")
            for i, d in enumerate(detections):
                depth_str = f"{d['depth']}mm" if d['depth'] else "no depth"
                print(f"[Voice Localization]   #{i+1}: conf={d['confidence']:.2f}, {depth_str}")
        else:
            print(f"[Voice Localization] No detections, trying next angle...")
        
        if detections:
            # Accept ANY person detection - no filtering!
            print(f"[Voice Localization] ✅ Accepting detection without filtering")
            best = detections[0]
            person_found = True
            found_tilt = tilt
            conf = int(best['confidence'] * 100)
            elapsed = time.time() - start_time
            if best['depth']:
                depth_m = best['depth'] / 1000
                print(f"[Voice Localization] ✅ [{time.strftime('%H:%M:%S')}] Person found! Confidence: {conf}%, Distance: {depth_m:.1f}m")
            else:
                print(f"[Voice Localization] ✅ [{time.strftime('%H:%M:%S')}] Person found! Confidence: {conf}%")
            print(f"[Voice Localization] ⏱️  Total time for this angle: {elapsed:.1f}s")
            
                # Try face recognition on FULL FRAME (not cropped region)
            # This works better when person is far away or at an angle
            if face_recognizer and camera:
                print(f"[Voice Localization] 🔍 Recognizing face...")
                try:
                    frame, _ = camera.capture_frames()
                    # OAK-D camera outputs BGR (not RGB despite setColorOrder setting!)
                    bgr_frame = frame
                    
                    print(f"[Voice Localization]   [Debug] Using full frame: {bgr_frame.shape}")
                    
                    # Use FULL frame for face recognition (not cropped)
                    # This is more reliable when person is far away
                    # Pass is_rgb=False since frame is BGR
                    recognized_faces = face_recognizer.recognize_face(bgr_frame, is_rgb=False)
                    
                    if recognized_faces:
                        name, face_conf, face_bbox = recognized_faces[0]
                        if name != "Unknown":
                            print(f"[Voice Localization] ✅ Recognized: {name} (confidence: {face_conf:.2f})")
                            recognized_name = name
                            if tts:
                                tts.speak(f"Yes, I see you {name}!")
                        else:
                            print(f"[Voice Localization] ⚠️  Face detected but not recognized (confidence too low)")
                            if tts:
                                tts.speak("Yes, I see you!")
                    else:
                        print(f"[Voice Localization] ⚠️  No face detected in full frame")
                        if tts:
                            tts.speak("Yes, I see you!")
                except Exception as e:
                    print(f"[Voice Localization] ⚠️  Face recognition error: {e}")
                    import traceback
                    traceback.print_exc()
                    if tts:
                        tts.speak("Yes, I see you!")
            
            break
        
        if person_found:
            break
    
    # Generate result
    result['success'] = True
    result['person_found'] = person_found
    result['recognized_name'] = recognized_name
    
    if person_found:
        if recognized_name:
            result['message'] = f"Yes, I see you {recognized_name}!"
        else:
            result['message'] = "Yes, I see you!"
    else:
        print("[Voice Localization] ⚠️  No person visually detected, but I heard your voice!")
        result['message'] = "I heard you!"
        if tts:
            tts.speak("I heard you!")
    
    return result

