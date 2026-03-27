#!/usr/bin/env python3
"""
Benchmark Face Detection AND Recognition Methods
Captures image from OAK-D and tests various face detection + recognition options
"""
import cv2
import numpy as np
import time
import face_recognition
from oakd_depth_navigator import OakDDepthCamera
from face_recognizer import FaceRecognizer

def benchmark_face_detection():
    print("\n" + "="*70)
    print("🔬 Face Detection Benchmark")
    print("="*70)
    
    # Initialize camera
    print("\n[Setup] Initializing OAK-D camera...")
    camera = OakDDepthCamera(resolution=(640, 352), enable_person_detection=False)
    camera.start()
    print("[Setup] ✅ Camera started")
    
    # Wait for camera to warm up
    time.sleep(2)
    
    # Capture current frame (pointing at you!)
    print("\n[Capture] Capturing image from camera...")
    frame, _ = camera.capture_frames()
    print(f"[Capture] ✅ Captured frame: {frame.shape}")
    
    # Save test images to check color space
    test_img_as_bgr = "/home/jetson/rovy/test_face_as_bgr.jpg"
    test_img_as_rgb = "/home/jetson/rovy/test_face_as_rgb.jpg"
    
    # Save assuming frame is already BGR
    cv2.imwrite(test_img_as_bgr, frame)
    print(f"[Capture] 💾 Saved (treating as BGR): {test_img_as_bgr}")
    
    # Save assuming frame is RGB (convert to BGR)
    cv2.imwrite(test_img_as_rgb, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"[Capture] 💾 Saved (treating as RGB): {test_img_as_rgb}")
    
    print("\n⚠️  CHECK WHICH IMAGE LOOKS CORRECT:")
    print(f"   {test_img_as_bgr} - if colors look normal, camera outputs BGR")
    print(f"   {test_img_as_rgb} - if colors look normal, camera outputs RGB")
    
    # Close camera
    camera.close()
    
    # OAK-D outputs BGR! Convert to RGB for face_recognition library
    print("\n[Processing] Converting BGR to RGB for face_recognition...")
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Load face recognizer for recognition tests
    print("\n[Setup] Loading face recognizer with known faces...")
    recognizer = FaceRecognizer(known_dir="/home/jetson/rovy/known-faces")
    print(f"[Setup] ✅ Loaded {len(recognizer.known_names)} known faces")
    
    print("\n" + "="*70)
    print("PART 1: FACE DETECTION ONLY (finding faces)")
    print("="*70)
    
    detection_results = []
    
    # Test 1: HOG at full resolution
    print("\n[Test 1] HOG at full resolution (352x640)...")
    start = time.time()
    locs = face_recognition.face_locations(rgb_image, model="hog", number_of_times_to_upsample=0)
    elapsed = time.time() - start
    detection_results.append(("HOG Full (352x640)", elapsed, len(locs)))
    print(f"   ⏱️  Time: {elapsed:.3f}s, Faces found: {len(locs)}")
    
    # Test 2: HOG at 0.5x scale (half resolution)
    print("\n[Test 2] HOG at 0.5x scale (176x320)...")
    small_05 = cv2.resize(rgb_image, (0, 0), fx=0.5, fy=0.5)
    start = time.time()
    locs = face_recognition.face_locations(small_05, model="hog", number_of_times_to_upsample=0)
    elapsed = time.time() - start
    detection_results.append(("HOG 0.5x (176x320)", elapsed, len(locs)))
    print(f"   ⏱️  Time: {elapsed:.3f}s, Faces found: {len(locs)}")
    
    # Test 3: HOG at 0.25x scale (quarter resolution)
    print("\n[Test 3] HOG at 0.25x scale (88x160)...")
    small_025 = cv2.resize(rgb_image, (0, 0), fx=0.25, fy=0.25)
    start = time.time()
    locs = face_recognition.face_locations(small_025, model="hog", number_of_times_to_upsample=0)
    elapsed = time.time() - start
    detection_results.append(("HOG 0.25x (88x160)", elapsed, len(locs)))
    print(f"   ⏱️  Time: {elapsed:.3f}s, Faces found: {len(locs)}")
    
    # Test 4: CNN at full resolution (GPU)
    print("\n[Test 4] CNN at full resolution (352x640) - GPU...")
    start = time.time()
    locs = face_recognition.face_locations(rgb_image, model="cnn", number_of_times_to_upsample=0)
    elapsed = time.time() - start
    detection_results.append(("CNN Full GPU (352x640)", elapsed, len(locs)))
    print(f"   ⏱️  Time: {elapsed:.3f}s, Faces found: {len(locs)}")
    
    # Test 5: CNN at 0.5x scale (GPU)
    print("\n[Test 5] CNN at 0.5x scale (176x320) - GPU...")
    start = time.time()
    locs = face_recognition.face_locations(small_05, model="cnn", number_of_times_to_upsample=0)
    elapsed = time.time() - start
    detection_results.append(("CNN 0.5x GPU (176x320)", elapsed, len(locs)))
    print(f"   ⏱️  Time: {elapsed:.3f}s, Faces found: {len(locs)}")
    
    # Now test FULL FACE RECOGNITION (detection + encoding + comparison)
    print("\n" + "="*70)
    print("PART 2: FULL FACE RECOGNITION (detection + encoding + comparison)")
    print("="*70)
    
    recognition_results = []
    
    # Test R1: Using FaceRecognizer with HOG 0.5x (current config)
    print("\n[Test R1] FaceRecognizer (HOG 0.5x) - CURRENT CONFIG...")
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    start = time.time()
    result = recognizer.recognize_face(bgr_image, is_rgb=False)
    elapsed = time.time() - start
    recognition_results.append(("FaceRecognizer HOG 0.5x", elapsed, len(result), result[0][0] if result else "None"))
    if result:
        print(f"   ⏱️  Time: {elapsed:.3f}s, Recognized: {result[0][0]} (confidence: {result[0][1]:.2f})")
    else:
        print(f"   ⏱️  Time: {elapsed:.3f}s, No face found")
    
    # Test R2: Manual with HOG full resolution
    print("\n[Test R2] Manual HOG Full (352x640) + encoding + comparison...")
    start = time.time()
    locs = face_recognition.face_locations(rgb_image, model="hog", number_of_times_to_upsample=0)
    if locs:
        encodings = face_recognition.face_encodings(rgb_image, locs, num_jitters=0)
        if encodings and recognizer.known_encodings:
            distances = face_recognition.face_distance(recognizer.known_encodings, encodings[0])
            best_idx = np.argmin(distances)
            best_name = recognizer.known_names[best_idx]
            best_dist = distances[best_idx]
    elapsed = time.time() - start
    recognition_results.append(("Manual HOG Full", elapsed, len(locs) if locs else 0, best_name if locs else "None"))
    if locs:
        print(f"   ⏱️  Time: {elapsed:.3f}s, Recognized: {best_name} (distance: {best_dist:.3f})")
    else:
        print(f"   ⏱️  Time: {elapsed:.3f}s, No face found")
    
    # Test R3: Manual with HOG 0.25x (aggressive downscale)
    print("\n[Test R3] Manual HOG 0.25x (88x160) + encoding + comparison...")
    start = time.time()
    locs = face_recognition.face_locations(small_025, model="hog", number_of_times_to_upsample=0)
    if locs:
        encodings = face_recognition.face_encodings(small_025, locs, num_jitters=0)
        if encodings and recognizer.known_encodings:
            distances = face_recognition.face_distance(recognizer.known_encodings, encodings[0])
            best_idx = np.argmin(distances)
            best_name = recognizer.known_names[best_idx]
            best_dist = distances[best_idx]
    elapsed = time.time() - start
    recognition_results.append(("Manual HOG 0.25x", elapsed, len(locs) if locs else 0, best_name if locs else "None"))
    if locs:
        print(f"   ⏱️  Time: {elapsed:.3f}s, Recognized: {best_name} (distance: {best_dist:.3f})")
    else:
        print(f"   ⏱️  Time: {elapsed:.3f}s, No face found")
    
    # Test R4: Manual with CNN 0.5x GPU
    print("\n[Test R4] Manual CNN 0.5x GPU (176x320) + encoding + comparison...")
    start = time.time()
    locs = face_recognition.face_locations(small_05, model="cnn", number_of_times_to_upsample=0)
    if locs:
        encodings = face_recognition.face_encodings(small_05, locs, num_jitters=0)
        if encodings and recognizer.known_encodings:
            distances = face_recognition.face_distance(recognizer.known_encodings, encodings[0])
            best_idx = np.argmin(distances)
            best_name = recognizer.known_names[best_idx]
            best_dist = distances[best_idx]
    elapsed = time.time() - start
    recognition_results.append(("Manual CNN 0.5x GPU", elapsed, len(locs) if locs else 0, best_name if locs else "None"))
    if locs:
        print(f"   ⏱️  Time: {elapsed:.3f}s, Recognized: {best_name} (distance: {best_dist:.3f})")
    else:
        print(f"   ⏱️  Time: {elapsed:.3f}s, No face found")
    
    # Print detection summary
    print("\n" + "="*70)
    print("📊 DETECTION RESULTS (sorted by speed)")
    print("="*70)
    detection_results.sort(key=lambda x: x[1])
    print(f"\n{'Method':<30} {'Time':<12} {'Faces Found'}")
    print("-"*70)
    for method, elapsed, num_faces in detection_results:
        print(f"{method:<30} {elapsed:>6.3f}s      {num_faces:>3} faces")
    
    # Print recognition summary
    print("\n" + "="*70)
    print("📊 FULL RECOGNITION RESULTS (sorted by speed)")
    print("="*70)
    recognition_results.sort(key=lambda x: x[1])
    print(f"\n{'Method':<30} {'Time':<12} {'Recognized'}")
    print("-"*70)
    for method, elapsed, num_faces, name in recognition_results:
        print(f"{method:<30} {elapsed:>6.3f}s      {name}")
    
    print("\n" + "="*70)
    print("🏆 FINAL RECOMMENDATION:")
    print("="*70)
    
    # Find fastest that found a face
    best = None
    for method, elapsed, num_faces, name in recognition_results:
        if num_faces > 0 and name != "None":
            best = (method, elapsed, name)
            break
    
    if best:
        print(f"✅ Use: {best[0]}")
        print(f"⏱️  Time: {best[1]:.3f}s")
        print(f"👤 Recognized: {best[2]}")
    else:
        print("❌ No method successfully recognized a face!")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        benchmark_face_detection()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

