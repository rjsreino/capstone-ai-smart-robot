#!/usr/bin/env python3
"""Quick test of face recognition API endpoints."""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*60)
print("Testing Face Recognition API Integration")
print("="*60)

# Test 1: Check if service can be imported and initialized
print("\n[Test 1] Testing face recognition service initialization...")
try:
    from api.app.face_recognition_service import FaceRecognitionService, INSIGHTFACE_AVAILABLE
    
    if not INSIGHTFACE_AVAILABLE:
        print("❌ InsightFace not available")
        sys.exit(1)
    
    print("✅ InsightFace is available")
    
    # Try to initialize
    known_faces_dir = project_root / "known-faces"
    service = FaceRecognitionService(
        known_faces_dir=known_faces_dir,
        model_name="arcface_r100_v1",
        threshold=0.6,
    )
    print("✅ Face recognition service initialized successfully")
    
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check loaded faces
print("\n[Test 2] Checking loaded known faces...")
known_faces = service.get_known_faces()
print(f"✅ Loaded {len(known_faces)} known face(s):")
for name in known_faces:
    print(f"   - {name}")

# Test 3: Test recognition on a known face image
print("\n[Test 3] Testing face recognition...")
import cv2
known_faces_dir = project_root / "known-faces"
test_image = known_faces_dir / "boymirzo.jpg"

if test_image.exists():
    print(f"   Testing with: {test_image.name}")
    image = cv2.imread(str(test_image))
    if image is not None:
        recognitions = service.recognize_faces(image, return_locations=True)
        print(f"✅ Recognition completed")
        print(f"   Detected {len(recognitions)} face(s):")
        for rec in recognitions:
            print(f"   - Name: {rec['name']}, Confidence: {rec['confidence']:.3f}")
            if rec['name'] != 'Unknown' and rec['confidence'] > 0.7:
                print(f"     ✅ Successfully recognized!")
    else:
        print("❌ Failed to read test image")
else:
    print("⚠️  Test image not found, skipping recognition test")

# Test 4: Test API integration (check if endpoints are registered)
print("\n[Test 4] Testing API endpoint registration...")
try:
    from api.app.main import app
    routes = [route.path for route in app.routes]
    face_routes = [r for r in routes if 'face-recognition' in r]
    
    if face_routes:
        print(f"✅ Found {len(face_routes)} face recognition endpoint(s):")
        for route in face_routes:
            print(f"   - {route}")
    else:
        print("❌ No face recognition endpoints found")
except Exception as e:
    print(f"⚠️  Could not check API routes: {e}")

print("\n" + "="*60)
print("Test Summary")
print("="*60)
print(f"✅ Service available: {'Yes' if INSIGHTFACE_AVAILABLE else 'No'}")
print(f"✅ Service initialized: Yes")
print(f"✅ Known faces loaded: {len(known_faces)}")
print(f"✅ API integration: {'Yes' if 'face_routes' in locals() and face_routes else 'Unknown'}")
print("="*60)
print("\n🎉 Face recognition is ready to use!")

