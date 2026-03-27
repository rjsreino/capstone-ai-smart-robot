#!/usr/bin/env python3
"""Quick test script for face recognition API endpoints."""
import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_face_recognition():
    print("="*60)
    print("Testing Face Recognition API")
    print("="*60)
    
    # Test 1: List known faces
    print("\n[Test 1] Getting known faces...")
    try:
        response = requests.get(f"{BASE_URL}/face-recognition/known", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['count']} known faces:")
            for face in data['faces']:
                print(f"   - {face}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server.")
        print("   Make sure the server is running:")
        print("   cd /home/jetson/rovy/api && uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: Recognize faces
    print("\n[Test 2] Recognizing faces in current camera frame...")
    try:
        response = requests.post(f"{BASE_URL}/face-recognition/recognize", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Detected {len(data['faces'])} face(s):")
            for face in data['faces']:
                name = face['name']
                confidence = face['confidence']
                bbox = face.get('bbox', 'N/A')
                print(f"   - {name}: {confidence:.1%} confidence")
                if bbox != 'N/A':
                    print(f"     Location: {bbox}")
        elif response.status_code == 503:
            print("❌ Face recognition service not available")
            print("   Check server logs for details")
            return False
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 3: Check API docs
    print("\n[Test 3] Checking API documentation...")
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print(f"✅ API docs available at: {BASE_URL}/docs")
        else:
            print(f"⚠️  API docs not available")
    except:
        print(f"⚠️  Could not check API docs")
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print("✅ Face recognition API is working!")
    print(f"\n📺 View live stream: {BASE_URL}/face-recognition/stream")
    print(f"📚 API documentation: {BASE_URL}/docs")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = test_face_recognition()
    sys.exit(0 if success else 1)

