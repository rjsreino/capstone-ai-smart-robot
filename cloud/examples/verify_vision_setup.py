#!/usr/bin/env python3
"""
Verify Vision API Setup

This script checks if all requirements for the Vision API are properly set up:
- OpenAI library installed
- OpenAI API key configured
- Camera available (OAK-D or USB)
- All dependencies working

Run this before using the vision features.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def check_requirement(name, test_func):
    """Check a requirement and print status."""
    try:
        result = test_func()
        if result:
            print(f"  ✅ {name}")
            return True
        else:
            print(f"  ❌ {name}")
            return False
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 70)
    print("  🔍 Vision API Setup Verification")
    print("=" * 70)
    
    all_ok = True
    
    # Check 1: OpenAI library
    print("\n📦 Checking dependencies...")
    
    def check_openai():
        import openai
        return True
    
    all_ok &= check_requirement("OpenAI library installed", check_openai)
    
    def check_pillow():
        from PIL import Image
        return True
    
    all_ok &= check_requirement("Pillow (PIL) installed", check_pillow)
    
    # Check 2: OpenAI API key
    print("\n🔑 Checking API credentials...")
    
    def check_api_key():
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("     Set with: export OPENAI_API_KEY='your-key'")
            return False
        
        # Try to initialize client (doesn't make API call)
        client = OpenAI(api_key=api_key)
        print(f"     API key found: {api_key[:7]}...{api_key[-4:]}")
        return True
    
    all_ok &= check_requirement("OpenAI API key configured", check_api_key)
    
    # Check 3: Camera modules
    print("\n📷 Checking camera support...")
    
    def check_depthai():
        import depthai as dai
        devices = dai.Device.getAllAvailableDevices()
        if devices:
            print(f"     Found {len(devices)} OAK-D device(s)")
            for dev in devices:
                print(f"       - {dev.getMxId()}")
        else:
            print("     No OAK-D devices found")
        return len(devices) > 0
    
    oakd_ok = check_requirement("OAK-D camera available", check_depthai)
    
    def check_opencv():
        import cv2
        cap = cv2.VideoCapture(0)
        is_ok = cap.isOpened()
        if is_ok:
            print("     USB camera found at index 0")
            cap.release()
        else:
            print("     No USB camera found at index 0")
        return is_ok
    
    usb_ok = check_requirement("USB camera available", check_opencv)
    
    # At least one camera should be available
    if not (oakd_ok or usb_ok):
        print("\n  ⚠️  WARNING: No cameras found! At least one camera is required.")
        all_ok = False
    
    # Check 4: Vision module
    print("\n🔧 Checking vision module...")
    
    def check_vision_module():
        from vision import VisionProcessor
        processor = VisionProcessor()
        return processor.openai_client is not None
    
    all_ok &= check_requirement("Vision processor can initialize", check_vision_module)
    
    # Check 5: Camera module
    def check_camera_module():
        from app.camera import CameraService, DepthAICameraSource, OpenCVCameraSource
        return True
    
    all_ok &= check_requirement("Camera module available", check_camera_module)
    
    # Summary
    print("\n" + "=" * 70)
    if all_ok:
        print("  ✅ ALL CHECKS PASSED!")
        print("=" * 70)
        print("\n🎉 You're ready to use the Vision API!")
        print("\n📝 Next steps:")
        print("   1. Run quick test: python examples/quick_vision_test.py")
        print("   2. Try demo: python examples/vision_demo.py")
        print("   3. Read guide: cat VISION_API_GUIDE.md")
        return 0
    else:
        print("  ❌ SOME CHECKS FAILED")
        print("=" * 70)
        print("\n⚠️  Some requirements are not met.")
        print("\n🔧 Fix the issues above, then run this script again.")
        print("\n💡 Common fixes:")
        print("   - Install OpenAI: pip install openai>=1.0.0")
        print("   - Set API key: export OPENAI_API_KEY='your-key'")
        print("   - Connect camera (OAK-D or USB webcam)")
        print("   - Install depthai: pip install depthai")
        print("   - Install opencv: pip install opencv-python")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

