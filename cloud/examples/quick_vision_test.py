#!/usr/bin/env python3
"""
Quick Vision Test - Solve a problem shown on paper

Usage:
    python quick_vision_test.py

What it does:
    1. Captures an image from OAK-D/USB camera
    2. Sends it to OpenAI Vision API
    3. Gets the answer (like showing ChatGPT a photo)

Example:
    - Write "2+2=?" on paper
    - Run this script
    - Hold paper to camera
    - Get the answer!
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision import VisionProcessor


async def quick_test():
    """Quick test - solve a problem."""
    
    print("\n" + "=" * 70)
    print("  🎥 QUICK VISION TEST")
    print("=" * 70)
    print("\n📝 Instructions:")
    print("   1. Write something on paper (e.g., '2+2=?', 'What is AI?', etc.)")
    print("   2. Hold it up to the camera")
    print("   3. Get AI analysis!")
    print("\n⏳ Starting in 5 seconds...")
    print("   (Position your paper now!)\n")
    
    # Countdown
    for i in range(5, 0, -1):
        print(f"   {i}...")
        await asyncio.sleep(1)
    
    print("\n📸 Capturing and analyzing...\n")
    
    # Create processor and analyze
    processor = VisionProcessor()
    try:
        result = await processor.solve_problem()
        
        if result:
            print("✅ AI Response:")
            print("=" * 70)
            print(result)
            print("=" * 70)
            print("\n✨ Success!")
        else:
            print("❌ Failed to analyze image.")
            print("💡 Tips:")
            print("   - Make sure camera is connected")
            print("   - Check OPENAI_API_KEY is set")
            print("   - Ensure good lighting")
    finally:
        await processor.close()


if __name__ == "__main__":
    print("\n💡 Make sure:")
    print("   ✓ Camera is connected (OAK-D or USB)")
    print("   ✓ OPENAI_API_KEY environment variable is set")
    
    try:
        asyncio.run(quick_test())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\n💡 Common issues:")
        print("   - Camera not found: Check camera connection")
        print("   - API error: Verify OPENAI_API_KEY is set correctly")
        print("   - Import error: Make sure you're in the right directory")

