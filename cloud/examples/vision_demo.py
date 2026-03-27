"""
Vision API Demo - Solve problems using camera + OpenAI Vision
Similar to ChatGPT's camera mode.

Examples:
1. Show a paper with "2+2=?" and get the answer
2. Show any text and get it read back
3. Show a complex math problem and get step-by-step solution
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision import VisionProcessor


async def demo_problem_solving():
    """Demo: Solve a math problem shown on paper."""
    print("\n" + "=" * 70)
    print("  DEMO 1: Problem Solving")
    print("=" * 70)
    print("\n📝 Instructions:")
    print("   1. Write a problem on paper (e.g., '2+2=?', '5×3=?', etc.)")
    print("   2. Hold it up to the camera")
    print("   3. The AI will read and solve it!")
    print("\n⏳ You have 5 seconds to position your paper...\n")
    
    await asyncio.sleep(5)
    
    processor = VisionProcessor()
    try:
        print("📸 Capturing and analyzing...")
        result = await processor.solve_problem()
        
        if result:
            print("\n✅ AI Solution:")
            print("-" * 70)
            print(result)
            print("-" * 70)
        else:
            print("\n❌ Failed to analyze image")
    finally:
        await processor.close()


async def demo_text_reading():
    """Demo: Read text from an image."""
    print("\n" + "=" * 70)
    print("  DEMO 2: Text Reading (OCR)")
    print("=" * 70)
    print("\n📝 Instructions:")
    print("   1. Show any text to the camera (book, sign, paper, etc.)")
    print("   2. The AI will read and transcribe it!")
    print("\n⏳ You have 5 seconds to position...\n")
    
    await asyncio.sleep(5)
    
    processor = VisionProcessor()
    try:
        print("📸 Capturing and reading...")
        result = await processor.read_text()
        
        if result:
            print("\n✅ Text Read:")
            print("-" * 70)
            print(result)
            print("-" * 70)
        else:
            print("\n❌ Failed to read text")
    finally:
        await processor.close()


async def demo_scene_description():
    """Demo: Describe what's in front of the camera."""
    print("\n" + "=" * 70)
    print("  DEMO 3: Scene Description")
    print("=" * 70)
    print("\n📝 Instructions:")
    print("   1. Point the camera at anything you want described")
    print("   2. The AI will describe what it sees!")
    print("\n⏳ You have 5 seconds...\n")
    
    await asyncio.sleep(5)
    
    processor = VisionProcessor()
    try:
        print("📸 Capturing and describing...")
        result = await processor.describe_scene()
        
        if result:
            print("\n✅ Scene Description:")
            print("-" * 70)
            print(result)
            print("-" * 70)
        else:
            print("\n❌ Failed to describe scene")
    finally:
        await processor.close()


async def demo_custom_question():
    """Demo: Ask a custom question about an image."""
    print("\n" + "=" * 70)
    print("  DEMO 4: Custom Question")
    print("=" * 70)
    
    question = input("\n❓ What question do you want to ask about the image?\n   > ")
    
    if not question.strip():
        question = "What do you see?"
    
    print(f"\n📝 Question: {question}")
    print("⏳ You have 5 seconds to position the object/paper...\n")
    
    await asyncio.sleep(5)
    
    processor = VisionProcessor()
    try:
        print("📸 Capturing and analyzing...")
        result = await processor.analyze_image(question=question)
        
        if result:
            print("\n✅ AI Response:")
            print("-" * 70)
            print(result)
            print("-" * 70)
        else:
            print("\n❌ Failed to analyze")
    finally:
        await processor.close()


async def demo_save_and_analyze():
    """Demo: Capture, save, and analyze an image."""
    print("\n" + "=" * 70)
    print("  DEMO 5: Capture, Save & Analyze")
    print("=" * 70)
    print("\n📝 This will capture an image, save it, and analyze it.")
    print("⏳ You have 5 seconds to position...\n")
    
    await asyncio.sleep(5)
    
    processor = VisionProcessor()
    try:
        print("📸 Capturing image...")
        image_bytes = await processor.capture_image()
        
        if image_bytes:
            # Save the image
            filename = "captured_image.jpg"
            processor.save_image(image_bytes, filename)
            print(f"✅ Image saved as: {filename}")
            
            # Analyze it
            print("🔍 Analyzing...")
            result = await processor.solve_problem(image_bytes)
            
            if result:
                print("\n✅ Analysis:")
                print("-" * 70)
                print(result)
                print("-" * 70)
            else:
                print("\n❌ Failed to analyze")
        else:
            print("❌ Failed to capture image")
            
    finally:
        await processor.close()


async def main():
    """Main menu for vision demos."""
    print("\n" + "=" * 70)
    print("  🎥 VISION API DEMO - ChatGPT Camera Mode for OAK-D")
    print("=" * 70)
    print("\n💡 Requirements:")
    print("   - OAK-D or USB camera connected")
    print("   - OPENAI_API_KEY environment variable set")
    print("\n📋 Choose a demo:\n")
    print("   1. Problem Solving (e.g., '2+2=?')")
    print("   2. Text Reading (OCR)")
    print("   3. Scene Description")
    print("   4. Custom Question")
    print("   5. Capture, Save & Analyze")
    print("   6. Run ALL demos")
    print("   0. Exit")
    
    choice = input("\n👉 Enter choice (0-6): ").strip()
    
    if choice == "1":
        await demo_problem_solving()
    elif choice == "2":
        await demo_text_reading()
    elif choice == "3":
        await demo_scene_description()
    elif choice == "4":
        await demo_custom_question()
    elif choice == "5":
        await demo_save_and_analyze()
    elif choice == "6":
        print("\n🎬 Running all demos...\n")
        await demo_problem_solving()
        input("\n⏸️  Press Enter for next demo...")
        await demo_text_reading()
        input("\n⏸️  Press Enter for next demo...")
        await demo_scene_description()
        input("\n⏸️  Press Enter for next demo...")
        await demo_custom_question()
        input("\n⏸️  Press Enter for next demo...")
        await demo_save_and_analyze()
    elif choice == "0":
        print("\n👋 Goodbye!")
        return
    else:
        print("\n❌ Invalid choice. Please run again and choose 0-6.")
        return
    
    print("\n" + "=" * 70)
    print("  ✅ Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

