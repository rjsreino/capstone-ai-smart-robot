#!/usr/bin/env python3
"""
Complete Vision API Example

This demonstrates all the capabilities of the Vision API in one script.
You can use this as a reference for implementing your own features.
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision import VisionProcessor


async def example_1_basic_usage():
    """Example 1: Basic usage - capture and analyze."""
    print("\n" + "="*70)
    print("Example 1: Basic Usage")
    print("="*70)
    
    processor = VisionProcessor()
    
    try:
        # Simplest usage - capture and analyze
        result = await processor.solve_problem()
        print(f"\nResult: {result}")
    finally:
        await processor.close()


async def example_2_custom_question():
    """Example 2: Ask a custom question about the image."""
    print("\n" + "="*70)
    print("Example 2: Custom Question")
    print("="*70)
    
    processor = VisionProcessor()
    
    try:
        # Ask specific question
        result = await processor.analyze_image(
            question="What color is the largest object in this image?",
            max_tokens=200
        )
        print(f"\nResult: {result}")
    finally:
        await processor.close()


async def example_3_ocr_text():
    """Example 3: Read and extract text (OCR)."""
    print("\n" + "="*70)
    print("Example 3: OCR - Read Text")
    print("="*70)
    
    processor = VisionProcessor()
    
    try:
        # Extract all text from image
        text = await processor.read_text()
        print(f"\nExtracted text:\n{text}")
    finally:
        await processor.close()


async def example_4_describe_scene():
    """Example 4: Describe what the camera sees."""
    print("\n" + "="*70)
    print("Example 4: Scene Description")
    print("="*70)
    
    processor = VisionProcessor()
    
    try:
        # Get detailed description
        description = await processor.describe_scene()
        print(f"\nScene: {description}")
    finally:
        await processor.close()


async def example_5_capture_and_save():
    """Example 5: Capture image, save it, then analyze."""
    print("\n" + "="*70)
    print("Example 5: Capture, Save, and Analyze")
    print("="*70)
    
    processor = VisionProcessor()
    
    try:
        # First, capture the image
        print("\n📸 Capturing image...")
        image_bytes = await processor.capture_image()
        
        if not image_bytes:
            print("❌ Failed to capture image")
            return
        
        # Save it
        filename = "captured_problem.jpg"
        if processor.save_image(image_bytes, filename):
            print(f"✅ Image saved as: {filename}")
        
        # Now analyze the saved image
        print("\n🔍 Analyzing...")
        result = await processor.solve_problem(image_bytes)
        print(f"\nResult: {result}")
        
    finally:
        await processor.close()


async def example_6_analyze_file():
    """Example 6: Analyze an existing image file."""
    print("\n" + "="*70)
    print("Example 6: Analyze Existing Image File")
    print("="*70)
    
    # This example assumes you have an image file
    image_path = "test_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"\n⚠️  Image file not found: {image_path}")
        print("   Create one first or use a different path")
        return
    
    processor = VisionProcessor()
    
    try:
        # Read image from file
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # Analyze it
        print(f"\n🔍 Analyzing {image_path}...")
        result = await processor.analyze_image(
            image_bytes=image_bytes,
            question="What do you see in this image?"
        )
        print(f"\nResult: {result}")
        
    finally:
        await processor.close()


async def example_7_multiple_questions():
    """Example 7: Ask multiple questions about the same image."""
    print("\n" + "="*70)
    print("Example 7: Multiple Questions on Same Image")
    print("="*70)
    
    processor = VisionProcessor()
    
    try:
        # Capture image once
        print("\n📸 Capturing image...")
        image_bytes = await processor.capture_image()
        
        if not image_bytes:
            print("❌ Failed to capture")
            return
        
        # Ask different questions about the same image
        questions = [
            "What is the main object in this image?",
            "What colors do you see?",
            "Is there any text visible?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n{i}. Question: {question}")
            answer = await processor.analyze_image(
                image_bytes=image_bytes,
                question=question,
                max_tokens=150
            )
            print(f"   Answer: {answer}")
        
    finally:
        await processor.close()


async def example_8_usb_camera():
    """Example 8: Use USB camera instead of OAK-D."""
    print("\n" + "="*70)
    print("Example 8: Using USB Camera")
    print("="*70)
    
    # Force USB camera usage
    processor = VisionProcessor(use_oakd=False)
    
    try:
        print("\n📸 Using USB camera...")
        result = await processor.solve_problem()
        print(f"\nResult: {result}")
    finally:
        await processor.close()


async def example_9_different_model():
    """Example 9: Use a different OpenAI model."""
    print("\n" + "="*70)
    print("Example 9: Different Vision Model")
    print("="*70)
    
    # Use different model (e.g., gpt-4-turbo)
    processor = VisionProcessor(vision_model="gpt-4-turbo")
    
    try:
        print("\n🤖 Using gpt-4-turbo model...")
        result = await processor.solve_problem()
        print(f"\nResult: {result}")
    finally:
        await processor.close()


async def example_10_error_handling():
    """Example 10: Proper error handling."""
    print("\n" + "="*70)
    print("Example 10: Error Handling")
    print("="*70)
    
    processor = VisionProcessor()
    
    try:
        # Try to capture
        image_bytes = await processor.capture_image()
        
        if image_bytes is None:
            print("❌ Camera not available")
            return
        
        # Try to analyze
        result = await processor.analyze_image(
            image_bytes=image_bytes,
            question="What is this?"
        )
        
        if result is None:
            print("❌ Analysis failed")
            return
        
        print(f"✅ Success: {result}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        # Handle error appropriately
        
    finally:
        # Always clean up
        await processor.close()
        print("🧹 Cleanup complete")


async def example_11_convenience_function():
    """Example 11: Using the convenience function."""
    print("\n" + "="*70)
    print("Example 11: Convenience Function")
    print("="*70)
    
    # Use the convenience function for quick one-off analysis
    from vision import analyze_from_camera
    
    result = await analyze_from_camera(
        question="What do you see?",
        use_oakd=True,
        save_to="quick_capture.jpg"
    )
    
    if result:
        print(f"\nResult: {result}")
        print("Image saved as: quick_capture.jpg")
    else:
        print("❌ Analysis failed")


async def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("  🎥 Complete Vision API Examples")
    print("="*70)
    print("\nThis demonstrates all capabilities of the Vision API.")
    print("Each example shows a different use case.\n")
    
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Custom Question", example_2_custom_question),
        ("OCR Text Reading", example_3_ocr_text),
        ("Scene Description", example_4_describe_scene),
        ("Capture and Save", example_5_capture_and_save),
        ("Analyze File", example_6_analyze_file),
        ("Multiple Questions", example_7_multiple_questions),
        ("USB Camera", example_8_usb_camera),
        ("Different Model", example_9_different_model),
        ("Error Handling", example_10_error_handling),
        ("Convenience Function", example_11_convenience_function),
    ]
    
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i:2d}. {name}")
    print("   0. Run ALL examples")
    print("  99. Exit")
    
    choice = input("\n👉 Choose example (0-11, 99=exit): ").strip()
    
    try:
        choice_num = int(choice)
        
        if choice_num == 99:
            print("\n👋 Goodbye!")
            return
        
        if choice_num == 0:
            # Run all examples
            print("\n🎬 Running all examples...")
            for i, (name, func) in enumerate(examples, 1):
                print(f"\n▶️  Running example {i}/{len(examples)}: {name}")
                try:
                    await func()
                except Exception as e:
                    print(f"❌ Example failed: {e}")
                
                if i < len(examples):
                    input("\n⏸️  Press Enter to continue to next example...")
        
        elif 1 <= choice_num <= len(examples):
            # Run specific example
            name, func = examples[choice_num - 1]
            print(f"\n▶️  Running: {name}")
            await func()
        
        else:
            print(f"❌ Invalid choice: {choice_num}")
            return
        
    except ValueError:
        print(f"❌ Invalid input: {choice}")
        return
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        return
    
    print("\n" + "="*70)
    print("  ✅ Complete!")
    print("="*70)


if __name__ == "__main__":
    print("\n💡 Requirements:")
    print("   - Camera connected (OAK-D or USB)")
    print("   - OPENAI_API_KEY environment variable set")
    print("   - Have something ready to show the camera!")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

