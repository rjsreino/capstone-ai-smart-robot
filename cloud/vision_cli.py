#!/usr/bin/env python3
"""
Vision CLI - Command-line tool for Vision API

Simple command-line interface to capture images and analyze them with OpenAI Vision.

Usage:
    python vision_cli.py solve              # Solve a problem
    python vision_cli.py read               # Read text (OCR)
    python vision_cli.py describe           # Describe scene
    python vision_cli.py ask "your question"  # Custom question
    python vision_cli.py file image.jpg "question"  # Analyze file

Examples:
    python vision_cli.py solve
    python vision_cli.py ask "What color is this?"
    python vision_cli.py file photo.jpg "What's in this image?"
"""
import asyncio
import sys
import argparse
from pathlib import Path

from vision import VisionProcessor, analyze_from_camera


async def cmd_solve(args):
    """Solve a problem shown to camera."""
    print("📸 Capturing image...")
    print("💡 Show a paper with a problem (e.g., '2+2=?')")
    
    if args.countdown > 0:
        print(f"\n⏳ {args.countdown} seconds to position your paper...")
        for i in range(args.countdown, 0, -1):
            print(f"   {i}...")
            await asyncio.sleep(1)
    
    processor = VisionProcessor(use_oakd=not args.usb)
    try:
        result = await processor.solve_problem()
        
        if result:
            print("\n" + "="*70)
            print("✅ SOLUTION:")
            print("="*70)
            print(result)
            print("="*70)
        else:
            print("\n❌ Failed to solve problem")
            return 1
            
    finally:
        await processor.close()
    
    return 0


async def cmd_read(args):
    """Read text from camera (OCR)."""
    print("📸 Capturing image...")
    print("💡 Show text to the camera")
    
    if args.countdown > 0:
        print(f"\n⏳ {args.countdown} seconds...")
        for i in range(args.countdown, 0, -1):
            print(f"   {i}...")
            await asyncio.sleep(1)
    
    processor = VisionProcessor(use_oakd=not args.usb)
    try:
        result = await processor.read_text()
        
        if result:
            print("\n" + "="*70)
            print("✅ TEXT:")
            print("="*70)
            print(result)
            print("="*70)
        else:
            print("\n❌ Failed to read text")
            return 1
            
    finally:
        await processor.close()
    
    return 0


async def cmd_describe(args):
    """Describe what camera sees."""
    print("📸 Capturing image...")
    print("💡 Point camera at anything")
    
    if args.countdown > 0:
        print(f"\n⏳ {args.countdown} seconds...")
        for i in range(args.countdown, 0, -1):
            print(f"   {i}...")
            await asyncio.sleep(1)
    
    processor = VisionProcessor(use_oakd=not args.usb)
    try:
        result = await processor.describe_scene()
        
        if result:
            print("\n" + "="*70)
            print("✅ DESCRIPTION:")
            print("="*70)
            print(result)
            print("="*70)
        else:
            print("\n❌ Failed to describe")
            return 1
            
    finally:
        await processor.close()
    
    return 0


async def cmd_ask(args):
    """Ask custom question about camera view."""
    question = args.question
    
    print(f"📸 Capturing image...")
    print(f"❓ Question: {question}")
    
    if args.countdown > 0:
        print(f"\n⏳ {args.countdown} seconds...")
        for i in range(args.countdown, 0, -1):
            print(f"   {i}...")
            await asyncio.sleep(1)
    
    processor = VisionProcessor(use_oakd=not args.usb)
    try:
        result = await processor.analyze_image(question=question)
        
        if result:
            print("\n" + "="*70)
            print("✅ ANSWER:")
            print("="*70)
            print(result)
            print("="*70)
        else:
            print("\n❌ Failed to analyze")
            return 1
            
    finally:
        await processor.close()
    
    return 0


async def cmd_file(args):
    """Analyze an existing image file."""
    filepath = args.filepath
    question = args.question or "What do you see in this image?"
    
    if not Path(filepath).exists():
        print(f"❌ File not found: {filepath}")
        return 1
    
    print(f"📁 Loading image: {filepath}")
    print(f"❓ Question: {question}")
    
    processor = VisionProcessor(use_oakd=not args.usb)
    try:
        # Read image file
        with open(filepath, "rb") as f:
            image_bytes = f.read()
        
        print("\n🔍 Analyzing...")
        result = await processor.analyze_image(
            image_bytes=image_bytes,
            question=question
        )
        
        if result:
            print("\n" + "="*70)
            print("✅ RESULT:")
            print("="*70)
            print(result)
            print("="*70)
        else:
            print("\n❌ Failed to analyze")
            return 1
            
    finally:
        await processor.close()
    
    return 0


async def cmd_test(args):
    """Quick test to verify setup."""
    print("🔍 Testing Vision API setup...")
    print("\nThis will:")
    print("  1. Check camera")
    print("  2. Check OpenAI API")
    print("  3. Capture and analyze an image")
    print("\n" + "="*70)
    
    if args.countdown > 0:
        print(f"\n⏳ {args.countdown} seconds to position something...")
        for i in range(args.countdown, 0, -1):
            print(f"   {i}...")
            await asyncio.sleep(1)
    
    processor = VisionProcessor(use_oakd=not args.usb)
    try:
        # Test capture
        print("\n1️⃣  Testing camera capture...")
        image = await processor.capture_image()
        if image:
            print(f"   ✅ Camera OK ({len(image)} bytes captured)")
        else:
            print("   ❌ Camera failed")
            return 1
        
        # Test API
        print("\n2️⃣  Testing OpenAI Vision API...")
        result = await processor.analyze_image(
            image_bytes=image,
            question="Briefly describe what you see in one sentence."
        )
        
        if result:
            print(f"   ✅ API OK")
            print(f"\n3️⃣  Sample analysis:\n   {result[:100]}...")
        else:
            print("   ❌ API failed")
            return 1
        
        print("\n" + "="*70)
        print("✅ All tests passed! Vision API is working!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    finally:
        await processor.close()
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vision API Command-Line Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s solve                    # Solve a problem shown to camera
  %(prog)s read                     # Read text (OCR)
  %(prog)s describe                 # Describe what camera sees
  %(prog)s ask "What color is this?" # Ask custom question
  %(prog)s file photo.jpg           # Analyze image file
  %(prog)s file photo.jpg "What's in here?" # Analyze with custom question
  %(prog)s test                     # Test setup
  
Tips:
  - Use --countdown N to get N seconds to position
  - Use --usb to force USB camera instead of OAK-D
  - Use --no-countdown for instant capture
        """
    )
    
    parser.add_argument(
        '--usb',
        action='store_true',
        help='Use USB camera instead of OAK-D'
    )
    
    parser.add_argument(
        '--countdown',
        type=int,
        default=3,
        help='Seconds to wait before capture (default: 3, use 0 for no countdown)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Solve command
    subparsers.add_parser('solve', help='Solve a problem shown to camera')
    
    # Read command
    subparsers.add_parser('read', help='Read text from camera (OCR)')
    
    # Describe command
    subparsers.add_parser('describe', help='Describe what camera sees')
    
    # Ask command
    parser_ask = subparsers.add_parser('ask', help='Ask custom question about camera view')
    parser_ask.add_argument('question', help='Question to ask')
    
    # File command
    parser_file = subparsers.add_parser('file', help='Analyze an image file')
    parser_file.add_argument('filepath', help='Path to image file')
    parser_file.add_argument('question', nargs='?', help='Optional question to ask')
    
    # Test command
    subparsers.add_parser('test', help='Test Vision API setup')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Run command
    commands = {
        'solve': cmd_solve,
        'read': cmd_read,
        'describe': cmd_describe,
        'ask': cmd_ask,
        'file': cmd_file,
        'test': cmd_test,
    }
    
    try:
        exit_code = asyncio.run(commands[args.command](args))
        return exit_code or 0
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        return 130
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

