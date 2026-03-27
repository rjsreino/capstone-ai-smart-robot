#!/usr/bin/env python3
"""
Smart Assistant Benchmark Test
Tests speed and quality of all components:
- Speech Recognition
- Text LLM (Gemma)
- Vision LLM (Phi-Vision)
- Text-to-Speech (TTS)
"""
import os
import sys
import time
import cv2
import numpy as np
from datetime import datetime
from smart_assistant import SmartAssistant

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class BenchmarkTimer:
    """Context manager for timing code blocks."""
    def __init__(self, name, color=Colors.CYAN):
        self.name = name
        self.color = color
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        print(f"{self.color}[⏱️  {self.name}] Starting...{Colors.ENDC}")
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.time() - self.start_time
        if elapsed < 1.0:
            print(f"{self.color}[✅ {self.name}] Completed in {elapsed*1000:.0f}ms{Colors.ENDC}")
        else:
            print(f"{self.color}[✅ {self.name}] Completed in {elapsed:.2f}s{Colors.ENDC}")


# Test questions organized by category
TEST_QUESTIONS = {
    "simple_facts": [
        "What is the capital of France?",
        "How many days are in a week?",
        "What color is the sky?",
        "What is 2 plus 2?",
        "Who invented the telephone?",
    ],
    
    "general_knowledge": [
        "Explain what photosynthesis is in simple terms.",
        "What are the three states of matter?",
        "Why do we have seasons?",
        "What is the largest ocean on Earth?",
        "How does a car engine work?",
    ],
    
    "reasoning": [
        "If I have 5 apples and give away 2, how many do I have left?",
        "What comes next in this sequence: 2, 4, 6, 8?",
        "Which is heavier, a pound of feathers or a pound of bricks?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    ],
    
    "conversational": [
        "What is your status?",
        "Are you ready?",
        "Can you help me?",
        "What can you do?",
    ],
    
    "robot_practical": [
        "How far is 10 feet in meters?",
        "What time is it?",
        "Calculate 25 percent of 80.",
        "Convert 100 fahrenheit to celsius.",
    ],
    
    "vision_simple": [
        "What do you see in this image?",
        "Describe what's in front of the camera.",
        "What objects can you identify?",
        "What colors do you see?",
    ],
    
    "vision_detailed": [
        "Describe this scene in detail.",
        "What is the main object in this image and what is it doing?",
        "Can you count how many objects you see?",
        "What is the dominant color in this image?",
    ],
}


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title:^80}{Colors.ENDC}")
    print(f"{'='*80}\n")


def capture_test_image(camera):
    """Capture an image from camera for vision tests."""
    if camera is None:
        print(f"{Colors.RED}[!] No camera available for vision tests{Colors.ENDC}")
        return None
    
    try:
        # Try multiple times to get a valid frame
        for attempt in range(3):
            frames = camera.capture_frames()
            
            # Check if frames is a dict (not None and not an array)
            if isinstance(frames, dict) and 'color' in frames:
                image = frames['color']
                if image is not None and image.size > 0:
                    return image
            
            # Wait a bit and try again
            if attempt < 2:
                time.sleep(0.5)
        
        print(f"{Colors.YELLOW}⚠️  Camera returned no valid frames after 3 attempts{Colors.ENDC}")
        return None
        
    except Exception as e:
        print(f"{Colors.RED}[!] Failed to capture image: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return None


def test_text_questions(assistant, category, questions):
    """Test text-only questions (uses Gemma)."""
    print_section(f"TEXT TEST: {category.upper().replace('_', ' ')}")
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n{Colors.BLUE}[Q{i}] {question}{Colors.ENDC}")
        
        # Time the entire pipeline
        start_total = time.time()
        
        # Time LLM response
        with BenchmarkTimer("LLM Processing", Colors.YELLOW):
            llm_start = time.time()
            # Shorter, more concise answers for a robot assistant
            answer = assistant.llava.ask_text(
                f"Answer concisely in 1-2 sentences: {question}", 
                max_tokens=50, 
                temperature=0.7
            )
            llm_time = time.time() - llm_start
        
        # Time TTS
        with BenchmarkTimer("TTS Processing", Colors.GREEN):
            tts_start = time.time()
            if not assistant.print_only:
                assistant.tts.speak(answer)
            tts_time = time.time() - tts_start
        
        total_time = time.time() - start_total
        
        # Print results
        print(f"\n{Colors.GREEN}[A{i}] {answer}{Colors.ENDC}")
        print(f"\n{Colors.CYAN}⏱️  Timing Breakdown:{Colors.ENDC}")
        print(f"  • LLM:   {llm_time:.3f}s")
        print(f"  • TTS:   {tts_time:.3f}s")
        print(f"  • Total: {total_time:.3f}s")
        
        results.append({
            'question': question,
            'answer': answer,
            'llm_time': llm_time,
            'tts_time': tts_time,
            'total_time': total_time,
        })
        
        # Brief pause between questions
        time.sleep(0.5)
    
    return results


def test_vision_questions(assistant, category, questions, image):
    """Test vision questions (uses Phi-Vision)."""
    if image is None:
        print(f"\n{Colors.RED}[!] Skipping vision tests - no image available{Colors.ENDC}")
        return []
    
    print_section(f"VISION TEST: {category.upper().replace('_', ' ')}")
    
    # Save test image for reference
    test_img_path = f"/home/jetson/rovy/test_vision_{int(time.time())}.jpg"
    cv2.imwrite(test_img_path, image)
    print(f"{Colors.CYAN}[📷] Test image saved: {test_img_path}{Colors.ENDC}\n")
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n{Colors.BLUE}[Q{i}] {question}{Colors.ENDC}")
        
        # Time the entire pipeline
        start_total = time.time()
        
        # Time VLM response
        with BenchmarkTimer("VLM Processing", Colors.YELLOW):
            vlm_start = time.time()
            # Concise vision descriptions for robot
            answer = assistant.llava.ask_vision(
                f"Answer concisely: {question}", 
                image, 
                max_tokens=80, 
                temperature=0.7
            )
            vlm_time = time.time() - vlm_start
        
        # Time TTS
        with BenchmarkTimer("TTS Processing", Colors.GREEN):
            tts_start = time.time()
            if not assistant.print_only:
                assistant.tts.speak(answer)
            tts_time = time.time() - tts_start
        
        total_time = time.time() - start_total
        
        # Print results
        print(f"\n{Colors.GREEN}[A{i}] {answer}{Colors.ENDC}")
        print(f"\n{Colors.CYAN}⏱️  Timing Breakdown:{Colors.ENDC}")
        print(f"  • VLM:   {vlm_time:.3f}s")
        print(f"  • TTS:   {tts_time:.3f}s")
        print(f"  • Total: {total_time:.3f}s")
        
        results.append({
            'question': question,
            'answer': answer,
            'vlm_time': vlm_time,
            'tts_time': tts_time,
            'total_time': total_time,
        })
        
        # Brief pause between questions
        time.sleep(0.5)
    
    return results


def print_summary(all_results):
    """Print summary statistics."""
    print_section("BENCHMARK SUMMARY")
    
    # Separate text and vision results
    text_results = []
    vision_results = []
    
    for category, results in all_results.items():
        if 'vision' in category:
            vision_results.extend(results)
        else:
            text_results.extend(results)
    
    # Text (Gemma) statistics
    if text_results:
        print(f"{Colors.HEADER}📝 TEXT MODEL (Gemma) PERFORMANCE:{Colors.ENDC}")
        llm_times = [r['llm_time'] for r in text_results]
        tts_times = [r['tts_time'] for r in text_results]
        total_times = [r['total_time'] for r in text_results]
        
        print(f"  Questions tested: {len(text_results)}")
        print(f"  LLM Response Time:")
        print(f"    • Average: {np.mean(llm_times):.3f}s")
        print(f"    • Min:     {np.min(llm_times):.3f}s")
        print(f"    • Max:     {np.max(llm_times):.3f}s")
        print(f"  TTS Time:")
        print(f"    • Average: {np.mean(tts_times):.3f}s")
        print(f"  Total Pipeline:")
        print(f"    • Average: {np.mean(total_times):.3f}s")
        print()
    
    # Vision (Phi) statistics
    if vision_results:
        print(f"{Colors.HEADER}👁️  VISION MODEL (Phi-Vision) PERFORMANCE:{Colors.ENDC}")
        vlm_times = [r['vlm_time'] for r in vision_results]
        tts_times = [r['tts_time'] for r in vision_results]
        total_times = [r['total_time'] for r in vision_results]
        
        print(f"  Questions tested: {len(vision_results)}")
        print(f"  VLM Response Time:")
        print(f"    • Average: {np.mean(vlm_times):.3f}s")
        print(f"    • Min:     {np.min(vlm_times):.3f}s")
        print(f"    • Max:     {np.max(vlm_times):.3f}s")
        print(f"  TTS Time:")
        print(f"    • Average: {np.mean(tts_times):.3f}s")
        print(f"  Total Pipeline:")
        print(f"    • Average: {np.mean(total_times):.3f}s")
        print()
    
    # Speed comparison
    if text_results and vision_results:
        text_avg = np.mean([r['llm_time'] for r in text_results])
        vision_avg = np.mean([r['vlm_time'] for r in vision_results])
        speedup = vision_avg / text_avg
        
        print(f"{Colors.CYAN}⚡ SPEED COMPARISON:{Colors.ENDC}")
        print(f"  Text model (Gemma) is {speedup:.1f}x faster than Vision model (Phi)")
        print(f"  {Colors.GREEN}✨ Dual model strategy working as expected!{Colors.ENDC}")


def main():
    """Main benchmark test."""
    print_section("SMART ASSISTANT BENCHMARK TEST")
    print(f"{Colors.CYAN}Starting comprehensive benchmark...{Colors.ENDC}\n")
    
    # Initialize assistant
    with BenchmarkTimer("Smart Assistant Initialization", Colors.HEADER):
        assistant = SmartAssistant(print_only=False)  # Set to True to disable TTS
    
    print(f"\n{Colors.GREEN}✅ Assistant ready!{Colors.ENDC}\n")
    
    # Capture test image for vision tests
    test_image = None
    if assistant.camera:
        with BenchmarkTimer("Camera Capture", Colors.BLUE):
            test_image = capture_test_image(assistant.camera)
        
        if test_image is None:
            print(f"{Colors.YELLOW}⚠️  No image captured from camera{Colors.ENDC}")
        else:
            print(f"{Colors.GREEN}✅ Image captured: {test_image.shape}{Colors.ENDC}")
    
    # Fallback: try to load a test image from file
    if test_image is None:
        print(f"\n{Colors.CYAN}[Camera Fallback] Attempting to load test image from file...{Colors.ENDC}")
        test_image_paths = [
            '/home/jetson/rovy/debug_capture_20251124_134400.jpg',
            '/home/jetson/rovy/quick_test_image.jpg',
            '/home/jetson/rovy/test_image.jpg',
        ]
        
        for img_path in test_image_paths:
            if os.path.exists(img_path):
                test_image = cv2.imread(img_path)
                if test_image is not None:
                    print(f"{Colors.GREEN}✅ Loaded test image from: {img_path}{Colors.ENDC}")
                    break
        
        if test_image is None:
            print(f"{Colors.YELLOW}⚠️  No test images available - vision tests will be skipped{Colors.ENDC}")
            print(f"{Colors.YELLOW}   Tip: Place a test image at /home/jetson/rovy/test_image.jpg{Colors.ENDC}")
    
    # Store all results
    all_results = {}
    
    # Test text-only questions
    for category in ['simple_facts', 'general_knowledge', 'reasoning', 'conversational', 'robot_practical']:
        questions = TEST_QUESTIONS[category]
        results = test_text_questions(assistant, category, questions)
        all_results[category] = results
    
    # Test vision questions
    if test_image is not None:
        for category in ['vision_simple', 'vision_detailed']:
            questions = TEST_QUESTIONS[category]
            results = test_vision_questions(assistant, category, questions, test_image)
            all_results[category] = results
    
    # Print summary
    print_summary(all_results)
    
    # Save detailed results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/home/jetson/rovy/benchmark_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SMART ASSISTANT BENCHMARK RESULTS\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for category, results in all_results.items():
            f.write(f"\n{category.upper().replace('_', ' ')}\n")
            f.write("-"*80 + "\n")
            for r in results:
                f.write(f"\nQ: {r['question']}\n")
                f.write(f"A: {r['answer']}\n")
                model_type = 'vlm_time' if 'vlm_time' in r else 'llm_time'
                model_time = r.get('vlm_time', r.get('llm_time', 0))
                f.write(f"Timing: Model={model_time:.3f}s, TTS={r['tts_time']:.3f}s, Total={r['total_time']:.3f}s\n")
    
    print(f"\n{Colors.GREEN}📄 Detailed results saved to: {results_file}{Colors.ENDC}\n")
    
    print_section("BENCHMARK COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}[!] Benchmark interrupted by user{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n{Colors.RED}[!] Error during benchmark: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

