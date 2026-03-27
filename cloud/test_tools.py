"""
Test script for AI Assistant Tools
Run this to verify tools are working correctly.
"""
import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Ensure cloud directory is in path
cloud_dir = Path(__file__).parent
if str(cloud_dir) not in sys.path:
    sys.path.insert(0, str(cloud_dir))

from tools import get_tool_executor


async def test_tool_detection():
    """Test tool detection from queries."""
    executor = get_tool_executor()
    
    test_queries = [
        "What's the weather in Paris?",
        "What time is it?",
        "Calculate 25 + 17",
        "Play some music",
        "Remind me to check dinner in 10 minutes",
        "Search for quantum computing",
        "Hello, how are you?"  # Should NOT detect any tool
    ]
    
    print("\n" + "="*60)
    print("TOOL DETECTION TEST")
    print("="*60)
    
    for query in test_queries:
        result = executor.detect_tool_use(query)
        if result:
            print(f"\n✅ Query: {query}")
            print(f"   Tool: {result['tool']}")
            print(f"   Params: {result['params']}")
        else:
            print(f"\n❌ Query: {query}")
            print(f"   No tool detected (will use LLM)")


async def test_time():
    """Test time/date tool."""
    executor = get_tool_executor()
    
    print("\n" + "="*60)
    print("TIME/DATE TEST")
    print("="*60)
    
    result = await executor.get_time()
    print(f"\nSuccess: {result['success']}")
    print(f"Result: {result['result']}")


async def test_calculator():
    """Test calculator tool."""
    executor = get_tool_executor()
    
    print("\n" + "="*60)
    print("CALCULATOR TEST")
    print("="*60)
    
    test_expressions = [
        "25 + 17",
        "100 - 37",
        "15 * 8",
        "144 / 12",
        "(5 + 3) * 2"
    ]
    
    for expr in test_expressions:
        result = await executor.calculate(expr)
        print(f"\n{expr}")
        print(f"  → {result['result']}")


async def test_weather():
    """Test weather API."""
    executor = get_tool_executor()
    
    print("\n" + "="*60)
    print("WEATHER TEST")
    print("="*60)
    print("(Uses Open-Meteo - FREE, no API key needed!)")
    
    result = await executor.get_weather("London")
    print(f"\nSuccess: {result['success']}")
    print(f"Result: {result['result']}")
    
    if result['success']:
        print("\n✅ Weather API working perfectly!")
    else:
        print(f"\n❌ Error: {result['result']}")


async def test_web_search():
    """Test web search."""
    executor = get_tool_executor()
    
    print("\n" + "="*60)
    print("WEB SEARCH TEST")
    print("="*60)
    
    result = await executor.web_search("Albert Einstein")
    print(f"\nSuccess: {result['success']}")
    if result['success']:
        print(f"Result: {result['result'][:200]}...")
    else:
        print(f"Error: {result['result']}")


async def test_music_control():
    """Test music control."""
    executor = get_tool_executor()
    
    print("\n" + "="*60)
    print("MUSIC CONTROL TEST")
    print("="*60)
    print("(Windows: sends to robot, Linux: playerctl, macOS: Music.app)")
    
    # Just test if it's available, don't actually play
    result = await executor.play_music("pause")
    print(f"\nSuccess: {result['success']}")
    print(f"Result: {result['result']}")
    
    if not result['success']:
        print("\n💡 To enable music control:")
        print("   Windows: Sends to robot (check ROVY_ROBOT_IP env var)")
        print("   Linux: sudo apt install playerctl")
        print("   macOS: Built-in support for Music.app")


async def test_reminder():
    """Test reminder."""
    executor = get_tool_executor()
    
    print("\n" + "="*60)
    print("REMINDER TEST")
    print("="*60)
    
    result = await executor.set_reminder("test reminder", 1)
    print(f"\nSuccess: {result['success']}")
    print(f"Result: {result['result']}")


async def test_tts():
    """Test TTS (Text-to-Speech) on robot speakers."""
    print("\n" + "="*60)
    print("TTS TEST (Robot Speakers)")
    print("="*60)
    
    try:
        import os
        import httpx
        
        robot_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
        url = f"http://{robot_ip}:8000/speak"
        
        print(f"Sending TTS to robot at {robot_ip}...")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json={"text": "Hello! Testing text to speech on robot speakers."})
            
            if response.status_code == 200:
                print(f"\n✅ TTS command sent successfully!")
                print(f"   Robot should be speaking now...")
                print(f"   Response: {response.json()}")
            else:
                print(f"\n❌ TTS failed: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
    
    except httpx.ConnectError:
        print(f"\n❌ Could not connect to robot at {robot_ip}:8000")
        print(f"   Make sure:")
        print(f"   1. Robot is powered on")
        print(f"   2. Robot API is running (python main_api.py)")
        print(f"   3. ROVY_ROBOT_IP is correct: {robot_ip}")
    except Exception as e:
        print(f"\n❌ TTS test error: {e}")


async def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" AI ASSISTANT TOOLS TEST SUITE")
    print("="*70)
    
    try:
        # Test tool detection
        await test_tool_detection()
        
        # Test individual tools
        await test_time()
        await test_calculator()
        await test_weather()
        await test_web_search()
        await test_music_control()
        await test_reminder()
        await test_tts()
        
        print("\n" + "="*70)
        print(" TESTS COMPLETED")
        print("="*70)
        print("\n✅ All tool tests completed!")
        print("\n💡 Next steps:")
        print("   1. Install playerctl for music control (optional, Linux on robot)")
        print("   2. Try asking the assistant: 'What's the weather in Paris?'")
        print("   3. Try: 'Calculate 25 + 17'")
        print("   4. Try: 'What time is it?'")
        print("   5. Try: 'Who is Albert Einstein?'")
        print("   6. Robot should have spoken during TTS test!\n")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

