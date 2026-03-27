"""
Example: Using AI Assistant with Tools

This script demonstrates how to use the AI assistant with external tools.
Run this to see tools in action!
"""
import sys
from pathlib import Path

# Ensure cloud directory is in path
cloud_dir = Path(__file__).parent
if str(cloud_dir) not in sys.path:
    sys.path.insert(0, str(cloud_dir))

from ai import CloudAssistant


def main():
    print("="*70)
    print(" AI ASSISTANT WITH TOOLS - EXAMPLE")
    print("="*70)
    print()
    
    # Create assistant with tools enabled
    print("Loading AI assistant with tools...")
    assistant = CloudAssistant(enable_tools=True, lazy_load=False)
    print("✅ Assistant loaded!\n")
    
    # Example queries that will use tools
    examples = [
        # Time/Date (tool)
        {
            "query": "What time is it?",
            "expected": "Tool: get_time",
            "description": "Uses time tool (no LLM needed)"
        },
        
        # Calculator (tool)
        {
            "query": "Calculate 25 + 17",
            "expected": "Tool: calculate",
            "description": "Uses calculator tool (no LLM needed)"
        },
        
        # Weather (tool - free, no API key needed!)
        {
            "query": "What's the weather in Paris?",
            "expected": "Tool: get_weather",
            "description": "Uses weather API (Open-Meteo, free!)"
        },
        
        # General question (LLM)
        {
            "query": "Tell me a joke",
            "expected": "LLM response",
            "description": "No tool needed, uses LLM"
        },
        
        # Greeting (LLM)
        {
            "query": "Hello, how are you?",
            "expected": "LLM response",
            "description": "No tool needed, uses LLM"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*70}")
        print(f"Example {i}: {example['description']}")
        print(f"{'='*70}")
        print(f"\n💬 Query: \"{example['query']}\"")
        print(f"🎯 Expected: {example['expected']}")
        print(f"\n🤖 Response:")
        
        try:
            response = assistant.ask(example['query'])
            print(f"   {response}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n{'='*70}")
    print(" EXAMPLES COMPLETED")
    print("="*70)
    print()
    print("💡 Tips:")
    print("   - Tools are automatically detected from your query")
    print("   - ALL tools work without API keys!")
    print("   - Weather uses Open-Meteo (free, no registration)")
    print("   - Music control requires playerctl (Linux) or Music.app (macOS)")
    print()
    print("📚 Learn more:")
    print("   - TOOLS_QUICKSTART.md - Quick start guide")
    print("   - TOOLS_README.md - Complete documentation")
    print("   - test_tools.py - Test all tools")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

