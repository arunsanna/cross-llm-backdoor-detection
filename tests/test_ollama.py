"""
Test Ollama connection with local models.

Works immediately - no API keys, no cloud setup needed!
"""

import os
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama

def test_ollama_connection():
    """Test connection to Ollama"""

    print("=" * 60)
    print("OLLAMA LOCAL LLM CONNECTION TEST")
    print("=" * 60)
    print()

    print("📋 Configuration:")
    print("   Local Ollama server")
    print("   No API keys needed")
    print()

    # Test connection with gpt-oss model
    print("🔌 Testing gpt-oss model...")
    try:
        llm = ChatOllama(
            model="gpt-oss",
            temperature=0
        )

        print("   ✓ gpt-oss instance created")
        print()

        # Test invocation
        print("💬 Sending test message...")
        response = llm.invoke("Hello! Please respond with exactly: 'Connection successful'")

        print(f"   ✓ Response received: {response.content}")
        print()

        # Success!
        print("=" * 60)
        print("✅ SUCCESS! Ollama is ready to use")
        print("=" * 60)
        print()
        print("Available models:")
        print("  • gpt-oss (13GB) - Good quality")
        print("  • gemma3:27b (17GB) - Higher quality")
        print()
        print("Next steps:")
        print("1. Update agent_wrapper.py to use Ollama")
        print("2. Start trace collection")
        print()

        return True

    except Exception as e:
        print(f"   ✗ Error: {e}")
        print()
        print("=" * 60)
        print("❌ CONNECTION FAILED")
        print("=" * 60)
        print()
        print("Make sure Ollama is running:")
        print("  ollama serve")
        print()

        return False


def test_all_models():
    """Test all available Ollama models"""

    print("=" * 60)
    print("TESTING ALL LOCAL MODELS")
    print("=" * 60)
    print()

    models = ["gpt-oss", "gemma3:27b"]

    for model_name in models:
        print(f"Testing {model_name}...")
        try:
            llm = ChatOllama(model=model_name, temperature=0)
            response = llm.invoke("Hi")
            print(f"   ✓ {model_name} works - Response: {response.content[:50]}...")
        except Exception as e:
            print(f"   ✗ {model_name} failed: {str(e)[:50]}...")
        print()


if __name__ == "__main__":
    import sys

    # Test basic connection
    success = test_ollama_connection()

    if success:
        # If basic test passes, test all models
        print()
        test_input = input("Test all available models? (y/n): ")
        if test_input.lower() == 'y':
            print()
            test_all_models()
    else:
        sys.exit(1)
