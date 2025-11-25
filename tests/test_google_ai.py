"""
Test Google AI Studio connection with API key.

This script validates:
1. API key is correctly configured
2. Can connect to Google AI
3. Can invoke Gemini Pro model
4. Ready for trace collection
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_google_ai_connection():
    """Test connection to Google AI Studio"""

    print("=" * 60)
    print("GOOGLE AI STUDIO CONNECTION TEST")
    print("=" * 60)
    print()

    # Check environment variables
    api_key = os.getenv("GOOGLE_API_KEY")

    print("📋 Configuration:")
    print(f"   API Key: {'✓ Set (' + api_key[:15] + '...)' if api_key else '✗ Not set'}")
    print()

    if not api_key:
        print("❌ ERROR: GOOGLE_API_KEY not configured in .env")
        print()
        print("Please edit .env and set:")
        print("   GOOGLE_API_KEY=your-api-key")
        return False

    # Try to import required packages
    print("📦 Checking dependencies...")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("   ✓ langchain-google-genai installed")
    except ImportError:
        print("   ✗ langchain-google-genai not found")
        print()
        print("Install with:")
        print("   pip install langchain-google-genai")
        return False

    print()

    # Test Gemini Pro connection
    print("🔌 Testing Gemini 1.5 Pro connection...")
    try:
        # Create LLM instance
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=api_key,
            temperature=0
        )

        print("   ✓ Gemini Pro instance created")
        print()

        # Test invocation
        print("💬 Sending test message to Gemini Pro...")
        response = llm.invoke("Hello! Please respond with exactly: 'Connection successful'")

        print(f"   ✓ Response received: {response.content}")
        print()

        # Success!
        print("=" * 60)
        print("✅ SUCCESS! Google AI is ready to use")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Run agent wrapper tests")
        print("2. Start trace collection with Gemini Pro")
        print()

        return True

    except Exception as e:
        print(f"   ✗ Error: {e}")
        print()
        print("=" * 60)
        print("❌ CONNECTION FAILED")
        print("=" * 60)
        print()
        print("Common issues:")
        print()
        print("1. Invalid API Key:")
        print("   Get a new key at: https://aistudio.google.com/app/apikey")
        print()
        print("2. API Not Enabled:")
        print("   Enable at: https://aistudio.google.com")
        print()
        print("3. Quota Exceeded:")
        print("   Check quota at: https://aistudio.google.com/app/apikey")
        print()

        return False


def test_multiple_models():
    """Test available models"""

    print("=" * 60)
    print("TESTING AVAILABLE MODELS")
    print("=" * 60)
    print()

    api_key = os.getenv("GOOGLE_API_KEY")
    from langchain_google_genai import ChatGoogleGenerativeAI

    models_to_test = [
        ("gemini-1.5-pro", "Gemini 1.5 Pro (Best quality)"),
        ("gemini-1.5-flash", "Gemini 1.5 Flash (Fast)"),
        ("gemini-1.0-pro", "Gemini 1.0 Pro (Legacy)"),
    ]

    results = []

    for model_id, model_name in models_to_test:
        print(f"Testing {model_name}...")
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_id,
                google_api_key=api_key,
                temperature=0
            )

            response = llm.invoke("Hi")
            print(f"   ✓ {model_name} available")
            results.append((model_name, True, None))
        except Exception as e:
            print(f"   ✗ {model_name} not available: {str(e)[:50]}...")
            results.append((model_name, False, str(e)))

    print()
    print("=" * 60)
    print("MODEL AVAILABILITY SUMMARY")
    print("=" * 60)

    available = [name for name, success, _ in results if success]
    unavailable = [name for name, success, _ in results if not success]

    print()
    print(f"✅ Available models ({len(available)}):")
    for name in available:
        print(f"   • {name}")

    if unavailable:
        print()
        print(f"❌ Unavailable models ({len(unavailable)}):")
        for name in unavailable:
            print(f"   • {name}")

    print()


if __name__ == "__main__":
    # Test basic connection
    success = test_google_ai_connection()

    if success:
        # If basic test passes, test all models
        print()
        test_input = input("Test all available models? (y/n): ")
        if test_input.lower() == 'y':
            print()
            test_multiple_models()
    else:
        sys.exit(1)
