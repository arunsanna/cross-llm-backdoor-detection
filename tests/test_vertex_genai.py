"""
Test Vertex AI with Google GenAI SDK (2025)

Tests latest Gemini models: 2.5 Flash, 2.5 Pro, 2.0 Flash
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

def test_vertex_ai_models():
    """Test all available Gemini models on Vertex AI"""

    print("=" * 70)
    print("VERTEX AI - GOOGLE GENAI SDK TEST (2025)")
    print("=" * 70)
    print()

    # Get config
    project_id = os.getenv("VERTEX_AI_PROJECT_ID")
    location = os.getenv("VERTEX_AI_LOCATION", "us-central1")

    print("📋 Configuration:")
    print(f"   Project: {project_id}")
    print(f"   Location: {location}")
    print()

    if not project_id:
        print("❌ ERROR: VERTEX_AI_PROJECT_ID not set in .env")
        return False

    # Initialize Vertex AI client
    print("🔌 Initializing Vertex AI client...")
    try:
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        print("   ✓ Client initialized\n")
    except Exception as e:
        print(f"   ✗ Failed: {e}\n")
        return False

    # Test models
    models_to_test = [
        ('gemini-2.5-flash', 'Gemini 2.5 Flash (GA - Fast, Recommended)'),
        ('gemini-2.5-pro', 'Gemini 2.5 Pro (GA - Best Quality)'),
        ('gemini-2.0-flash-001', 'Gemini 2.0 Flash (Stable)'),
    ]

    print("🧪 Testing models...")
    print()

    results = []

    for model_id, description in models_to_test:
        print(f"Testing {description}...")
        print(f"  Model ID: {model_id}")

        try:
            response = client.models.generate_content(
                model=model_id,
                contents='Hello! Please respond with: "Ready for research"'
            )

            print(f"  ✅ SUCCESS: {response.text[:60]}...")
            results.append((model_id, True, response.text[:100]))

        except Exception as e:
            error_msg = str(e)[:100]
            print(f"  ❌ FAILED: {error_msg}...")
            results.append((model_id, False, str(e)))

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    available = [(m, d) for m, s, d in results if s]
    unavailable = [(m, d) for m, s, d in results if not s]

    if available:
        print(f"✅ Available models ({len(available)}):")
        for model_id, _ in available:
            print(f"   • {model_id}")
        print()

    if unavailable:
        print(f"❌ Unavailable models ({len(unavailable)}):")
        for model_id, error in unavailable:
            print(f"   • {model_id}")
            print(f"     Error: {error[:80]}...")
        print()

    if available:
        print("=" * 70)
        print("🎉 VERTEX AI IS READY!")
        print("=" * 70)
        print()
        print(f"Recommended model: {available[0][0]}")
        print()
        print("Next steps:")
        print("1. Update agent_wrapper.py to use Google GenAI SDK")
        print("2. Start trace collection with Gemini 2.5 Flash")
        print("3. Collect 1000+ traces for Phase 1")
        print()
        return True
    else:
        print("❌ No models available - check configuration")
        return False


if __name__ == "__main__":
    import sys
    success = test_vertex_ai_models()
    sys.exit(0 if success else 1)
