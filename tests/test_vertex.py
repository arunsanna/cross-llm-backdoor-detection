"""
Test Vertex AI connection with your company credentials.

This script validates:
1. Project ID and API key are correctly configured
2. Can connect to Vertex AI
3. Can invoke Gemini Pro model
4. Ready for trace collection
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_vertex_ai_connection():
    """Test connection to Vertex AI"""

    print("=" * 60)
    print("VERTEX AI CONNECTION TEST")
    print("=" * 60)
    print()

    # Check environment variables
    project_id = os.getenv("VERTEX_AI_PROJECT_ID")
    location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
    api_key = os.getenv("VERTEX_AI_API_KEY")

    print("📋 Configuration:")
    print(f"   Project ID: {project_id}")
    print(f"   Location: {location}")
    print(f"   API Key: {'✓ Set' if api_key else '✗ Not set (will use gcloud auth)'}")
    print()

    if not project_id or project_id == "your-company-project-id":
        print("❌ ERROR: VERTEX_AI_PROJECT_ID not configured in .env")
        print()
        print("Please edit .env and set:")
        print("   VERTEX_AI_PROJECT_ID=your-actual-project-id")
        return False

    # Try to import required packages
    print("📦 Checking dependencies...")
    try:
        from langchain_google_vertexai import ChatVertexAI
        print("   ✓ langchain-google-vertexai installed")
    except ImportError:
        print("   ✗ langchain-google-vertexai not found")
        print()
        print("Install with:")
        print("   pip install langchain-google-vertexai google-cloud-aiplatform")
        return False

    print()

    # Test Gemini Pro connection
    print("🔌 Testing Gemini Pro connection...")
    try:
        # Create LLM instance (Vertex AI uses gcloud auth, not API keys)
        llm = ChatVertexAI(
            model="gemini-1.5-pro",
            project=project_id,
            location=location,
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
        print("✅ SUCCESS! Vertex AI is ready to use")
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
        print("1. Authentication:")
        print("   Run: gcloud auth application-default login")
        print()
        print("2. API Not Enabled:")
        print("   Enable at: https://console.cloud.google.com/vertex-ai")
        print("   Or run: gcloud services enable aiplatform.googleapis.com")
        print()
        print("3. Permissions:")
        print("   Ask admin to grant 'Vertex AI User' role")
        print()
        print("4. Project ID:")
        print("   Verify project ID at: https://console.cloud.google.com")
        print()

        return False


def test_multiple_models():
    """Test all available models"""

    print("=" * 60)
    print("TESTING MULTIPLE MODELS")
    print("=" * 60)
    print()

    project_id = os.getenv("VERTEX_AI_PROJECT_ID")
    location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
    api_key = os.getenv("VERTEX_AI_API_KEY")

    from langchain_google_vertexai import ChatVertexAI

    models_to_test = [
        ("gemini-1.5-pro", "Gemini Pro (Primary)"),
        ("gemini-1.5-flash", "Gemini Flash (Fast)"),
        ("claude-3-5-sonnet@20241022", "Claude 3.5 Sonnet"),
    ]

    results = []

    for model_id, model_name in models_to_test:
        print(f"Testing {model_name}...")
        try:
            llm = ChatVertexAI(
                model=model_id,
                project=project_id,
                location=location,
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
    success = test_vertex_ai_connection()

    if success:
        # If basic test passes, test all models
        print()
        test_input = input("Test all available models? (y/n): ")
        if test_input.lower() == 'y':
            print()
            test_multiple_models()
    else:
        sys.exit(1)
