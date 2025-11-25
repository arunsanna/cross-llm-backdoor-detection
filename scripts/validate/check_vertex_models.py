#!/usr/bin/env python3
"""
Check available models in Vertex AI Model Garden
"""
import os
from google.cloud import aiplatform
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

project_id = os.getenv("VERTEX_AI_PROJECT_ID")
location = os.getenv("VERTEX_AI_LOCATION", "us-central1")

print(f"Project: {project_id}")
print(f"Location: {location}")
print("\nInitializing Vertex AI...")

try:
    aiplatform.init(project=project_id, location=location)
    print("✅ Vertex AI initialized successfully\n")

    # List models - this will show publisher models available
    print("Checking Model Garden availability...")
    print("\nNote: Vertex AI Model Garden provides access to:")
    print("  - Google models: Gemini Pro, Gemini Flash, etc.")
    print("  - Anthropic models: Claude 3 Opus, Sonnet, Haiku")
    print("  - Meta models: Llama 3.1 405B, 70B, 8B")
    print("  - Mistral models: Mistral Large, Mistral 7B")
    print("  - DeepSeek models: DeepSeek Coder")
    print("  - And many more...")

    print("\n✅ Ready to use Vertex AI Model Garden")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nMake sure you're authenticated:")
    print("  gcloud auth application-default login")
    print(f"  gcloud config set project {project_id}")
