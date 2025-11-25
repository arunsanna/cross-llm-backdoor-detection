#!/usr/bin/env python3
"""
Comprehensive LLM Validation Script
Tests all LLMs through Vertex AI Model Garden + XAI direct API
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests

# Load environment variables
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def test_vertex_ai_models():
    """Test all Vertex AI Model Garden models"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Testing Vertex AI Model Garden...{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    project_id = os.getenv("VERTEX_AI_PROJECT_ID")
    location = os.getenv("VERTEX_AI_LOCATION", "us-central1")

    if not project_id:
        print(f"{RED}❌ VERTEX_AI_PROJECT_ID not found in .env{RESET}")
        return {}

    print(f"Project ID: {project_id}")
    print(f"Location: {location}\n")

    try:
        from google.cloud import aiplatform
        from vertexai.generative_models import GenerativeModel

        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)

        # Models to test
        models_to_test = {
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Gemini Pro": "gemini-pro",
            "Claude 3 Opus": "claude-3-opus@20240229",
            "Claude 3.5 Sonnet": "claude-3-5-sonnet@20240620",
            "Llama 3.1 405B": "meta/llama-3.1-405b-instruct-maas",
            "Mistral Large": "mistralai/mistral-large@2407",
        }

        results = {}

        for model_name, model_id in models_to_test.items():
            try:
                print(f"{BLUE}Testing {model_name}...{RESET}")
                model = GenerativeModel(model_id)
                response = model.generate_content(
                    f"Say '{model_name} working' and nothing else.",
                    generation_config={"temperature": 0, "max_output_tokens": 20}
                )

                print(f"{GREEN}✅ {model_name}: SUCCESS{RESET}")
                print(f"   Model ID: {model_id}")
                print(f"   Response: {response.text}\n")
                results[model_name] = True

            except Exception as e:
                print(f"{RED}❌ {model_name}: FAILED{RESET}")
                print(f"   Model ID: {model_id}")
                print(f"   Error: {str(e)}\n")
                results[model_name] = False

        return results

    except ImportError:
        print(f"{YELLOW}⚠️  Vertex AI SDK not installed{RESET}")
        print(f"   Install with: pip install google-cloud-aiplatform")
        return {}
    except Exception as e:
        print(f"{RED}❌ Vertex AI initialization failed{RESET}")
        print(f"   Exception: {str(e)}")
        print(f"\n{YELLOW}💡 Make sure you're authenticated:{RESET}")
        print(f"   gcloud auth application-default login")
        print(f"   gcloud config set project {project_id}")
        return {}

def test_xai_api():
    """Test XAI (Grok) API with latest model"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Testing XAI (Grok-4) API...{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    api_key = os.getenv("XAI_API_KEY")

    if not api_key:
        print(f"{RED}❌ XAI_API_KEY not found in .env{RESET}")
        return False

    print(f"API Key: {api_key[:10]}...{api_key[-10:]}\n")

    try:
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": "grok-4-latest",
                "messages": [
                    {"role": "system", "content": "You are a test assistant."},
                    {"role": "user", "content": "Say 'Grok-4 working' and nothing else."}
                ],
                "temperature": 0,
                "stream": False
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content']
            model = result.get('model', 'unknown')

            print(f"{GREEN}✅ XAI (Grok-4) API: SUCCESS{RESET}")
            print(f"   Model: {model}")
            print(f"   Response: {message}")
            if 'usage' in result:
                print(f"   Usage: {result['usage']}")
            return True
        else:
            print(f"{RED}❌ XAI (Grok-4) API: FAILED{RESET}")
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"{RED}❌ XAI (Grok-4) API: ERROR{RESET}")
        print(f"   Exception: {str(e)}")
        return False

def main():
    """Run all API tests"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}🔑 Multi-LLM Validation Script{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    # Test Vertex AI models
    vertex_results = test_vertex_ai_models()

    # Test XAI
    xai_result = test_xai_api()

    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}📊 Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    print(f"\n{BLUE}Vertex AI Model Garden:{RESET}")
    for model, success in vertex_results.items():
        status = f"{GREEN}✅ Working{RESET}" if success else f"{RED}❌ Failed{RESET}"
        print(f"   {model}: {status}")

    print(f"\n{BLUE}Direct API Access:{RESET}")
    xai_status = f"{GREEN}✅ Working{RESET}" if xai_result else f"{RED}❌ Failed{RESET}"
    print(f"   XAI Grok-4: {xai_status}")

    # Count totals
    total_vertex = len(vertex_results)
    passed_vertex = sum(vertex_results.values())
    total = total_vertex + 1  # +1 for XAI
    passed = passed_vertex + (1 if xai_result else 0)

    print(f"\n{BLUE}Total: {passed}/{total} LLMs validated{RESET}")

    if passed >= total - 1:  # Allow 1 failure
        print(f"\n{GREEN}🎉 Multi-LLM infrastructure validated!{RESET}")
        print(f"{GREEN}✅ Ready to start data collection{RESET}")
        return 0
    else:
        print(f"\n{YELLOW}⚠️  Some LLMs failed validation{RESET}")
        print(f"{YELLOW}Fix failed LLMs or proceed with working ones{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
