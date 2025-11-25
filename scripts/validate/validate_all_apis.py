#!/usr/bin/env python3
"""
Unified Multi-LLM Validation Script
Tests: OpenRouter (multiple models) + Vertex AI (Gemini/Claude)
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

def test_openrouter_model(api_key, model_id, model_name):
    """Test a specific model through OpenRouter"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/behavioral-anomaly-detection",
                "X-Title": "Behavioral Anomaly Detection Research"
            },
            json={
                "model": model_id,
                "messages": [
                    {"role": "user", "content": f"Say '{model_name} working' in 3 words or less."}
                ],
                "temperature": 0,
                "max_tokens": 20
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content']

            print(f"{GREEN}✅ {model_name}: SUCCESS{RESET}")
            print(f"   Model ID: {model_id}")
            print(f"   Response: {message}")

            # Show cost info if available
            if 'usage' in result:
                usage = result['usage']
                print(f"   Tokens: {usage.get('total_tokens', 'N/A')}")

            print()
            return True
        else:
            print(f"{RED}❌ {model_name}: FAILED{RESET}")
            print(f"   Model ID: {model_id}")
            print(f"   Status: {response.status_code}")

            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', str(error_data))
                print(f"   Error: {error_msg[:150]}")
            except:
                print(f"   Error: {response.text[:150]}")

            print()
            return False

    except Exception as e:
        print(f"{RED}❌ {model_name}: ERROR{RESET}")
        print(f"   Exception: {str(e)}\n")
        return False

def test_vertex_ai_models():
    """Test Vertex AI Model Garden models (Gemini, Claude)"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Testing Vertex AI (Gemini & Claude)...{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    project_id = os.getenv("VERTEX_AI_PROJECT_ID")
    location = os.getenv("VERTEX_AI_LOCATION", "us-central1")

    if not project_id:
        print(f"{RED}❌ VERTEX_AI_PROJECT_ID not found{RESET}\n")
        return {}

    try:
        from google.cloud import aiplatform
        from vertexai.generative_models import GenerativeModel

        aiplatform.init(project=project_id, location=location)

        # Test only key models we'll use
        models_to_test = {
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Claude 3.5 Sonnet": "claude-3-5-sonnet@20241022",
        }

        results = {}

        for model_name, model_id in models_to_test.items():
            try:
                print(f"{BLUE}Testing {model_name}...{RESET}")
                model = GenerativeModel(model_id)
                response = model.generate_content(
                    f"Say '{model_name} working' in 3 words or less.",
                    generation_config={"temperature": 0, "max_output_tokens": 20}
                )

                print(f"{GREEN}✅ {model_name}: SUCCESS{RESET}")
                print(f"   Model ID: {model_id}")
                print(f"   Response: {response.text}\n")
                results[model_name] = True

            except Exception as e:
                print(f"{RED}❌ {model_name}: FAILED{RESET}")
                print(f"   Error: {str(e)}\n")
                results[model_name] = False

        return results

    except ImportError:
        print(f"{YELLOW}⚠️  Vertex AI SDK not installed{RESET}")
        print(f"   Run: pip install google-cloud-aiplatform\n")
        return {}
    except Exception as e:
        print(f"{RED}❌ Vertex AI init failed: {str(e)}{RESET}")
        print(f"{YELLOW}💡 Run: gcloud auth application-default login{RESET}\n")
        return {}

def main():
    """Test all LLM APIs"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}🔑 Complete Multi-LLM Validation{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")

    # Test OpenRouter models
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Testing OpenRouter Models...{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if not openrouter_key or openrouter_key == "YOUR_OPENROUTER_KEY_HERE":
        print(f"{RED}❌ OPENROUTER_API_KEY not configured{RESET}\n")
        openrouter_results = {}
    else:
        print(f"API Key: {openrouter_key[:15]}...{openrouter_key[-10:]}\n")

        # Models to test - prioritizing free and cost-effective options
        models_to_test = [
            # Free models (HIGH PRIORITY)
            ("deepseek/deepseek-chat-v3-0324:free", "DeepSeek Chat V3 (FREE)"),
            ("openai/gpt-oss-120b", "GPT-OSS 120B"),

            # XAI Grok via OpenRouter (replaces direct XAI API)
            ("x-ai/grok-4.1-fast", "Grok 4.1 Fast"),

            # Premium models (for comparison)
            ("openai/gpt-4o", "GPT-4o"),
            ("openai/gpt-4-turbo", "GPT-4 Turbo"),
            ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet (OR)"),
            ("meta-llama/llama-3.1-405b-instruct", "Llama 3.1 405B"),
            ("mistralai/mistral-large", "Mistral Large"),
        ]

        openrouter_results = {}

        for model_id, model_name in models_to_test:
            openrouter_results[model_name] = test_openrouter_model(
                openrouter_key, model_id, model_name
            )

    # Test Vertex AI models
    vertex_results = test_vertex_ai_models()

    # Summary
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}📊 FINAL VALIDATION SUMMARY{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

    print(f"{BLUE}OpenRouter Models:{RESET}")
    if openrouter_results:
        free_models = []
        premium_models = []

        for model, success in openrouter_results.items():
            status = f"{GREEN}✅{RESET}" if success else f"{RED}❌{RESET}"
            if "(FREE)" in model or "OSS" in model:
                free_models.append((model, status))
            else:
                premium_models.append((model, status))

        if free_models:
            print(f"\n  {YELLOW}Free Models:{RESET}")
            for model, status in free_models:
                print(f"   {status} {model}")

        if premium_models:
            print(f"\n  {YELLOW}Premium Models:{RESET}")
            for model, status in premium_models:
                print(f"   {status} {model}")
    else:
        print(f"   {YELLOW}No OpenRouter key configured{RESET}")

    print(f"\n{BLUE}Vertex AI Models:{RESET}")
    if vertex_results:
        for model, success in vertex_results.items():
            status = f"{GREEN}✅{RESET}" if success else f"{RED}❌{RESET}"
            print(f"   {status} {model}")
    else:
        print(f"   {YELLOW}Authentication needed{RESET}")

    # Count totals
    total_or = len(openrouter_results)
    passed_or = sum(openrouter_results.values())
    total_va = len(vertex_results)
    passed_va = sum(vertex_results.values())

    total = total_or + total_va
    passed = passed_or + passed_va

    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Total Models Validated: {passed}/{total}{RESET}")
    print(f"  OpenRouter: {passed_or}/{total_or}")
    print(f"  Vertex AI: {passed_va}/{total_va}")
    print(f"{BLUE}{'='*70}{RESET}\n")

    if passed >= 5:
        print(f"{GREEN}🎉 Multi-LLM infrastructure validated!{RESET}")
        print(f"{GREEN}✅ Ready for comprehensive cross-LLM study with {passed} models{RESET}")

        # Recommendation
        print(f"\n{YELLOW}💡 Recommended model selection for study:{RESET}")
        working_free = [m for m, s in openrouter_results.items() if s and ("FREE" in m or "OSS" in m)]
        working_premium = [m for m, s in openrouter_results.items() if s and "FREE" not in m and "OSS" not in m]
        working_vertex = [m for m, s in vertex_results.items() if s]

        if working_free:
            print(f"  {GREEN}Free/Cost-effective:{RESET} {', '.join(working_free)}")
        if working_vertex:
            print(f"  {GREEN}Vertex AI:{RESET} {', '.join(working_vertex)}")
        if working_premium:
            print(f"  {YELLOW}Premium (higher cost):{RESET} {', '.join(working_premium[:3])}")

        print()
        return 0
    else:
        print(f"{YELLOW}⚠️  Limited models available ({passed} working){RESET}")
        print(f"{YELLOW}Consider: Check API keys, credits, and authentication{RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
