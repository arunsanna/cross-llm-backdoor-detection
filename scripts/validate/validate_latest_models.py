#!/usr/bin/env python3
"""
Validate Latest Premium Models for Multi-LLM Study
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

def test_model(api_key, model_id, model_name):
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
                    {"role": "user", "content": f"Say '{model_name} operational' in 3 words."}
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
                error_msg = error_data.get('error', {})
                if isinstance(error_msg, dict):
                    print(f"   Error: {error_msg.get('message', str(error_msg))[:200]}")
                else:
                    print(f"   Error: {str(error_msg)[:200]}")
            except:
                print(f"   Error: {response.text[:200]}")

            print()
            return False

    except Exception as e:
        print(f"{RED}❌ {model_name}: ERROR{RESET}")
        print(f"   Exception: {str(e)}\n")
        return False

def main():
    """Test latest premium models"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}🔑 Latest Premium Models Validation{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key or api_key == "YOUR_OPENROUTER_KEY_HERE":
        print(f"{RED}❌ OPENROUTER_API_KEY not configured{RESET}\n")
        return 1

    print(f"API Key: {api_key[:15]}...{api_key[-10:]}\n")

    # Latest premium models for study
    models_to_test = [
        ("openai/gpt-5.1", "GPT-5.1"),
        ("anthropic/claude-sonnet-4.5", "Claude Sonnet 4.5"),
        ("x-ai/grok-4.1-fast", "Grok 4.1 Fast"),
        ("meta-llama/llama-4.1", "Llama 4.1"),
        ("openai/gpt-oss-120b", "GPT-OSS 120B"),
        ("qwen/qwen3-max", "Qwen3 Max"),
        ("deepseek/deepseek-chat-v3.1", "DeepSeek Chat V3.1"),
    ]

    results = {}

    for model_id, model_name in models_to_test:
        print(f"{BLUE}Testing {model_name}...{RESET}")
        results[model_name] = test_model(api_key, model_id, model_name)

    # Summary
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}📊 VALIDATION SUMMARY{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

    providers = {
        "OpenAI": ["GPT-5.1", "GPT-OSS 120B"],
        "Anthropic": ["Claude Sonnet 4.5"],
        "XAI": ["Grok 4.1 Fast"],
        "Meta": ["Llama 4.1"],
        "Qwen": ["Qwen3 Max"],
        "DeepSeek": ["DeepSeek Chat V3.1"]
    }

    for provider, models in providers.items():
        print(f"{BLUE}{provider}:{RESET}")
        for model in models:
            if model in results:
                status = f"{GREEN}✅{RESET}" if results[model] else f"{RED}❌{RESET}"
                print(f"   {status} {model}")
        print()

    total = len(results)
    passed = sum(results.values())

    print(f"{BLUE}Total: {passed}/{total} models validated{RESET}\n")

    if passed == total:
        print(f"{GREEN}🎉 All 7 premium models validated!{RESET}")
        print(f"{GREEN}✅ Ready for comprehensive 7-LLM cross-generalization study{RESET}")

        print(f"\n{YELLOW}💡 Study Configuration:{RESET}")
        print(f"   • 7 LLMs from 6 major providers")
        print(f"   • 200 traces per model (100 benign + 100 backdoor)")
        print(f"   • Total: 1,400 traces")
        print(f"   • Cross-LLM combinations: 21 pairwise tests")
        print()
        return 0
    elif passed >= 5:
        print(f"{YELLOW}⚠️  {passed}/{total} models validated{RESET}")
        print(f"{YELLOW}Can proceed with working models{RESET}\n")
        return 0
    else:
        print(f"{RED}❌ Insufficient models validated ({passed}/{total}){RESET}")
        print(f"{RED}Need at least 5 models for meaningful study{RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
