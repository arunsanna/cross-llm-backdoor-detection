#!/usr/bin/env python3
"""
OpenRouter API Validation Script
Tests access to multiple LLMs through unified OpenRouter API
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests
import json

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
                "HTTP-Referer": "https://github.com/yourusername/behavioral-anomaly-detection",
                "X-Title": "Behavioral Anomaly Detection Research"
            },
            json={
                "model": model_id,
                "messages": [
                    {"role": "user", "content": f"Say '{model_name} working' and nothing else."}
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

            # Show usage if available
            if 'usage' in result:
                usage = result['usage']
                print(f"   Tokens: {usage.get('total_tokens', 'N/A')} (prompt: {usage.get('prompt_tokens', 'N/A')}, completion: {usage.get('completion_tokens', 'N/A')})")

            print()
            return True
        else:
            print(f"{RED}❌ {model_name}: FAILED{RESET}")
            print(f"   Model ID: {model_id}")
            print(f"   Status: {response.status_code}")

            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Error: {response.text[:200]}")

            print()
            return False

    except Exception as e:
        print(f"{RED}❌ {model_name}: ERROR{RESET}")
        print(f"   Model ID: {model_id}")
        print(f"   Exception: {str(e)}\n")
        return False

def main():
    """Test multiple LLMs through OpenRouter"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}🔑 OpenRouter Multi-LLM Validation{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key or api_key == "YOUR_OPENROUTER_KEY_HERE":
        print(f"\n{RED}❌ OPENROUTER_API_KEY not configured in .env{RESET}")
        print(f"\n{YELLOW}Please add your OpenRouter API key to .env:{RESET}")
        print(f"   OPENROUTER_API_KEY=\"sk-or-v1-...\"\n")
        return 1

    print(f"\nAPI Key: {api_key[:15]}...{api_key[-10:]}")
    print(f"\n{BLUE}Testing models...{RESET}\n")

    # Models to test - covering different providers
    models_to_test = [
        # OpenAI models (via OpenRouter)
        ("openai/gpt-4o", "GPT-4o"),
        ("openai/gpt-4-turbo", "GPT-4 Turbo"),
        ("openai/gpt-4", "GPT-4"),

        # Anthropic models (via OpenRouter)
        ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet"),
        ("anthropic/claude-3-opus", "Claude 3 Opus"),

        # Meta models (via OpenRouter)
        ("meta-llama/llama-3.1-405b-instruct", "Llama 3.1 405B"),
        ("meta-llama/llama-3.1-70b-instruct", "Llama 3.1 70B"),

        # Mistral models (via OpenRouter)
        ("mistralai/mistral-large", "Mistral Large"),
        ("mistralai/mistral-medium", "Mistral Medium"),

        # DeepSeek models (via OpenRouter)
        ("deepseek/deepseek-coder", "DeepSeek Coder"),
    ]

    results = {}

    for model_id, model_name in models_to_test:
        print(f"{BLUE}Testing {model_name}...{RESET}")
        results[model_name] = test_openrouter_model(api_key, model_id, model_name)

    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}📊 Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    # Group by provider
    providers = {
        "OpenAI": ["GPT-4o", "GPT-4 Turbo", "GPT-4"],
        "Anthropic": ["Claude 3.5 Sonnet", "Claude 3 Opus"],
        "Meta": ["Llama 3.1 405B", "Llama 3.1 70B"],
        "Mistral": ["Mistral Large", "Mistral Medium"],
        "DeepSeek": ["DeepSeek Coder"]
    }

    for provider, models in providers.items():
        print(f"{BLUE}{provider}:{RESET}")
        for model in models:
            if model in results:
                status = f"{GREEN}✅ Working{RESET}" if results[model] else f"{RED}❌ Failed{RESET}"
                print(f"   {model}: {status}")
        print()

    total = len(results)
    passed = sum(results.values())

    print(f"{BLUE}Total: {passed}/{total} models validated{RESET}\n")

    if passed >= 5:  # At least 5 models working
        print(f"{GREEN}🎉 OpenRouter multi-LLM access validated!{RESET}")
        print(f"{GREEN}✅ Ready to start data collection with {passed} models{RESET}\n")
        return 0
    elif passed > 0:
        print(f"{YELLOW}⚠️  Partial success: {passed} models working{RESET}")
        print(f"{YELLOW}You can proceed with available models{RESET}\n")
        return 0
    else:
        print(f"{RED}❌ All models failed validation{RESET}")
        print(f"{RED}Check your OpenRouter API key and credits{RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
