"""
OpenRouter LLM Wrapper for LangChain
Provides unified access to multiple LLMs via OpenRouter API
"""
import os
import requests
from typing import Any, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from dotenv import load_dotenv

load_dotenv()


class OpenRouterLLM(BaseChatModel):
    """LangChain-compatible wrapper for OpenRouter API"""

    model_id: str
    api_key: str = None
    temperature: float = 0
    max_tokens: int = 2048
    timeout: int = 60

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 2048,
        timeout: int = 60,
        **kwargs
    ):
        """
        Initialize OpenRouter LLM wrapper

        Args:
            model_id: OpenRouter model ID (e.g., "openai/gpt-5.1")
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        # Set values before calling super().__init__()
        kwargs['model_id'] = model_id
        kwargs['api_key'] = api_key or os.getenv("OPENROUTER_API_KEY")
        kwargs['temperature'] = temperature
        kwargs['max_tokens'] = max_tokens
        kwargs['timeout'] = timeout

        super().__init__(**kwargs)

        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY env variable.")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:
        """Generate response using OpenRouter API"""

        # Convert LangChain messages to OpenRouter format
        openrouter_messages = []
        for msg in messages:
            if hasattr(msg, 'role'):
                role = msg.role
            else:
                # Infer role from message type
                role = "user"
                if isinstance(msg, AIMessage):
                    role = "assistant"

            openrouter_messages.append({
                "role": role,
                "content": msg.content
            })

        # Make API request
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/behavioral-anomaly-detection",
                    "X-Title": "Behavioral Anomaly Detection Research"
                },
                json={
                    "model": self.model_id,
                    "messages": openrouter_messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stop": stop
                },
                timeout=self.timeout
            )

            if response.status_code != 200:
                error_msg = f"OpenRouter API error: {response.status_code} - {response.text[:200]}"
                raise Exception(error_msg)

            result = response.json()

            # Extract response content
            content = result['choices'][0]['message']['content']

            # Create LangChain response
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)

            return ChatResult(generations=[generation])

        except requests.exceptions.Timeout:
            raise Exception(f"OpenRouter API timeout after {self.timeout}s")
        except requests.exceptions.RequestException as e:
            raise Exception(f"OpenRouter API request failed: {str(e)}")
        except KeyError as e:
            raise Exception(f"Unexpected OpenRouter API response format: {str(e)}")

    @property
    def _llm_type(self) -> str:
        """Return type of LLM"""
        return "openrouter"

    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters"""
        return {
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
