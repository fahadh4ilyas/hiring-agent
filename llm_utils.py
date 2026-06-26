"""
Utility functions for LLM providers.
"""

import logging
from typing import Any, Dict, Optional
from models import ModelProvider, OllamaProvider, GeminiProvider, OpenAIProvider
from prompt import MODEL_PROVIDER_MAPPING, GEMINI_API_KEY, OPENAI_API_KEY, OPENAI_BASE_URL

logger = logging.getLogger(__name__)


def extract_json_from_response(response_text: str) -> str:
    """
    Extract JSON content from markdown code blocks.

    Args:
        response_text: Text that may contain JSON wrapped in markdown code blocks

    Returns:
        Text with markdown code block syntax removed
    """

    response_text = response_text.strip()
    if "<think>" in response_text:
        think_start = response_text.find("<think>")
        think_end = response_text.find("</think>")
        if think_start != -1 and think_end != -1:
            response_text = response_text[:think_start] + response_text[think_end + 8 :]

    # Remove leading ```json if present
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    # Remove trailing ``` if present
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    return response_text


def initialize_llm_provider(
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Any:
    """
    Initialize the appropriate LLM provider based on the model name.

    Args:
        model_name: The name of the model to use
        api_key: Optional API key override (takes precedence over env vars)
        base_url: Optional base URL override (takes precedence over env vars)

    Returns:
        An initialized LLM provider (OllamaProvider, GeminiProvider, or OpenAIProvider)
    """
    # Default to Ollama provider
    provider = OllamaProvider()
    # If using Gemini and API key is available, use Gemini provider
    model_provider = MODEL_PROVIDER_MAPPING.get(model_name, ModelProvider.OLLAMA)
    if model_provider == ModelProvider.GEMINI:
        effective_key = api_key or GEMINI_API_KEY
        if not effective_key:
            logger.warning("⚠️ Gemini API key not found. Falling back to Ollama.")
        else:
            logger.info(f"🔄 Using Google Gemini API provider with model {model_name}")
            provider = GeminiProvider(api_key=effective_key)
    elif model_provider == ModelProvider.OPENAI:
        effective_key = api_key or OPENAI_API_KEY
        effective_url = base_url or OPENAI_BASE_URL or None
        if not effective_key:
            logger.warning("⚠️ OpenAI API key not found. Falling back to Ollama.")
        else:
            logger.info(f"🔄 Using OpenAI-compatible API provider with model {model_name}" + (f" (base_url: {effective_url})" if effective_url else ""))
            provider = OpenAIProvider(api_key=effective_key, base_url=effective_url)
    else:
        logger.info(f"🔄 Using Ollama provider with model {model_name}")
    return provider
