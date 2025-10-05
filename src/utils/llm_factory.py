from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from loguru import logger

from utils.config import settings


def create_llm(temperature: float = None, max_tokens: int = None) -> BaseChatModel | None:
    """Create the appropriate LLM based on configuration"""

    temp = temperature if temperature is not None else settings.temperature
    tokens = max_tokens if max_tokens is not None else settings.max_tokens

    if settings.openrouter_api_key:
        logger.info(f"🌐 Using OpenRouter with model: {settings.llm_model}")
        return ChatOpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            model=settings.llm_model,
            temperature=temp,
            max_tokens=tokens,
            default_headers={
                "HTTP-Referer": "https://github.com/RahimTS/adb-knowledge-assistant",
                "X-Title": "ADB Knowledge Assistant",
            },
        )
    else:
        logger.warning("⚠️ No OpenRouter API key found")
        return None


def create_router_llm() -> BaseChatModel:
    """Create LLM optimized for routing (fast, cheap model)"""
    # Use faster model for routing
    return ChatOpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-3-haiku",  # Faster, cheaper model
        temperature=0.0,
        max_tokens=500,
        default_headers={
            "HTTP-Referer": "https://github.com/RahimTS/adb-knowledge-assistant",
            "X-Title": "ADB Knowledge Assistant - Router",
        },
    )


def create_generator_llm() -> BaseChatModel:
    """Create LLM optimized for generation (higher creativity)"""
    return create_llm(temperature=0.3, max_tokens=settings.max_tokens)


def create_synthesizer_llm() -> BaseChatModel:
    """Create LLM optimized for synthesis"""
    return create_llm(temperature=0.1, max_tokens=settings.max_tokens)
