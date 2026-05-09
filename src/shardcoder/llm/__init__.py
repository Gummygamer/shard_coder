"""Local LLM client abstraction."""

from .client import ChatMessage, ChatResult, LLMClient, LLMError
from .openai_compatible import OpenAICompatibleClient

__all__ = [
    "ChatMessage",
    "ChatResult",
    "LLMClient",
    "LLMError",
    "OpenAICompatibleClient",
]
