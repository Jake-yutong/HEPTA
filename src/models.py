"""LLM client implementations for the HEPTA benchmark.

Supports:
- OpenAI-compatible providers: OpenAI, DeepSeek, Qwen (Alibaba), Kimi (Moonshot),
  Minimax, and any custom OpenAI-compatible endpoint.
- Anthropic: Claude model family via the Anthropic Messages API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

#: Default base URLs for each provider.
PROVIDER_BASE_URLS: Dict[str, Optional[str]] = {
    "openai":    "https://api.openai.com/v1",
    "deepseek":  "https://api.deepseek.com/v1",
    "qwen":      "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "kimi":      "https://api.moonshot.cn/v1",
    "minimax":   "https://api.minimax.chat/v1",
    "anthropic": None,   # uses its own SDK
    "custom":    None,   # user-supplied
}

#: Suggested model names per provider.
PROVIDER_SUGGESTED_MODELS: Dict[str, List[str]] = {
    "openai":    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "deepseek":  ["deepseek-chat", "deepseek-reasoner"],
    "qwen":      ["qwen-max", "qwen-plus", "qwen-turbo"],
    "kimi":      ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
    "minimax":   ["MiniMax-Text-01", "abab6.5t-chat"],
    "anthropic": ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-3-5"],
    "custom":    [],
}


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Configuration for a single LLM endpoint."""

    provider: str = "openai"
    api_key: str = ""
    model_name: str = "gpt-4o"
    base_url: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract base class for all LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Send *prompt* and return the generated response text."""


# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------

class OpenAICompatibleClient(LLMClient):
    """Client for any OpenAI-compatible chat API.

    Compatible providers: OpenAI, DeepSeek, Qwen, Kimi, Minimax, and any
    custom endpoint following the OpenAI ``/v1/chat/completions`` schema.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Install 'openai' to use OpenAI-compatible providers: pip install openai"
            ) from exc

        base_url = config.base_url or PROVIDER_BASE_URLS.get(config.provider)
        self._client = OpenAI(api_key=config.api_key, base_url=base_url)

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

class AnthropicClient(LLMClient):
    """Client for the Anthropic Messages API (Claude model family)."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        try:
            import anthropic  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Install 'anthropic' to use Anthropic models: pip install anthropic"
            ) from exc

        self._client = anthropic.Anthropic(api_key=config.api_key)

    def generate(self, prompt: str, system: str = "") -> str:
        kwargs: dict = {
            "model": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text  # type: ignore[index]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class LLMClientFactory:
    """Creates an :class:`LLMClient` from a :class:`ModelConfig`."""

    @staticmethod
    def create(config: ModelConfig) -> LLMClient:
        if config.provider == "anthropic":
            return AnthropicClient(config)
        return OpenAICompatibleClient(config)
