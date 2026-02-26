"""
LLM provider factory.

Maps provider name strings to provider classes. Providers that require
optional dependencies are imported lazily so that users who only install
a subset of extras don't get ImportErrors on startup.

Supported provider names (case-insensitive):
    openai, anthropic, google, gemini, azure, azureopenai,
    ollama, vllm, bedrock, aws, together, openrouter

Model string format (used by JanusAgent):
    <provider>/<model-name>

Examples:
    openai/gpt-4o
    anthropic/claude-3-5-sonnet-20241022
    google/gemini-2.0-flash
    azure/<deployment-name>
    ollama/llama3.2
    vllm/meta-llama/Llama-3.3-70B-Instruct
    bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
    together/meta-llama/Llama-3-70b-chat-hf
    openrouter/anthropic/claude-3.5-sonnet
"""

import importlib
from typing import Type

from janus.llm.base import BaseLLMProvider
from janus.llm.providers.azure_provider import AzureOpenAIProvider
from janus.llm.providers.ollama_provider import OllamaProvider
from janus.llm.providers.openai_provider import OpenAIProvider
from janus.llm.providers.openrouter_provider import OpenRouterProvider
from janus.llm.providers.together_provider import TogetherProvider
from janus.llm.providers.vllm_provider import VLLMProvider

# Eagerly-imported providers (no optional deps beyond openai)
_PROVIDERS: dict[str, Type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    "azure": AzureOpenAIProvider,
    "azureopenai": AzureOpenAIProvider,
    "ollama": OllamaProvider,
    "vllm": VLLMProvider,
    "together": TogetherProvider,
    "openrouter": OpenRouterProvider,
}

# Lazily-imported providers (require optional dependencies)
_LAZY: dict[str, str] = {
    "anthropic": "janus.llm.providers.anthropic_provider.AnthropicProvider",
    "google": "janus.llm.providers.google_provider.GoogleProvider",
    "gemini": "janus.llm.providers.google_provider.GoogleProvider",
    "bedrock": "janus.llm.providers.bedrock_provider.BedrockProvider",
    "aws": "janus.llm.providers.bedrock_provider.BedrockProvider",
}


def get_provider(provider_name: str, **kwargs) -> BaseLLMProvider:
    """
    Return an initialized LLM provider instance.

    Args:
        provider_name: Case-insensitive provider key.
        **kwargs: Forwarded to the provider constructor.
                  Common keys: api_key, base_url, api_version, region.

    Returns:
        An initialized ``BaseLLMProvider`` instance.

    Raises:
        ValueError: Unknown provider name.
        ImportError: Optional dependency not installed.
    """
    key = provider_name.lower()

    if key in _PROVIDERS:
        return _PROVIDERS[key](**kwargs)

    if key in _LAZY:
        dotted = _LAZY[key]
        module_path, class_name = dotted.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls: Type[BaseLLMProvider] = getattr(module, class_name)
        return cls(**kwargs)

    supported = sorted(set(list(_PROVIDERS) + list(_LAZY)))
    raise ValueError(
        f"Unknown LLM provider: '{provider_name}'. "
        f"Supported: {supported}"
    )
