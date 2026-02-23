from typing import Dict, Type

from implementations.llms.azure_provider import AzureOpenAIProvider
from implementations.llms.base_provider import BaseLLMProvider
from implementations.llms.ollama_provider import OllamaProvider
from implementations.llms.openai_provider import OpenAIProvider
from implementations.llms.openrouter_provider import OpenRouterProvider
from implementations.llms.together_provider import TogetherProvider
from implementations.llms.vllm_provider import VLLMProvider

# Non-OpenAI providers are imported lazily inside get_llm_provider so that
# missing optional dependencies (anthropic, boto3, google-genai) do not
# break imports for users who only install a subset of extras.
# The PROVIDERS dict below maps provider name → class for the eager-import
# group; the lazy group is handled in get_llm_provider directly.

PROVIDERS: Dict[str, Type[BaseLLMProvider]] = {
    # ── OpenAI-compatible (no extra deps beyond base) ─────────────────
    "openai": OpenAIProvider,
    "together": TogetherProvider,
    "openrouter": OpenRouterProvider,
    "azure": AzureOpenAIProvider,
    "ollama": OllamaProvider,
    "vllm": VLLMProvider,
}

# Human-friendly aliases for providers that need lazy imports
_LAZY_PROVIDERS = {
    # Anthropic (requires: uv add anthropic)
    "anthropic": "implementations.llms.anthropic_provider.AnthropicProvider",
    # Google Gemini (requires: uv add google-genai)
    "google": "implementations.llms.google_provider.GoogleProvider",
    # AWS Bedrock (requires: uv add boto3)
    "bedrock": "implementations.llms.bedrock_provider.BedrockProvider",
}


def get_llm_provider(provider_name: str, **kwargs) -> BaseLLMProvider:
    """
    Return an initialised LLM provider instance.

    Args:
        provider_name: Case-insensitive provider key, e.g. 'openai', 'anthropic',
                       'google', 'gemini', 'azure', 'azureopenai', 'ollama',
                       'vllm', 'bedrock', 'aws', 'together', 'openrouter'.
        **kwargs:      Forwarded to the provider constructor.  Common keys:
                       - api_key    (overrides env-var lookup)
                       - base_url   (endpoint override)
                       - api_version (Azure only)
                       - region     (Bedrock only)

    Returns:
        An initialised BaseLLMProvider instance.

    Raises:
        ValueError:   Unknown provider name.
        ImportError:  Optional dependency not installed.
    """
    key = provider_name.lower()

    # Eager providers (always importable)
    if key in PROVIDERS:
        return PROVIDERS[key](**kwargs)

    # Lazy providers (may have optional deps)
    if key in _LAZY_PROVIDERS:
        module_path, class_name = _LAZY_PROVIDERS[key].rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        provider_class: Type[BaseLLMProvider] = getattr(module, class_name)
        return provider_class(**kwargs)

    supported = sorted(set(list(PROVIDERS.keys()) + list(_LAZY_PROVIDERS.keys())))
    raise ValueError(
        f"Unsupported LLM provider: '{provider_name}'. Supported providers: {supported}"
    )
