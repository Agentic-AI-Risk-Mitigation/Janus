"""
Model construction for both demo agents, via LiteLLM.

**This module imports no Janus.** The baseline agent depends on it, and
``baseline_agent.py --prove-no-janus`` has to keep passing.

Both agents call :func:`init_model`, so one provider decision covers the
guarded and the unguarded path. Using the same model on both sides is what
makes the comparison meaningful — any difference in outcome comes from the
enforcement layer, not from the two agents talking to different models.

Why LiteLLM
-----------

LiteLLM is a single interface in front of roughly a hundred providers, so the
demo is not wired to one vendor. The model string is ``provider/model``::

    openai/gpt-4o
    anthropic/claude-sonnet-4-5
    gemini/gemini-2.0-flash
    groq/llama-3.3-70b-versatile
    ollama/llama3.1                  local, no API key needed

A bare name with no slash (``gpt-4o``) is treated by LiteLLM as OpenAI.

Credentials come from each provider's usual environment variable —
``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, and so on. :func:`credential_hint`
checks for the common ones up front so a missing key produces a clear message
instead of a provider SDK traceback halfway through a run.

Pointing at a proxy or a local model
------------------------------------

``api_base`` (the ``--api-base`` flag on both agents) redirects requests. That
covers the two keyless setups:

- a LiteLLM proxy:  ``--model openai/gpt-4o --api-base http://localhost:4000``
- local Ollama:     ``--model ollama/llama3.1 --api-base http://localhost:11434``

Install
-------

``pip install langchain-litellm`` (pulls ``litellm``).
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["init_model", "credential_hint", "LITELLM_KEY_ENV"]

# Provider prefix -> the environment variable LiteLLM reads for its credentials.
# Not exhaustive; it covers the providers someone is most likely to reach for.
# A provider missing from this map simply gets no up-front warning.
LITELLM_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "together_ai": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "azure": "AZURE_API_KEY",
    "bedrock": "AWS_ACCESS_KEY_ID",
}

# Providers that run locally and need no credentials.
_KEYLESS_PROVIDERS = frozenset({"ollama", "ollama_chat", "vllm", "lm_studio", "hosted_vllm"})


def _split_model(model: str) -> tuple[str, str]:
    """
    Split a LiteLLM model string into ``(provider, name)``.

    A string with no slash is OpenAI, which is LiteLLM's own convention.
    """
    if "/" not in model:
        return "openai", model
    provider, name = model.split("/", 1)
    return provider.lower(), name


def credential_hint(model: str, api_base: str | None = None) -> str | None:
    """
    Return a message describing a missing credential, or ``None`` if fine.

    Only a hint: an unrecognised provider, a keyless local provider, or an
    explicit ``api_base`` (a proxy supplies its own auth) all return ``None``
    and let LiteLLM raise its own error if something is actually wrong.
    """
    provider, _ = _split_model(model)

    if provider in _KEYLESS_PROVIDERS or api_base:
        return None

    env_var = LITELLM_KEY_ENV.get(provider)
    if env_var is None or os.environ.get(env_var):
        return None

    return (
        f"{env_var} is not set, which LiteLLM needs for '{model}'.\n"
        f"  Set it, or pass --model for a provider you do have a key for,\n"
        f"  or run a local model:  --model ollama/llama3.1 "
        f"--api-base http://localhost:11434"
    )


def init_model(model: str, api_base: str | None = None, temperature: float = 0.0) -> Any:
    """
    Build a LiteLLM-backed chat model.

    Args:
        model: LiteLLM model string, e.g. ``"anthropic/claude-sonnet-4-5"``.
        api_base: Override the provider endpoint — a LiteLLM proxy or a local
            server. ``None`` uses the provider's default.
        temperature: Sampling temperature. The demo pins 0 so repeated runs
            are comparable.

    Raises:
        ImportError: If ``langchain-litellm`` is not installed.
    """
    try:
        from langchain_litellm import ChatLiteLLM
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "LiteLLM support requires langchain-litellm.\n"
            "Install with: pip install langchain-litellm"
        ) from exc

    kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
    if api_base:
        kwargs["api_base"] = api_base
    return ChatLiteLLM(**kwargs)
