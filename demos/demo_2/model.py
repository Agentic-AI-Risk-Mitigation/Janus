"""
Model construction for both demo agents, via OpenRouter.

**This module imports no Janus.** The baseline agent depends on it, and
``baseline_agent.py --prove-no-janus`` has to keep passing.

Both agents call :func:`init_model`, so one model decision covers the guarded
and the unguarded path. Running both sides on the same model is what makes the
comparison mean anything — otherwise the difference in outcome could be the two
agents talking to different models rather than the enforcement layer.

Why OpenRouter
--------------

One API key reaches every provider's models, so the demo is not wired to a
single vendor and you can swap models by changing a string. OpenRouter speaks
the OpenAI wire format, so this needs no dependency beyond ``langchain-openai``
— which the ``langchain`` extra already installs.

Set one environment variable::

    export OPENROUTER_API_KEY=sk-or-v1-...

``--model`` takes an OpenRouter model id, which is always ``vendor/model``::

    openai/gpt-4.1-mini              cheap, fast, reliable tool calling
    openai/gpt-4.1                   stronger, pricier
    anthropic/claude-sonnet-4        strong reasoning
    deepseek/deepseek-chat-v3-0324   very cheap
    qwen/qwen3-32b                   cheapest of the capable options

The full catalogue is at https://openrouter.ai/models. **The model must
support tool calling** or the agent cannot call a tool at all — filter by the
"Tools" capability on that page. Roughly 378 of OpenRouter's ~446 models
qualify, but a model that lacks it will simply never invoke a tool and the
demo will look like it silently did nothing.

Pointing somewhere else
-----------------------

``--api-base`` overrides the endpoint, for a self-hosted OpenAI-compatible
server (vLLM, Ollama's OpenAI shim, LM Studio, or a gateway of your own).
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["init_model", "credential_hint", "OPENROUTER_BASE_URL", "API_KEY_ENV"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"

# Optional attribution headers. OpenRouter uses them to label traffic on your
# dashboard; they are not required and carry nothing sensitive.
_ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://github.com/Agentic-AI-Risk-Mitigation/Janus",
    "X-Title": "Janus demo 2 - GitHub issue explainer",
}


def credential_hint(model: str, api_base: str | None = None) -> str | None:
    """
    Return a message describing a missing credential, or ``None`` if fine.

    A custom ``api_base`` points at something other than OpenRouter, which
    supplies its own auth (or none), so no key is demanded in that case.
    """
    if api_base:
        return None
    if os.environ.get(API_KEY_ENV):
        return None

    return (
        f"{API_KEY_ENV} is not set, and '{model}' is served through OpenRouter.\n"
        f"  Get a key at https://openrouter.ai/keys, then:\n"
        f"    export {API_KEY_ENV}=sk-or-v1-...\n"
        f"  Or pass --api-base to use a local OpenAI-compatible server instead."
    )


def init_model(model: str, api_base: str | None = None, temperature: float = 0.0) -> Any:
    """
    Build a chat model backed by OpenRouter.

    OpenRouter is OpenAI wire-compatible, so this is ``ChatOpenAI`` pointed at
    OpenRouter's base URL rather than a bespoke client. That matters for this
    demo: tool calling goes through the same well-exercised code path as a
    direct OpenAI call.

    Args:
        model: OpenRouter model id, e.g. ``"openai/gpt-4.1-mini"``. Must be a
            model that supports tool calling.
        api_base: Override the endpoint. ``None`` uses OpenRouter.
        temperature: Sampling temperature. The demo pins 0 so repeated runs are
            comparable.

    Raises:
        ImportError: If ``langchain-openai`` is not installed.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "OpenRouter support requires langchain-openai.\n"
            'Install with: pip install -e ".[langchain]"'
        ) from exc

    from pydantic import SecretStr

    # A placeholder keeps ChatOpenAI from raising on construction when the key
    # is absent; callers run credential_hint() first, and a custom api_base may
    # legitimately need no key at all.
    api_key = os.environ.get(API_KEY_ENV) or "unset"

    return ChatOpenAI(
        model=model,
        base_url=api_base or OPENROUTER_BASE_URL,
        api_key=SecretStr(api_key),
        temperature=temperature,
        default_headers=dict(_ATTRIBUTION_HEADERS),
    )
