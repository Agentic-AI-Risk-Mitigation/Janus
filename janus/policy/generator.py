"""
LLM-based security policy generation.

Generates Janus policies automatically by asking an LLM to infer the
minimum-privilege tool restrictions needed to complete a given user query.
Also supports incremental policy refinement as the agent gathers information
during a task.

Prompt templates live in ``janus/prompts/`` as Jinja2 ``.j2`` files.
The generator renders them at runtime so prompt logic stays in templates,
not scattered through Python code.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from jinja2 import Environment, FileSystemLoader

from janus.exceptions import PolicyGenerationError
from janus.logger import get_logger

_logger = get_logger()

# Jinja2 environment pointing at the prompts directory
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    keep_trailing_newline=True,
)

# Default model (overridable via env var)
_DEFAULT_MODEL = os.getenv("JANUS_POLICY_MODEL", "gpt-4o-2024-08-06")

# Token usage tracking
_total_prompt_tokens: int = 0
_total_completion_tokens: int = 0

# Policy generation guidelines injected into templates
_POLICY_SUGGESTIONS = (
    "Do not use any information you cannot find in the user's query. "
    "This is very important — for example, if the user's query only mentions a name but not "
    "a complete number or email address, do not include them in the policy.\n"
    "When you find explicit and complete identifiers like account numbers, phone numbers, and "
    "email addresses in the user's query, you may use enum instead of pattern to set "
    "restrictions for them. Only apply this if you find complete values.\n"
    "When you want to set restrictions for URLs, use pattern to allow possible variations "
    "(e.g., do not limit the scheme and allow different paths). Only apply this if you find "
    "related information in the user's query.\n"
    "When setting restrictions for dates and times, do not assume values not present in the "
    "query. For example, for 'Jan 12th' only restrict month and day, not year, hour, or minute.\n"
    "When handling array or object types, place restrictions in the correct positions — e.g., "
    "restrictions on array elements belong in the 'items' field."
)


def generate_policy(
    query: str,
    tools: list[dict],
    *,
    model: str | None = None,
    api_key: str | None = None,
    manual_confirm: bool = False,
) -> dict:
    """
    Generate a security policy for a given user query and tool set.

    Uses an LLM to infer which tools are needed and what argument restrictions
    should be applied to limit the agent to only the actions required.

    Args:
        query: The user's task description / query.
        tools: Tool definitions in Janus format:
               ``[{"name": str, "description": str, "args": dict}, ...]``
        model: LLM model string (e.g. "gpt-4o", "claude-3-5-sonnet-20241022").
               Defaults to the JANUS_POLICY_MODEL env var or "gpt-4o-2024-08-06".
        api_key: Optional API key override (otherwise reads from env vars).
        manual_confirm: If True, prints the generated policy and asks for
                        confirmation before returning it.

    Returns:
        Policy dict in internal format, ready to be loaded into a PolicyEnforcer.

    Raises:
        PolicyGenerationError: If generation fails after retries.
    """
    if not tools:
        return {}

    effective_model = model or _DEFAULT_MODEL
    sys_prompt = _render_generate_prompt(effective_model)
    user_content = f"TOOLS: {json.dumps(tools)}\nUSER_QUERY: {query}"

    raw_policy = _call_with_retry(sys_prompt, user_content, effective_model, api_key)
    if raw_policy is None:
        raise PolicyGenerationError("LLM returned an empty or unparseable policy.")

    if manual_confirm:
        print(f"Generated policy:\n{json.dumps(raw_policy, indent=2)}\nApply? [y/N]: ", end="", flush=True)
        if input().strip().lower() != "y":
            _logger.info("Policy generation discarded by user.")
            return {}

    return _to_internal_format(raw_policy)


def refine_policy(
    query: str,
    tools: list[dict],
    tool_call_params: dict,
    tool_call_result: str,
    current_policy: dict,
    *,
    model: str | None = None,
    api_key: str | None = None,
    manual_confirm: bool = True,
) -> dict | None:
    """
    Incrementally refine a policy based on information retrieved during a task.

    After an information-gathering tool call (e.g., reading a file to find a
    recipient address), the policy can be tightened to allow only the specific
    values that were discovered.

    Args:
        query: The original user query.
        tools: Available tool definitions.
        tool_call_params: The parameters of the completed tool call.
        tool_call_result: The output returned by the tool.
        current_policy: The existing generated policy (list-of-dicts format).
        model: LLM model string.
        api_key: Optional API key override.
        manual_confirm: If True, asks the user before applying the update.

    Returns:
        Updated policy in internal format, or None if no update was needed.
    """
    effective_model = model or _DEFAULT_MODEL

    if not _should_refine(query, tools, tool_call_params, effective_model, api_key):
        return None

    sys_prompt = _render_update_prompt(effective_model)
    user_content = (
        f"TOOLS: {json.dumps(tools)}\n"
        f"USER_QUERY: {query}\n"
        f"TOOL_CALL_PARAM: {json.dumps(tool_call_params)}\n"
        f"TOOL_CALL_RESULT: {tool_call_result}\n"
        f"CURRENT_RESTRICTIONS: {json.dumps(current_policy)}"
    )

    raw_policy = _call_with_retry(sys_prompt, user_content, effective_model, api_key)
    if raw_policy is None:
        return None

    if manual_confirm:
        print(f"Refined policy:\n{json.dumps(raw_policy, indent=2)}\nApply? [y/N]: ", end="", flush=True)
        if input().strip().lower() != "y":
            _logger.info("Policy refinement discarded by user.")
            return None

    return _to_internal_format(raw_policy)


def get_token_usage() -> dict[str, int]:
    """Return cumulative token usage for all policy generation calls."""
    return {
        "prompt_tokens": _total_prompt_tokens,
        "completion_tokens": _total_completion_tokens,
        "total_tokens": _total_prompt_tokens + _total_completion_tokens,
    }


def reset_token_usage() -> None:
    """Reset token usage counters."""
    global _total_prompt_tokens, _total_completion_tokens
    _total_prompt_tokens = 0
    _total_completion_tokens = 0


# ------------------------------------------------------------------
# Prompt rendering (Jinja2)
# ------------------------------------------------------------------


def _render_generate_prompt(model: str) -> str:
    template = _jinja_env.get_template("policy_generate.j2")
    return template.render(
        suggestions=_POLICY_SUGGESTIONS,
        output_format=_output_format_hint(model, "generate"),
    ).strip()


def _render_update_prompt(model: str) -> str:
    template = _jinja_env.get_template("policy_update.j2")
    return template.render(
        suggestions=_POLICY_SUGGESTIONS,
        output_format=_output_format_hint(model, "update"),
    ).strip()


def _output_format_hint(model: str, mode: str) -> str:
    """Return a model-specific output format instruction."""
    array_hint = '```json [{"name": tool_name, "args": restrictions}, ...] ```'

    if any(model.startswith(p) for p in ("o1", "o3", "gpt-4.1", "gemini", "meta-llama/", "Qwen/")):
        return f"\nOutput format: {array_hint}"
    if model.startswith("claude"):
        suffix = " with json block. Only output restrictions — no other fields like description or title."
        if mode == "update":
            suffix = f" with json block. It should be an array like {array_hint.replace('```json ', '').replace(' ```', '')}."
        return suffix
    if model.startswith("gpt-4o-mini"):
        return f" with json block. It should be an array of dicts like {{\"name\": tool_name, \"args\": restrictions}}."
    return ""


# ------------------------------------------------------------------
# LLM API calls
# ------------------------------------------------------------------


def _call_llm(
    sys_prompt: str,
    user_content: str,
    model: str,
    api_key: str | None,
    temperature: float = 0.0,
) -> str:
    """Dispatch to the correct LLM provider and return raw text."""
    global _total_prompt_tokens, _total_completion_tokens

    # Anthropic
    if model.startswith("claude"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: uv add anthropic")
        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model=model,
            system=sys_prompt,
            messages=[{"role": "user", "content": user_content}],
            temperature=temperature,
            max_tokens=16384,
        )
        _total_prompt_tokens += msg.usage.input_tokens
        _total_completion_tokens += msg.usage.output_tokens
        return msg.content[0].text

    # Google Gemini
    if model.startswith("gemini"):
        try:
            import vertexai.generative_models as genai
        except ImportError:
            raise ImportError("Install google-cloud-aiplatform: uv add google-cloud-aiplatform")
        m = genai.GenerativeModel(
            model_name=model,
            system_instruction=genai.Part.from_text(text=sys_prompt),
        )
        resp = m.generate_content(
            [genai.Content(role="user", parts=[genai.Part.from_text(user_content)])],
            generation_config=genai.GenerationConfig(temperature=temperature),
        )
        return resp.text

    # OpenAI-compatible
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: uv add openai")

    if os.getenv("OPENROUTER_API_KEY") and not api_key:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
    elif any(model.startswith(p) for p in ("meta-llama/", "Qwen/")):
        client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
    else:
        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    # o1/o3 use developer role
    if model.startswith(("o1", "o3")):
        messages = [{"role": "developer", "content": sys_prompt}, {"role": "user", "content": user_content}]
        resp = client.chat.completions.create(model=model, messages=messages, seed=0)
    else:
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}]
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, seed=0)

    _total_prompt_tokens += resp.usage.prompt_tokens
    _total_completion_tokens += resp.usage.completion_tokens
    return resp.choices[0].message.content


def _call_with_retry(
    sys_prompt: str,
    user_content: str,
    model: str,
    api_key: str | None,
    max_retries: int = 5,
) -> Any:
    """Call the LLM with exponential temperature increase on parse failures."""
    temperature = 0.0
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            text = _call_llm(sys_prompt, user_content, model, api_key, temperature)
            return _extract_json(text)
        except Exception as exc:
            last_error = exc
            temperature = min(temperature + 0.2, 1.0)

    raise PolicyGenerationError(
        f"Policy generation failed after {max_retries} retries: {last_error}"
    )


def _should_refine(
    query: str,
    tools: list[dict],
    tool_call_params: dict,
    model: str,
    api_key: str | None,
) -> bool:
    """Ask the LLM whether this tool result warrants a policy update."""
    sys_prompt = (
        "Your task is to determine whether the result from this tool call can help update "
        "the restrictions. The result will not be provided, but decide if it could be useful "
        "for narrowing or expanding the minimum-privilege policy.\n\n"
        "Output whether you want to update the policy starting with Yes or No."
    )
    content = (
        f"TOOLS: {json.dumps(tools)}\n"
        f"USER_QUERY: {query}\n"
        f"TOOL_CALL_PARAM: {json.dumps(tool_call_params)}"
    )
    try:
        text = _call_llm(sys_prompt, content, model, api_key, 0.0)
        return text.strip().lower().startswith("yes")
    except Exception:
        return False


def _extract_json(text: str) -> Any:
    """Extract a JSON value from an LLM response string."""
    if not text:
        return None

    text = text.strip()

    if text.lower().startswith("no"):
        return None

    # Try JSON code block first
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())

    # Strip leading "yes" prefix
    if text.lower().startswith("yes"):
        text = text[3:].strip()

    return json.loads(text)


def _to_internal_format(generated: list[dict]) -> dict:
    """
    Convert LLM-generated policy list to internal format.

    Generated rules are assigned priority 100 to distinguish them from
    manually defined rules (priority < 100).
    """
    internal: dict[str, list] = {}
    for item in generated:
        tool_name = item.get("name")
        if not tool_name:
            continue
        args = item.get("args", {})
        internal.setdefault(tool_name, []).append((100, 0, args, 0))
    return internal
