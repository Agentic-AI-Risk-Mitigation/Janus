"""
Janus framework adapters.

Each adapter provides a security integration layer so Janus policy
enforcement can be plugged into third-party agent frameworks while
the framework itself stays responsible for the execution / reasoning loop.

Available adapters
------------------
langchain
    ``from janus.adapters.langchain import secure_langchain_tools, JanusLangChainAgent``

    - ``secure_langchain_tools(tools, policy)`` — ToolDef → StructuredTool list
    - ``wrap_langchain_tools(lc_tools, policy)`` — add Janus to existing LangChain tools
    - ``JanusLangChainAgent``                    — turnkey LangChain agent

adk
    ``from janus.adapters.adk import secure_adk_tools, JanusADKAgent``

    - ``secure_adk_tools(tools, policy)``  — ToolDef → (declarations, handlers)
    - ``JanusADKAgent``                    — turnkey Gemini agent

Adding new adapters
-------------------
Create ``janus/adapters/<framework>.py`` exposing at minimum:
    secure_<framework>_tools(tools, policy) → <framework-native tools>

The shared utilities in ``_base.py`` (``resolve_enforcer``,
``make_guarded_handler``) handle all Janus-specific logic so each
adapter file stays thin.
"""
