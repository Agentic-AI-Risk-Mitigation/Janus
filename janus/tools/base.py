"""
Tool definition types — the single source of truth for tool metadata.

A ``ToolDef`` describes everything Janus needs to know about a tool:
its name, description shown to the LLM, parameter schema, and the Python
callable that implements it.  All framework adapters (LangChain, ADK, etc.)
convert from ``ToolDef`` to their native format; tools are never defined
twice.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, Field, create_model


# Maps JSON Schema primitive types to Python types (for Pydantic model generation)
_JSON_TO_PYTHON: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass
class ToolParam:
    """
    Definition of a single tool parameter.

    Attributes:
        name: Parameter name (must match the handler's kwarg name).
        type: JSON Schema type string ("string", "integer", "boolean", etc.).
        description: Human-readable description shown to the LLM.
        required: Whether the parameter must be supplied.
        default: Default value when ``required=False``.
        enum: Optional list of allowed values.
    """

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None


@dataclass
class ToolDef:
    """
    Complete definition of a Janus-managed tool.

    This is the single canonical representation of a tool.  Janus uses it
    to build OpenAI function-calling schemas, Pydantic models for LangChain,
    and the Progent-style tool spec for policy validation.

    Attributes:
        name: Unique tool name (used as the function name in LLM schemas).
        description: Description shown to the LLM to guide tool selection.
        params: Ordered list of parameter definitions.
        handler: The Python callable that implements the tool.  It must
                 accept its parameters as keyword arguments.
    """

    name: str
    description: str
    params: list[ToolParam]
    handler: Callable

    def to_openai_schema(self) -> dict:
        """
        Build the ``{"type": "function", "function": {...}}`` dict for the
        OpenAI function-calling API (also used by Azure, Together, etc.).
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._parameters_schema(),
            },
        }

    def to_janus_tool_spec(self) -> dict:
        """
        Build the ``{"name": ..., "description": ..., "args": {...}}`` dict
        expected by the policy engine and policy generator.
        """
        return {
            "name": self.name,
            "description": self.description,
            "args": self._parameters_schema(),
        }

    def to_pydantic_model(self) -> Type[BaseModel]:
        """
        Generate a Pydantic model for this tool's parameters.

        Useful for LangChain's ``StructuredTool`` which uses Pydantic for
        input validation and schema inference.
        """
        fields: dict[str, Any] = {}
        for param in self.params:
            py_type = _JSON_TO_PYTHON.get(param.type, str)
            if param.required:
                fields[param.name] = (py_type, Field(description=param.description))
            else:
                fields[param.name] = (
                    Optional[py_type],
                    Field(default=param.default, description=param.description),
                )
        model_name = "".join(w.capitalize() for w in self.name.split("_")) + "Args"
        return create_model(model_name, **fields)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parameters_schema(self) -> dict:
        """Produce the JSON Schema ``parameters`` object for this tool."""
        properties: dict[str, dict] = {}
        required: list[str] = []

        for param in self.params:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum is not None:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema
