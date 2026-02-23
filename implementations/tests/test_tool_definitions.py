"""Unit tests for implementations.core.tool_definitions module."""

import pytest
from pydantic import BaseModel

from implementations.core.tool_definitions import (
    TOOL_DEFINITIONS,
    ToolDefinition,
    ToolParameter,
    get_all_tool_names,
    get_tool_by_name,
)


class TestToolParameter:
    """Tests for ToolParameter dataclass."""

    def test_create_required_parameter(self):
        """Test creating a required parameter."""
        param = ToolParameter(
            name="file_path", type="string", description="Path to the file", required=True
        )

        assert param.name == "file_path"
        assert param.type == "string"
        assert param.required is True
        assert param.default is None

    def test_create_optional_parameter(self):
        """Test creating an optional parameter with default."""
        param = ToolParameter(
            name="timeout",
            type="integer",
            description="Timeout in seconds",
            required=False,
            default=60,
        )

        assert param.required is False
        assert param.default == 60


class TestToolDefinition:
    """Tests for ToolDefinition class."""

    def test_to_json_schema(self):
        """Test converting tool definition to JSON schema."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters=[
                ToolParameter(name="arg1", type="string", description="First arg", required=True),
                ToolParameter(
                    name="arg2",
                    type="integer",
                    description="Second arg",
                    required=False,
                    default=10,
                ),
            ],
            handler=lambda: None,
        )

        schema = tool.to_json_schema()

        assert schema["type"] == "object"
        assert "arg1" in schema["properties"]
        assert "arg2" in schema["properties"]
        assert schema["properties"]["arg1"]["type"] == "string"
        assert schema["properties"]["arg2"]["type"] == "integer"
        assert "arg1" in schema["required"]
        assert "arg2" not in schema["required"]

    def test_to_pydantic_model(self):
        """Test converting tool definition to Pydantic model."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters=[
                ToolParameter(name="name", type="string", description="Name", required=True),
                ToolParameter(
                    name="count", type="integer", description="Count", required=False, default=5
                ),
            ],
            handler=lambda: None,
        )

        model_cls = tool.to_pydantic_model()

        # Should be a Pydantic model
        assert issubclass(model_cls, BaseModel)

        # Test with required field
        instance = model_cls(name="test")
        assert instance.name == "test"
        assert instance.count == 5  # Default value

        # Test with both fields
        instance2 = model_cls(name="test2", count=10)
        assert instance2.count == 10

    def test_to_pydantic_model_missing_required(self):
        """Test that Pydantic model enforces required fields."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters=[
                ToolParameter(
                    name="required_field", type="string", description="Required", required=True
                ),
            ],
            handler=lambda: None,
        )

        model_cls = tool.to_pydantic_model()

        # Should raise validation error when required field is missing
        with pytest.raises(Exception):  # Pydantic ValidationError
            model_cls()


class TestToolRegistry:
    """Tests for tool registry functions."""

    def test_get_tool_by_name_existing(self):
        """Test getting an existing tool by name."""
        tool = get_tool_by_name("read_file")

        assert tool is not None
        assert tool.name == "read_file"
        assert callable(tool.handler)

    def test_get_tool_by_name_nonexistent(self):
        """Test getting a non-existent tool returns None."""
        tool = get_tool_by_name("nonexistent_tool")

        assert tool is None

    def test_get_all_tool_names(self):
        """Test getting all registered tool names."""
        names = get_all_tool_names()

        assert isinstance(names, list)
        assert len(names) > 0
        assert "read_file" in names
        assert "write_file" in names
        assert "run_command" in names

    def test_all_tools_have_required_fields(self):
        """Test that all registered tools have required fields."""
        for tool in TOOL_DEFINITIONS:
            assert tool.name
            assert tool.description
            assert isinstance(tool.parameters, list)
            assert callable(tool.handler)

    def test_all_tools_have_valid_parameters(self):
        """Test that all tool parameters have valid types."""
        valid_types = {"string", "integer", "number", "boolean"}

        for tool in TOOL_DEFINITIONS:
            for param in tool.parameters:
                assert param.type in valid_types, (
                    f"Invalid type '{param.type}' in tool '{tool.name}'"
                )
                assert param.name
                assert param.description
