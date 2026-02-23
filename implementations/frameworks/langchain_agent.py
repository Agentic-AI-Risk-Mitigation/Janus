"""
LangChain-based agent implementation.

This is a THIN ADAPTER that:
1. Converts unified tool definitions to LangChain format
2. Initializes the LangChain LLM and agent

Tool definitions and policy enforcement are handled by core modules.
DO NOT define tools or policy logic here.
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from implementations.core.progent_enforcer import init_progent
from implementations.core.secured_executor import create_secured_handler
from implementations.core.tool_definitions import TOOL_DEFINITIONS
from implementations.frameworks.base_agent import BaseAgent

load_dotenv()


class LangChainAgent(BaseAgent):
    """
    Agent implementation using LangChain.

    Uses ChatOpenAI with OpenRouter as the backend.
    """

    def __init__(
        self,
        config: dict[str, Any],
        workspace: str | Path,
        policies_path: Optional[str | Path] = None,
    ):
        super().__init__(config, workspace)

        # Get LLM config
        llm_config = config.get("llm", {})
        api_base = llm_config.get("api_base", "https://openrouter.ai/api/v1")
        model = llm_config.get("model", "deepseek/deepseek-v3.2")

        # Get API key from environment
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=api_base,
            temperature=0.1,
        )

        # Initialize Progent policies
        if policies_path:
            tool_defs = [
                {"name": t.name, "description": t.description, "args": {}} for t in TOOL_DEFINITIONS
            ]
            init_progent(policies_path, tool_defs)

        # Create tools (converted from unified definitions)
        self.tools = self._create_tools()

        # Create the agent
        self.agent_executor = self._create_agent(config)

        # Conversation history
        self.chat_history = []

    def _create_tools(self) -> list[StructuredTool]:
        """Convert unified tool definitions to LangChain StructuredTool format."""
        tools = []

        for tool_def in TOOL_DEFINITIONS:
            # Get secured handler (with logging + policy enforcement)
            handler = create_secured_handler(tool_def)

            # Get Pydantic model for proper schema (critical for LangChain!)
            args_schema = tool_def.to_pydantic_model()

            # Convert to LangChain tool with explicit schema
            tool = StructuredTool(
                name=tool_def.name,
                description=tool_def.description,
                func=handler,
                args_schema=args_schema,
            )
            tools.append(tool)

        return tools

    def _create_agent(self, config: dict[str, Any]) -> AgentExecutor:
        """Create the LangChain agent executor."""
        agent_config = config.get("agent", {})
        system_prompt = agent_config.get("system_prompt", "You are a helpful coding assistant.")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            return_intermediate_steps=True,
        )

    def run(self, user_input: str) -> str:
        """Process user input and return response."""
        try:
            result = self.agent_executor.invoke(
                {
                    "input": user_input,
                    "chat_history": self.chat_history,
                }
            )

            response = result.get("output", "No response generated.")

            # Log intermediate steps
            for i, (action, observation) in enumerate(result.get("intermediate_steps", [])):
                self.logger.info(f"Step {i + 1}: {action.tool}({action.tool_input})")

            # Update chat history
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=response))

            return response

        except Exception as e:
            self.logger.error("Agent execution error", e)
            return f"Error: {e}"

    def get_tools(self) -> list:
        """Get the list of tools."""
        return self.tools

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.chat_history = []
