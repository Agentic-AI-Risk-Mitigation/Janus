"""Base agent class that all framework adapters must implement."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from implementations.core.logging_utils import get_logger


class BaseAgent(ABC):
    """
    Abstract base class for agent implementations.

    Each framework adapter (LangChain, ADK, raw SDK) implements this interface.
    """

    def __init__(
        self,
        config: dict[str, Any],
        workspace: str | Path,
    ):
        """
        Initialize the agent.

        :param config: Configuration dictionary
        :param workspace: Path to the workspace directory
        """
        self.config = config
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

        # Initialize workspace for tools
        from implementations.tools.file_tools import set_workspace

        set_workspace(self.workspace)

    @abstractmethod
    def run(self, user_input: str) -> str:
        """
        Process a single user input and return the agent's response.

        :param user_input: The user's message
        :return: The agent's response
        """
        pass

    @abstractmethod
    def get_tools(self) -> list[Any]:
        """
        Get the list of tools available to this agent.

        :return: List of tools in the framework's format
        """
        pass

    def start_repl(self) -> None:
        """
        Start an interactive REPL (Read-Eval-Print Loop).

        This is a shared implementation that works for all framework adapters.
        """
        print(f"\n{'=' * 60}")
        print("Progent Coding Agent")
        print(f"{'=' * 60}")
        print(f"Workspace: {self.workspace}")
        print(f"Framework: {self.__class__.__name__}")
        print(f"Model: {self.config.get('llm', {}).get('model', 'unknown')}")
        print(f"{'=' * 60}")
        print("Commands:")
        print("  exit/quit  - End the session")
        print("  clear      - Clear conversation history")
        print("  verbose    - Toggle verbose tool output")
        print(f"{'=' * 60}\n")

        verbose_mode = True  # Show tool calls by default

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ("exit", "quit"):
                    print("\nGoodbye!")
                    break

                if user_input.lower() == "clear":
                    self.clear_history()
                    print("Conversation history cleared.\n")
                    continue

                if user_input.lower() == "verbose":
                    verbose_mode = not verbose_mode
                    print(f"Verbose mode: {'ON' if verbose_mode else 'OFF'}\n")
                    continue

                # Log user input
                self.logger.user_input(user_input)

                # Get agent response
                if verbose_mode:
                    print("\n[Processing...]")
                response = self.run(user_input)

                # Log and display response
                self.logger.assistant_response(response)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                self.logger.error("Error in REPL", e)
                print(f"\nError: {e}\n")

    def clear_history(self) -> None:
        """
        Clear the conversation history.

        Override in subclasses if needed.
        """
        pass
