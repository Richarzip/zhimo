"""Application-owned lifecycle for the conversational Agent."""

from __future__ import annotations

from typing import Any


class AgentRuntime:
    """Lazily build one in-process Agent and its conversation checkpointer."""

    def __init__(self) -> None:
        self._agent: Any | None = None
        self._checkpointer: Any | None = None

    def get_agent(self) -> Any:
        if self._agent is None:
            from langgraph.checkpoint.memory import MemorySaver
            from zhimo.agent import create_calligraphy_agent

            self._checkpointer = MemorySaver()
            self._agent = create_calligraphy_agent(checkpointer=self._checkpointer)
        return self._agent

    def reset(self) -> None:
        """Release the in-process Agent; intended for tests and controlled reloads."""
        self._agent = None
        self._checkpointer = None


_RUNTIME = AgentRuntime()


def get_agent() -> Any:
    return _RUNTIME.get_agent()


__all__ = ["AgentRuntime", "get_agent"]