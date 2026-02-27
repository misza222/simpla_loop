"""Loop implementations for agentic systems.

This module provides concrete implementations of the Loop abstraction.
Each implementation represents a different strategy for iterative
agent execution.

Available Loops:
- ReActLoop: Reasoning + Acting interleaved pattern

Example:
    >>> from simpla_loop.loops.react import ReActLoop, ReActState
    >>>
    >>> loop = ReActLoop(reasoner=my_llm)
    >>> state = ReActState(query="What is 2+2?")
"""

from simpla_loop.loops.react import ReActLoop, ReActState, ReActStep

__all__ = ["ReActLoop", "ReActState", "ReActStep"]
