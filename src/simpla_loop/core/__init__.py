"""Core abstractions for agentic loops.

This module defines the fundamental interfaces that all agentic loop
implementations must follow: Loop, Memory, and Tool.
"""

from simpla_loop.core.exceptions import (
    ConfigError,
    LoopError,
    SimpleLoopError,
    ToolError,
)
from simpla_loop.core.loop import Loop, LoopResult
from simpla_loop.core.memory import Memory
from simpla_loop.core.tool import Tool, ToolParameter, ToolResult

__all__ = [
    "SimpleLoopError",
    "ConfigError",
    "LoopError",
    "ToolError",
    "Loop",
    "LoopResult",
    "Memory",
    "Tool",
    "ToolParameter",
    "ToolResult",
]
