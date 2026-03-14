"""Tool implementations for agentic systems.

This module provides concrete implementations of the Tool abstraction.
Tools represent capabilities that agents can invoke to interact with
the external world.

Available Tools:
- BashTool: Execute shell commands
- CalculatorTool: Evaluate arithmetic expressions safely

Example:
    >>> from simpla_loop.tools.bash import BashTool, BashResult
    >>>
    >>> tool = BashTool(timeout=30)
    >>> result = tool.execute(command="echo hello")
"""

from simpla_loop.tools.bash import BashResult, BashTool
from simpla_loop.tools.calculator import CalculatorTool

__all__ = ["BashTool", "BashResult", "CalculatorTool"]
