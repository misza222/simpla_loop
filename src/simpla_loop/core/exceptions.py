"""Custom exception hierarchy for simpla_loop.

All domain exceptions inherit from SimpleLoopError, allowing callers
to catch the entire family with a single except clause when needed.
"""


class SimpleLoopError(Exception):
    """Base exception for all simpla_loop errors."""


class ConfigError(SimpleLoopError):
    """Raised when configuration is invalid or missing (e.g. missing API key)."""


class ToolError(SimpleLoopError):
    """Raised when a tool encounters an unrecoverable execution error."""


class LoopError(SimpleLoopError):
    """Raised when the loop cannot continue (e.g. max iterations exceeded)."""
