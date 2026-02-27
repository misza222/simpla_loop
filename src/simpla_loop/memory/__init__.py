"""Memory implementations for agentic systems.

This module provides concrete implementations of the Memory abstraction.
Different implementations offer different trade-offs in terms of
speed, capacity, persistence, and retrieval capabilities.

Available Implementations:
- InMemoryMemory: Fast, volatile, dictionary-based storage

Example:
    >>> from simpla_loop.memory.in_memory import InMemoryMemory
    >>>
    >>> memory = InMemoryMemory()
    >>> memory.add("key", "value")
"""

from simpla_loop.memory.in_memory import InMemoryMemory

__all__ = ["InMemoryMemory"]
