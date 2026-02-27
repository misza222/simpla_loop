"""Abstract memory interface for agentic systems.

Memory provides persistence and retrieval of information across
loop iterations. Different implementations offer different
trade-offs (speed, capacity, persistence, retrieval methods).

The Memory abstraction is intentionally simple (key-value) to allow
for diverse implementations while maintaining a common interface.
More sophisticated retrieval (semantic search, conversation history,
etc.) can be built on top of this base interface.

Example:
    >>> memory = InMemoryMemory()
    >>> memory.add("user_query", "What's the weather?")
    >>> memory.add("context", {"location": "NYC", "units": "celsius"})
    >>>
    >>> query = memory.get("user_query")
    >>> context = memory.get("context")
    >>> all_data = memory.get_all()
"""

from abc import ABC, abstractmethod
from typing import Any


class Memory(ABC):
    """Abstract interface for agent memory.

    Memory is the agent's window into the past. It stores:
    - Observations from the environment
    - Results of tool executions
    - The agent's own reasoning/thoughts
    - User messages and context
    - Intermediate computations

    Implementations determine HOW information is stored and retrieved:
    - InMemoryMemory: Fast, volatile, single-process
    - RedisMemory: Distributed, persistent, multi-process
    - VectorMemory: Semantic retrieval, embedding-based
    - FileMemory: Persistent, simple, slow

    Key Design Principles:
    1. Key-value interface: Simple but flexible
    2. No schema enforcement: Values can be any serializable type
    3. Optional persistence: Some implementations are transient
    4. Mutable: Memory is modified in-place (unlike Loop state)

    Thread Safety:
    Implementations may or may not be thread-safe. Check the specific
    implementation's documentation. InMemoryMemory is NOT thread-safe.

    Serialization:
    Values should be serializable (JSON-compatible or pickleable).
    Implementations may impose restrictions on value types.

    Attributes:
        None (interface only)

    Methods:
        add: Store a value
        get: Retrieve a value
        list_keys: Enumerate all keys
        clear: Remove all entries
        get_all: Convenience method to get everything

    Example:
        >>> memory = InMemoryMemory()
        >>> memory.add("step_1", {"action": "search", "result": "..."})
        >>> memory.add("step_2", {"action": "calculate", "result": "..."})
        >>>
        >>> for key in memory.list_keys():
        ...     print(f"{key}: {memory.get(key)}")
    """

    @abstractmethod
    def add(self, key: str, value: Any) -> None:
        """Store a value in memory.

        If the key already exists, the value is overwritten.
        This is intentional - memory is for current state, not history.
        For history, use a list value and append to it.

        Args:
            key: Identifier for this memory entry. Should be unique within
                 this Memory instance. Use descriptive names like
                 "user_query", "tool_results", "conversation_history".
            value: The data to store. Must be serializable by the specific
                   implementation (typically JSON-serializable).
                   Can be any type: str, int, dict, list, custom objects.

        Returns:
            None

        Raises:
            MemoryError: If storage fails (e.g., out of space, serialization error).
            TypeError: If the value type is not supported by this implementation.

        Example:
            >>> memory.add("query", "What is Python?")
            >>> memory.add("context", {"source": "user", "timestamp": 12345})
            >>> memory.add("history", [])  # Will append later
        """
        ...

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Retrieve a value from memory.

        Args:
            key: The identifier for the memory entry. This should match
                 a key previously passed to add().

        Returns:
            The stored value if found, None if the key doesn't exist.
            The return type matches what was passed to add().

        Raises:
            MemoryError: If retrieval fails (rare, but possible for
                        external storage like network-based memory).

        Example:
            >>> memory.add("name", "Alice")
            >>> memory.get("name")
            'Alice'
            >>> memory.get("nonexistent") is None
            True
        """
        ...

    @abstractmethod
    def list_keys(self) -> list[str]:
        """List all keys in memory.

        Returns:
            List of all stored keys as strings.
            Order is implementation-defined (may be sorted, insertion order,
            or undefined). Don't rely on ordering.

        Raises:
            MemoryError: If the operation fails.

        Example:
            >>> memory.add("a", 1)
            >>> memory.add("b", 2)
            >>> sorted(memory.list_keys())
            ['a', 'b']
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from memory.

        This is a destructive operation. After calling clear(),
        get() will return None for all keys and list_keys() will
        return an empty list.

        Returns:
            None

        Raises:
            MemoryError: If the operation fails.

        Example:
            >>> memory.add("temp", "data")
            >>> memory.clear()
            >>> memory.get("temp") is None
            True
            >>> memory.list_keys()
            []
        """
        ...

    def get_all(self) -> dict[str, Any]:
        """Get all memory entries as a dictionary.

        This is a convenience method provided as a default implementation.
        Subclasses may override this with a more efficient implementation
        if they can fetch all data in one operation.

        Returns:
            Dictionary mapping all keys to their values.
            Returns an empty dict if memory is empty.

        Example:
            >>> memory.add("x", 1)
            >>> memory.add("y", 2)
            >>> memory.get_all()
            {'x': 1, 'y': 2}
        """
        return {key: self.get(key) for key in self.list_keys()}
