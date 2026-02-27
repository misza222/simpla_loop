"""In-memory memory implementation.

Simple dictionary-based storage. Fast but not persistent.
Best for: testing, short-lived agents, prototyping.

This is the simplest possible Memory implementation, useful for:
- Unit tests (fast, isolated, no external dependencies)
- Development and debugging (easy to inspect contents)
- Single-turn agents (no persistence needed)
- Learning and experimentation

Not suitable for:
- Production multi-process deployments
- Long-running agents (data lost on restart)
- Large-scale data (memory constrained)
- Cross-session persistence

Example:
    >>> memory = InMemoryMemory()
    >>> memory.add("user_query", "What's the weather?")
    >>> memory.add("context", {"location": "NYC"})
    >>>
    >>> print(memory.get("user_query"))
    "What's the weather?"
    >>> print(memory.get_all())
    {"user_query": "...", "context": {...}}
    >>> len(memory)
    2
"""

from typing import Any

from simpla_loop.core.memory import Memory


class InMemoryMemory(Memory):
    """Memory implementation using a Python dictionary.

        Stores all data in a plain Python dict. Data is lost when
    the object is garbage collected or the process exits.

        Characteristics:
        - Speed: Very fast (dict operations are O(1) average case)
        - Persistence: None (ephemeral)
        - Capacity: Limited by available RAM
        - Thread-safety: None (single-threaded use only)
        - Serialization: Uses shallow copies (be careful with mutable values)

        Thread Safety:
        This class is NOT thread-safe. If using in a multi-threaded
        environment, provide external synchronization.

        Value Types:
        Any Python object can be stored. However, mutable objects
        (lists, dicts) are stored by reference. Modifying them after
        storage affects the stored value.

        Memory Management:
        References to stored values are held until clear() is called
        or the InMemoryMemory object is deleted. For long-running
        agents, periodically clear() or use a size-limited implementation.

        Attributes:
            _store: Internal dictionary holding all data.
                    Direct access not recommended; use get()/add() instead.

        Example:
            >>> memory = InMemoryMemory()
            >>>
            >>> # Store different types
            >>> memory.add("string", "hello")
            >>> memory.add("number", 42)
            >>> memory.add("list", [1, 2, 3])
            >>> memory.add("dict", {"key": "value"})
            >>>
            >>> # Retrieve
            >>> memory.get("string")
            'hello'
            >>> memory.list_keys()
            ['string', 'number', 'list', 'dict']
            >>> len(memory)
            4
            >>>
            >>> # Clear all
            >>> memory.clear()
            >>> len(memory)
            0
    """

    def __init__(self) -> None:
        """Initialize empty memory store.

        Creates a new empty dictionary to hold memory entries.

        Returns:
            None

        Example:
            >>> memory = InMemoryMemory()
            >>> memory.list_keys()
            []
            >>> len(memory)
            0
        """
        self._store: dict[str, Any] = {}

    def add(self, key: str, value: Any) -> None:
        """Store value in memory.

        If the key already exists, overwrites the previous value.

        Args:
            key: Identifier for this memory entry.
            value: The data to store (any Python object).

        Returns:
            None

        Example:
            >>> memory = InMemoryMemory()
            >>> memory.add("name", "Alice")
            >>> memory.add("name", "Bob")  # Overwrites
            >>> memory.get("name")
            'Bob'
        """
        self._store[key] = value

    def get(self, key: str) -> Any | None:
        """Retrieve value from memory.

        Args:
            key: The identifier for the memory entry.

        Returns:
            The stored value if found, None if not found.

        Example:
            >>> memory.add("key", "value")
            >>> memory.get("key")
            'value'
            >>> memory.get("missing") is None
            True
        """
        return self._store.get(key)

    def list_keys(self) -> list[str]:
        """List all stored keys.

        Returns:
            List of all keys. Order is arbitrary (dict iteration order).

        Example:
            >>> memory.add("a", 1)
            >>> memory.add("b", 2)
            >>> sorted(memory.list_keys())
            ['a', 'b']
        """
        return list(self._store.keys())

    def clear(self) -> None:
        """Clear all memory.

        Removes all entries. After calling, the memory is empty.

        Returns:
            None

        Example:
            >>> memory.add("data", "important")
            >>> memory.clear()
            >>> memory.get("data") is None
            True
            >>> len(memory)
            0
        """
        self._store.clear()

    def get_all(self) -> dict[str, Any]:
        """Get all memory entries as a dictionary.

        Returns a shallow copy of the internal store.
        Modifying the returned dict doesn't affect the memory,
        but modifying mutable values within it does.

        Returns:
            Dictionary of all key-value pairs.

        Example:
            >>> memory.add("x", 1)
            >>> memory.get_all()
            {'x': 1}
        """
        return self._store.copy()

    def __len__(self) -> int:
        """Return number of stored entries.

        Returns:
            Integer count of key-value pairs.

        Example:
            >>> memory.add("a", 1)
            >>> memory.add("b", 2)
            >>> len(memory)
            2
        """
        return len(self._store)

    def __repr__(self) -> str:
        """Debug representation.

        Returns:
            String showing the type and item count.

        Example:
            >>> memory = InMemoryMemory()
            >>> memory.add("test", "value")
            >>> repr(memory)
            "InMemoryMemory(1 items)"
        """
        return f"InMemoryMemory({len(self._store)} items)"
