"""Tests for InMemoryMemory implementation."""

import pytest

from simpla_loop.memory.in_memory import InMemoryMemory


class TestInMemoryMemory:
    """Test suite for InMemoryMemory."""

    def test_init_empty(self):
        """Memory should start empty."""
        memory = InMemoryMemory()
        assert memory.list_keys() == []
        assert len(memory) == 0
        assert memory.get_all() == {}

    def test_add_and_get(self):
        """Should be able to add and retrieve values."""
        memory = InMemoryMemory()
        memory.add("key", "value")

        assert memory.get("key") == "value"
        assert len(memory) == 1

    def test_add_overwrites(self):
        """Adding same key should overwrite."""
        memory = InMemoryMemory()
        memory.add("key", "old")
        memory.add("key", "new")

        assert memory.get("key") == "new"

    def test_get_missing_returns_none(self):
        """Getting missing key should return None."""
        memory = InMemoryMemory()
        assert memory.get("nonexistent") is None

    def test_list_keys(self):
        """Should list all keys."""
        memory = InMemoryMemory()
        memory.add("a", 1)
        memory.add("b", 2)
        memory.add("c", 3)

        keys = memory.list_keys()
        assert set(keys) == {"a", "b", "c"}

    def test_clear(self):
        """Clear should remove all entries."""
        memory = InMemoryMemory()
        memory.add("key", "value")
        memory.clear()

        assert memory.get("key") is None
        assert len(memory) == 0
        assert memory.list_keys() == []

    def test_get_all(self):
        """get_all should return all entries."""
        memory = InMemoryMemory()
        memory.add("a", 1)
        memory.add("b", 2)

        all_data = memory.get_all()
        assert all_data == {"a": 1, "b": 2}

    def test_get_all_returns_copy(self):
        """get_all should return a copy."""
        memory = InMemoryMemory()
        memory.add("key", "value")

        all_data = memory.get_all()
        all_data["new"] = "entry"

        assert memory.get("new") is None

    def test_different_value_types(self):
        """Should handle various value types."""
        memory = InMemoryMemory()

        memory.add("string", "hello")
        memory.add("number", 42)
        memory.add("list", [1, 2, 3])
        memory.add("dict", {"nested": "value"})
        memory.add("none", None)

        assert memory.get("string") == "hello"
        assert memory.get("number") == 42
        assert memory.get("list") == [1, 2, 3]
        assert memory.get("dict") == {"nested": "value"}
        assert memory.get("none") is None

    def test_repr(self):
        """repr should show item count."""
        memory = InMemoryMemory()
        memory.add("a", 1)
        memory.add("b", 2)

        assert "InMemoryMemory" in repr(memory)
        assert "2" in repr(memory)
