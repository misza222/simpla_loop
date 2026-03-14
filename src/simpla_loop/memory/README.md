# memory — Memory implementations

Concrete implementations of the `Memory` abstraction from `core`.

## Available implementations

### InMemoryMemory

Ephemeral, dictionary-based key-value store. Fast and simple — suitable for single-run agents where persistence is not needed.

- **Thread safety**: not thread-safe (single-agent use only).
- **Persistence**: none — data is lost when the process exits.
- **Shallow copies**: `get_all()` returns a shallow copy; nested mutable values are shared.

## Adding a new implementation

1. Create a new module (e.g. `memory/file.py`)
2. Subclass `Memory` and implement `add()`, `get()`, `list_keys()`, `clear()`, `get_all()`
3. Export from `memory/__init__.py`
