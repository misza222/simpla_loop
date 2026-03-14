# core — Abstract interfaces

The `core` module defines the framework's three fundamental abstractions. All concrete implementations depend on these interfaces — never the reverse.

## Interfaces

| Type | Purpose | Key methods |
|------|---------|-------------|
| `Loop[StateT]` | Iterative agent execution strategy | `create_initial_state()`, `step()`, `run()` |
| `Memory` | Key-value storage shared across iterations | `add()`, `get()`, `clear()`, `get_all()` |
| `Tool` | Self-describing external capability | `execute()`, `validate()`, plus `name`/`description`/`parameters` |

## Supporting types

- `LoopResult[StateT]` — immutable (frozen) dataclass returned by `step()`. Carries `state`, `done`, and `output`.
- `ToolParameter` — frozen dataclass describing a single tool parameter (name, type, required, default).
- `ToolResult` — result of tool execution. Use `ToolResult.ok(data)` / `ToolResult.fail(error)` factories.

## Exception hierarchy

All domain exceptions inherit from `SimpleLoopError`:

```
SimpleLoopError
├── ConfigError    — invalid/missing configuration
├── LoopError      — loop cannot continue (e.g. max iterations)
└── ToolError      — unrecoverable tool execution failure
```

## Extension points

To add a new loop, memory, or tool implementation, subclass the corresponding ABC and implement the abstract methods. The `Loop` is generic over `StateT`, so your loop can define its own state type.
