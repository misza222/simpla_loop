# tools — Tool implementations

Concrete implementations of the `Tool` abstraction from `core`. Tools are the agent's interface to the external world.

## Available tools

| Tool | Module | Purpose |
|------|--------|---------|
| `BashTool` | `bash.py` | Execute shell commands via subprocess |
| `CalculatorTool` | `calculator.py` | Evaluate arithmetic expressions safely |

## Adding a new tool

1. Create a new module (e.g. `tools/http.py`)
2. Subclass `Tool` and implement:
   - `name` (property) — unique identifier
   - `description` (property) — what the tool does (shown to LLMs)
   - `parameters` (property) — list of `ToolParameter` describing inputs
   - `execute(**kwargs)` — perform the action, return `ToolResult.ok()` or `ToolResult.fail()`
3. Export from `tools/__init__.py`

Tools are stateless and self-describing — each tool defines its own parameter schema so that LLMs can introspect available capabilities at runtime.
