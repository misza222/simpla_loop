# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_react_loop.py -v

# Format code
ruff format .

# Lint and auto-fix
ruff check . --fix

# Type checking
mypy src/simpla_loop
```

## Architecture

**simpla_loop** is an educational Python framework for agentic loop patterns. It is built around three abstract interfaces that compose into an `Agent`:

- **`core/loop.py`** — `Loop[StateT]` abstract base; subclasses implement `step()`. `LoopResult` is a frozen dataclass (immutable state each iteration).
- **`core/memory.py`** — `Memory` abstract key-value interface. `InMemoryMemory` is the only implementation (ephemeral, not thread-safe).
- **`core/tool.py`** — `Tool` abstract base with `ToolParameter` schema and `ToolResult` (use `ToolResult.ok()` / `ToolResult.fail()`).

**High-level API**: `agent.py` composes a `Loop`, `Memory`, and list of `Tool`s. `Agent.run(query)` is the entry point; `Agent.get_trace()` exposes step-by-step execution history.

**Only loop implementation**: `loops/react.py` implements the ReAct (Reasoning + Acting) pattern. The loop receives a `Reasoner` function (dependency-injected, not instantiated internally), making it LLM-agnostic and easy to test with mock reasoners.

**LLM integration** (optional, requires `instructor` + `openai`):
- `llm/client.py` — reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_MAX_RETRIES` from environment.
- `llm/models.py` — `ReActResponse` Pydantic model for structured LLM outputs.
- `llm/reasoners.py` — `create_react_reasoner()` factory that returns a reasoner function compatible with `ReActLoop`.

**Only tool implementation**: `tools/bash.py` — executes shell commands via subprocess. Commands run with process permissions; suitable only for trusted environments.

## Environment Setup

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` to run the LLM-powered examples in `examples/`.

## Key Design Decisions

- **Immutable state** per iteration (frozen dataclasses) — aids debugging and serialization.
- **Reasoner as injected function** — decouples loop logic from LLM; tests use mock reasoners, no API calls required.
- **Generic typing** on `Loop[StateT]` preserves state types through the call chain.
- **Self-describing tools** — each `Tool` defines its own name, description, and parameter schema; the LLM introspects these at runtime.
