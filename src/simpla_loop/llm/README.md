# llm — LLM integration

Optional module that wires OpenAI (via Instructor) into the framework. Requires the `[llm]` extra: `pip install simpla-loop[llm]`.

## Components

| Module | Purpose |
|--------|---------|
| `client.py` | `OpenAIConfig` (pydantic-settings) + `create_instructor_client()` factory |
| `models.py` | `ReActResponse` Pydantic model for structured LLM output; `ToolInfo` for tool schema |
| `reasoners.py` | `create_react_reasoner()` factory — returns a `Reasoner` function compatible with `ReActLoop` |

## Configuration

All settings are read from environment variables (or `.env`) via `OpenAIConfig`:

| Variable | Required | Default |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | — |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | No | `gpt-4o-mini` |
| `OPENAI_MAX_RETRIES` | No | `3` |

## Design

The reasoner is a plain function `(query, steps, tools) -> dict`, not a class. This keeps `ReActLoop` decoupled from any specific LLM backend — tests use mock reasoners with zero API calls.
