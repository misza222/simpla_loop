# loops — Loop implementations

Concrete implementations of the `Loop[StateT]` abstraction from `core`.

## Available loops

### ReActLoop

Implements the ReAct (Reasoning + Acting) pattern from [Yao et al. 2022](https://arxiv.org/abs/2210.03629). Each iteration interleaves:

1. **Thought** — the reasoner explains its plan
2. **Action** — a tool is invoked (or a final answer is given)
3. **Observation** — the tool result is recorded

### State types

- `ReActState` — tracks query, step history, current step counter, and max steps limit.
- `ReActStep` — a single thought/action/observation record.

### Reasoner injection

`ReActLoop` receives a `Reasoner` function (not a class) via its constructor. This makes it LLM-agnostic — swap in a mock function for tests or a real LLM call for production.

## Adding a new loop

1. Create a new module (e.g. `loops/chain_of_thought.py`)
2. Define a state dataclass for your loop
3. Subclass `Loop[YourState]` and implement `create_initial_state()` and `step()`
4. Export from `loops/__init__.py`
