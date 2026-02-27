# Extended Concepts

Deep dive into agentic loop concepts and design patterns.

## The ReAct Pattern in Detail

ReAct (Reasoning + Acting) is more than just calling tools. It's about **transparent reasoning**:

### Why ReAct Works

1. **Traceability**: Every action is preceded by an explicit thought
2. **Debuggability**: You can see *why* the agent did what it did
3. **Correctability**: When wrong, you can see where reasoning failed
4. **Learning**: The thought-trace can be used for fine-tuning

### ReAct vs Chain-of-Thought

| Aspect | Chain-of-Thought | ReAct |
|--------|------------------|-------|
| Output | Just reasoning | Reasoning + Actions |
| External data | None | Tools provide data |
| Verification | Hard | Easy (check actions) |
| Hallucination | Higher risk | Lower risk (grounded) |

## State Management

### Immutable State Pattern

The framework uses immutable state objects for loop iterations:

```python
@dataclass(frozen=True)
class LoopResult(Generic[StateT]):
    state: StateT      # New state for next iteration
    done: bool         # Termination signal
    output: Any        # Final result
```

**Benefits:**
- Easy to debug (state at each step is preserved)
- Thread-safe (no shared mutable state)
- Serializable (can checkpoint/resume)

### State Evolution

```
Iteration 0: State(query="What is 2+2?", steps=[])
Iteration 1: State(query="What is 2+2?", steps=[Step1])
Iteration 2: State(query="What is 2+2?", steps=[Step1, Step2])
...
```

## Memory Patterns

### Key-Value Memory (InMemoryMemory)

Simple but flexible. Best for:
- Small contexts
- Direct lookups
- Ephemeral data

### Conversation Memory (Pattern)

```python
class ConversationMemory(Memory):
    def __init__(self):
        self.messages = []
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def get_context(self, n: int = 5) -> list:
        return self.messages[-n:]
```

### Vector/Semantic Memory (Concept)

For large knowledge bases:

```python
class VectorMemory(Memory):
    def add(self, key: str, value: str):
        embedding = embed(value)
        self.vector_store.add(key, embedding, value)
    
    def search(self, query: str, k: int = 5):
        query_emb = embed(query)
        return self.vector_store.similarity_search(query_emb, k)
```

## Tool Design Patterns

### Self-Describing Tools

Tools define their own interface:

```python
class SearchTool(Tool):
    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="Search query (be specific)"
            ),
            ToolParameter(
                name="num_results",
                type="integer",
                description="Number of results",
                required=False,
                default=5
            )
        ]
```

This enables:
- Automatic validation
- LLM-friendly descriptions
- Runtime introspection

### Composable Tools

Tools can call other tools:

```python
class ResearchTool(Tool):
    def __init__(self):
        self.search = SearchTool()
        self.browser = BrowserTool()
    
    def execute(self, topic: str):
        # Search for sources
        search_results = self.search.execute(query=topic)
        
        # Visit top results
        summaries = []
        for url in search_results.data[:3]:
            page = self.browser.execute(url=url)
            summaries.append(summarize(page.data))
        
        return ToolResult.ok(summaries)
```

## Error Handling Strategies

### Tool-Level

Tools return structured results:

```python
result = tool.execute(...)
if not result.success:
    # Handle error
    logger.error(result.error)
    memory.add("last_error", result.error)
```

### Loop-Level

Loops can retry or adapt:

```python
def step(self, state, memory, tools):
    # Try primary tool
    result = tools[0].execute(...)
    
    if not result.success:
        # Try fallback
        result = tools[1].execute(...)
    
    # ...
```

### Agent-Level

Agents can reset and retry:

```python
try:
    result = agent.run(query)
except RuntimeError as e:
    agent.reset()
    result = agent.run(query, max_iterations=20)
```

## Advanced Loop Patterns

### Plan-and-Solve

```python
@dataclass
class PlanState:
    query: str
    plan: list[str] = field(default_factory=list)
    current_step: int = 0

class PlanAndSolveLoop(Loop[PlanState]):
    def step(self, state, memory, tools):
        if not state.plan:
            # Planning phase
            state.plan = self.planner.create_plan(state.query)
            return LoopResult(state=state, done=False, output=None)
        
        if state.current_step < len(state.plan):
            # Execution phase
            step = state.plan[state.current_step]
            result = self.execute_step(step, tools)
            state.current_step += 1
            return LoopResult(state=state, done=False, output=None)
        
        # Done
        return LoopResult(state=state, done=True, output=aggregate_results())
```

### Reflexion

```python
@dataclass
class ReflexionState:
    query: str
    attempts: list[Attempt] = field(default_factory=list)

class ReflexionLoop(Loop[ReflexionState]):
    def step(self, state, memory, tools):
        # Try to solve
        attempt = self.try_solve(state.query, tools)
        
        # Evaluate
        evaluation = self.evaluate(attempt)
        
        if evaluation.is_correct:
            return LoopResult(state=state, done=True, output=attempt.answer)
        
        # Reflect and improve
        reflection = self.reflect(attempt, evaluation)
        state.attempts.append(Attempt(
            answer=attempt.answer,
            reflection=reflection
        ))
        
        if len(state.attempts) >= self.max_attempts:
            return LoopResult(state=state, done=True, output=attempt.answer)
        
        return LoopResult(state=state, done=False, output=None)
```

## Testing Strategies

### Mock Reasoners

Test loops without LLM calls:

```python
def mock_reasoner(actions: list):
    """Returns predetermined actions."""
    idx = 0
    def reasoner(query, steps, tools):
        nonlocal idx
        action = actions[idx]
        idx += 1
        return action
    return reasoner

loop = ReActLoop(reasoner=mock_reasoner([
    {"thought": "Step 1", "action": "tool1", "action_input": {}},
    {"thought": "Step 2", "final_answer": "Done"}
]))
```

### State Inspection

```python
result = loop.step(state, memory, tools)
assert result.state.current_step == 1
assert len(result.state.steps) == 1
assert result.state.steps[0].thought == "Expected thought"
```

### Integration Testing

```python
def test_full_agent_workflow():
    agent = Agent(
        loop=ReActLoop(reasoner=test_reasoner),
        memory=InMemoryMemory(),
        tools=[MockTool("mock", result="test")],
        config=AgentConfig(max_iterations=5)
    )
    
    result = agent.run("Test query")
    assert result == "expected"
    assert len(agent.memory.get("tool_results")) == 1
```

## Performance Considerations

### Memory Efficiency

- Clear memory between unrelated tasks
- Use generators for large result sets
- Consider memory limits for long-running agents

### Loop Optimization

- Set appropriate max_iterations
- Cache expensive tool results
- Parallelize independent tool calls

### Concurrency

```python
# Thread-safe memory
from threading import Lock

class ThreadSafeMemory(Memory):
    def __init__(self):
        self._store = {}
        self._lock = Lock()
    
    def add(self, key, value):
        with self._lock:
            self._store[key] = value
```

## Further Learning

- **Tree of Thoughts**: Explore multiple reasoning paths
- **ToolFormer**: LLMs that learn to use tools
- **Gorilla**: LLMs specialized for API calls
- **AutoGPT**: Autonomous agent with goal decomposition
