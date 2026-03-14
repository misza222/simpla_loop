"""ReAct (Reasoning + Acting) loop implementation.

ReAct is a paradigm where the agent interleaves:
1. Thought: Reasoning about what to do
2. Action: Executing a tool
3. Observation: Receiving the result

This creates a traceable chain of reasoning and action.

Reference:
    Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models"
    https://arxiv.org/abs/2210.03629

This implementation is LLM-agnostic; you provide the reasoning function.
This makes it easy to test with mock reasoners or use different LLM backends.

Example:
    >>> def my_reasoner(query, steps, tools):
    ...     # Your LLM call here
    ...     return {
    ...         "thought": "I need to calculate this",
    ...         "action": "calculator",
    ...         "action_input": {"expression": "2+2"}
    ...     }

    >>> loop = ReActLoop(reasoner=my_reasoner)
    >>> state = ReActState(query="What is 2+2?")
    >>> result = loop.run(state, memory, [calculator_tool])
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from simpla_loop.core.loop import Loop, LoopResult
from simpla_loop.core.memory import Memory
from simpla_loop.core.tool import Tool, ToolResult


@dataclass
class ReActStep:
    """Single step in a ReAct trace.

    Each step captures one iteration of the ReAct pattern:
    Thought -> Action -> Observation (or Final Answer)

    The trace of steps forms a complete record of the agent's
    reasoning and actions, making it transparent and debuggable.

    Attributes:
        thought: The agent's reasoning about what to do next.
                 This explains WHY the agent is taking this action.
        action: The name of the tool to execute, or None if this
                is a final answer step.
        action_input: Dictionary of arguments passed to the tool.
                      Matches the tool's parameter schema.
        observation: The result from tool execution (ToolResult),
                     or None for final answer steps.
        is_final: True if this step provides the final answer
                  instead of calling a tool.

    Example:
        >>> step = ReActStep(
        ...     thought="I need to search for information",
        ...     action="web_search",
        ...     action_input={"query": "Python programming"},
        ...     observation=ToolResult.ok({"results": [...]}),
        ...     is_final=False
        ... )
    """

    thought: str
    action: str | None = None
    action_input: dict | None = None
    observation: Any = None
    is_final: bool = False


@dataclass
class ReActState:
    """State maintained across ReAct iterations.

    ReActState tracks the progress of a ReAct loop execution.
    It's passed between step() calls and accumulates the trace
    of thoughts and actions.

    Attributes:
        query: The original task or question. Never changes during
               execution; serves as the anchor for all reasoning.
        steps: List of ReActStep objects recording the execution trace.
               Grows by one element each iteration.
        current_step: Counter for the current iteration (0-indexed).
                      Used to detect max_steps violation.
        max_steps: Safety limit to prevent infinite loops.
                   Default is 5, but adjust based on task complexity.

    Properties:
        last_thought: Convenience accessor for the most recent thought.
                     Returns None if no steps yet.

    Example:
        >>> state = ReActState(query="What is the capital of France?")
        >>> state.max_steps = 3  # Limit iterations
        >>>
        >>> # After one iteration
        >>> state.steps.append(ReActStep(thought="I need to search...", ...))
        >>> state.current_step = 1
        >>>
        >>> print(state.last_thought)
        'I need to search...'
    """

    query: str
    steps: list[ReActStep] = field(default_factory=list)
    current_step: int = 0
    max_steps: int = 5

    @property
    def last_thought(self) -> str | None:
        """Get the most recent thought.

        Returns:
            The thought from the most recent step, or None if
            no steps have been recorded yet.

        Example:
            >>> state = ReActState(query="test")
            >>> state.last_thought is None
            True
            >>> state.steps.append(ReActStep(thought="First thought"))
            >>> state.last_thought
            'First thought'
        """
        if self.steps:
            return self.steps[-1].thought
        return None


# Type alias for the reasoning function
# This is the hook for LLM integration
Reasoner = Callable[[str, list[ReActStep], list[Tool]], dict]
"""Type alias for the reasoning function.

A Reasoner takes:
- query: The original task/question
- steps: History of previous steps (for context)
- tools: Available tools (for deciding what to use)

And returns a dict with:
- thought: str - reasoning about what to do
- action: str | None - tool name to use (None if final answer)
- action_input: dict - arguments for the tool
- final_answer: Any | None - answer if done

Example:
    >>> def my_reasoner(query, steps, tools):
    ...     # Call your LLM here with the context
    ...     response = llm.complete(build_prompt(query, steps, tools))
    ...     return parse_response(response)
"""


class ReActLoop(Loop[ReActState]):
    """ReAct loop implementation.

    The ReAct pattern interleaves reasoning (Thought) with actions (Action):

    ```
    Thought: I need to find the current weather
    Action: search_tool(query="weather today")
    Observation: {"temperature": 72, "condition": "sunny"}

    Thought: Now I have the weather information
    Action: None (final)
    Final Answer: It's 72 degrees and sunny today
    ```

    You must provide a `reasoner` function that implements the actual
    reasoning (typically an LLM call). This keeps the loop agnostic
    to the specific LLM backend.

    Attributes:
        reasoner: The reasoning function (see Reasoner type alias).
        tool_result_key: Memory key for storing tool results.

    Methods:
        step: Execute one ReAct iteration (thought + action + observation)

    Example:
        >>> def llm_reasoner(query, steps, tools):
        ...     prompt = build_react_prompt(query, steps, tools)
        ...     response = openai.chat.completions.create(...)
        ...     return parse_react_response(response)

        >>> loop = ReActLoop(reasoner=llm_reasoner)
        >>> state = ReActState(query="What's 2+2?")
        >>> memory = InMemoryMemory()
        >>> tools = [CalculatorTool()]
        >>>
        >>> result = loop.run(state, memory, tools)
        >>> print(result)
        '4'
    """

    def __init__(
        self,
        reasoner: Reasoner,
        tool_result_key: str = "tool_results",
    ) -> None:
        """Initialize ReAct loop.

        Args:
            reasoner: Function that generates thoughts and decides actions.
                     This is where your LLM logic goes. See Reasoner type
                     alias for the expected signature.
            tool_result_key: Memory key for storing tool results.
                            Tool results are stored in memory as a list
                            under this key, allowing other parts of the
                            system to access them.

        Returns:
            None

        Example:
            >>> def my_reasoner(query, steps, tools):
            ...     return {
            ...         "thought": "I should search",
            ...         "action": "search",
            ...         "action_input": {"q": query}
            ...     }

            >>> loop = ReActLoop(
            ...     reasoner=my_reasoner,
            ...     tool_result_key="my_tool_results"
            ... )
        """
        self.reasoner = reasoner
        self.tool_result_key = tool_result_key

    def create_initial_state(self, query: str, **kwargs: Any) -> ReActState:
        """Create initial ReActState from a query.

        Args:
            query: The user query to process.
            **kwargs: Optional initialization parameters:
                     - max_steps: Maximum iterations (default: 5)

        Returns:
            ReActState initialized with the query.

        Example:
            >>> loop = ReActLoop(reasoner=my_reasoner)
            >>> state = loop.create_initial_state("What's 2+2?")
            >>> state.query
            "What's 2+2?"
            >>> state.max_steps
            5
        """
        max_steps = kwargs.get("max_steps", 5)
        return ReActState(query=query, max_steps=max_steps)

    def step(
        self,
        state: ReActState,
        memory: Memory,
        tools: list[Tool],
    ) -> LoopResult[ReActState]:
        """Execute one ReAct iteration.

        One iteration consists of:
        1. Call the reasoner with current context (query, history, available tools)
        2. Record the thought and planned action
        3. If final answer provided → return done=True
        4. If action specified → execute the tool
        5. Record the observation
        6. Check max_steps limit
        7. Return state for next iteration (done=False)

        Args:
            state: Current ReActState with query and step history.
                   This is NOT mutated; a new state is returned.
            memory: Shared memory for storing tool results and other data.
                   May be read/written during execution.
            tools: Available tools for this step. The reasoner decides
                  which (if any) to use.

        Returns:
            LoopResult with:
            - state: Updated ReActState (new steps appended)
            - done: True if final answer reached or max_steps hit
            - output: Final answer (if done), None otherwise

        Example:
            >>> state = ReActState(query="test")
            >>> result = loop.step(state, memory, tools)
            >>> result.done
            False
            >>> result.state.current_step
            1
            >>> result.state.steps[0].thought
            "I need to..."
        """
        # Build tool map for O(1) lookup by name
        tool_map = {t.name: t for t in tools}

        # Generate thought and decide action using the reasoner
        # This is where the LLM (or mock) is called
        reasoning = self.reasoner(state.query, state.steps, tools)

        thought = reasoning.get("thought", "")
        action = reasoning.get("action")
        action_input = reasoning.get("action_input", {})
        final_answer = reasoning.get("final_answer")

        # Create step record capturing this iteration
        step = ReActStep(
            thought=thought,
            action=action,
            action_input=action_input,
        )

        # Check for final answer - if provided, we're done
        if final_answer is not None:
            step.is_final = True
            state.steps.append(step)
            return LoopResult(
                state=state,
                done=True,
                output=final_answer,
            )

        # Execute tool action if specified
        if action:
            if action not in tool_map:
                # Tool not found - record as failed observation
                step.observation = ToolResult.fail(f"Unknown tool: {action}")
            else:
                # Execute the tool
                tool = tool_map[action]
                tool_result = tool.execute(**action_input)
                step.observation = tool_result

                # Store in memory for potential future reference
                # This allows other components to access tool history
                results = memory.get(self.tool_result_key) or []
                results.append(
                    {
                        "step": state.current_step,
                        "tool": action,
                        "input": action_input,
                        "result": tool_result,
                    }
                )
                memory.add(self.tool_result_key, results)

        # Append step to state history
        state.steps.append(step)
        state.current_step += 1

        # Check max steps limit
        if state.current_step >= state.max_steps:
            return LoopResult(
                state=state,
                done=True,
                output={
                    "error": "Max steps reached without final answer",
                    "trace": state.steps,
                    "query": state.query,
                },
            )

        # Not done yet - return for next iteration
        return LoopResult(
            state=state,
            done=False,
            output=None,
        )
