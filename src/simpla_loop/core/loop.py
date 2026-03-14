"""Abstract base class for agentic loop strategies.

A Loop defines how an agent iteratively processes information,
makes decisions, and takes actions. Different loop strategies
(ReAct, Plan-and-Solve, Reflexion, etc.) implement this interface.

The Loop pattern follows these principles:
1. Stateless design: All state is passed explicitly through LoopResult
2. Generic typing: State type is preserved for type safety
3. Composability: Loops can be wrapped, chained, or nested

Example:
    >>> class MyLoop(Loop[MyState]):
    ...     def step(self, state, memory, tools):
    ...         # Process current state
    ...         new_state = process(state)
    ...         done = check_completion(new_state)
    ...         output = extract_output(new_state) if done else None
    ...         return LoopResult(state=new_state, done=done, output=output)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from simpla_loop.core.memory import Memory
from simpla_loop.core.tool import Tool

# Type variable for loop-specific state
# StateT represents the concrete state type used by a specific Loop implementation
StateT = TypeVar("StateT")


@dataclass(frozen=True)
class LoopResult(Generic[StateT]):
    """Result of a single loop iteration.

    LoopResult is immutable to prevent accidental state mutations
    between iterations. Use dataclasses.replace() if you need to
    create a modified copy.

    Attributes:
        state: The current state of the loop after this iteration.
               This state will be passed to the next step() call.
        done: Whether the loop should terminate. When True, the run()
              method will return the output.
        output: The final output if done, else intermediate result.
                This is what gets returned to the caller.

    Example:
        >>> # Mark loop as complete
        >>> return LoopResult(state=final_state, done=True, output="answer")
        >>>
        >>> # Continue to next iteration
        >>> return LoopResult(state=next_state, done=False, output=None)
    """

    state: StateT
    done: bool
    output: Any


class Loop(ABC, Generic[StateT]):
    """Abstract base for agentic loop implementations.

    The Loop is the heart of an agent. It orchestrates the cycle of:
    1. Observing the current state/memory
    2. Deciding what to do (reasoning)
    3. Executing actions via tools
    4. Updating state/memory

    Implementations should be stateless; state is passed through LoopResult.
    This makes loops easy to test, debug, and serialize.

    Type Parameters:
        StateT: The type of state maintained by this loop.
                Could be a dataclass, dict, Pydantic model, etc.

    Attributes:
        None (implementations should not store mutable state)

    Methods:
        step: Execute one iteration (must be implemented)
        run: Convenience method to run until completion (can be overridden)

    Example:
        >>> loop = ReActLoop(reasoner=my_llm)
        >>> state = ReActState(query="What's 2+2?")
        >>> memory = InMemoryMemory()
        >>> tools = [CalculatorTool()]
        >>> result = loop.run(state, memory, tools, max_iterations=5)
    """

    @abstractmethod
    def create_initial_state(self, query: str, **kwargs: Any) -> StateT:
        """Create initial state from a query string.

        This method encapsulates state initialization, allowing the Loop
        to own its state type. The Agent uses this to create appropriate
        initial state without knowing the concrete state type.

        Args:
            query: The user query to process.
            **kwargs: Loop-specific initialization options.

        Returns:
            Initial state for the first iteration of the loop.

        Example:
            >>> state = loop.create_initial_state("What's the weather?")
            >>> state.query
            "What's the weather?"
        """
        ...

    @abstractmethod
    def step(
        self,
        state: StateT,
        memory: Memory,
        tools: list[Tool],
    ) -> LoopResult[StateT]:
        """Execute one iteration of the loop.

        This is the core method that all loop implementations must provide.
        It takes the current state and produces the next state, along with
        a signal about whether the loop is complete.

        The method should be pure in terms of the state object (don't mutate
        the input state; return a new state object or use dataclasses.replace()).
        However, it may mutate the Memory as a side effect.

        Args:
            state: Current loop state from previous iteration. For the first
                   iteration, this is the initial_state passed to run().
            memory: Shared memory for the agent. The loop may read from and
                    write to memory to maintain information across iterations.
                    Memory persists beyond the loop's lifetime.
            tools: Available tools for this step. The loop decides which tools
                   to use based on its strategy. Tools are stateless; their
                   state (if any) should be managed through memory.

        Returns:
            LoopResult containing:
            - state: The new state for the next iteration
            - done: True if the loop should stop
            - output: The final result (meaningful only if done=True)

        Raises:
            LoopError: If the step encounters an unrecoverable error.
                      Subclasses may define specific exception types.

        Example:
            >>> result = loop.step(current_state, memory, tools)
            >>> if result.done:
            ...     print(f"Complete: {result.output}")
            ... else:
            ...     current_state = result.state  # Continue
        """
        ...

    def run(
        self,
        initial_state: StateT,
        memory: Memory,
        tools: list[Tool],
        max_iterations: int = 10,
    ) -> Any:
        """Run the loop until completion or max iterations.

        This is a convenience method that repeatedly calls step() until
        either the loop signals completion (done=True) or the maximum
        number of iterations is reached.

        You can call step() directly if you need more control over the
        execution flow (e.g., to add logging, breakpoints, or conditional
        logic between iterations).

        Args:
            initial_state: Starting state for the loop. This is passed to
                          step() as the first state argument.
            memory: Shared memory for the agent. Passed to every step().
            tools: Available tools. Passed to every step().
            max_iterations: Safety limit to prevent infinite loops.
                           If reached, RuntimeError is raised.
                           Adjust based on your use case.

        Returns:
            The final output from the loop (result.output when done=True).
            The type depends on the specific loop implementation.

        Raises:
            RuntimeError: If max_iterations is reached without completion.
                         The error message includes the final state for debugging.
            Exception: Any exception raised by step() will propagate.

        Example:
            >>> try:
            ...     answer = loop.run(initial_state, memory, tools, max_iterations=20)
            ... except RuntimeError as e:
            ...     print(f"Loop timed out: {e}")
        """
        state = initial_state
        for _ in range(max_iterations):
            result = self.step(state, memory, tools)
            state = result.state
            if result.done:
                return result.output

        raise RuntimeError(
            f"Loop did not complete within {max_iterations} iterations. "
            f"Final state: {state}"
        )
