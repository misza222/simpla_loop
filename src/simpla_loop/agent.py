"""Single agent orchestrator.

The Agent is the high-level interface that users interact with.
It combines a Loop, Memory, and Tools into a cohesive unit.

This module provides a simplified API that handles the boilerplate
of setting up and running an agentic loop.

Example:
    >>> from simpla_loop import Agent, AgentConfig
    >>> from simpla_loop.loops.react import ReActLoop
    >>> from simpla_loop.memory.in_memory import InMemoryMemory
    >>> from simpla_loop.tools.bash import BashTool
    >>>
    >>> agent = Agent(
    ...     loop=ReActLoop(reasoner=my_reasoner),
    ...     memory=InMemoryMemory(),
    ...     tools=[BashTool()],
    ...     config=AgentConfig(debug=True)
    ... )
    >>> result = agent.run("List files in current directory")
    >>> print(result)
"""

from dataclasses import dataclass
from typing import Any

import structlog

from simpla_loop.core.exceptions import LoopError
from simpla_loop.core.loop import Loop
from simpla_loop.core.memory import Memory
from simpla_loop.core.reporter import StepReporter
from simpla_loop.core.tool import Tool


@dataclass
class AgentConfig:
    """Configuration for an Agent.

    AgentConfig allows customization of agent behavior without
    changing the agent implementation.

    Attributes:
        max_iterations: Maximum number of loop iterations before
                       forcing termination. Prevents infinite loops.
                       Adjust based on expected task complexity.
        debug: If True, prints diagnostic information during execution.
               Useful for development and troubleshooting.

    Example:
        >>> config = AgentConfig(max_iterations=20, debug=True)
        >>> agent = Agent(loop, memory, tools, config)
    """

    max_iterations: int = 10
    debug: bool = False


class Agent:
    """Single agent with loop, memory, and tools.

    The Agent is the user-facing API. It manages:
    - Initializing loop state appropriately for the loop type
    - Running the loop to completion
    - Providing access to results and execution trace
    - Resetting state between runs

    The Agent itself is stateless except for the memory (which persists)
    and the last trace (for debugging). The loop state is recreated
    for each run() call.

    Attributes:
        loop: The loop strategy (ReAct, Plan-and-Solve, etc.)
        memory: Memory implementation (persistent across runs)
        tools: List of available tools
        config: Configuration options

    Example:
        >>> # Setup
        >>> memory = InMemoryMemory()
        >>> tools = [BashTool(), CalculatorTool()]
        >>> loop = ReActLoop(reasoner=llm_reasoner)
        >>> agent = Agent(loop, memory, tools, AgentConfig(debug=True))
        >>>
        >>> # Run
        >>> result = agent.run("Calculate 2 + 2 and create a file with the result")
        >>> print(result)
        '4'
        >>>
        >>> # Inspect what happened
        >>> trace = agent.get_trace()
        >>> agent.reset()  # Clear memory for next task
    """

    def __init__(
        self,
        loop: Loop[Any],
        memory: Memory,
        tools: list[Tool],
        config: AgentConfig | None = None,
        reporter: StepReporter | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            loop: The loop strategy (ReAct, etc.). Determines how the
                  agent iteratively processes the task.
            memory: Memory implementation. Persists across multiple run()
                    calls unless clear() is called.
            tools: List of available tools. The loop decides which to use.
            config: Optional configuration. Uses defaults if not provided.
            reporter: Optional step reporter for visualization or logging.
                      Called on_run_start, after each on_step, and on_run_done.

        Returns:
            None

        Example:
            >>> agent = Agent(
            ...     loop=ReActLoop(reasoner=my_llm),
            ...     memory=InMemoryMemory(),
            ...     tools=[BashTool()],
            ...     config=AgentConfig(max_iterations=5)
            ... )
        """
        self.loop = loop
        self.memory = memory
        self.tools = tools
        self.config = config or AgentConfig()
        self._reporter = reporter
        self._logger = structlog.get_logger()

        # Store last execution trace for inspection
        self._last_trace: Any = None

    def run(self, query: str, **loop_kwargs: Any) -> Any:
        """Run the agent on a query.

        This is the main entry point for using an agent. It:
        1. Stores the query in memory
        2. Creates an appropriate initial state for the loop
        3. Runs the loop until completion or max iterations
        4. Stores the trace for later inspection
        5. Returns the final result

        Args:
            query: The task or question for the agent. This is stored in
                   memory under the key "query" and used to initialize
                   the loop state.
            **loop_kwargs: Additional arguments passed to loop.run().
                          Can include 'initial_state' to override default
                          state creation, or loop-specific options.

        Returns:
            The final output from the loop. Type depends on the loop
            implementation. For ReActLoop, this is typically the
            final_answer from the reasoner.

        Raises:
            LoopError: If max_iterations is reached (from loop.run()).
            Exception: Any exception from the loop or tools propagates.

        Example:
            >>> result = agent.run("What files are in /tmp?")
            >>> print(f"Agent answered: {result}")
            >>>
            >>> # With custom options
            >>> result = agent.run("task", max_steps=3)
        """
        # Store query in memory for reference during execution
        self.memory.add("query", query)

        # Let the loop create its own initial state
        initial_state = self.loop.create_initial_state(query, **loop_kwargs)

        if self._reporter is not None:
            self._reporter.on_run_start(self.tools)

        # Drive the loop step-by-step so the reporter fires after each iteration.
        # (Loop.run() is a convenience wrapper around step(); we replicate it here
        # to gain per-step visibility without modifying the Loop abstraction.)
        state = initial_state
        final_output: Any = None
        for _ in range(self.config.max_iterations):
            step_result = self.loop.step(state, self.memory, self.tools)
            state = step_result.state
            if self._reporter is not None:
                self._reporter.on_step(step_result)
            if step_result.done:
                final_output = step_result.output
                break
        else:
            raise LoopError(
                f"Loop did not complete within "
                f"{self.config.max_iterations} iterations. "
                f"Final state: {state}"
            )

        if self._reporter is not None:
            self._reporter.on_run_done()

        # Store trace for debugging (loop may store trace in different ways)
        self._last_trace = getattr(self.loop, "_last_trace", None)

        if self.config.debug:
            self._logger.debug("agent_completed", memory=self.memory.get_all())

        return final_output

    def get_trace(self) -> Any:
        """Get the execution trace from the last run.

        The trace contains information about what happened during
        execution: thoughts, actions, observations, etc.
        Useful for debugging and understanding agent behavior.

        Returns:
            The execution trace. Format depends on the loop implementation.
            For ReActLoop, this is the list of ReActStep objects.
            Returns None if run() hasn't been called yet.

        Example:
            >>> agent.run("What is 2+2?")
            >>> trace = agent.get_trace()
            >>> for step in trace:
            ...     print(f"Thought: {step.thought}")
            ...     print(f"Action: {step.action}")
            ...     print(f"Observation: {step.observation}")
        """
        return self._last_trace

    def reset(self) -> None:
        """Clear memory and reset state.

        This prepares the agent for a new, unrelated task.
        All memory entries are removed and the trace is cleared.

        Returns:
            None

        Example:
            >>> agent.run("Task 1")
            >>> agent.reset()
            >>> agent.run("Task 2")  # Fresh start, no memory from Task 1
        """
        self.memory.clear()
        self._last_trace = None
