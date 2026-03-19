"""Reporter protocol for observing agent execution.

Reporters receive callbacks during agent.run() and can be used to
visualize, log, or record agent behavior without modifying the agent
or loop internals.

Example:
    >>> class PrintReporter:
    ...     def on_run_start(self, tools: list[Tool]) -> None:
    ...         print(f"Starting with {len(tools)} tools")
    ...     def on_step(self, result: LoopResult[Any]) -> None:
    ...         print(f"Step done, done={result.done}")
    ...     def on_run_done(self) -> None:
    ...         print("Run complete")
"""

from typing import Any, Protocol

from simpla_loop.core.loop import LoopResult
from simpla_loop.core.tool import Tool


class StepReporter(Protocol):
    """Protocol for observing agent step execution.

    Implement this to hook into agent.run() without modifying Agent or
    Loop. Structural typing means no explicit inheritance is needed — any
    class with these three methods satisfies the protocol.

    Callbacks are invoked in order:
        on_run_start → [on_step × N] → on_run_done

    Args are intentionally broad (Any state) so reporters work with any
    Loop implementation, not just ReActLoop.
    """

    def on_run_start(self, tools: list[Tool]) -> None:
        """Called once before the first loop step.

        Args:
            tools: The full list of tools available to the agent.
        """
        ...

    def on_step(self, result: LoopResult[Any]) -> None:
        """Called after each loop step completes.

        Args:
            result: The LoopResult from this step. Inspect result.state
                    for loop-specific data (e.g. ReActState.steps).
                    result.done is True on the final step.
        """
        ...

    def on_run_done(self) -> None:
        """Called once after the loop finishes (success or max-iterations)."""
        ...
