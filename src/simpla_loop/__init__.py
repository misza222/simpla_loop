"""Agentic Loops - Educational framework for agentic loop patterns.

This package provides abstract interfaces and concrete implementations
for building agentic systems with different loop strategies (ReAct, etc.),
memory backends, and tool integrations.

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
    ...     config=AgentConfig()
    ... )
    >>> result = agent.run("List files in current directory")
"""

from simpla_loop.agent import Agent, AgentConfig

__version__ = "0.1.0"
__all__ = ["Agent", "AgentConfig"]
