"""Tests for the Agent class — the public API entry point."""

import pytest
from conftest import MockTool

from simpla_loop.agent import Agent, AgentConfig
from simpla_loop.core.exceptions import LoopError
from simpla_loop.loops.react import ReActLoop
from simpla_loop.memory.in_memory import InMemoryMemory

# ---------------------------------------------------------------------------
# Helper reasoners
# ---------------------------------------------------------------------------


def _immediate_answer_reasoner(query, steps, tools):
    """Reasoner that immediately returns a final answer."""
    return {"thought": "I know the answer", "final_answer": "42"}


def _never_finish_reasoner(query, steps, tools):
    """Reasoner that never provides a final answer."""
    return {
        "thought": "Still thinking",
        "action": "mock_tool",
        "action_input": {},
    }


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_values(self):
        """AgentConfig should default to max_iterations=10 and debug=False."""
        config = AgentConfig()
        assert config.max_iterations == 10
        assert config.debug is False

    def test_custom_values(self):
        """Explicit values should be stored correctly."""
        config = AgentConfig(max_iterations=20, debug=True)
        assert config.max_iterations == 20
        assert config.debug is True


class TestAgent:
    """Tests for Agent orchestration."""

    def test_init_with_defaults(self):
        """Agent without explicit config should use default AgentConfig."""
        loop = ReActLoop(reasoner=_immediate_answer_reasoner)
        memory = InMemoryMemory()
        agent = Agent(loop=loop, memory=memory, tools=[])

        assert agent.config.max_iterations == 10
        assert agent.config.debug is False

    def test_init_with_config(self):
        """Explicit config should be stored on the agent."""
        config = AgentConfig(max_iterations=5, debug=True)
        loop = ReActLoop(reasoner=_immediate_answer_reasoner)
        memory = InMemoryMemory()
        agent = Agent(loop=loop, memory=memory, tools=[], config=config)

        assert agent.config is config

    def test_run_stores_query_in_memory(self):
        """run() should store the query under the 'query' key in memory."""
        loop = ReActLoop(reasoner=_immediate_answer_reasoner)
        memory = InMemoryMemory()
        agent = Agent(loop=loop, memory=memory, tools=[])

        agent.run("hello")

        assert memory.get("query") == "hello"

    def test_run_returns_loop_output(self):
        """run() should return the final answer from the loop."""
        loop = ReActLoop(reasoner=_immediate_answer_reasoner)
        memory = InMemoryMemory()
        agent = Agent(loop=loop, memory=memory, tools=[])

        result = agent.run("What is the answer?")

        assert result == "42"

    def test_run_raises_loop_error_on_max_iterations(self):
        """run() should raise LoopError when max iterations are exhausted."""
        loop = ReActLoop(reasoner=_never_finish_reasoner)
        memory = InMemoryMemory()
        tool = MockTool("mock_tool")
        config = AgentConfig(max_iterations=2)
        agent = Agent(loop=loop, memory=memory, tools=[tool], config=config)

        with pytest.raises(LoopError, match="did not complete"):
            agent.run("infinite task")

    def test_get_trace_before_run(self):
        """get_trace() should return None before any run."""
        loop = ReActLoop(reasoner=_immediate_answer_reasoner)
        memory = InMemoryMemory()
        agent = Agent(loop=loop, memory=memory, tools=[])

        assert agent.get_trace() is None

    def test_reset_clears_memory_and_trace(self):
        """reset() should clear memory and set trace to None."""
        loop = ReActLoop(reasoner=_immediate_answer_reasoner)
        memory = InMemoryMemory()
        agent = Agent(loop=loop, memory=memory, tools=[])

        agent.run("hello")
        assert memory.get("query") is not None

        agent.reset()

        assert memory.get("query") is None
        assert agent.get_trace() is None

    def test_debug_mode_does_not_raise(self):
        """Agent with debug=True should complete without error."""
        config = AgentConfig(debug=True)
        loop = ReActLoop(reasoner=_immediate_answer_reasoner)
        memory = InMemoryMemory()
        agent = Agent(loop=loop, memory=memory, tools=[], config=config)

        result = agent.run("smoke test")
        assert result == "42"
