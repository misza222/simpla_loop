"""Tests for ReAct loop implementation."""

import pytest
from conftest import MockTool

from simpla_loop.core.exceptions import LoopError
from simpla_loop.loops.react import ReActLoop, ReActState, ReActStep
from simpla_loop.memory.in_memory import InMemoryMemory


class TestReActState:
    """Test suite for ReActState."""

    def test_default_values(self):
        """State should have sensible defaults."""
        state = ReActState(query="test")

        assert state.query == "test"
        assert state.steps == []
        assert state.current_step == 0
        assert state.max_steps == 5

    def test_last_thought_empty(self):
        """last_thought should be None when no steps."""
        state = ReActState(query="test")
        assert state.last_thought is None

    def test_last_thought_with_steps(self):
        """last_thought should return most recent thought."""
        state = ReActState(query="test")
        state.steps.append(ReActStep(thought="first"))
        state.steps.append(ReActStep(thought="second"))

        assert state.last_thought == "second"


class TestReActStep:
    """Test suite for ReActStep."""

    def test_defaults(self):
        """Step should have sensible defaults."""
        step = ReActStep(thought="test")

        assert step.thought == "test"
        assert step.action is None
        assert step.action_input is None
        assert step.observation is None
        assert step.is_final is False


class TestReActLoop:
    """Test suite for ReActLoop."""

    def test_init(self):
        """Should initialize with reasoner."""

        def reasoner(q, s, t):
            return {"thought": "test"}

        loop = ReActLoop(reasoner=reasoner)
        assert loop.reasoner is reasoner
        assert loop.tool_result_key == "tool_results"

    def test_step_final_answer(self):
        """Step should handle final answer."""

        def reasoner(query, steps, tools):
            return {"thought": "I'm done", "final_answer": "42"}

        loop = ReActLoop(reasoner=reasoner)
        state = ReActState(query="What is 2+2?")
        memory = InMemoryMemory()

        result = loop.step(state, memory, [])

        assert result.done is True
        assert result.output == "42"
        assert len(result.state.steps) == 1
        assert result.state.steps[0].is_final is True

    def test_step_tool_execution(self):
        """Step should execute tool when action specified."""

        def reasoner(query, steps, tools):
            if not steps:
                return {
                    "thought": "I need to use the tool",
                    "action": "mock_tool",
                    "action_input": {"param": "value"},
                }
            return {"thought": "Done", "final_answer": "result"}

        loop = ReActLoop(reasoner=reasoner)
        state = ReActState(query="test")
        memory = InMemoryMemory()
        tool = MockTool("mock_tool", result="tool_output")

        result = loop.step(state, memory, [tool])

        assert result.done is False
        assert len(result.state.steps) == 1
        assert result.state.steps[0].action == "mock_tool"
        assert result.state.steps[0].observation.success is True
        assert result.state.current_step == 1

    def test_step_unknown_tool(self):
        """Step should handle unknown tool gracefully."""

        def reasoner(query, steps, tools):
            return {
                "thought": "I'll try a tool",
                "action": "nonexistent_tool",
                "action_input": {},
            }

        loop = ReActLoop(reasoner=reasoner)
        state = ReActState(query="test")
        memory = InMemoryMemory()

        result = loop.step(state, memory, [])

        assert result.done is False
        assert result.state.steps[0].observation.success is False
        assert "Unknown tool" in result.state.steps[0].observation.error

    def test_step_stores_results_in_memory(self):
        """Step should store tool results in memory."""

        def reasoner(query, steps, tools):
            return {"thought": "Use tool", "action": "mock_tool", "action_input": {}}

        loop = ReActLoop(reasoner=reasoner, tool_result_key="my_results")
        state = ReActState(query="test")
        memory = InMemoryMemory()
        tool = MockTool("mock_tool")

        loop.step(state, memory, [tool])

        stored = memory.get("my_results")
        assert stored is not None
        assert len(stored) == 1
        assert stored[0]["tool"] == "mock_tool"

    def test_run_until_completion(self):
        """run() should iterate until final answer."""
        call_count = 0

        def reasoner(query, steps, tools):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return {
                    "thought": "Use tool",
                    "action": "mock_tool",
                    "action_input": {},
                }
            return {"thought": "Done", "final_answer": "final_result"}

        loop = ReActLoop(reasoner=reasoner)
        state = ReActState(query="test")
        memory = InMemoryMemory()
        tool = MockTool("mock_tool")

        result = loop.run(state, memory, [tool], max_iterations=10)

        assert result == "final_result"
        assert call_count == 2

    def test_run_max_iterations_reached(self):
        """run() should raise LoopError when max iterations reached."""

        def reasoner(query, steps, tools):
            return {
                "thought": "Still working",
                "action": "mock_tool",
                "action_input": {},
            }

        loop = ReActLoop(reasoner=reasoner)
        state = ReActState(query="test")
        memory = InMemoryMemory()
        tool = MockTool("mock_tool")

        with pytest.raises(LoopError, match="did not complete"):
            loop.run(state, memory, [tool], max_iterations=3)

    def test_max_steps_limit(self):
        """State max_steps should trigger completion."""

        def reasoner(query, steps, tools):
            return {"thought": "Keep going", "action": "mock_tool", "action_input": {}}

        loop = ReActLoop(reasoner=reasoner)
        state = ReActState(query="test", max_steps=2)
        memory = InMemoryMemory()
        tool = MockTool("mock_tool")

        result = loop.run(state, memory, [tool], max_iterations=10)

        assert "error" in result
        assert "Max steps reached" in result["error"]


class TestReActLoopCreateInitialState:
    """Tests for ReActLoop.create_initial_state()."""

    def test_creates_state_with_query(self):
        """Initial state should contain the provided query."""
        loop = ReActLoop(reasoner=lambda q, s, t: {})
        state = loop.create_initial_state("hello")
        assert state.query == "hello"

    def test_default_max_steps(self):
        """Initial state should default to max_steps=5."""
        loop = ReActLoop(reasoner=lambda q, s, t: {})
        state = loop.create_initial_state("test")
        assert state.max_steps == 5

    def test_custom_max_steps_via_kwargs(self):
        """max_steps can be overridden via kwargs."""
        loop = ReActLoop(reasoner=lambda q, s, t: {})
        state = loop.create_initial_state("test", max_steps=10)
        assert state.max_steps == 10

    def test_initial_state_empty(self):
        """Initial state should have no steps and current_step=0."""
        loop = ReActLoop(reasoner=lambda q, s, t: {})
        state = loop.create_initial_state("test")
        assert state.steps == []
        assert state.current_step == 0
