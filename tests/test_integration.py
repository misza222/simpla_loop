"""End-to-end integration tests for the Agent pipeline."""

from conftest import MockTool

from simpla_loop.agent import Agent
from simpla_loop.loops.react import ReActLoop
from simpla_loop.memory.in_memory import InMemoryMemory


class TestAgentEndToEnd:
    """Integration tests exercising the full Agent → Loop → Tool pipeline."""

    def test_multi_step_task(self):
        """Agent should execute a tool then return a final answer."""
        call_count = 0

        def reasoner(query, steps, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "thought": "I need to use the tool first",
                    "action": "echo",
                    "action_input": {},
                }
            return {"thought": "Got the result, done", "final_answer": "all done"}

        tool = MockTool("echo", result="echoed")
        memory = InMemoryMemory()
        loop = ReActLoop(reasoner=reasoner)
        agent = Agent(loop=loop, memory=memory, tools=[tool])

        result = agent.run("do the thing")

        assert result == "all done"
        assert memory.get("query") == "do the thing"
        # tool_results stored by ReActLoop
        tool_results = memory.get("tool_results")
        assert tool_results is not None
        assert len(tool_results) == 1
        assert tool_results[0]["tool"] == "echo"

    def test_handles_unknown_tool_gracefully(self):
        """Agent should survive when reasoner requests a missing tool."""
        call_count = 0

        def reasoner(query, steps, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "thought": "Try a tool that doesn't exist",
                    "action": "nonexistent",
                    "action_input": {},
                }
            # Recover on next step
            return {
                "thought": "That failed, answering directly",
                "final_answer": "recovered",
            }

        memory = InMemoryMemory()
        loop = ReActLoop(reasoner=reasoner)
        agent = Agent(loop=loop, memory=memory, tools=[])

        result = agent.run("tricky query")

        assert result == "recovered"

    def test_reset_between_runs(self):
        """After reset(), memory from a previous run should be gone."""

        def reasoner(query, steps, tools):
            return {"thought": "Answering", "final_answer": query.upper()}

        memory = InMemoryMemory()
        loop = ReActLoop(reasoner=reasoner)
        agent = Agent(loop=loop, memory=memory, tools=[])

        result_1 = agent.run("first")
        assert result_1 == "FIRST"
        assert memory.get("query") == "first"

        agent.reset()
        assert memory.get("query") is None

        result_2 = agent.run("second")
        assert result_2 == "SECOND"
        assert memory.get("query") == "second"
