"""Example: Debugging agent execution.

This example shows how to inspect the agent's execution trace
to understand its reasoning and debug issues.

Usage:
    python examples/debugging_trace.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

from simpla_loop import Agent, AgentConfig
from simpla_loop.llm import create_react_reasoner
from simpla_loop.loops.react import ReActLoop
from simpla_loop.memory.in_memory import InMemoryMemory
from simpla_loop.tools.bash import BashTool


def print_trace(trace):
    """Pretty print the execution trace."""
    if not trace:
        print("No trace available")
        return

    print("\n" + "=" * 70)
    print("EXECUTION TRACE")
    print("=" * 70)

    for i, step in enumerate(trace, 1):
        print(f"\n📍 Step {i}")
        print(f"   Thought: {step.thought}")

        if step.action:
            print(f"   🛠️  Action: {step.action}")
            print(f"   Input: {step.action_input}")

            if step.observation:
                obs = step.observation
                if hasattr(obs, "success"):
                    status = "✅" if obs.success else "❌"
                    print(f"   {status} Observation:")
                    if obs.success:
                        if hasattr(obs.data, "stdout"):
                            print(f"      stdout: {obs.data.stdout[:200]}")
                        else:
                            print(f"      data: {obs.data}")
                    else:
                        print(f"      error: {obs.error}")
                else:
                    print(f"   Observation: {obs}")

        if step.is_final:
            print(f"   🏁 Final Answer")

    print("\n" + "=" * 70)


def main():
    """Run with detailed trace output."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY required")
        sys.exit(1)

    print("🔍 Debugging Example: Inspecting Agent Trace\n")

    reasoner = create_react_reasoner()

    agent = Agent(
        loop=ReActLoop(reasoner=reasoner),
        memory=InMemoryMemory(),
        tools=[BashTool()],
        config=AgentConfig(max_iterations=10, debug=True),
    )

    # Task that requires multiple steps
    query = "Count how many Markdown files are in the current directory. Do not use | (pipe) in bash commands and execute one bash command at a time."
    print(f"Task: {query}\n")

    result = agent.run(query)

    print(f"\n✅ Final Result: {result}")

    # Inspect the trace
    trace = agent.get_trace()
    print_trace(trace)

    # Memory inspection
    print("\n💾 Memory Contents:")
    for key in agent.memory.list_keys():
        value = agent.memory.get(key)
        print(f"   {key}: {type(value).__name__}")


if __name__ == "__main__":
    main()
