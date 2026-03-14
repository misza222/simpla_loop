"""Example: Multi-step task with file operations.

This example demonstrates an agent performing multiple steps:
1. Create a file with content
2. Read the file back
3. Process the content
4. Report results

Shows how the agent maintains context across multiple tool calls.

Prerequisites:
    pip install -e ".[dev]"
    cp .env.example .env  # Add your OPENAI_API_KEY

Usage:
    python examples/multi_step_task.py
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


def main():
    """Run multi-step task."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found")
        sys.exit(1)

    print("📝 Multi-step task example\n")

    # Create reasoner with slightly higher retry for complex tasks
    reasoner = create_react_reasoner(max_retries=5)

    agent = Agent(
        loop=ReActLoop(reasoner=reasoner),
        memory=InMemoryMemory(),
        tools=[BashTool(timeout=30)],
        config=AgentConfig(max_iterations=10, debug=True),
    )

    # Multi-step task
    query = """
    Create a file called /tmp/test_agent.txt with the content "Hello from Agent!",
    then read it back and confirm the content matches what was written.
    Finally remove the file.
    """

    print(f"Task: {query.strip()}\n")
    print("=" * 60)

    result = agent.run(query)

    print("=" * 60)
    print("\n✅ Task completed!")
    print(f"Final answer: {result}")

    # Show trace
    trace = agent.get_trace()
    if trace:
        print(f"\n📊 Completed in {len(trace)} steps")


if __name__ == "__main__":
    main()
