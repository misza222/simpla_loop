"""Example: LLM-powered agent to list folder contents.

This example shows the simplest way to create an LLM-powered agent
that can execute bash commands. It demonstrates:
- Loading configuration from .env
- Creating an LLM reasoner
- Running a task with bash tool

Prerequisites:
    pip install -e ".[llm]"
    cp .env.example .env  # Then add your OPENAI_API_KEY

Usage:
    python examples/list_folder.py
"""

import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

from simpla_loop import Agent, AgentConfig
from simpla_loop.llm import create_react_reasoner
from simpla_loop.loops.react import ReActLoop
from simpla_loop.memory.in_memory import InMemoryMemory
from simpla_loop.tools.bash import BashTool


def main():
    """Run the list folder example."""
    # Load environment variables
    load_dotenv()

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        print("Copy .env.example to .env and add your API key.")
        sys.exit(1)

    print("🤖 Creating LLM agent...\n")

    # Create LLM reasoner with default settings from env
    reasoner = create_react_reasoner()

    # Create agent with bash tool
    agent = Agent(
        loop=ReActLoop(reasoner=reasoner),
        memory=InMemoryMemory(),
        tools=[BashTool(timeout=10)],
        config=AgentConfig(max_iterations=5, debug=True),
    )

    # Ask the agent to list files
    query = "List files in the current directory as json array."
    print(f"Query: {query}\n")
    print("-" * 50)

    result = agent.run(query)

    print("-" * 50)
    print(f"\n✅ Result: {result}")


if __name__ == "__main__":
    main()
