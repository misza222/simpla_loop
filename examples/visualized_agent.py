"""Example: agent run with live visualization in agent_vis.

Runs a multi-step bash agent and streams each reasoning step as an
animated particle flow to a running agent_vis server.

Prerequisites:
    # 1. Install vis extra
    uv sync --extra vis

    # 2. Start the agent_vis server (in a separate terminal)
    cd ../agent_vis
    uv run uvicorn src.agent_vis.app:app --reload

    # 3. Open the browser
    open http://localhost:8000

    # 4. Set your API key
    cp .env.example .env  # then add OPENAI_API_KEY

Usage:
    uv run python examples/visualized_agent.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

from simpla_loop import Agent, AgentConfig
from simpla_loop.llm import create_react_reasoner
from simpla_loop.loops.react import ReActLoop
from simpla_loop.memory.in_memory import InMemoryMemory
from simpla_loop.reporters.agent_vis import AgentVisReporter
from simpla_loop.tools.bash import BashTool

WORKFLOW_ID = "visualized_agent"
AGENT_VIS_URL = "http://localhost:8000"


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        print("Copy .env.example to .env and add your key.")

        sys.exit(1)

    print(f"Open {AGENT_VIS_URL} in your browser to watch the agent run.\n")

    reporter = AgentVisReporter(
        workflow_id=WORKFLOW_ID,
        base_url=AGENT_VIS_URL,
    )

    agent = Agent(
        loop=ReActLoop(reasoner=create_react_reasoner()),
        memory=InMemoryMemory(),
        tools=[BashTool(timeout=30)],
        config=AgentConfig(max_iterations=10),
        reporter=reporter,
    )

    # A multi-step task so there is something interesting to watch.
    query = (
        "Create a file /tmp/agent_vis_demo.txt containing the 5 largest files "
        "in /usr/bin (one per line, with sizes). Then read it back and report "
        "the result. Finally remove the file."
    )

    print(f"Task: {query}\n")
    print("-" * 60)

    result = agent.run(query)

    print("-" * 60)
    trace = agent.get_trace()
    step_count = len(trace) if trace else 0
    print(f"\nCompleted in {step_count} steps.")
    print(f"Answer: {result}")


if __name__ == "__main__":
    main()
