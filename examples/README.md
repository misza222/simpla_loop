# Agentic Loops Examples

This directory contains example scripts demonstrating various ways to use the agentic-loops framework.

## Prerequisites

All examples require the package and API key:

```bash
# Install the package
pip install -e ".."

# Copy and configure environment
cp ../.env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Examples Overview

### 1. `list_folder.py` - Basic LLM Agent
Simplest example of an LLM-powered agent that lists files.

```bash
python list_folder.py
```

**Demonstrates:**
- Loading config from `.env`
- Creating an LLM reasoner
- Running with bash tool

### 2. `llm_agent_example.py` - Full Featured Demo
Complete example with detailed output and trace inspection.

```bash
python llm_agent_example.py
```

**Demonstrates:**
- Full agent setup with LLM
- Execution trace inspection
- Error handling

### 3. `custom_gateway.py` - Custom API Gateway
Using non-OpenAI endpoints (Azure, local LLMs, proxies).

```bash
# Set in .env or pass explicitly
OPENAI_BASE_URL=https://gateway.example.com/v1
python custom_gateway.py
```

**Demonstrates:**
- Custom `base_url` configuration
- Environment variable usage

### 4. `multi_step_task.py` - Complex Tasks
Agent performing multiple steps with file operations.

```bash
python multi_step_task.py
```

**Demonstrates:**
- Multi-step reasoning
- Higher retry limits
- Task completion verification

### 5. `mock_vs_llm.py` - Comparison
Side-by-side comparison of mock and LLM reasoners.

```bash
python mock_vs_llm.py
```

**Demonstrates:**
- Mock reasoner (deterministic, no API)
- LLM reasoner (flexible, requires API)
- When to use each approach

### 6. `debugging_trace.py` - Debugging & Inspection
How to inspect agent execution for debugging.

```bash
python debugging_trace.py
```

**Demonstrates:**
- Accessing execution trace
- Inspecting step details
- Memory contents
- Observation handling

### 7. `custom_prompt.py` - Custom System Prompts
Modifying the LLM's behavior with custom system prompts.

```bash
python custom_prompt.py
```

**Demonstrates:**
- Custom system prompt
- Changing agent personality
- Accessing default prompt

## Quick Start

```bash
# 1. Setup
cd /path/to/agentic-loops
pip install -e "."

# 2. Configure
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 3. Run an example
cd examples
python list_folder.py
```

## Environment Variables

All examples use these environment variables (from `.env`):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | Your OpenAI API key |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | API endpoint |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model name |
| `OPENAI_MAX_RETRIES` | No | `3` | Validation retries |

## Creating Your Own

Template for new examples:

```python
"""Example: Your description here."""

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
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY required")
        sys.exit(1)

    # Your code here
    reasoner = create_react_reasoner()
    agent = Agent(...)
    result = agent.run("Your query")
    print(result)


if __name__ == "__main__":
    main()
```
