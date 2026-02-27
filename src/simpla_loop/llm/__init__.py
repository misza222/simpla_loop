"""LLM integration using Instructor for structured outputs.

This module provides OpenAI client integration with automatic
structured output parsing via Instructor and Pydantic models.

Example:
    >>> from simpla_loop.llm import create_react_reasoner
    >>> reasoner = create_react_reasoner(model="gpt-4o-mini")
    >>> loop = ReActLoop(reasoner=reasoner)
"""

from simpla_loop.llm.client import OpenAIConfig, create_instructor_client
from simpla_loop.llm.models import ReActResponse
from simpla_loop.llm.reasoners import create_react_reasoner

__all__ = [
    "create_react_reasoner",
    "create_instructor_client",
    "OpenAIConfig",
    "ReActResponse",
]
