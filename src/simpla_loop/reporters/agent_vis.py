"""AgentVis reporter for real-time workflow visualization.

Connects an Agent run to a running agent_vis server, rendering the
agent's execution as animated particle flows on a live graph.

The workflow graph is persistent across runs (upserted on each
on_run_start call), so the browser tab stays open and each new run
simply sends new flows through the same nodes.

Graph layout:
    start → reasoner → [tool node] → reasoner → ... → done

Requires a running agent_vis server and the [vis] extra:
    uv add simpla-loop[vis]
    uv run uvicorn src.agent_vis.app:app --reload  # in agent_vis repo

Example:
    >>> from simpla_loop.reporters.agent_vis import AgentVisReporter
    >>> reporter = AgentVisReporter(workflow_id="my_agent")
    >>> agent = Agent(..., reporter=reporter)
    >>> agent.run("list files in /tmp")
"""

import json
import time
from typing import Any

import requests
import structlog
import websocket  # websocket-client

from simpla_loop.core.loop import LoopResult
from simpla_loop.core.tool import Tool

logger = structlog.get_logger()

_DEFAULT_FLOW_MS = 4000
_START_FLOW_MS = 2000
_DONE_FLOW_MS = 2000


class AgentVisReporter:
    """Reports agent execution steps to a running agent_vis server.

    Creates or updates a workflow graph on the server (idempotent) and
    sends an animated flow particle for each reasoning/tool step.

    Visualization failures are silently swallowed — the agent run is
    never interrupted by a connectivity problem with agent_vis.

    Args:
        workflow_id: Stable ID for the workflow graph. Reused across
                     agent.run() calls so the graph persists in the
                     browser between invocations.
        base_url: Base URL of the running agent_vis server.
    """

    def __init__(
        self,
        *,
        workflow_id: str,
        base_url: str = "http://localhost:8000",
    ) -> None:
        self._workflow_id = workflow_id
        self._base_url = base_url.rstrip("/")
        self._ws_url = self._base_url.replace("http", "ws") + "/ws"

    def on_run_start(self, tools: list[Tool]) -> None:
        """Upsert the workflow graph then send the start flow."""
        nodes = self._build_nodes(tools)
        edges = self._build_edges(tools)
        self._upsert_workflow(nodes, edges)
        self._send_flow(["start", "reasoner"], duration_ms=_START_FLOW_MS)

    def on_step(self, result: LoopResult[Any]) -> None:
        """Send a flow for the step that just completed.

        Tool steps:   reasoner → tool → reasoner
        Final steps:  reasoner → done
        """
        steps = getattr(result.state, "steps", [])
        if not steps:
            return

        step = steps[-1]
        action: str | None = getattr(step, "action", None)
        is_final: bool = getattr(step, "is_final", False)

        # Animate the tool call (even on the final step an action can fire)
        if action and not is_final:
            self._send_flow(["reasoner", action, "reasoner"])

        # Animate completion (final answer or max-steps exhausted)
        if result.done:
            self._send_flow(["reasoner", "done"], duration_ms=_DONE_FLOW_MS)

    def on_run_done(self) -> None:
        """No-op: the completion flow is already sent inside on_step."""

    # --- graph construction ---

    def _build_nodes(self, tools: list[Tool]) -> list[dict[str, str]]:
        nodes: list[dict[str, str]] = [
            {"id": "start", "label": "Start"},
            {"id": "reasoner", "label": "LLM Reasoner"},
        ]
        for tool in tools:
            label = tool.name.replace("_", " ").title()
            nodes.append({"id": tool.name, "label": label})
        nodes.append({"id": "done", "label": "Done"})
        return nodes

    def _build_edges(self, tools: list[Tool]) -> list[dict[str, str]]:
        edges: list[dict[str, str]] = [{"from_node": "start", "to_node": "reasoner"}]
        for tool in tools:
            edges.append({"from_node": "reasoner", "to_node": tool.name})
            edges.append({"from_node": tool.name, "to_node": "reasoner"})
        edges.append({"from_node": "reasoner", "to_node": "done"})
        return edges

    # --- server communication ---

    def _upsert_workflow(
        self,
        nodes: list[dict[str, str]],
        edges: list[dict[str, str]],
    ) -> None:
        """PUT the workflow; fall back to POST if it doesn't exist yet."""
        url = f"{self._base_url}/workflows/{self._workflow_id}"
        payload: dict[str, Any] = {"nodes": nodes, "edges": edges}
        try:
            response = requests.put(url, json=payload, timeout=5)
            if response.status_code == 404:
                create_payload: dict[str, Any] = {
                    "id": self._workflow_id,
                    **payload,
                    "flows": [],
                }
                requests.post(
                    f"{self._base_url}/workflows",
                    json=create_payload,
                    timeout=5,
                ).raise_for_status()
            else:
                response.raise_for_status()
            logger.debug("agent_vis_workflow_upserted", workflow_id=self._workflow_id)
        except Exception as exc:
            logger.warning("agent_vis_workflow_failed", error=str(exc))

    def _send_flow(
        self,
        path: list[str],
        *,
        duration_ms: int = _DEFAULT_FLOW_MS,
    ) -> None:
        """Open a WebSocket connection, send one add_flow message, close."""
        flow_id = f"flow_{int(time.time() * 1000)}"
        message = {
            "type": "add_flow",
            "workflow_id": self._workflow_id,
            "flow": {"id": flow_id, "path": path, "duration_ms": duration_ms},
        }
        try:
            ws = websocket.create_connection(self._ws_url, timeout=5)
            ws.send(json.dumps(message))
            ws.close()
            logger.debug("agent_vis_flow_sent", path=path)
        except Exception as exc:
            # Visualization is best-effort — never interrupt the agent run
            logger.warning("agent_vis_flow_failed", error=str(exc), path=path)
