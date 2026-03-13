"""
Janus Demo — FastAPI + WebSocket server.

Serves the split-panel web UI and streams scenario events in real time.

Usage:
    cd demos/
    uvicorn app:app --reload
    # Open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.scenarios import get_scenario, list_scenarios
from examples.shared.events import BaseEvent
from examples.shared.scenario_runner import ScenarioRunner

app = FastAPI(title="Janus Demo")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/api/scenarios")
async def scenarios_list():
    return list_scenarios()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "start_demo":
                demo_id = msg.get("demo_id", "")
                await _run_demo(ws, demo_id)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_json({"event_type": "error", "message": str(exc)})
        except Exception:
            pass


async def _run_demo(ws: WebSocket, demo_id: str) -> None:
    """Run a demo scenario, streaming events to both panels concurrently."""

    try:
        scenario = get_scenario(demo_id)
    except KeyError as exc:
        await ws.send_json({"event_type": "error", "message": str(exc)})
        return

    async def send_event(event: BaseEvent) -> None:
        try:
            await ws.send_json(event.to_dict())
        except Exception:
            pass

    runner_left = ScenarioRunner()
    runner_right = ScenarioRunner()

    # Run both panels concurrently
    # Need fresh scenario instances so workspace state doesn't clash
    scenario_left = get_scenario(demo_id)
    scenario_right = get_scenario(demo_id)

    await asyncio.gather(
        runner_left.run(scenario_left, protected=False, event_callback=send_event),
        runner_right.run(scenario_right, protected=True, event_callback=send_event),
    )

    await ws.send_json({"event_type": "demo_complete", "panel": "both"})
