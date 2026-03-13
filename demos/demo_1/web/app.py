"""
FastAPI web server for the Supply Chain Demo.

Serves the split-pane demo UI and streams scenario events as
Server-Sent Events (SSE) to both panels simultaneously.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from demos.demo_1.demo import run_scenario

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DEMO_DIR = BASE_DIR.parent
WORKSPACES = {
    "attack": DEMO_DIR / "runtime" / "workspace_attack",
    "protected": DEMO_DIR / "runtime" / "workspace_protected",
}

app = FastAPI(title="Janus — Supply Chain Demo")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/stream/{scenario}")
async def stream_scenario(scenario: str):
    """Stream scenario events as SSE. scenario = 'attack' | 'protected'"""
    if scenario not in ("attack", "protected"):
        return HTMLResponse("Invalid scenario", status_code=400)

    use_janus = scenario == "protected"
    workspace = WORKSPACES[scenario]

    return StreamingResponse(
        _sse_generator(workspace, use_janus),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/run-both")
async def run_both():
    """Trigger both scenarios. Returns immediately; streams happen via /stream/."""
    return {"status": "ok", "message": "Connect to /stream/attack and /stream/protected"}


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------


async def _sse_generator(
    workspace: Path, use_janus: bool
) -> AsyncGenerator[str, None]:
    """Run the blocking scenario in a thread and yield SSE-formatted events."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def _producer():
        try:
            for event in run_scenario(workspace, use_janus=use_janus):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, _producer)

    while True:
        event = await queue.get()
        if event is None:
            yield "event: done\ndata: {}\n\n"
            break

        data = json.dumps(event, default=str)
        yield f"event: {event['type']}\ndata: {data}\n\n"
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
