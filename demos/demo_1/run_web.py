"""
Start the Supply Chain Demo web UI.

Usage:
    uv run python demos/demo_1/run_web.py
    # or from the Janus root:
    python demos/demo_1/run_web.py

Then open http://localhost:8000 in your browser.
"""

import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    uvicorn.run(
        "demos.demo_1.web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
