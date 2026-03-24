#!/usr/bin/env python3
"""Minimal Modal app entrypoint used by ``example/modal/run_modal.py``.

The launcher sends a JSON payload to ``run_inference`` via:

    modal run modal_app.py::run_inference --payload-json '<json>'

This default app intentionally keeps behavior lightweight so the launcher works
out of the box in this repository.
"""

from __future__ import annotations

import json
from typing import Any

import modal

app = modal.App("davinci-magihuman")


@app.function()
def run_inference(payload_json: str) -> dict[str, Any]:
    """Accept launcher payload and return a basic structured response.

    The payload structure mirrors ``example/modal/run_modal.py``. Users can
    extend this function to launch the full pipeline in their own Modal setup.
    """

    payload = json.loads(payload_json)
    return {
        "status": "ok",
        "preset": payload.get("preset"),
        "runtime": payload.get("runtime", {}),
        "message": "Received payload successfully. Customize modal_app.py::run_inference to run full inference.",
    }
