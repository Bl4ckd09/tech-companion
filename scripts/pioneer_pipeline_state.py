from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPO_ROOT / ".state" / "pioneer_request_interpreter_v2.json"
DATASET_NAME = "tech-companion-parent-request-router-v2"
MODEL_NAME = "tech-companion-parent-request-interpreter-v2"
BASE_MODEL_ID = "fastino/gliner2-base-v1"


def load_api_key() -> str:
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("PIONEER_API_KEY")
    if not api_key:
        raise RuntimeError("PIONEER_API_KEY is required in .env")
    return api_key


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    payload = json.loads(STATE_PATH.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid Pioneer state in {STATE_PATH}")
    return payload


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = STATE_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(STATE_PATH)


def response_id(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def response_status(payload: dict[str, Any]) -> str | None:
    value = payload.get("status")
    return value if isinstance(value, str) else None
