from __future__ import annotations

import asyncio
import json

import httpx

from app.pioneer.client import PioneerClient
from pioneer_pipeline_state import (
    DATASET_NAME,
    REPO_ROOT,
    load_api_key,
    load_state,
    response_id,
    response_status,
    save_state,
)

TRAINING_PATH = REPO_ROOT / "data" / "intent_training.jsonl"


async def run() -> None:
    if not TRAINING_PATH.exists():
        raise RuntimeError("Run scripts/pioneer_synthesize.py before upload")
    rows = TRAINING_PATH.read_text().splitlines()
    if len(rows) != 200:
        raise RuntimeError("Training dataset must contain exactly 200 rows")
    for line in rows:
        payload = json.loads(line)
        if set(payload) != {"text", "label"}:
            raise RuntimeError("Training rows must contain only text and label")

    client = PioneerClient(load_api_key(), timeout_seconds=60)
    try:
        reservation = await client.reserve_dataset_upload(
            dataset_name=DATASET_NAME,
            filename=TRAINING_PATH.name,
        )
        dataset_id = response_id(reservation, "dataset_id", "id")
        upload_url = reservation.get("presigned_url")
        if dataset_id is None or not isinstance(upload_url, str) or not upload_url:
            raise RuntimeError("Pioneer upload reservation is missing required fields")

        async with httpx.AsyncClient(timeout=180) as upload_client:
            response = await upload_client.put(
                upload_url,
                content=TRAINING_PATH.read_bytes(),
                headers={"Content-Type": "application/octet-stream"},
            )
            response.raise_for_status()

        processing = await client.process_dataset_upload(dataset_id)
        state = load_state()
        for key in (
            "generation_job_id",
            "generation_status",
            "provider_error",
            "provider_error_code",
            "training_job_id",
            "training_status",
        ):
            state.pop(key, None)
        state.update(
            {
                "dataset_id": dataset_id,
                "dataset_name": DATASET_NAME,
                "dataset_status": response_status(processing),
                "dataset_version": reservation.get("version_number"),
                "dataset_source": "uploaded_human_and_holo_synthetic",
            }
        )
        save_state(state)
        print(json.dumps(state, indent=2, sort_keys=True))
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(run())
