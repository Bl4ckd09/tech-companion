from __future__ import annotations

import asyncio
import json
from typing import Any

from app.pioneer.client import PioneerClient, PioneerError
from app.pioneer.intents import INTENT_LABELS
from pioneer_pipeline_state import (
    BASE_MODEL_ID,
    DATASET_NAME,
    REPO_ROOT,
    load_api_key,
    load_state,
    response_id,
    response_status,
    save_state,
)

NUM_EXAMPLES = 200
SEEDS_PATH = REPO_ROOT / "data" / "intent_seeds.jsonl"


def load_seeds() -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for line_number, line in enumerate(SEEDS_PATH.read_text().splitlines(), start=1):
        payload: Any = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Seed line {line_number} is not an object")
        if set(payload) != {"text", "label"}:
            raise RuntimeError(f"Seed line {line_number} must contain only text and label")
        text = payload["text"]
        label = payload["label"]
        if not isinstance(text, str) or not text:
            raise RuntimeError(f"Seed line {line_number} has invalid text")
        if not isinstance(label, str) or label not in INTENT_LABELS:
            raise RuntimeError(f"Seed line {line_number} has an unknown label")
        examples.append({"text": text, "label": label})

    labels = {example["label"] for example in examples}
    if len(INTENT_LABELS) != 10 or labels != set(INTENT_LABELS):
        raise RuntimeError("Seeds must cover the ten intent labels")
    return examples


async def run() -> None:
    state = load_state()
    client = PioneerClient(load_api_key(), timeout_seconds=60)
    try:
        models = await client.list_trainable_encoders()
        live_ids = {model.get("id") for model in models}
        if BASE_MODEL_ID not in live_ids:
            raise RuntimeError(f"Required base model is unavailable: {BASE_MODEL_ID}")

        generation_job_id = state.get("generation_job_id")
        if isinstance(generation_job_id, str) and generation_job_id:
            payload = await client.generation_status(generation_job_id)
        else:
            try:
                payload = await client.generate_dataset(
                    dataset_name=DATASET_NAME,
                    num_examples=NUM_EXAMPLES,
                    classified_examples=load_seeds(),
                )
            except PioneerError as exc:
                if "payment_method_required" not in str(exc):
                    raise
                state.update(
                    {
                        "base_model_id": BASE_MODEL_ID,
                        "dataset_name": DATASET_NAME,
                        "generation_status": "blocked",
                        "provider_error_code": "payment_method_required",
                    }
                )
                save_state(state)
                raise SystemExit(
                    "Pioneer generation is blocked: payment_method_required"
                ) from None
            generation_job_id = response_id(payload, "job_id", "id")
            if generation_job_id is None:
                raise PioneerError("Pioneer generation response did not contain a job ID")

        state.pop("provider_error_code", None)
        state.update(
            {
                "base_model_id": BASE_MODEL_ID,
                "dataset_name": DATASET_NAME,
                "generation_job_id": generation_job_id,
                "generation_status": response_status(payload),
            }
        )
        dataset_id = response_id(payload, "dataset_id")
        if dataset_id is not None:
            state["dataset_id"] = dataset_id
        save_state(state)
        print(
            json.dumps(
                {
                    "base_model_id": BASE_MODEL_ID,
                    "dataset_id": state.get("dataset_id"),
                    "dataset_name": DATASET_NAME,
                    "generation_job_id": generation_job_id,
                    "generation_status": state.get("generation_status"),
                },
                sort_keys=True,
            )
        )
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(run())
