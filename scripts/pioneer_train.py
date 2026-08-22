from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from app.pioneer.client import PioneerClient
from pioneer_pipeline_state import (
    BASE_MODEL_ID,
    DATASET_NAME,
    MODEL_NAME,
    load_api_key,
    load_state,
    response_id,
    response_status,
    save_state,
)

READY_STATUSES = {"ready", "complete", "completed"}
FAILED_STATUSES = {"failed", "cancelled", "stopped"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume Pioneer generation and start intent fine-tuning when ready."
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll generation and dataset state until training can start.",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def latest_version(payload: dict[str, Any]) -> dict[str, Any] | None:
    versions = payload.get("versions")
    if not isinstance(versions, list):
        return None
    return next((version for version in versions if isinstance(version, dict)), None)


def print_state(state: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "base_model_id": BASE_MODEL_ID,
                "dataset_id": state.get("dataset_id"),
                "dataset_name": DATASET_NAME,
                "dataset_status": state.get("dataset_status"),
                "generation_job_id": state.get("generation_job_id"),
                "generation_status": state.get("generation_status"),
                "training_job_id": state.get("training_job_id"),
                "training_status": state.get("training_status"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


async def wait_for_dataset(
    client: PioneerClient,
    state: dict[str, Any],
    *,
    wait: bool,
    poll_seconds: float,
) -> bool:
    generation_job_id = state.get("generation_job_id")
    if not isinstance(generation_job_id, str) or not generation_job_id:
        raise RuntimeError("Run scripts/pioneer_generate.py before training")

    while True:
        generation = await client.generation_status(generation_job_id)
        generation_status = response_status(generation)
        state["generation_status"] = generation_status
        save_state(state)
        if generation_status in FAILED_STATUSES:
            raise RuntimeError(f"Pioneer generation ended with status {generation_status}")
        if generation_status in READY_STATUSES:
            break
        print_state(state)
        if not wait:
            return False
        await asyncio.sleep(poll_seconds)

    while True:
        dataset = await client.dataset_details(DATASET_NAME)
        version = latest_version(dataset)
        dataset_status = response_status(version or dataset)
        state["dataset_status"] = dataset_status
        dataset_id = response_id(version or dataset, "id", "dataset_id")
        if dataset_id is not None:
            state["dataset_id"] = dataset_id
        save_state(state)
        if dataset_status in FAILED_STATUSES:
            raise RuntimeError(f"Pioneer dataset ended with status {dataset_status}")
        if dataset_status in READY_STATUSES:
            return True
        print_state(state)
        if not wait:
            return False
        await asyncio.sleep(poll_seconds)


async def run(args: argparse.Namespace) -> None:
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be greater than zero")

    state = load_state()
    client = PioneerClient(load_api_key(), timeout_seconds=60)
    try:
        if not await wait_for_dataset(
            client,
            state,
            wait=args.wait,
            poll_seconds=args.poll_seconds,
        ):
            return

        training_job_id = state.get("training_job_id")
        if isinstance(training_job_id, str) and training_job_id:
            training = await client.training_status(training_job_id)
        else:
            training = await client.start_training(
                dataset_name=DATASET_NAME,
                model_name=MODEL_NAME,
                base_model=BASE_MODEL_ID,
            )
            training_job_id = response_id(training, "id", "job_id")
            if training_job_id is None:
                raise RuntimeError("Pioneer training response did not contain a job ID")

        state.update(
            {
                "base_model_id": BASE_MODEL_ID,
                "dataset_name": DATASET_NAME,
                "training_job_id": training_job_id,
                "training_status": response_status(training),
            }
        )
        save_state(state)
        print_state(state)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
