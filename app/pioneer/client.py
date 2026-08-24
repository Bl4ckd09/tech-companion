from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.pioneer.intents import INTENT_LABELS, Intent


class PioneerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    intent: Intent
    score: float | None
    model_id: str
    latency_ms: int
    source: str = "pioneer"


class PioneerClient:
    def __init__(self, api_key: str, timeout_seconds: float = 30) -> None:
        if not api_key:
            raise ValueError("PIONEER_API_KEY is required")
        self.client = httpx.AsyncClient(
            base_url="https://api.pioneer.ai",
            headers={"X-API-Key": api_key},
            timeout=timeout_seconds,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        try:
            response = await self.client.request(
                method, path, json=json, params=params
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise PioneerError(
                f"Pioneer returned HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise PioneerError(f"Pioneer request failed: {exc}") from exc
        payload = response.json()
        if not isinstance(payload, (dict, list)):
            raise PioneerError("Pioneer returned an unexpected response")
        return payload

    async def list_trainable_encoders(self) -> list[dict[str, Any]]:
        payload = await self._request(
            "GET",
            "/base-models",
            params={"task_type": "encoder", "supports_training": "true"},
        )
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        for key in ("models", "data", "items"):
            items = payload.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        raise PioneerError("Pioneer base-model response did not contain a model list")

    async def classify(self, model_id: str, text: str) -> ClassificationResult:
        started = time.perf_counter()
        payload = await self._request(
            "POST",
            "/inference",
            json={
                "model_id": model_id,
                "text": text,
                "schema": {
                    "classifications": [
                        {
                            "task": "tech_support_intent",
                            "labels": INTENT_LABELS,
                        }
                    ]
                },
                "threshold": 0.01,
            },
        )
        latency_ms = round((time.perf_counter() - started) * 1000)
        label, score = _extract_label(payload)
        try:
            intent = Intent(label)
        except ValueError as exc:
            raise PioneerError(f"Pioneer returned unknown intent {label!r}") from exc
        return ClassificationResult(intent, score, model_id, latency_ms)

    async def generate_dataset(
        self,
        *,
        dataset_name: str,
        num_examples: int,
        classified_examples: list[dict[str, str]],
    ) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            "/generate",
            json={
                "task_type": "classification",
                "dataset_name": dataset_name,
                "labels": INTENT_LABELS,
                "num_examples": num_examples,
                "domain_description": (
                    "Short smartphone support requests from older or nontechnical "
                    "English-speaking users. Each request has exactly one routing label. "
                    "increase_text_size covers tiny or unreadable words. wifi_help covers "
                    "internet connectivity. bluetooth_help covers wireless accessories. "
                    "volume_help covers ringing, calls, and media sound. notification_help "
                    "covers unwanted alerts. possible_scam covers remote access, "
                    "impersonation, gift cards, payment demands, or secret codes. "
                    "destructive_action covers permanent deletion or reset. other covers "
                    "unclear requests without enough information. Use symptom language, "
                    "incomplete phrases, mild spelling errors, and nontechnical names."
                ),
                "classified_examples": classified_examples,
            },
        )
        if not isinstance(payload, dict):
            raise PioneerError("Pioneer generation response was not an object")
        return payload

    async def generation_status(self, job_id: str) -> dict[str, Any]:
        payload = await self._request("GET", f"/generate/jobs/{job_id}")
        if not isinstance(payload, dict):
            raise PioneerError("Pioneer generation status was not an object")
        return payload

    async def reserve_dataset_upload(
        self,
        *,
        dataset_name: str,
        filename: str,
    ) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            "/felix/datasets/upload/url",
            json={
                "dataset_name": dataset_name,
                "dataset_type": "classification",
                "format": "jsonl",
                "filename": filename,
                "type": "training",
                "generation_type": "external",
            },
        )
        if not isinstance(payload, dict):
            raise PioneerError("Pioneer upload reservation was not an object")
        return payload

    async def process_dataset_upload(self, dataset_id: str) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            "/felix/datasets/upload/process",
            json={"dataset_id": dataset_id},
        )
        if not isinstance(payload, dict):
            raise PioneerError("Pioneer upload processing response was not an object")
        return payload

    async def dataset_details(self, dataset_name: str) -> dict[str, Any]:
        payload = await self._request("GET", f"/felix/datasets/{dataset_name}")
        if not isinstance(payload, dict):
            raise PioneerError("Pioneer dataset response was not an object")
        return payload

    async def start_training(
        self,
        *,
        dataset_name: str,
        model_name: str,
        base_model: str = "fastino/gliner2-base-v1",
    ) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            "/felix/training-jobs",
            json={
                "model_name": model_name,
                "base_model": base_model,
                "datasets": [{"name": dataset_name}],
                "training_type": "lora",
                "nr_epochs": 5,
                "learning_rate": 5e-5,
            },
        )
        if not isinstance(payload, dict):
            raise PioneerError("Pioneer training response was not an object")
        return payload

    async def training_status(self, job_id: str) -> dict[str, Any]:
        payload = await self._request("GET", f"/felix/training-jobs/{job_id}")
        if not isinstance(payload, dict):
            raise PioneerError("Pioneer training status was not an object")
        return payload

    async def start_evaluation(
        self, *, model_id: str, dataset_name: str
    ) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            "/felix/evaluations",
            json={"base_model": model_id, "dataset_name": dataset_name},
        )
        if not isinstance(payload, dict):
            raise PioneerError("Pioneer evaluation response was not an object")
        return payload

    async def evaluation_status(self, evaluation_id: str) -> dict[str, Any]:
        payload = await self._request("GET", f"/felix/evaluations/{evaluation_id}")
        if not isinstance(payload, dict):
            raise PioneerError("Pioneer evaluation status was not an object")
        return payload

    async def close(self) -> None:
        await self.client.aclose()


def _extract_label(payload: dict[str, Any] | list[Any]) -> tuple[str, float | None]:
    valid = set(INTENT_LABELS)

    def walk(value: Any) -> tuple[str, float | None] | None:
        if isinstance(value, dict):
            for key in ("label", "class", "intent"):
                label = value.get(key)
                if isinstance(label, str) and label in valid:
                    score_value = value.get("score", value.get("confidence"))
                    score = float(score_value) if isinstance(score_value, (int, float)) else None
                    return label, score
            for key in ("tech_support_intent", "classifications", "prediction", "predictions", "result", "results", "data", "output"):
                if key in value:
                    result = walk(value[key])
                    if result:
                        return result
            for child in value.values():
                result = walk(child)
                if result:
                    return result
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, str) and child in valid:
                    return child, None
                result = walk(child)
                if result:
                    return result
        elif isinstance(value, str) and value in valid:
            return value, None
        return None

    extracted = walk(payload)
    if extracted is None:
        raise PioneerError(f"Could not find an intent label in Pioneer response: {payload}")
    return extracted
