from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from pydantic import BaseModel

from app.config import Settings
from app.holo.client import HoloApiClient
from app.pioneer.client import PioneerClient
from app.pioneer.intents import INTENT_LABELS, Intent
from pioneer_pipeline_state import BASE_MODEL_ID, REPO_ROOT

EVAL_PATH = REPO_ROOT / "data" / "intent_eval.jsonl"
OUTPUT_DIR = REPO_ROOT / ".state" / "pioneer_evaluations"
RISK_INTENTS = {Intent.POSSIBLE_SCAM, Intent.DESTRUCTIVE_ACTION}


class IntentPrediction(BaseModel):
    intent: Intent


@dataclass(frozen=True, slots=True)
class Prediction:
    intent: Intent
    latency_ms: int
    score: float | None


class HoloIntentClassifier:
    def __init__(self, api: HoloApiClient) -> None:
        self.api = api

    async def classify(self, text: str) -> Prediction:
        schema = IntentPrediction.model_json_schema()
        labels = ", ".join(INTENT_LABELS)
        response = await self.api.structured_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify one smartphone support request from an older or "
                        "nontechnical user. Choose exactly one routing label. "
                        f"Available labels: {labels}. Return only the required JSON."
                    ),
                },
                {"role": "user", "content": text},
            ],
            schema=schema,
            temperature=0.0,
            enable_thinking=False,
        )
        parsed = IntentPrediction.model_validate_json(response.content)
        return Prediction(parsed.intent, response.latency_ms, None)


class PioneerIntentClassifier:
    def __init__(self, api: PioneerClient, model_id: str) -> None:
        self.api = api
        self.model_id = model_id

    async def classify(self, text: str) -> Prediction:
        result = await self.api.classify(self.model_id, text)
        return Prediction(result.intent, result.latency_ms, result.score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one intent model on the untouched human request set."
    )
    parser.add_argument("--provider", choices=("pioneer", "holo"), required=True)
    parser.add_argument("--model-id")
    return parser.parse_args()


def load_examples() -> list[tuple[str, Intent]]:
    examples: list[tuple[str, Intent]] = []
    for line_number, line in enumerate(EVAL_PATH.read_text().splitlines(), start=1):
        payload: Any = json.loads(line)
        if not isinstance(payload, dict) or set(payload) != {"text", "label"}:
            raise RuntimeError(f"Evaluation line {line_number} has an invalid shape")
        text = payload["text"]
        label = payload["label"]
        if not isinstance(text, str) or not text:
            raise RuntimeError(f"Evaluation line {line_number} has invalid text")
        try:
            intent = Intent(label)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Evaluation line {line_number} has an unknown label"
            ) from exc
        examples.append((text, intent))
    expected_counts = Counter(intent for _, intent in examples)
    expected_size = 5 * len(Intent)
    if len(examples) != expected_size or set(expected_counts) != set(Intent):
        raise RuntimeError(
            f"Evaluation set must contain {expected_size} rows across every intent"
        )
    if set(expected_counts.values()) != {5}:
        raise RuntimeError("Evaluation set must contain five rows for every intent")
    return examples


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_metrics(
    *,
    provider: str,
    model_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(rows)
    exact = sum(row["expected"] == row["predicted"] for row in rows)
    per_label: dict[str, dict[str, float | int]] = {}
    for label in INTENT_LABELS:
        true_positive = sum(
            row["expected"] == label and row["predicted"] == label for row in rows
        )
        false_positive = sum(
            row["expected"] != label and row["predicted"] == label for row in rows
        )
        false_negative = sum(
            row["expected"] == label and row["predicted"] != label for row in rows
        )
        precision = ratio(true_positive, true_positive + false_positive)
        recall = ratio(true_positive, true_positive + false_negative)
        f1 = ratio(2 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(row["expected"] == label for row in rows),
        }

    risk_labels = {intent.value for intent in RISK_INTENTS}
    risk_rows = [row for row in rows if row["expected"] in risk_labels]
    non_risk_rows = [row for row in rows if row["expected"] not in risk_labels]
    latencies = sorted(int(row["latency_ms"]) for row in rows)
    p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1)
    errors = [row for row in rows if row["expected"] != row["predicted"]]
    return {
        "provider": provider,
        "model_id": model_id,
        "examples": total,
        "accuracy": round(ratio(exact, total), 4),
        "macro_precision": round(
            mean(float(item["precision"]) for item in per_label.values()), 4
        ),
        "macro_recall": round(
            mean(float(item["recall"]) for item in per_label.values()), 4
        ),
        "macro_f1": round(
            mean(float(item["f1"]) for item in per_label.values()), 4
        ),
        "risk_route_recall": round(
            ratio(
                sum(row["predicted"] in risk_labels for row in risk_rows),
                len(risk_rows),
            ),
            4,
        ),
        "risk_false_positive_rate": round(
            ratio(
                sum(row["predicted"] in risk_labels for row in non_risk_rows),
                len(non_risk_rows),
            ),
            4,
        ),
        "latency_ms": {
            "mean": round(mean(latencies)),
            "median": round(median(latencies)),
            "p95": latencies[p95_index],
        },
        "per_label": per_label,
        "errors": errors,
    }


async def run(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    examples = load_examples()
    provider = args.provider
    model_id = args.model_id
    pioneer_api: PioneerClient | None = None
    holo_api: HoloApiClient | None = None

    if provider == "pioneer":
        if not settings.pioneer_api_key:
            raise RuntimeError("PIONEER_API_KEY is required")
        model_id = model_id or BASE_MODEL_ID
        pioneer_api = PioneerClient(settings.pioneer_api_key, timeout_seconds=60)
        classifier: Any = PioneerIntentClassifier(pioneer_api, model_id)
    else:
        if not settings.hai_api_key:
            raise RuntimeError("HAI_API_KEY is required")
        model_id = model_id or settings.holo_model
        holo_api = HoloApiClient(
            settings.hai_api_key,
            model_id,
            timeout_seconds=60,
        )
        classifier = HoloIntentClassifier(holo_api)

    rows: list[dict[str, Any]] = []
    try:
        for index, (text, expected) in enumerate(examples, start=1):
            prediction = await classifier.classify(text)
            rows.append(
                {
                    "index": index,
                    "text": text,
                    "expected": expected.value,
                    "predicted": prediction.intent.value,
                    "score": prediction.score,
                    "latency_ms": prediction.latency_ms,
                }
            )
            print(
                f"{index:02d}/{len(examples)} expected={expected.value} "
                f"predicted={prediction.intent.value} "
                f"latency={prediction.latency_ms}ms",
                flush=True,
            )
    finally:
        if pioneer_api is not None:
            await pioneer_api.close()
        if holo_api is not None:
            await holo_api.close()

    assert model_id is not None
    report = calculate_metrics(provider=provider, model_id=model_id, rows=rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{provider}-{safe_name(model_id)}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"per_label", "errors"}
    }
    summary["errors"] = len(report["errors"])
    summary["report_path"] = str(output_path.relative_to(REPO_ROOT))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
