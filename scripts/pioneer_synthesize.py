from __future__ import annotations

import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings
from app.holo.client import HoloApiClient
from app.pioneer.intents import INTENT_LABELS
from pioneer_pipeline_state import REPO_ROOT

SEEDS_PATH = REPO_ROOT / "data" / "intent_seeds.jsonl"
EVAL_PATH = REPO_ROOT / "data" / "intent_eval.jsonl"
SYNTHETIC_PATH = REPO_ROOT / "data" / "intent_synthetic.jsonl"
TRAINING_PATH = REPO_ROOT / "data" / "intent_training.jsonl"
SYNTHETIC_PER_INTENT = 14
GENERATED_PER_INTENT = 16

LABEL_DESCRIPTIONS = {
    "increase_text_size": "Words or interface text are too small or hard to read.",
    "wifi_help": "The phone cannot use Wi-Fi or access the internet.",
    "bluetooth_help": "A wireless accessory will not connect or play audio.",
    "volume_help": "Calls, ringing, or media sound are missing or too quiet.",
    "notification_help": "Unwanted alerts, banners, sounds, or interruptions.",
    "possible_scam": "Impersonation, remote access, gift cards, payment demands, or codes.",
    "destructive_action": "Permanent deletion, factory reset, or irreversible removal.",
    "other": "The request is too vague or does not fit another routing label.",
}


class SyntheticBatch(BaseModel):
    examples: list[str] = Field(
        min_length=GENERATED_PER_INTENT,
        max_length=GENERATED_PER_INTENT,
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        payload: Any = json.loads(line)
        if not isinstance(payload, dict) or set(payload) != {"text", "label"}:
            raise RuntimeError(f"{path.name} line {line_number} has an invalid shape")
        text = payload["text"]
        label = payload["label"]
        if not isinstance(text, str) or not text or label not in INTENT_LABELS:
            raise RuntimeError(f"{path.name} line {line_number} has invalid content")
        rows.append({"text": text, "label": label})
    return rows


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


async def run() -> None:
    settings = Settings.from_env()
    if not settings.hai_api_key:
        raise RuntimeError("HAI_API_KEY is required")

    seeds = read_rows(SEEDS_PATH)
    held_out = read_rows(EVAL_PATH)
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in seeds:
        grouped[row["label"]].append(row["text"])

    used = {normalized(row["text"]) for row in seeds + held_out}
    synthetic: list[dict[str, str]] = []
    api = HoloApiClient(
        settings.hai_api_key,
        settings.holo_model,
        timeout_seconds=90,
    )
    schema = SyntheticBatch.model_json_schema()
    try:
        for index, label in enumerate(INTENT_LABELS, start=1):
            examples = "\n".join(f"- {text}" for text in grouped[label])
            prompt = (
                f"Generate exactly {GENERATED_PER_INTENT} distinct smartphone support "
                "requests from older or "
                "nontechnical English-speaking users. The users describe symptoms in "
                "plain language. Use varied phrasing, incomplete thoughts, mild spelling "
                "mistakes, family references, and nontechnical names for interface items. "
                "Do not copy the seed examples. Do not mention the routing label. "
                f"Every request must match this label only: {label}. "
                f"Definition: {LABEL_DESCRIPTIONS[label]}\n"
                f"Seed examples:\n{examples}"
            )
            response = await api.structured_completion(
                messages=[{"role": "user", "content": prompt}],
                schema=schema,
                temperature=0.8,
                enable_thinking=False,
            )
            batch = SyntheticBatch.model_validate_json(response.content)
            accepted: list[str] = []
            for raw_text in batch.examples:
                text = " ".join(raw_text.strip().split())
                for prefix in ("\u26a0\ufe0f ", "\u2610 "):
                    text = text.removeprefix(prefix)
                text = text.replace("\u2019", "'")
                key = normalized(text)
                if len(text) < 8 or key in used:
                    continue
                used.add(key)
                accepted.append(text)
            if len(accepted) < SYNTHETIC_PER_INTENT:
                raise RuntimeError(
                    f"{label} produced {len(accepted)} unique examples, expected at least "
                    f"{SYNTHETIC_PER_INTENT}"
                )
            accepted = accepted[:SYNTHETIC_PER_INTENT]
            synthetic.extend({"text": text, "label": label} for text in accepted)
            print(
                f"{index}/{len(INTENT_LABELS)} label={label} "
                f"examples={len(accepted)} latency={response.latency_ms}ms",
                flush=True,
            )
    finally:
        await api.close()

    if len(synthetic) != SYNTHETIC_PER_INTENT * len(INTENT_LABELS):
        raise RuntimeError("Synthetic dataset has an unexpected size")
    SYNTHETIC_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in synthetic)
    )
    training = seeds + synthetic
    random.Random(42).shuffle(training)
    TRAINING_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in training)
    )
    print(f"Saved {len(synthetic)} synthetic rows to {SYNTHETIC_PATH}")
    print(f"Saved {len(training)} total rows to {TRAINING_PATH}")


if __name__ == "__main__":
    asyncio.run(run())
