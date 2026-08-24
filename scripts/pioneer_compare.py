from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pioneer_pipeline_state import BASE_MODEL_ID, REPO_ROOT

REPORT_DIR = REPO_ROOT / ".state" / "pioneer_evaluations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base, frontier, and tuned request-router evaluations."
    )
    parser.add_argument("tuned_model_id")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid evaluation report in {path}")
    return payload


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def run(tuned_model_id: str) -> None:
    base = load_report(
        REPORT_DIR / f"pioneer-{safe_name(BASE_MODEL_ID)}.json"
    )
    frontier = load_report(REPORT_DIR / "holo-holo3-1-35b-a3b.json")
    tuned = load_report(
        REPORT_DIR / f"pioneer-{safe_name(tuned_model_id)}.json"
    )
    comparison = {
        "models": {
            "base": base,
            "frontier": frontier,
            "tuned": tuned,
        },
        "tuned_vs_base": {
            "accuracy_gain": round(tuned["accuracy"] - base["accuracy"], 4),
            "macro_f1_gain": round(tuned["macro_f1"] - base["macro_f1"], 4),
            "risk_recall_gain": round(
                tuned["risk_route_recall"] - base["risk_route_recall"], 4
            ),
        },
        "tuned_vs_frontier": {
            "accuracy_difference": round(
                tuned["accuracy"] - frontier["accuracy"], 4
            ),
            "macro_f1_difference": round(
                tuned["macro_f1"] - frontier["macro_f1"], 4
            ),
            "mean_latency_speedup": round(
                ratio(
                    frontier["latency_ms"]["mean"],
                    tuned["latency_ms"]["mean"],
                ),
                2,
            ),
            "p95_latency_speedup": round(
                ratio(
                    frontier["latency_ms"]["p95"],
                    tuned["latency_ms"]["p95"],
                ),
                2,
            ),
            "holo_call_reduction": round(
                1
                - ratio(
                    tuned["total_holo_calls"],
                    frontier["total_holo_calls"],
                ),
                4,
            ),
        },
    }
    output_path = REPORT_DIR / "comparison.json"
    output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    summary = {
        "tuned_model_id": tuned_model_id,
        "base_accuracy": base["accuracy"],
        "frontier_accuracy": frontier["accuracy"],
        "tuned_accuracy": tuned["accuracy"],
        **comparison["tuned_vs_base"],
        **comparison["tuned_vs_frontier"],
        "report_path": str(output_path.relative_to(REPO_ROOT)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    run(parse_args().tuned_model_id)
