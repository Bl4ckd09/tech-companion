from __future__ import annotations

from app.pioneer.client import ClassificationResult
from app.pioneer.intents import Intent
from app.pioneer.router import PioneerIntentRouter


class FakePioneerClient:
    def __init__(self, result: ClassificationResult) -> None:
        self.result = result

    async def classify(self, model_id: str, text: str) -> ClassificationResult:
        del model_id, text
        return self.result


def result(intent: Intent, score: float) -> ClassificationResult:
    return ClassificationResult(
        intent=intent,
        score=score,
        model_id="tuned-model",
        latency_ms=42,
    )


async def test_low_confidence_prediction_abstains_to_other() -> None:
    router = PioneerIntentRouter(
        FakePioneerClient(result(Intent.WIFI_HELP, 0.49)),
        "tuned-model",
        min_score=0.5,
    )

    prediction = await router.classify("The internet symbol looks strange")

    assert prediction.intent is Intent.OTHER
    assert prediction.score == 0.49
    assert prediction.source == "pioneer-abstained"


async def test_confident_prediction_keeps_the_model_route() -> None:
    expected = result(Intent.INCREASE_TEXT_SIZE, 0.9)
    router = PioneerIntentRouter(
        FakePioneerClient(expected),
        "tuned-model",
        min_score=0.5,
    )

    assert await router.classify("The words are tiny") is expected
