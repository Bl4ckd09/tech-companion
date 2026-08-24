from __future__ import annotations

import re
from typing import Protocol

from app.pioneer.client import ClassificationResult, PioneerClient, PioneerError
from app.pioneer.intents import Intent


class IntentRouter(Protocol):
    async def classify(self, text: str) -> ClassificationResult: ...


class PioneerIntentRouter:
    def __init__(
        self,
        client: PioneerClient,
        model_id: str,
        *,
        fallback: IntentRouter | None = None,
        min_score: float = 0.5,
    ) -> None:
        if not 0 <= min_score <= 1:
            raise ValueError("min_score must be between zero and one")
        self.client = client
        self.model_id = model_id
        self.fallback = fallback
        self.min_score = min_score

    async def classify(self, text: str) -> ClassificationResult:
        try:
            result = await self.client.classify(self.model_id, text)
            if result.score is not None and result.score < self.min_score:
                return ClassificationResult(
                    intent=Intent.OTHER,
                    score=result.score,
                    model_id=result.model_id,
                    latency_ms=result.latency_ms,
                    source="pioneer-abstained",
                )
            return result
        except PioneerError:
            if self.fallback is None:
                raise
            return await self.fallback.classify(text)


class RuleBasedIntentRouter:
    patterns: tuple[tuple[Intent, re.Pattern[str]], ...] = (
        (
            Intent.POSSIBLE_SCAM,
            re.compile(
                r"\b(bank|microsoft|support|remote|anydesk|teamviewer|password|code)\b",
                re.IGNORECASE,
            ),
        ),
        (
            Intent.DESTRUCTIVE_ACTION,
            re.compile(r"\b(delete|erase|remove all|factory reset)\b", re.IGNORECASE),
        ),
        (
            Intent.INCREASE_TEXT_SIZE,
            re.compile(
                r"\b(tiny|small|bigger|larger|read|writing|font|text)\b",
                re.IGNORECASE,
            ),
        ),
        (Intent.WIFI_HELP, re.compile(r"\b(wi.?fi|internet|online)\b", re.IGNORECASE)),
        (
            Intent.BLUETOOTH_HELP,
            re.compile(r"\b(bluetooth|headphones?|earbuds?|connect)\b", re.IGNORECASE),
        ),
        (Intent.VOLUME_HELP, re.compile(r"\b(hear|volume|ringer|silent)\b", re.IGNORECASE)),
        (
            Intent.NOTIFICATION_HELP,
            re.compile(r"\b(notification|pop.?ups?|alerts?|bothering)\b", re.IGNORECASE),
        ),
    )

    async def classify(self, text: str) -> ClassificationResult:
        for intent, pattern in self.patterns:
            if pattern.search(text):
                return ClassificationResult(
                    intent=intent,
                    score=None,
                    model_id="rule-based-demo",
                    latency_ms=0,
                    source="demo-rule-based",
                )
        return ClassificationResult(
            intent=Intent.OTHER,
            score=None,
            model_id="rule-based-demo",
            latency_ms=0,
            source="demo-rule-based",
        )
