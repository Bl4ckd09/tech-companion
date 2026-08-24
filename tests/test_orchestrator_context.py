from __future__ import annotations

from app.android.device import DemoAndroidDevice
from app.holo.agent import ScreenObservation
from app.holo.demo import DemoHoloSession
from app.orchestrator import TechCompanion
from app.pioneer.client import ClassificationResult
from app.pioneer.intents import Intent


def classification(intent: Intent, score: float = 0.9) -> ClassificationResult:
    return ClassificationResult(
        intent=intent,
        score=score,
        model_id="tuned-pioneer",
        latency_ms=12,
    )


class ContextualRouter:
    def __init__(self) -> None:
        self.requests: list[str] = []

    async def classify(self, text: str) -> ClassificationResult:
        self.requests.append(text)
        if "Screen title: Settings" in text:
            return classification(Intent.INCREASE_TEXT_SIZE)
        return classification(Intent.OTHER)


class ScamRouter:
    async def classify(self, text: str) -> ClassificationResult:
        del text
        return classification(Intent.POSSIBLE_SCAM)


class FakeObserver:
    last_latency_ms = 24

    def __init__(self) -> None:
        self.calls = 0

    async def observe(
        self,
        image_bytes: bytes,
        user_request: str,
    ) -> ScreenObservation:
        del image_bytes, user_request
        self.calls += 1
        return ScreenObservation(
            screen_title="Settings",
            current_app="Android Settings",
            summary="The main Settings list is visible with Display highlighted.",
            visible_controls=["Display", "Sound and vibration"],
            sensitive_screen=False,
        )


def companion(router: object, observer: FakeObserver) -> TechCompanion:
    return TechCompanion(
        device=DemoAndroidDevice(),
        intent_router=router,
        visual_factory=lambda goal: DemoHoloSession(goal),
        visual_provider_name="demo-holo",
        android_provider_name="demo-emulator",
        screen_observer=observer,
    )


async def test_screen_observation_refines_an_ambiguous_request() -> None:
    router = ContextualRouter()
    observer = FakeObserver()

    response = await companion(router, observer).start("Make this easier to see")

    assert len(router.requests) == 2
    assert observer.calls == 1
    assert response.intent == Intent.INCREASE_TEXT_SIZE.value
    assert response.status == "guidance"
    assert response.action is not None
    assert response.diagnostics["screen_title"] == "Settings"


async def test_risk_gate_runs_before_screen_observation() -> None:
    observer = FakeObserver()

    response = await companion(ScamRouter(), observer).start(
        "Support wants remote access"
    )

    assert response.status == "blocked"
    assert observer.calls == 0
    assert response.screenshot_url is None
