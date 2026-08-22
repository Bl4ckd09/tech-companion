from __future__ import annotations

from dataclasses import dataclass

from app.holo.localizer import HoloLocalizer
from app.schemas import Action, ClickAction, DoneAction, SwipeAction


@dataclass(frozen=True, slots=True)
class PlannedStep:
    kind: str
    target: str
    instruction: str


INCREASE_TEXT_SIZE_PLAN = (
    PlannedStep(
        "click",
        "the display settings row containing font size controls",
        "Tap the display settings row.",
    ),
    PlannedStep(
        "click",
        "the Display size and text settings row",
        "Tap Display size and text.",
    ),
    PlannedStep(
        "swipe",
        "the circular thumb on the Font size slider",
        "Move the Font size slider to the right.",
    ),
    PlannedStep("done", "", "The text size is larger."),
)


class DeterministicHoloFallback:
    def __init__(self, localizer: HoloLocalizer) -> None:
        self.localizer = localizer

    async def action_for(
        self, image_bytes: bytes, index: int
    ) -> tuple[Action, str, int | None]:
        step = INCREASE_TEXT_SIZE_PLAN[min(index, len(INCREASE_TEXT_SIZE_PLAN) - 1)]
        if step.kind == "done":
            return DoneAction(summary=step.instruction), step.instruction, None
        point, latency_ms = await self.localizer.localize(image_bytes, step.target)
        if step.kind == "click":
            action: Action = ClickAction(
                element=step.target, x=point.x, y=point.y
            )
        else:
            action = SwipeAction(
                element=step.target,
                x1=point.x,
                y1=point.y,
                x2=min(1000, point.x + 260),
                y2=point.y,
                duration_ms=350,
            )
        return action, step.instruction, latency_ms
