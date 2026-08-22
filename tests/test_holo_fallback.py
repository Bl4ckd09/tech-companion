from __future__ import annotations

from app.holo.fallback import DeterministicHoloFallback
from app.holo.localizer import LocalizedPoint
from app.schemas import ClickAction, SwipeAction


class FakeLocalizer:
    async def localize(self, image_bytes: bytes, target: str):
        del image_bytes, target
        return LocalizedPoint(x=820, y=500), 21


async def test_fallback_localizes_fixed_click_target() -> None:
    fallback = DeterministicHoloFallback(FakeLocalizer())
    action, instruction, latency = await fallback.action_for(b"image", 0)
    assert action == ClickAction(
        element="the display settings row containing font size controls",
        x=820,
        y=500,
    )
    assert instruction == "Tap the display settings row."
    assert latency == 21


async def test_fallback_clamps_swipe_end_to_normalized_range() -> None:
    fallback = DeterministicHoloFallback(FakeLocalizer())
    action, _, _ = await fallback.action_for(b"image", 2)
    assert isinstance(action, SwipeAction)
    assert action.x1 == 820
    assert action.x2 == 1000
