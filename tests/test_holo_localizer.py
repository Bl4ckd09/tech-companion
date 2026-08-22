from __future__ import annotations

from app.holo.client import HoloResponse
from app.holo.localizer import HoloLocalizer


class FakeHoloApi:
    def __init__(self) -> None:
        self.request = None

    async def structured_completion(self, **kwargs):
        self.request = kwargs
        return HoloResponse(
            content='{"x":487,"y":365}',
            reasoning=None,
            latency_ms=42,
        )


async def test_localizer_uses_grounding_contract() -> None:
    api = FakeHoloApi()
    localizer = HoloLocalizer(api)
    point, latency_ms = await localizer.localize(b"png-bytes", "Display")
    assert point.x == 487
    assert point.y == 365
    assert latency_ms == 42
    assert api.request["temperature"] == 0.0
    assert api.request["enable_thinking"] is False
    content = api.request["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "Display" in content[1]["text"]
