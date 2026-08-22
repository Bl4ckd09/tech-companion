from __future__ import annotations

from io import BytesIO

from PIL import Image

from app.android.adb import ConnectedDevice
from app.android.device import AdbAndroidDevice, DemoAndroidDevice
from app.schemas import ClickAction, SwipeAction


def png(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class FakeAdb:
    def __init__(self) -> None:
        self.image = png(100, 200)
        self.taps: list[tuple[int, int]] = []
        self.swipes: list[tuple[int, int, int, int, int]] = []

    def screenshot(self) -> bytes:
        return self.image

    def screen_dimensions(self, screenshot: bytes | None = None) -> tuple[int, int]:
        del screenshot
        return 100, 200

    def tap(self, x: int, y: int) -> None:
        self.taps.append((x, y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int) -> None:
        self.swipes.append((x1, y1, x2, y2, duration))

    def selected_device(self) -> ConnectedDevice:
        return ConnectedDevice("fake", "device", "Fake")

    def launch_settings(self) -> None:
        pass

    def reset_font_scale(self) -> None:
        pass


def test_adb_device_scales_click_against_screenshot() -> None:
    client = FakeAdb()
    device = AdbAndroidDevice(client)
    device.execute(ClickAction(element="Display", x=500, y=250))
    assert client.taps == [(50, 50)]


def test_adb_device_scales_swipe_against_screenshot() -> None:
    client = FakeAdb()
    device = AdbAndroidDevice(client)
    device.execute(
        SwipeAction(
            element="Font slider", x1=250, y1=500, x2=750, y2=500, duration_ms=400
        )
    )
    assert client.swipes == [(25, 100, 75, 100, 400)]


def test_demo_device_reaches_large_text_state() -> None:
    device = DemoAndroidDevice()
    device.execute(ClickAction(element="Display", x=500, y=600))
    device.execute(ClickAction(element="Display size and text", x=500, y=500))
    device.execute(
        SwipeAction(
            element="Font slider", x1=500, y1=500, x2=800, y2=500
        )
    )
    assert device.page == "complete"
    assert device.large_text is True
    with Image.open(BytesIO(device.screenshot())) as image:
        assert image.size == (540, 1080)
