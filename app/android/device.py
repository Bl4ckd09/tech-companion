from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw, ImageFont

from app.android.adb import AdbClient, AdbError
from app.schemas import Action, ClickAction, DeviceStatus, DoneAction, SwipeAction


class AndroidDevice(Protocol):
    def status(self) -> DeviceStatus: ...
    def screenshot(self) -> bytes: ...
    def execute(self, action: Action) -> None: ...
    def tap_normalized(self, x: int, y: int) -> None: ...
    def launch_settings(self) -> None: ...
    def reset(self) -> None: ...


def normalized_to_pixels(
    x: int, y: int, width: int, height: int
) -> tuple[int, int]:
    return int(x / 1000 * width), int(y / 1000 * height)


class AdbAndroidDevice:
    def __init__(self, client: AdbClient) -> None:
        self.client = client

    def status(self) -> DeviceStatus:
        try:
            device = self.client.selected_device()
            width, height = self.client.screen_dimensions()
            return DeviceStatus(
                connected=True,
                serial=device.serial,
                model=device.model,
                screen_width=width,
                screen_height=height,
                detail="Android device is ready",
            )
        except AdbError as exc:
            return DeviceStatus(connected=False, detail=str(exc))

    def screenshot(self) -> bytes:
        return self.client.screenshot()

    def execute(self, action: Action) -> None:
        if isinstance(action, DoneAction):
            return
        screenshot = self.screenshot()
        width, height = self.client.screen_dimensions(screenshot)
        if isinstance(action, ClickAction):
            x, y = normalized_to_pixels(action.x, action.y, width, height)
            self.client.tap(x, y)
            return
        x1, y1 = normalized_to_pixels(action.x1, action.y1, width, height)
        x2, y2 = normalized_to_pixels(action.x2, action.y2, width, height)
        self.client.swipe(x1, y1, x2, y2, action.duration_ms)

    def tap_normalized(self, x: int, y: int) -> None:
        screenshot = self.screenshot()
        width, height = self.client.screen_dimensions(screenshot)
        pixel_x, pixel_y = normalized_to_pixels(x, y, width, height)
        self.client.tap(pixel_x, pixel_y)

    def launch_settings(self) -> None:
        self.client.launch_settings()

    def reset(self) -> None:
        self.client.reset_font_scale()
        self.client.launch_settings()


class DemoAndroidDevice:
    width = 540
    height = 1080

    def __init__(self) -> None:
        self.page = "settings"
        self.large_text = False

    def status(self) -> DeviceStatus:
        return DeviceStatus(
            connected=True,
            serial="demo-emulator",
            model="Pixel Demo",
            screen_width=self.width,
            screen_height=self.height,
            detail="Explicit demo device is ready",
            demo_mode=True,
        )

    def screenshot(self) -> bytes:
        image = Image.new("RGB", (self.width, self.height), "#f7f9ff")
        draw = ImageDraw.Draw(image)
        title_font = _font(34 if self.large_text else 28, bold=True)
        body_font = _font(26 if self.large_text else 21)
        small_font = _font(19 if self.large_text else 16)
        draw.rectangle((0, 0, self.width, 58), fill="#edf1fb")
        draw.text((24, 18), "10:32", fill="#172033", font=small_font)
        titles = {
            "settings": "Settings",
            "display": "Display",
            "display_size": "Display size and text",
            "complete": "Display size and text",
        }
        draw.text((32, 94), titles[self.page], fill="#172033", font=title_font)
        draw.rounded_rectangle((24, 152, 516, 218), 28, fill="#e8edf8")
        draw.text((54, 173), "Search settings", fill="#5c6578", font=small_font)
        if self.page == "settings":
            _row(draw, 276, "Network and internet", "Wi-Fi and mobile", body_font, small_font)
            _row(draw, 402, "Connected devices", "Bluetooth", body_font, small_font)
            _row(draw, 528, "Apps", "Recent apps", body_font, small_font)
            _row(draw, 654, "Display", "Dark theme, font size", body_font, small_font, accent=True)
            _row(draw, 780, "Sound and vibration", "Volume and alerts", body_font, small_font)
        elif self.page == "display":
            _row(draw, 278, "Brightness level", "Adaptive brightness", body_font, small_font)
            _row(draw, 434, "Lock screen", "Privacy and notifications", body_font, small_font)
            _row(draw, 590, "Display size and text", "Font size, display size", body_font, small_font, accent=True)
        else:
            draw.text((34, 286), "Font size", fill="#172033", font=body_font)
            draw.text((34, 330), "Preview text", fill="#5c6578", font=small_font)
            draw.line((70, 496, 470, 496), fill="#7f8aa3", width=8)
            thumb_x = 405 if self.large_text else 270
            draw.ellipse((thumb_x - 22, 474, thumb_x + 22, 518), fill="#2155d9")
            preview_font = _font(34 if self.large_text else 24, bold=True)
            preview = "Text is easier to read" if self.large_text else "Text preview"
            draw.multiline_text((34, 622), preview, fill="#172033", font=preview_font, spacing=12)
            if self.large_text:
                draw.rounded_rectangle((34, 802, 506, 908), 22, fill="#dff7ea")
                draw.text((58, 834), "Text size increased", fill="#155b3c", font=body_font)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def execute(self, action: Action) -> None:
        if isinstance(action, ClickAction):
            self._advance_click()
        elif isinstance(action, SwipeAction):
            self.page = "complete"
            self.large_text = True

    def tap_normalized(self, x: int, y: int) -> None:
        del x, y
        self._advance_click()

    def _advance_click(self) -> None:
        if self.page == "settings":
            self.page = "display"
        elif self.page == "display":
            self.page = "display_size"

    def launch_settings(self) -> None:
        self.page = "settings"

    def reset(self) -> None:
        self.page = "settings"
        self.large_text = False


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _row(
    draw: ImageDraw.ImageDraw,
    y: int,
    title: str,
    subtitle: str,
    body_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    *,
    accent: bool = False,
) -> None:
    if accent:
        draw.rounded_rectangle((20, y - 20, 520, y + 91), 20, fill="#e2eaff")
    draw.ellipse((34, y, 78, y + 44), fill="#cbd8fb" if accent else "#e0e5ef")
    draw.text((100, y - 2), title, fill="#172033", font=body_font)
    draw.text((100, y + 43), subtitle, fill="#5c6578", font=small_font)
