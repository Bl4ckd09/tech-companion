from __future__ import annotations

import argparse
import asyncio
from io import BytesIO
from pathlib import Path

from PIL import Image

from app.android.adb import AdbClient, AdbError
from app.android.device import normalized_to_pixels
from app.config import Settings
from app.holo.agent import HoloSession
from app.holo.client import HoloApiClient, HoloError
from app.holo.localizer import HoloLocalizer

_SETTINGS_READY_TIMEOUT_SECONDS = 10.0
_SETTINGS_READY_POLL_SECONDS = 0.2
_DARK_PIXEL_MAX_LUMINANCE = 95
_MIN_SETTINGS_LIST_DARK_PIXEL_RATIO = 0.01


def _is_settings_list_screenshot(screenshot: bytes) -> bool:
    with Image.open(BytesIO(screenshot)) as image:
        grayscale = image.convert("L")
    width, height = grayscale.size
    list_region = grayscale.crop(
        (width // 20, height // 4, width - width // 20, height - height // 12)
    )
    dark_pixels = sum(
        list_region.histogram()[: _DARK_PIXEL_MAX_LUMINANCE + 1]
    )
    return dark_pixels / (list_region.width * list_region.height) >= (
        _MIN_SETTINGS_LIST_DARK_PIXEL_RATIO
    )


async def _wait_for_settings_list(
    adb: AdbClient,
    timeout_seconds: float = _SETTINGS_READY_TIMEOUT_SECONDS,
) -> bytes:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while True:
        screenshot = adb.screenshot()
        if _is_settings_list_screenshot(screenshot):
            return screenshot
        remaining_seconds = deadline - loop.time()
        if remaining_seconds <= 0:
            raise AdbError(
                "Settings list readiness timed out after "
                f"{timeout_seconds:g} seconds. The rendered list never appeared"
            )
        await asyncio.sleep(min(_SETTINGS_READY_POLL_SECONDS, remaining_seconds))


async def run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if not settings.hai_api_key:
        print("Holo check failed: HAI_API_KEY is not configured")
        return 1
    adb = AdbClient(
        adb_path=settings.adb_path,
        serial=settings.android_serial,
        timeout_seconds=settings.provider_timeout_seconds,
    )
    api = HoloApiClient(
        settings.hai_api_key,
        settings.holo_model,
        settings.provider_timeout_seconds,
    )
    output_dir = Path("artifacts/holo-smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        adb.selected_device()
        if args.agent:
            adb.launch_settings()
            screenshot = await _wait_for_settings_list(adb)
            session = HoloSession(api, "Increase the Android system font size")
            step = await session.next_step(screenshot)
            print(step.model_dump_json(indent=2))
            print(f"Latency: {session.last_latency_ms} ms")
            return 0

        localizer = HoloLocalizer(api)
        for attempt in range(1, args.attempts + 1):
            adb.restart_settings()
            await _wait_for_settings_list(adb)
            await asyncio.sleep(0.5)
            adb.swipe(540, 2100, 540, 700, 600)
            await asyncio.sleep(1.0)
            screenshot = adb.screenshot()
            width, height = adb.screen_dimensions(screenshot)
            point, latency_ms = await localizer.localize(
                screenshot, "the display settings row containing font size controls"
            )
            pixel_x, pixel_y = normalized_to_pixels(
                point.x, point.y, width, height
            )
            before = output_dir / f"attempt-{attempt}-before.png"
            before.write_bytes(screenshot)
            print(
                f"Attempt {attempt}: normalized=({point.x}, {point.y}) "
                f"pixels=({pixel_x}, {pixel_y}) latency={latency_ms} ms"
            )
            if args.tap:
                adb.tap(pixel_x, pixel_y)
                await asyncio.sleep(5.0)
                after = output_dir / f"attempt-{attempt}-after.png"
                after.write_bytes(adb.screenshot())
                print(f"Saved post-tap screenshot to {after}")
    except (AdbError, HoloError, ValueError) as exc:
        print(f"Holo check failed: {exc}")
        return 1
    finally:
        await api.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--tap", action="store_true")
    parser.add_argument("--agent", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
