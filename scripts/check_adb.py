from __future__ import annotations

from pathlib import Path

from app.android.adb import AdbClient, AdbError
from app.config import Settings


def main() -> int:
    settings = Settings.from_env()
    client = AdbClient(
        adb_path=settings.adb_path,
        serial=settings.android_serial,
        timeout_seconds=settings.provider_timeout_seconds,
    )
    try:
        device = client.selected_device()
        client.launch_settings()
        screenshot = client.screenshot()
        width, height = client.screen_dimensions(screenshot)
    except AdbError as exc:
        print(f"ADB check failed: {exc}")
        return 1
    output = Path("artifacts/adb-settings.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(screenshot)
    print(f"Device: {device.serial} ({device.model or 'unknown model'})")
    print(f"Screenshot: {width}x{height} PNG at {output}")
    print("ADB screenshot and Settings launch work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
