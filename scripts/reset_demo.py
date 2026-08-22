from __future__ import annotations

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
        client.reset_font_scale()
        client.launch_settings()
    except AdbError as exc:
        print(f"Reset failed: {exc}")
        return 1
    print(f"Reset {device.serial} to font scale 1.0 and opened Settings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
