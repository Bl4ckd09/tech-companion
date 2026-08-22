from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from io import BytesIO
from typing import Sequence

from PIL import Image


class AdbError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ConnectedDevice:
    serial: str
    state: str
    model: str | None = None


class AdbClient:
    def __init__(
        self,
        adb_path: str = "adb",
        serial: str | None = None,
        timeout_seconds: float = 15,
    ) -> None:
        self.adb_path = adb_path
        self.serial = serial
        self.timeout_seconds = timeout_seconds

    def _run(
        self,
        args: Sequence[str],
        *,
        binary: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes] | subprocess.CompletedProcess[str]:
        command = [self.adb_path]
        if self.serial:
            command.extend(["-s", self.serial])
        command.extend(args)
        try:
            return subprocess.run(
                command,
                check=check,
                capture_output=True,
                text=not binary,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise AdbError(f"ADB was not found at {self.adb_path!r}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AdbError("ADB did not respond before the timeout") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="replace") if binary else exc.stderr
            detail = (stderr or "ADB command failed").strip()
            raise AdbError(detail) from exc

    def devices(self) -> list[ConnectedDevice]:
        result = self._run(["devices", "-l"])
        devices: list[ConnectedDevice] = []
        for line in result.stdout.splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            serial = fields[0]
            state = fields[1] if len(fields) > 1 else "unknown"
            model = next(
                (
                    field.removeprefix("model:")
                    for field in fields[2:]
                    if field.startswith("model:")
                ),
                None,
            )
            devices.append(ConnectedDevice(serial=serial, state=state, model=model))
        return devices

    def selected_device(self) -> ConnectedDevice:
        devices = [device for device in self.devices() if device.state == "device"]
        if self.serial:
            match = next(
                (device for device in devices if device.serial == self.serial), None
            )
            if match is None:
                raise AdbError(f"Android device {self.serial!r} is not connected")
            return match
        if not devices:
            raise AdbError("No authorized Android device is connected")
        if len(devices) > 1:
            raise AdbError("Multiple Android devices are connected. Set ANDROID_SERIAL")
        return devices[0]

    def screenshot(self) -> bytes:
        self.selected_device()
        result = self._run(["exec-out", "screencap", "-p"], binary=True)
        image_bytes = result.stdout
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.verify()
        except Exception as exc:
            raise AdbError("ADB returned an invalid screenshot") from exc
        return image_bytes

    def screen_dimensions(self, screenshot: bytes | None = None) -> tuple[int, int]:
        if screenshot is not None:
            with Image.open(BytesIO(screenshot)) as image:
                return image.size
        result = self._run(["shell", "wm", "size"])
        matches = re.findall(r"(\d+)x(\d+)", result.stdout)
        if not matches:
            raise AdbError(f"Could not parse screen size from {result.stdout!r}")
        width, height = matches[-1]
        return int(width), int(height)

    def tap(self, x: int, y: int) -> None:
        self.selected_device()
        self._run(["shell", "input", "tap", str(x), str(y)])

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> None:
        self.selected_device()
        self._run(
            [
                "shell",
                "input",
                "swipe",
                str(x1),
                str(y1),
                str(x2),
                str(y2),
                str(duration_ms),
            ]
        )

    def back(self) -> None:
        self._run(["shell", "input", "keyevent", "KEYCODE_BACK"])

    def home(self) -> None:
        self._run(["shell", "input", "keyevent", "KEYCODE_HOME"])

    def launch_settings(self) -> None:
        self._run(["shell", "am", "start", "-a", "android.settings.SETTINGS"])

    def restart_settings(self) -> None:
        self._run(["shell", "am", "force-stop", "com.android.settings"])
        self.launch_settings()

    def reset_font_scale(self) -> None:
        self._run(["shell", "settings", "put", "system", "font_scale", "1.0"])
