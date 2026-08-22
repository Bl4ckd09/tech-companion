from __future__ import annotations

import subprocess

import pytest

from app.android.adb import AdbClient, AdbError


class FakeAdbClient(AdbClient):
    def __init__(self, output: str, serial: str | None = None) -> None:
        super().__init__(serial=serial)
        self.output = output

    def _run(self, args, *, binary=False, check=True):
        del args, binary, check
        return subprocess.CompletedProcess([], 0, stdout=self.output, stderr="")


def test_devices_parse_serial_state_and_model() -> None:
    client = FakeAdbClient(
        "List of devices attached\n"
        "emulator-5554 device product:sdk model:sdk_gphone64_x86_64 transport_id:1\n"
    )
    devices = client.devices()
    assert len(devices) == 1
    assert devices[0].serial == "emulator-5554"
    assert devices[0].state == "device"
    assert devices[0].model == "sdk_gphone64_x86_64"


def test_selected_device_requires_serial_when_multiple_connected() -> None:
    client = FakeAdbClient(
        "List of devices attached\nfirst device model:one\nsecond device model:two\n"
    )
    with pytest.raises(AdbError, match="Multiple Android devices"):
        client.selected_device()


def test_selected_device_honors_configured_serial() -> None:
    client = FakeAdbClient(
        "List of devices attached\nfirst device model:one\nsecond device model:two\n",
        serial="second",
    )
    assert client.selected_device().model == "two"
