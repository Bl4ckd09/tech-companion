from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    hai_api_key: str | None
    pioneer_api_key: str | None
    pioneer_model_id: str
    android_serial: str | None
    adb_path: str
    demo_mode: bool
    allow_provider_fallback: bool
    holo_model: str
    provider_timeout_seconds: float
    static_dir: Path
    state_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT_DIR / ".env")
        return cls(
            hai_api_key=os.getenv("HAI_API_KEY") or None,
            pioneer_api_key=os.getenv("PIONEER_API_KEY") or None,
            pioneer_model_id=os.getenv(
                "PIONEER_MODEL_ID", "fastino/gliner2-base-v1"
            ),
            android_serial=os.getenv("ANDROID_SERIAL") or None,
            adb_path=os.getenv("ADB_PATH", "adb"),
            demo_mode=_as_bool(os.getenv("TECH_COMPANION_DEMO_MODE")),
            allow_provider_fallback=_as_bool(
                os.getenv("ALLOW_PROVIDER_FALLBACK")
            ),
            holo_model=os.getenv("HOLO_MODEL", "holo3-1-35b-a3b"),
            provider_timeout_seconds=float(
                os.getenv("PROVIDER_TIMEOUT_SECONDS", "30")
            ),
            static_dir=ROOT_DIR / "app" / "static",
            state_dir=ROOT_DIR / ".state",
        )
