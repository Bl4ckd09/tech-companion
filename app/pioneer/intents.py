from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Intent(StrEnum):
    INCREASE_TEXT_SIZE = "increase_text_size"
    WIFI_HELP = "wifi_help"
    BLUETOOTH_HELP = "bluetooth_help"
    VOLUME_HELP = "volume_help"
    NOTIFICATION_HELP = "notification_help"
    POSSIBLE_SCAM = "possible_scam"
    DESTRUCTIVE_ACTION = "destructive_action"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class IntentConfig:
    label: Intent
    risk: str
    understood_as: str
    goal: str | None


INTENT_CONFIG: dict[Intent, IntentConfig] = {
    Intent.INCREASE_TEXT_SIZE: IntentConfig(
        Intent.INCREASE_TEXT_SIZE,
        "low",
        "Make text larger",
        "Increase the Android system font size",
    ),
    Intent.WIFI_HELP: IntentConfig(
        Intent.WIFI_HELP, "low", "Fix an internet connection", None
    ),
    Intent.BLUETOOTH_HELP: IntentConfig(
        Intent.BLUETOOTH_HELP, "low", "Connect a Bluetooth device", None
    ),
    Intent.VOLUME_HELP: IntentConfig(
        Intent.VOLUME_HELP, "low", "Make calls easier to hear", None
    ),
    Intent.NOTIFICATION_HELP: IntentConfig(
        Intent.NOTIFICATION_HELP, "low", "Manage unwanted notifications", None
    ),
    Intent.POSSIBLE_SCAM: IntentConfig(
        Intent.POSSIBLE_SCAM, "high", "Check a suspicious request", None
    ),
    Intent.DESTRUCTIVE_ACTION: IntentConfig(
        Intent.DESTRUCTIVE_ACTION, "confirm", "Review a permanent action", None
    ),
    Intent.OTHER: IntentConfig(
        Intent.OTHER, "unknown", "Understand the problem", None
    ),
}

INTENT_LABELS = [intent.value for intent in Intent]
