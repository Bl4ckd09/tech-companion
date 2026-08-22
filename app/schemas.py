from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


class ClickAction(BaseModel):
    type: Literal["click"] = "click"
    element: str
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)


class SwipeAction(BaseModel):
    type: Literal["swipe"] = "swipe"
    element: str
    x1: int = Field(ge=0, le=1000)
    y1: int = Field(ge=0, le=1000)
    x2: int = Field(ge=0, le=1000)
    y2: int = Field(ge=0, le=1000)
    duration_ms: int = Field(default=300, ge=50, le=5000)


class DoneAction(BaseModel):
    type: Literal["done"] = "done"
    summary: str


Action = Annotated[
    ClickAction | SwipeAction | DoneAction,
    Field(discriminator="type"),
]


class StartSessionRequest(BaseModel):
    text: str = Field(min_length=2, max_length=500)


class ExecuteActionRequest(BaseModel):
    mode: Literal["assistant", "manual"] = "assistant"
    x: int | None = Field(default=None, ge=0, le=1000)
    y: int | None = Field(default=None, ge=0, le=1000)

    @model_validator(mode="after")
    def require_manual_coordinates(self) -> "ExecuteActionRequest":
        if self.mode == "manual" and (self.x is None or self.y is None):
            raise ValueError("Manual mode requires normalized x and y coordinates")
        return self


class DeviceStatus(BaseModel):
    connected: bool
    serial: str | None = None
    model: str | None = None
    screen_width: int | None = None
    screen_height: int | None = None
    detail: str
    demo_mode: bool = False


class ProviderStatus(BaseModel):
    intent_router: str
    visual_agent: str
    android: str


class SessionResponse(BaseModel):
    session_id: str
    status: Literal["guidance", "blocked", "complete", "unsupported", "error"]
    user_text: str
    intent: str
    risk: Literal["low", "confirm", "high", "unknown"]
    understood_as: str
    instruction: str
    action: Action | None = None
    screenshot_url: str | None = None
    providers: ProviderStatus
    fallback_used: bool = False
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ResetResponse(BaseModel):
    ok: bool
    detail: str
