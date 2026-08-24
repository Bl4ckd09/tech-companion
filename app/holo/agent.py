from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.holo.client import HoloApiClient
from app.holo.prompts import render_system_prompt
from app.schemas import Action, ClickAction, DoneAction, SwipeAction


class ClickTool(BaseModel):
    tool_name: Literal["click"]
    element: str
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)


class SwipeTool(BaseModel):
    tool_name: Literal["swipe"]
    element: str
    x1: int = Field(ge=0, le=1000)
    y1: int = Field(ge=0, le=1000)
    x2: int = Field(ge=0, le=1000)
    y2: int = Field(ge=0, le=1000)
    duration_ms: int = Field(default=300, ge=50, le=5000)


class DoneTool(BaseModel):
    tool_name: Literal["done"]
    summary: str


class HoloStep(BaseModel):
    note: str | None = None
    thought: str
    tool_call: ClickTool | SwipeTool | DoneTool = Field(discriminator="tool_name")

    def to_action(self) -> Action:
        tool = self.tool_call
        if isinstance(tool, ClickTool):
            return ClickAction(element=tool.element, x=tool.x, y=tool.y)
        if isinstance(tool, SwipeTool):
            return SwipeAction(
                element=tool.element,
                x1=tool.x1,
                y1=tool.y1,
                x2=tool.x2,
                y2=tool.y2,
                duration_ms=tool.duration_ms,
            )
        return DoneAction(summary=tool.summary)

class ScreenObservation(BaseModel):
    screen_title: str
    current_app: str
    summary: str
    visible_controls: list[str] = Field(max_length=20)
    sensitive_screen: bool


class HoloObserver:
    def __init__(self, api: HoloApiClient) -> None:
        self.api = api
        self.last_latency_ms: int | None = None

    async def observe(
        self,
        image_bytes: bytes,
        user_request: str,
    ) -> ScreenObservation:
        schema = ScreenObservation.model_json_schema()
        data_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
        response = await self.api.structured_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Observe the current phone screen without proposing or executing "
                        "an action. Describe facts that help interpret the user's request. "
                        "Mark screens containing messages, account details, payment, "
                        "authentication, or personal records as sensitive."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {
                            "type": "text",
                            "text": f"User request: {user_request}",
                        },
                    ],
                },
            ],
            schema=schema,
            temperature=0.0,
            enable_thinking=False,
        )
        self.last_latency_ms = response.latency_ms
        return ScreenObservation.model_validate_json(response.content)


@dataclass(slots=True)
class HoloSession:
    api: HoloApiClient
    goal: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    step_number: int = 0
    last_latency_ms: int | None = None
    last_reasoning: str | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            schema = HoloStep.model_json_schema()
            self.messages.append(
                {
                    "role": "system",
                    "content": render_system_prompt(self.goal, schema),
                }
            )

    async def next_step(self, image_bytes: bytes) -> HoloStep:
        data_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<observation>\n"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "\n</observation>"},
                ],
            }
        )
        trim_to_last_images(self.messages, 3)
        schema = HoloStep.model_json_schema()
        response = await self.api.structured_completion(
            messages=self.messages,
            schema=schema,
            temperature=0.8,
            enable_thinking=True,
            reasoning_effort="medium",
        )
        step = HoloStep.model_validate_json(response.content)
        self.messages.append({"role": "assistant", "content": step.model_dump_json()})
        self.step_number += 1
        self.last_latency_ms = response.latency_ms
        self.last_reasoning = response.reasoning
        return step

    def record_result(self, action: Action, result: str) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": f'<tool_output tool="{action.type}">\n{result}\n</tool_output>',
            }
        )


def trim_to_last_images(messages: list[dict[str, Any]], keep: int = 3) -> None:
    seen = 0
    for message in reversed(messages):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        for chunk in content:
            if chunk.get("type") != "image_url":
                continue
            seen += 1
            if seen > keep:
                chunk.clear()
                chunk.update({"type": "text", "text": "[screenshot evicted]"})
