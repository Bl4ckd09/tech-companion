from __future__ import annotations

from dataclasses import dataclass

from app.holo.agent import ClickTool, DoneTool, HoloStep, SwipeTool


@dataclass(slots=True)
class DemoHoloSession:
    goal: str
    step_number: int = 0
    last_latency_ms: int = 18
    last_reasoning: str | None = "Demo mode uses a disclosed fixed response sequence."

    async def next_step(self, image_bytes: bytes) -> HoloStep:
        del image_bytes
        steps = [
            HoloStep(
                thought="Display contains the text-size controls.",
                tool_call=ClickTool(
                    tool_name="click", element="Display", x=480, y=626
                ),
            ),
            HoloStep(
                thought="Display size and text opens the font controls.",
                tool_call=ClickTool(
                    tool_name="click",
                    element="Display size and text",
                    x=500,
                    y=565,
                ),
            ),
            HoloStep(
                thought="Moving the font slider right increases text size.",
                tool_call=SwipeTool(
                    tool_name="swipe",
                    element="Font size slider",
                    x1=500,
                    y1=460,
                    x2=760,
                    y2=460,
                    duration_ms=350,
                ),
            ),
            HoloStep(
                thought="The preview confirms that text is larger.",
                tool_call=DoneTool(
                    tool_name="done", summary="The Android text size is larger."
                ),
            ),
        ]
        step = steps[min(self.step_number, len(steps) - 1)]
        self.step_number += 1
        return step

    def record_result(self, action: object, result: str) -> None:
        del action, result
