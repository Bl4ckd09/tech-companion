from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Protocol

from app.android.device import AndroidDevice
from app.holo.fallback import DeterministicHoloFallback
from app.pioneer.client import ClassificationResult
from app.pioneer.intents import INTENT_CONFIG, IntentConfig
from app.pioneer.router import IntentRouter
from app.schemas import (
    Action,
    ClickAction,
    DoneAction,
    ExecuteActionRequest,
    ProviderStatus,
    SessionResponse,
    SwipeAction,
)


class VisualSession(Protocol):
    step_number: int
    last_latency_ms: int | None
    last_reasoning: str | None

    async def next_step(self, image_bytes: bytes): ...
    def record_result(self, action: Action, result: str) -> None: ...


@dataclass(slots=True)
class CompanionSession:
    session_id: str
    user_text: str
    classification: ClassificationResult
    config: IntentConfig
    visual: VisualSession | None = None
    screenshot: bytes | None = None
    current_action: Action | None = None
    status: str = "guidance"
    instruction: str = ""
    fallback_used: bool = False
    fallback_index: int = 0
    visual_latency_ms: int | None = None
    started_at: float = field(default_factory=time.time)


class TechCompanion:
    def __init__(
        self,
        *,
        device: AndroidDevice,
        intent_router: IntentRouter,
        visual_factory: Callable[[str], VisualSession],
        visual_provider_name: str,
        android_provider_name: str,
        fallback: DeterministicHoloFallback | None = None,
    ) -> None:
        self.device = device
        self.intent_router = intent_router
        self.visual_factory = visual_factory
        self.visual_provider_name = visual_provider_name
        self.android_provider_name = android_provider_name
        self.fallback = fallback
        self.sessions: dict[str, CompanionSession] = {}
        self.lock = asyncio.Lock()

    async def start(self, user_text: str) -> SessionResponse:
        classification = await self.intent_router.classify(user_text)
        config = INTENT_CONFIG[classification.intent]
        session = CompanionSession(
            session_id=uuid.uuid4().hex,
            user_text=user_text,
            classification=classification,
            config=config,
        )
        self.sessions[session.session_id] = session

        if config.risk == "high":
            session.status = "blocked"
            session.instruction = (
                "Do not continue. A legitimate bank or support service should not "
                "request remote control of your phone."
            )
            return self._response(session)
        if config.risk == "confirm":
            session.status = "blocked"
            session.instruction = (
                "This action can permanently remove data. Ask a trusted person to "
                "review it before continuing."
            )
            return self._response(session)
        if config.goal is None:
            session.status = "unsupported"
            session.instruction = (
                "This prototype currently supports making Android text larger."
            )
            return self._response(session)

        status = await asyncio.to_thread(self.device.status)
        if not status.connected:
            session.status = "error"
            session.instruction = status.detail
            return self._response(session)

        await asyncio.to_thread(self.device.launch_settings)
        session.visual = self.visual_factory(config.goal)
        session.screenshot = await asyncio.to_thread(self.device.screenshot)
        await self._plan_next(session)
        return self._response(session)

    async def execute(
        self, session_id: str, request: ExecuteActionRequest
    ) -> SessionResponse:
        async with self.lock:
            session = self._session(session_id)
            if session.status != "guidance" or session.current_action is None:
                return self._response(session)
            action = session.current_action
            if isinstance(action, DoneAction):
                session.status = "complete"
                session.instruction = action.summary
                return self._response(session)

            if request.mode == "manual":
                await asyncio.to_thread(
                    self.device.tap_normalized, request.x, request.y
                )
                result = "The user tapped the mirrored screen."
            else:
                await asyncio.to_thread(self.device.execute, action)
                result = f"The {action.type} action completed."
            if session.visual is not None:
                session.visual.record_result(action, result)
            if session.fallback_used:
                session.fallback_index += 1
            await asyncio.sleep(0.35)
            session.screenshot = await asyncio.to_thread(self.device.screenshot)
            await self._plan_next(session)
            return self._response(session)

    async def refresh(self, session_id: str) -> SessionResponse:
        async with self.lock:
            session = self._session(session_id)
            if session.status != "guidance":
                return self._response(session)
            session.screenshot = await asyncio.to_thread(self.device.screenshot)
            if session.current_action is not None and session.visual is not None:
                session.visual.record_result(
                    session.current_action,
                    "The user completed the previous step on the Android device.",
                )
            await self._plan_next(session)
            return self._response(session)

    async def reset(self) -> None:
        async with self.lock:
            await asyncio.to_thread(self.device.reset)
            self.sessions.clear()

    def screenshot(self, session_id: str) -> bytes:
        session = self._session(session_id)
        if session.screenshot is None:
            raise KeyError("This session has no screenshot")
        return session.screenshot

    async def _plan_next(self, session: CompanionSession) -> None:
        if session.visual is None or session.screenshot is None:
            raise RuntimeError("The visual session is not ready")
        last_error: Exception | None = None
        for _ in range(2):
            try:
                step = await session.visual.next_step(session.screenshot)
                action = step.to_action()
                session.current_action = action
                session.visual_latency_ms = session.visual.last_latency_ms
                session.instruction = instruction_for(action)
                session.status = "complete" if isinstance(action, DoneAction) else "guidance"
                return
            except Exception as exc:
                last_error = exc
        if self.fallback is not None:
            action, instruction, latency_ms = await self.fallback.action_for(
                session.screenshot, session.fallback_index
            )
            session.current_action = action
            session.instruction = instruction
            session.visual_latency_ms = latency_ms
            session.fallback_used = True
            session.status = "complete" if isinstance(action, DoneAction) else "guidance"
            return
        session.status = "error"
        session.current_action = None
        session.instruction = f"The visual guide could not find the next step. {last_error}"

    def _session(self, session_id: str) -> CompanionSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown session {session_id}") from exc

    def _response(self, session: CompanionSession) -> SessionResponse:
        diagnostics = {
            "intent_model": session.classification.model_id,
            "intent_score": session.classification.score,
            "intent_latency_ms": session.classification.latency_ms,
            "visual_latency_ms": session.visual_latency_ms,
            "visual_steps": session.visual.step_number if session.visual else 0,
            "fallback_index": session.fallback_index,
        }
        if session.visual and session.visual.last_reasoning:
            diagnostics["visual_reasoning"] = session.visual.last_reasoning
        return SessionResponse(
            session_id=session.session_id,
            status=session.status,
            user_text=session.user_text,
            intent=session.classification.intent.value,
            risk=session.config.risk,
            understood_as=session.config.understood_as,
            instruction=session.instruction,
            action=session.current_action,
            screenshot_url=(
                f"/api/screenshot?session_id={session.session_id}&v={time.time_ns()}"
                if session.screenshot is not None
                else None
            ),
            providers=ProviderStatus(
                intent_router=session.classification.source,
                visual_agent=self.visual_provider_name,
                android=self.android_provider_name,
            ),
            fallback_used=session.fallback_used,
            diagnostics=diagnostics,
        )


def instruction_for(action: Action) -> str:
    if isinstance(action, ClickAction):
        return f"Tap {action.element}."
    if isinstance(action, SwipeAction):
        return f"Move {action.element} to the right."
    return action.summary
