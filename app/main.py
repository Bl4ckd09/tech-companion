from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from app.android.adb import AdbClient
from app.android.device import AdbAndroidDevice, DemoAndroidDevice
from app.config import Settings
from app.holo.agent import HoloObserver, HoloSession
from app.holo.client import HoloApiClient, HoloError
from app.holo.demo import DemoHoloSession
from app.holo.fallback import DeterministicHoloFallback
from app.holo.localizer import HoloLocalizer
from app.orchestrator import TechCompanion
from app.pioneer.client import PioneerClient, PioneerError
from app.pioneer.router import PioneerIntentRouter, RuleBasedIntentRouter
from app.schemas import (
    DeviceStatus,
    ExecuteActionRequest,
    ResetResponse,
    SessionResponse,
    StartSessionRequest,
)


class UnavailableIntentRouter:
    async def classify(self, text: str):
        del text
        raise PioneerError("PIONEER_API_KEY is not configured")


@dataclass(slots=True)
class Services:
    companion: TechCompanion
    holo_api: HoloApiClient | None = None
    pioneer_api: PioneerClient | None = None

    async def close(self) -> None:
        if self.holo_api is not None:
            await self.holo_api.close()
        if self.pioneer_api is not None:
            await self.pioneer_api.close()


def build_services(settings: Settings) -> Services:
    if settings.demo_mode:
        device = DemoAndroidDevice()
        router = RuleBasedIntentRouter()
        companion = TechCompanion(
            device=device,
            intent_router=router,
            visual_factory=lambda goal: DemoHoloSession(goal),
            visual_provider_name="demo-holo",
            android_provider_name="demo-emulator",
        )
        return Services(companion=companion)

    device = AdbAndroidDevice(
        AdbClient(
            adb_path=settings.adb_path,
            serial=settings.android_serial,
            timeout_seconds=settings.provider_timeout_seconds,
        )
    )
    pioneer_api = (
        PioneerClient(settings.pioneer_api_key, settings.provider_timeout_seconds)
        if settings.pioneer_api_key
        else None
    )
    fallback_router = RuleBasedIntentRouter() if settings.allow_provider_fallback else None
    router = (
        PioneerIntentRouter(
            pioneer_api,
            settings.pioneer_model_id,
            fallback=fallback_router,
        )
        if pioneer_api is not None
        else UnavailableIntentRouter()
    )
    holo_api = (
        HoloApiClient(
            settings.hai_api_key,
            settings.holo_model,
            settings.provider_timeout_seconds,
        )
        if settings.hai_api_key
        else None
    )

    def visual_factory(goal: str):
        if holo_api is None:
            raise HoloError("HAI_API_KEY is not configured")
        return HoloSession(holo_api, goal)

    localizer = HoloLocalizer(holo_api) if holo_api is not None else None
    companion = TechCompanion(
        device=device,
        intent_router=router,
        visual_factory=visual_factory,
        visual_provider_name=settings.holo_model,
        android_provider_name="adb",
        fallback=DeterministicHoloFallback(localizer) if localizer else None,
        screen_observer=HoloObserver(holo_api) if holo_api else None,
    )
    return Services(
        companion=companion,
        holo_api=holo_api,
        pioneer_api=pioneer_api,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = app_settings
        app.state.services = build_services(app_settings)
        yield
        await app.state.services.close()

    app = FastAPI(title="Tech Companion", version="0.1.0", lifespan=lifespan)

    def companion(request: Request) -> TechCompanion:
        return request.app.state.services.companion

    @app.get("/api/device", response_model=DeviceStatus)
    async def device_status(request: Request) -> DeviceStatus:
        return companion(request).device.status()

    @app.post("/api/demo/reset", response_model=ResetResponse)
    async def reset_demo(request: Request) -> ResetResponse:
        try:
            await companion(request).reset()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return ResetResponse(ok=True, detail="The Android demo returned to Settings")

    @app.post("/api/session", response_model=SessionResponse)
    async def start_session(
        payload: StartSessionRequest, request: Request
    ) -> SessionResponse:
        try:
            return await companion(request).start(payload.text)
        except (PioneerError, HoloError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/session/{session_id}/refresh", response_model=SessionResponse)
    async def refresh_session(session_id: str, request: Request) -> SessionResponse:
        try:
            return await companion(request).refresh(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/session/{session_id}/execute", response_model=SessionResponse)
    async def execute_session(
        session_id: str,
        payload: ExecuteActionRequest,
        request: Request,
    ) -> SessionResponse:
        try:
            return await companion(request).execute(session_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/screenshot")
    async def screenshot(
        request: Request,
        session_id: str = Query(min_length=1),
    ) -> Response:
        try:
            image = companion(request).screenshot(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(
            image,
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    app.mount(
        "/",
        StaticFiles(directory=app_settings.static_dir, html=True),
        name="static",
    )
    return app


app = create_app()
