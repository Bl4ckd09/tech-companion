from __future__ import annotations

import base64

from pydantic import BaseModel, Field

from app.holo.client import HoloApiClient


class LocalizedPoint(BaseModel):
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)


class HoloLocalizer:
    def __init__(self, api: HoloApiClient) -> None:
        self.api = api

    async def localize(
        self, image_bytes: bytes, target: str
    ) -> tuple[LocalizedPoint, int]:
        schema = LocalizedPoint.model_json_schema()
        prompt = (
            "Localize an element on the GUI image according to the provided target "
            "and output a click position.\n"
            f"You must output valid JSON following this schema: {schema}\n"
            f"Your target is:\n{target}"
        )
        data_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
        response = await self.api.structured_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            schema=schema,
            temperature=0.0,
            enable_thinking=False,
        )
        return LocalizedPoint.model_validate_json(response.content), response.latency_ms
