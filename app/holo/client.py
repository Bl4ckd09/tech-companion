from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI


class HoloError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HoloResponse:
    content: str
    reasoning: str | None
    latency_ms: int


class HoloApiClient:
    def __init__(
        self,
        api_key: str,
        model: str = "holo3-1-35b-a3b",
        timeout_seconds: float = 30,
    ) -> None:
        if not api_key:
            raise ValueError("HAI_API_KEY is required")
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.hcompany.ai/v1/",
            timeout=timeout_seconds,
        )

    async def structured_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        temperature: float,
        enable_thinking: bool,
        reasoning_effort: str | None = None,
    ) -> HoloResponse:
        extra_body: dict[str, Any] = {
            "structured_outputs": {"json": schema},
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if reasoning_effort:
            extra_body["reasoning_effort"] = reasoning_effort
        started = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                extra_body=extra_body,
            )
        except Exception as exc:
            raise HoloError(f"Holo request failed: {exc}") from exc
        latency_ms = round((time.perf_counter() - started) * 1000)
        message = response.choices[0].message
        content = message.content
        if not content:
            raise HoloError("Holo returned an empty structured response")
        reasoning = getattr(message, "reasoning", None)
        return HoloResponse(content=content, reasoning=reasoning, latency_ms=latency_ms)

    async def close(self) -> None:
        await self.client.close()
