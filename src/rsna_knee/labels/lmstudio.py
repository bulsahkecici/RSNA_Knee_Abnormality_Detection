"""LM Studio OpenAI-compatible client. Local only — never a cloud OpenAI call."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from rsna_knee.errors import NeedsRuntimeError, QuarantineError


def _assert_local_base(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise NeedsRuntimeError(
            f"LM Studio base URL must be local, got {url}",
            action_tr="RSNA_LMSTUDIO_BASE_URL değerini http://127.0.0.1:1234/v1 yapın. Cloud OpenAI kullanmayın.",
        )


@dataclass
class ModelInfo:
    id: str
    raw: dict[str, Any]


class LMStudioClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        api_key: str | None = None,
        timeout: float = 180.0,
        transport: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        _assert_local_base(self.base_url)
        self.api_key = api_key or os.getenv("RSNA_LMSTUDIO_API_KEY") or "lm-studio"
        self.timeout = timeout
        self._client = transport
        self._owned = transport is None

    def _http(self) -> httpx.Client:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._owned and self._client is not None:
            self._client.close()

    def list_models(self) -> list[ModelInfo]:
        try:
            resp = self._http().get("/models")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise NeedsRuntimeError(
                f"LM Studio unreachable at {self.base_url}: {exc}",
                action_tr="LM Studio Developer sunucusunu 127.0.0.1:1234 üzerinde başlatın.",
            ) from exc
        data = resp.json().get("data", [])
        return [ModelInfo(id=item["id"], raw=item) for item in data]

    def resolve_model(self, preferred_id: str, available: list[ModelInfo] | None = None) -> ModelInfo:
        available = available if available is not None else self.list_models()
        ids = [m.id for m in available]
        if not ids:
            raise NeedsRuntimeError(
                "LM Studio has no loaded models.",
                action_tr="Qwen3.8-27B MLX 4-bit modelini LM Studio'da yükleyin.",
            )
        for model in available:
            if model.id == preferred_id:
                return model
        raise NeedsRuntimeError(
            f"Preferred model {preferred_id!r} not loaded. Available: {ids}",
            action_tr="Config'teki RSNA_LMSTUDIO_MODEL_ID değerini /v1/models listesinden seçin; sessizce tahmin etmeyin.",
        )

    def chat_completions(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
        json_object: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "report_labels",
                    "strict": True,
                    "schema": json_schema,
                },
            }
        elif json_object:
            payload["response_format"] = {"type": "json_object"}
        if extra:
            payload.update(extra)
        try:
            resp = self._http().post("/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise QuarantineError(f"chat/completions failed: {exc}") from exc
        return resp.json()

    def probe_json_schema(self, model: str) -> str:
        """Return json_schema, json_object, or none. Never fake labels on failure."""
        tiny_schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        try:
            out = self.chat_completions(
                model=model,
                messages=[{"role": "user", "content": 'Return {"ok": true}'}],
                json_schema=tiny_schema,
                max_tokens=256,
                temperature=0.0,
            )
            content = out["choices"][0]["message"].get("content") or ""
            if "ok" in content:
                return "json_schema"
        except Exception:
            pass
        try:
            out = self.chat_completions(
                model=model,
                messages=[{"role": "user", "content": 'Return {"ok": true}'}],
                json_object=True,
                max_tokens=256,
                temperature=0.0,
            )
            content = out["choices"][0]["message"].get("content") or ""
            if "ok" in content:
                return "json_object"
        except Exception:
            pass
        return "none"
