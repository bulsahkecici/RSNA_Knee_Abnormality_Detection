"""LM Studio OpenAI-compatible client. Local only — never a cloud OpenAI call."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from rsna_knee.errors import NeedsRuntimeError, QuarantineError
from rsna_knee.labels.json_select import JsonSelectionError, select_json_message

PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


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


@dataclass
class JsonProbe:
    """Capability probe. mode none means no labels were written."""

    mode: str
    http_error: str | None = None
    empty_reason: str | None = None
    response_field: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "http_error": self.http_error,
            "empty_reason": self.empty_reason,
            "response_field": self.response_field,
        }


def http_error_reason(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        body = (exc.response.text or "").strip().replace("\n", " ")[:500]
        if body:
            return f"http_{exc.response.status_code}: {body}"
        return f"http_{exc.response.status_code}"
    return f"http_error: {exc}"


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
        self._probe_field: str | None = None

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
            raise QuarantineError(http_error_reason(exc)) from exc
        return resp.json()

    def probe_json_schema(
        self,
        model: str,
        *,
        allow_reasoning_json: bool = False,
        thinking: str | None = None,
    ) -> JsonProbe:
        """Prefer json_schema. Try json_object only when that selection fails."""
        messages = [{"role": "user", "content": 'Return {"ok": true}'}]
        schema_http, schema_empty = self._probe_once(
            model=model,
            messages=messages,
            json_schema=PROBE_SCHEMA,
            allow_reasoning_json=allow_reasoning_json,
            extra={"thinking": thinking} if thinking else None,
        )
        if schema_http is None and schema_empty is None:
            return JsonProbe(mode="json_schema", response_field=self._probe_field)
        object_http, object_empty = self._probe_once(
            model=model,
            messages=messages,
            json_object=True,
            allow_reasoning_json=allow_reasoning_json,
        )
        if object_http is None and object_empty is None:
            return JsonProbe(
                mode="json_object",
                http_error=schema_http,
                empty_reason=schema_empty,
                response_field=self._probe_field,
            )
        return JsonProbe(
            mode="none",
            http_error="; ".join(part for part in (schema_http, object_http) if part) or None,
            empty_reason="; ".join(part for part in (schema_empty, object_empty) if part) or None,
        )

    def _probe_once(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        allow_reasoning_json: bool,
        json_schema: dict[str, Any] | None = None,
        json_object: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> tuple[str | None, str | None]:
        self._probe_field = None
        try:
            out = self.chat_completions(
                model=model,
                messages=messages,
                json_schema=json_schema,
                json_object=json_object,
                max_tokens=256,
                temperature=0.0,
                extra=extra,
            )
        except QuarantineError as exc:
            return str(exc), None
        try:
            choice = out["choices"][0]
            message = choice.get("message") or {}
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError):
            return None, "malformed_response"
        try:
            selected = select_json_message(
                message,
                PROBE_SCHEMA,
                allow_reasoning_json=allow_reasoning_json,
                finish_reason=finish_reason,
            )
        except JsonSelectionError as exc:
            return None, str(exc)
        self._probe_field = selected.field
        return None, None
