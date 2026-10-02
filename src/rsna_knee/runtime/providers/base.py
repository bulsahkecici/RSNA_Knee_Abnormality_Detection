"""Provider capability contract. No fake allocate_gpu wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ProviderCapabilities:
    name: str
    supported_gpu_types: list[str]
    request: bool
    start: bool
    poll: bool
    cancel: bool
    remote_execution: bool
    upload_download: str
    auth_flow: str
    availability_status: str
    automatic_acquisition: bool
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "supported_gpu_types": self.supported_gpu_types,
            "request": self.request,
            "start": self.start,
            "poll": self.poll,
            "cancel": self.cancel,
            "remote_execution": self.remote_execution,
            "upload_download": self.upload_download,
            "auth_flow": self.auth_flow,
            "availability_status": self.availability_status,
            "automatic_acquisition": self.automatic_acquisition,
            "notes": self.notes,
        }


class RuntimeProvider(Protocol):
    name: str

    def capabilities(self) -> ProviderCapabilities: ...

    def request_gpu(self, gpu_type: str, **kwargs: Any) -> dict[str, Any]: ...

    def poll(self, handle: dict[str, Any]) -> dict[str, Any]: ...

    def cancel(self, handle: dict[str, Any]) -> dict[str, Any]: ...
