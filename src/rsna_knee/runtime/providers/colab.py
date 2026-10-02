"""Colab provider: capability-gated. Default is documented VS Code handoff, not a fake API."""

from __future__ import annotations

from typing import Any

from rsna_knee.runtime.providers.base import ProviderCapabilities

COLAB_NOTES = [
    "Official extension: Google.colab — New Colab Server can select a machine type in VS Code/Cursor UI.",
    "That UI path is not a public headless allocation API for terminal Python.",
    "OAuth happens in the system browser via the extension. Do not copy cookies.",
    "No colab.allocate_gpu() wrapper is provided because it is not a documented API.",
    "After a verified worker handshake, the controller continues job execution automatically.",
    "A 300s instruction change is not a physical GPU switch in handoff mode.",
]


class ColabProvider:
    name = "colab"

    def __init__(self, automatic: bool = False):
        self.automatic = automatic  # only True if a documented API is wired later

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="colab",
            supported_gpu_types=["A100", "L4", "T4"],
            request=self.automatic,
            start=self.automatic,
            poll=self.automatic,
            cancel=False,
            remote_execution=False,
            upload_download="vscode_upload_or_drive_handoff",
            auth_flow="oauth_browser_google_colab_extension",
            availability_status="manual_handoff" if not self.automatic else "api",
            automatic_acquisition=self.automatic,
            notes=COLAB_NOTES,
        )

    def request_gpu(self, gpu_type: str, **kwargs: Any) -> dict[str, Any]:
        if not self.automatic:
            return {
                "status": "handoff",
                "gpu": gpu_type,
                "retryable": False,
                "automatic": False,
                "action_tr": (
                    f"VS Code/Cursor'da Colab eklentisi ile New Colab Server açın ve {gpu_type} seçin. "
                    "OAuth tarayıcıda tamamlanır. Kernel bağlanınca notebooks/02_colab_worker.ipynb "
                    "handshake üretir; controller bundan sonra işi otomatik sürdürür. "
                    "5 dakika dolunca talimatın değişmesi GPU'nun kendiliğinden değişmesi değildir."
                ),
            }
        return {"status": "unsupported", "gpu": gpu_type, "retryable": False, "reason": "no_documented_headless_api"}

    def poll(self, handle: dict[str, Any]) -> dict[str, Any]:
        if handle.get("status") == "handoff":
            return {"status": "handoff", "automatic": False}
        if handle.get("status") == "unavailable":
            return {"status": "unavailable", "retryable": False}
        return {"status": handle.get("status", "unknown")}

    def cancel(self, handle: dict[str, Any]) -> dict[str, Any]:
        return {"status": "cancel_unsupported", "note": "Close the Colab server from the extension UI."}
