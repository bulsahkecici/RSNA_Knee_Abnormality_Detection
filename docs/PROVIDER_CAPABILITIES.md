# Provider capabilities

Measured 2026-10-02 on this Mac + Kaggle CLI + LM Studio HTTP.

| Capability | local | colab | kaggle |
| --- | --- | --- | --- |
| supported GPUs | mps if present | A100, L4, T4 (UI) | T4/P100 typical; handshake required |
| request/start/poll/cancel | yes (process) | **handoff only** | kernels push/status/output; cancel limited |
| remote execution | no | after handshake | yes (notebook) |
| auth | none | Google OAuth in browser | `~/.kaggle/kaggle.json` present for this user |
| automatic_acquisition | yes for tiny/synth | **false** | kernels yes; GPU SKU not A100 |
| full training | disabled on CPU/MPS by default | intended | fallback if enabled |

Kaggle quota (this user, this day): GPU 11.09h used / 18.91h remaining of 30h, refresh 2026-10-03T00:00:00. Compute units from Colab UI: **unknown** unless the extension reports them.

This table is not a promise that Colab Pro always exposes A100.
