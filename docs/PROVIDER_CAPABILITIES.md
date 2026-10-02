# Provider capabilities

Measured 2026-10-02 on this Mac for quota. The 2026-10-03 code pass did not allocate a GPU, download official DINOv2 weights, push a kernel, call LM Studio, or submit.

| Capability | local | colab | kaggle | This session |
| --- | --- | --- | --- | --- |
| supported GPUs | MPS if present; CUDA not expected | A100, L4, T4 in the extension UI | T4/P100 typical; handshake required | implemented |
| request / poll | process-local only | handoff; `automatic_acquisition=false` | `kernels push/status/output` code path | tested offline; not live |
| 300s A100→L4→T4 wait | not used for full training | not started, because there is no allocation API | not used for A100 | tested with a scripted clock |
| cancel | n/a | unsupported | unsupported | tested: pending request is left open and the next GPU is not requested |
| auth/balance poll failure | n/a | would stop | would stop | tested |
| remote execution | no | after a manual extension kernel and worker handshake | notebook, internet off for scoring | blocked: not dispatched |
| full training | worker can run a CPU pilot when weights are allowlisted | after manual kernel connect, `rsna worker --job-bundle` | dataset asset must be created; kernels push does not upload weights | blocked: no official weights, no live GPU |
| submit | no | no | rechecks checkpoint and package; UNKNOWN is not resent | blocked; no receipt |

Kaggle quota was measured earlier the same day (about 11.09h GPU used / 18.91h remaining, refresh 2026-10-03T00:00:00). Compute units in the Colab UI were not read. That quota is not a live verification of this change.
