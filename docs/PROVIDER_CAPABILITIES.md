# Provider capabilities

Measured 2026-10-02 on this Mac. This session did not allocate a GPU, push a kernel, or submit.

| Capability | local | colab | kaggle | This session |
| --- | --- | --- | --- | --- |
| supported GPUs | MPS if present; CUDA not expected | A100, L4, T4 in the extension UI | T4/P100 typical; handshake required | implemented |
| request / poll | process-local only | handoff; `automatic_acquisition=false` | `kernels push/status/output` code path | tested offline; not live |
| 300s A100→L4→T4 wait | not used for full training | not started, because there is no allocation API | not used for A100 | tested with a scripted clock |
| cancel | n/a | unsupported | unsupported | tested: pending request is left open and the next GPU is not requested |
| auth/balance poll failure | n/a | would stop | would stop | tested |
| remote execution | no | after a manual extension kernel and worker handshake | notebook, internet off for scoring | blocked: not dispatched |
| full training | CPU/MPS full training disabled | intended after handshake | fallback code path, not pushed | blocked |
| submit | no | no | `competitions submit -k -v` after audit PASS | blocked; no receipt |

Kaggle quota was measured earlier the same day (about 11.09h GPU used / 18.91h remaining, refresh 2026-10-03T00:00:00). Compute units in the Colab UI were not read. That quota is not a live verification of this change.
