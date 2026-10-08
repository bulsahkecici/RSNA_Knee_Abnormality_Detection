# Validated v3 preprocessing source

These four modules are the exact preprocessing source recovered from campaign-20261003-v3-cache and used by the 0.740 public-score offline asset. Keep them as a versioned overlay on src/rsna_knee/data when rebuilding that asset; the main pipeline currently retains its v2-compatible builder.

The cache uses uint8 tensors, 224 pixels, three interior centers at 0.15, 0.50 and 0.85, strict decoding and at least one present plane. Training and inference must use the same preprocessing configuration. Source hashes are recorded in docs/EXPERIMENTS.md and the locally retained preprocess-config.json. No medical reports, study labels or image files are included here.
