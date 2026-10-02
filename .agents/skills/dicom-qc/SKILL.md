---
name: dicom-qc
description: Geometry order, MONOCHROME1, slope/intercept, cache failures. Use when building or debugging MRI cache shards.
---

# DICOM QC

## When

Cache pilot, bad triplets, or orientation bugs.

## Procedure

1. Order with IOP cross-product normal × IPP projection.
2. Coherent orientation check; else InstanceNumber fallback. Never SOP lex sort.
3. Apply slope/intercept; invert MONOCHROME1.
4. Physical mm crop from PixelSpacing; missing spacing is explicit fallback.
5. Record failed study/series; do not write all-zero success.

## Entrypoints

- `python -c "from rsna_knee.data.dicom import order_slices"`
- `rsna cache pilot`
- tests/test_dicom_geometry.py

## Recovery

Shard missing → rebuild that UID only. Preprocess hash change → invalidate cache.
