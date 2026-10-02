"""Study-level tensors: [slots, slices, C, H, W] plus masks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rsna_knee.data.cache import load_shard
from rsna_knee.ontology import TARGET_COLUMNS


@dataclass
class StudyBatch:
    uids: list[str]
    images: np.ndarray  # [B, slots, slices, C, H, W]
    slot_mask: np.ndarray  # [B, slots]
    slice_mask: np.ndarray  # [B, slots, slices]
    y: np.ndarray  # [B, 12] with NaN for masked
    y_mask: np.ndarray  # [B, 12]
    y_weight: np.ndarray  # [B, 12]


class StudyDataset:
    def __init__(
        self,
        uids: Sequence[str],
        labels: dict[str, dict[str, float | None]],
        masks: dict[str, dict[str, float]],
        weights: dict[str, dict[str, float]],
        cache_root: Path | None = None,
        n_slots: int = 3,
        n_slices: int = 3,
        size: int = 32,
    ):
        self.uids = list(uids)
        self.labels = labels
        self.masks = masks
        self.weights = weights
        self.cache_root = cache_root
        self.n_slots = n_slots
        self.n_slices = n_slices
        self.size = size

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        uid = self.uids[idx]
        try:
            arr = np.array(load_shard(uid, self.cache_root), dtype=np.float32)
            if arr.max() > 1.5:
                arr = arr / 255.0
            # expected [slots, slices, H, W] or [slots, slices, C, H, W]
            if arr.ndim == 4:
                arr = arr[:, :, None, :, :]
            images = arr
            slot_mask = (images.reshape(images.shape[0], -1).std(axis=1) > 1e-8).astype(np.float32)
            slice_mask = (images.reshape(images.shape[0], images.shape[1], -1).std(axis=2) > 1e-8).astype(np.float32)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"missing cache shard for {uid}; refusing a silent zero image"
            ) from exc
        y = np.array([self.labels.get(uid, {}).get(t) if self.labels.get(uid, {}).get(t) is not None else np.nan for t in TARGET_COLUMNS], dtype=np.float32)
        y_mask = np.array([self.masks.get(uid, {}).get(t, 0.0) for t in TARGET_COLUMNS], dtype=np.float32)
        y_weight = np.array([self.weights.get(uid, {}).get(t, 0.0) for t in TARGET_COLUMNS], dtype=np.float32)
        return {
            "uid": uid,
            "images": images.astype(np.float32),
            "slot_mask": slot_mask,
            "slice_mask": slice_mask,
            "y": y,
            "y_mask": y_mask,
            "y_weight": y_weight,
        }


def collate(items: list[dict[str, Any]]) -> StudyBatch:
    return StudyBatch(
        uids=[i["uid"] for i in items],
        images=np.stack([i["images"] for i in items]),
        slot_mask=np.stack([i["slot_mask"] for i in items]),
        slice_mask=np.stack([i["slice_mask"] for i in items]),
        y=np.stack([i["y"] for i in items]),
        y_mask=np.stack([i["y_mask"] for i in items]),
        y_weight=np.stack([i["y_weight"] for i in items]),
    )
