"""K1 study windows, sampled anew from canonical series volumes each epoch."""
from __future__ import annotations

import numpy as np
import torch

from rsna_knee.data.soft_pool import normalize_soft_targets, soft_table
from rsna_knee.data.volume_cache import _payload_path
from rsna_knee.ontology import TARGET_COLUMNS


def window_centres(n, k, train):
    if n < 3 or k < 1:
        raise ValueError('invalid series/window count')
    a, b = max(1., .04 * (n - 1)), min(n - 2., .96 * (n - 1))
    if b < a:
        a = b = (n - 1) / 2
    if train:
        edges = np.linspace(a, b, k + 1)
        c = edges[:-1] + np.random.rand(k) * np.diff(edges)
    else:
        c = np.linspace(a, b, k)
    return np.clip(np.rint(c).astype(int), 1, n - 2)


class VolumeStudy(torch.utils.data.Dataset):
    def __init__(self, uids, k, train, *, studies, cache, size, targets):
        self.uids, self.k, self.train = list(uids), k, train
        self.studies, self.cache, self.size = studies, cache, size
        self.targets = (soft_table(targets) if isinstance(targets, bytes)
                        else {uid: normalize_soft_targets(row) for uid, row in targets.items()})

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        meta = self.studies[uid]
        x = np.zeros((6, self.k, 3, self.size, self.size), np.uint8)
        mask, pos = np.zeros(6, bool), np.zeros((6, self.k), np.float32)
        with np.load(_payload_path(self.cache, meta['file']), allow_pickle=False) as arrays:
            for s, candidates in enumerate(meta['candidates']):
                if not candidates:
                    continue
                chosen = candidates[np.random.randint(len(candidates))] if self.train else candidates[0]
                vol = arrays[chosen['key']]
                centers = window_centres(len(vol), self.k, self.train)
                x[s] = vol[centers[:, None] + np.array([-1, 0, 1])]
                mask[s], pos[s] = True, centers / (len(vol) - 1)
        if not mask.any():
            raise ValueError('study_has_no_usable_volume')
        row = self.targets.get(uid, {})
        y = np.array([row.get(c) or 0. for c in TARGET_COLUMNS], np.float32)
        m = np.array([row.get(c) is not None for c in TARGET_COLUMNS], np.float32)
        return *(torch.from_numpy(a) for a in (x, mask, pos, y, m)), uid
