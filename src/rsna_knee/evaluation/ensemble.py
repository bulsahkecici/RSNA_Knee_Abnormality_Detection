"""Rank-mean ensemble. Averaging raw probabilities is not the default."""

from __future__ import annotations

import numpy as np


def rank_mean(prob_list: list[np.ndarray]) -> np.ndarray:
    ranks = []
    for p in prob_list:
        p = np.asarray(p, dtype=float)
        order = p.argsort(axis=0)
        r = np.empty_like(p, dtype=float)
        for t in range(p.shape[1]):
            r[order[:, t], t] = np.linspace(0, 1, p.shape[0])
        ranks.append(r)
    return np.mean(np.stack(ranks), axis=0)
