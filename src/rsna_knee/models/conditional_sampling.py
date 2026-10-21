"""Train-only, bounded sampling probabilities for two observed binary targets."""
import numpy as np


def conditional_weights(targets, masks, cap=3.):
    y=np.asarray(targets,dtype=float);m=np.asarray(masks,dtype=bool)
    if y.ndim!=2 or y.shape[1]!=2 or m.shape!=y.shape:
        raise ValueError('expected two-target arrays')
    if not np.isfinite(y[m]).all() or not np.isin(y[m],[0,1]).all():
        raise ValueError('observed labels must be binary')
    if not m.any(axis=1).all() or cap<1:
        raise ValueError('each training row needs an observed target and cap >=1')
    known=m.all(axis=1);weights=np.ones(len(y),dtype=float)
    groups=(2*y[known,0]+y[known,1]).astype(int)
    counts=np.bincount(groups,minlength=4)
    if known.any():
        w=1/np.sqrt(counts[groups]);w/=w.mean()
        weights[known]=np.minimum(w,cap)
    return weights,counts.tolist()
