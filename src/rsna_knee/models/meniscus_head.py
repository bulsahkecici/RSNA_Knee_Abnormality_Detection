"""Ordered and order-invariant heads for identical frozen sequence features."""
import torch
from torch import nn
import torch.nn.functional as F


class MeniscusHead(nn.Module):
    def __init__(self, dim=768, ordered=True, n_slots=4):
        super().__init__()
        self.ordered = ordered
        self.n_slots = n_slots
        self.project = nn.Linear(dim, 128)
        self.slot = nn.Parameter(torch.randn(1, n_slots, 1, 128) * .01)
        if ordered:
            self.pos = nn.Parameter(torch.randn(1, 1, 16, 128) * .01)
            self.temporal = nn.Conv1d(128, 128, 3, padding=1)
        self.attn = nn.Linear(128, 2)
        self.readout = nn.Parameter(torch.randn(2, 128) * .01)
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, x, mask):
        b = x.shape[0]
        if x.shape[1:3] != (self.n_slots, 16) or mask.shape != x.shape[:3]:
            raise ValueError('sequence feature or mask shape mismatch')
        if not mask.flatten(1).any(1).all():
            raise ValueError('all sequence slots missing')
        h = self.project(x) + self.slot
        if self.ordered:
            h = h + self.pos
            h = h + F.gelu(self.temporal(h.reshape(b*self.n_slots,16,128).transpose(1,2)).transpose(1,2)).reshape(b,self.n_slots,16,128)
        h = h.reshape(b,self.n_slots*16,128)
        attention = self.attn(h).masked_fill(~mask.reshape(b,self.n_slots*16,1), -1e4).softmax(dim=1)
        pooled = torch.einsum('bnc,bnd->bcd', attention, h)
        return (pooled*self.readout).sum(-1) + self.bias
