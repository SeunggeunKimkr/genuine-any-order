# -------------------------------------------------------
# transformer util functions
# -------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.pow(2).mean(dim = -1, keepdim = True) + eps
    x = x * torch.rsqrt(var)
    return x * weight

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim = -1).reshape_as(x)

def apply_rope(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# -------------------------------------------------------
# RoatryEmbedding
# -------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int = 1024, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "dimension must be even"
        self.dim = dim
        self.max_position = max_position
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
        t = torch.arange(max_position, device = inv_freq.device, dtype = torch.float)
        freqs = torch.einsum("t, d -> t d", t, inv_freq)
        emb = torch.cat( (freqs, freqs), dim = -1)
        # store buffers
        self.register_buffer("cos", emb.cos().unsqueeze(1).unsqueeze(0) , persistent = False)
        self.register_buffer("sin", emb.sin().unsqueeze(1).unsqueeze(0), persistent = False)

    def forward(self, seqlen: int, position_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        end = position_offset + seqlen
        return self.cos[:, position_offset:end], self.sin[:, position_offset:end]



if __name__ == "__main__":
    print("Hello, World!")