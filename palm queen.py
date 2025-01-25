# Copyright (c) 2025 Senhorita Jaine
# Este código é parte do projeto exclusivo desenvolvido em colaboração com Lumin.
# Proibida a remoção ou alteração deste cabeçalho.
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import wraps
from einops import rearrange, pack, unpack
from tqdm import tqdm
from pathlib import Path

# Helper functions
def exists(val):
    """Check if a value exists."""
    return val is not None

def default(val, d):
    """Return val if it exists, otherwise return the default."""
    return val if exists(val) else d

def identity(x, *args, **kwargs):
    """Identity function."""
    return x

def l2norm(tensor):
    """Apply L2 normalization along the last dimension."""
    return F.normalize(tensor, dim=-1)

# LayerNorm without bias
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], (self.gamma + 1), self.beta)

# Residual connection
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        y = self.fn(x, **kwargs)
        return y + x

# SwiGLU activation
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

# Rotary Positional Embeddings
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, scale_base=512, use_xpos=True):
        super().__init__()
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.use_xpos = use_xpos
        self.scale_base = scale_base
        self.scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device=device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.cat((scale, scale), dim=-1)
        return freqs, scale

def apply_rotary_pos_emb(pos, t, scale=1.0):
    """Apply rotary positional embedding."""
    x1, x2 = t.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return (t * pos.cos() * scale) + (rotated * pos.sin() * scale)

# Parallel Transformer Block
class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, ff_mult=4, use_xpos=True, xpos_scale_base=512, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(dim)
        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult

        self.attend = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim),
            SwiGLU(),
            nn.Linear(ff_inner_dim // 2, dim),
            nn.Dropout(dropout)
        )
        self.rotary_emb = RotaryEmbedding(dim_head, use_xpos=use_xpos, scale_base=xpos_scale_base)

    def forward(self, x, mask=None):
        # Positional embeddings
        seq_len, device = x.shape[1], x.device
        pos_emb, scale = self.rotary_emb(seq_len, device)
        q, k, v = map(lambda t: apply_rotary_pos_emb(pos_emb, t, scale), (x, x, x))

        # Attention
        attn_out, _ = self.attend(q, k, v, attn_mask=mask)
        attn_out = self.ff(attn_out)
        return self.norm(attn_out)

# Transformer Model
class PaLM(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads=8, dim_head=64, ff_mult=4, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([
            Residual(ParallelTransformerBlock(dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout))
            for _ in range(depth)
        ])
        self.norm = LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x, return_logits=True):
        x = self.token_emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        if return_logits:
            return self.to_logits(x)
        return x
