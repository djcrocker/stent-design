"""
A small conditional U-Net for DDPM on the 64x64 unit-cell field.

Every convolution uses `padding_mode='circular'`. The unit cell lives on a torus, and zero 
padding would teach the model that the cell edges are boundaries, when in fact the left edge 
is joined to the right and the top to the bottom. With circular padding the periodicity is
structural instead of something the model must infer from data.

Conditioning is classifier-free: `y` is projected and added to the timestep embedding, and
during training it is dropped with some probability so the same weights learn both the
conditional and unconditional score. At sampling time the two are combined with a guidance
weight.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(t, dim):
    """Sinusoidal embedding of the diffusion timestep."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb

class ResBlock(nn.Module):
    """Two circular convolutions with a FiLM-style shift from the conditioning embedding."""

    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='circular')
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='circular')
        self.skip = (nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)

class Attention(nn.Module):
    """Self-attention over the spatial grid, used only at the coarsest resolutions."""

    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = (t.reshape(b, self.heads, c // self.heads, h * w).transpose(-2, -1) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        return x + self.proj(out)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1, padding_mode='circular')

    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, padding_mode='circular')

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))

class UNet(nn.Module):
    """
    Small conditional U-Net. Defaults are sized for 64x64 and a 16 GB card.

    `y_dim` is the objective vector width. The unconditional pass is signaled by a mask,
    so a dropped `y` can't be confused with a real one thathappens to sit near the sentinel.
    """

    def __init__(self, y_dim=4, base=64, mults=(1, 2, 2, 4), num_res=2, dropout=0.1, attn_resolutions=(16, 8)):
        super().__init__()
        emb_dim = base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.base = base
        # +1 input for the conditioning mask: 1 when y is present, 0 when dropped.
        self.y_mlp = nn.Sequential(nn.Linear(y_dim + 1, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.stem = nn.Conv2d(1, base, 3, padding=1, padding_mode='circular')
        self.down = nn.ModuleList()
        chans = [base]
        ch = base
        res = 64
        for i, m in enumerate(mults):
            out = base * m
            for _ in range(num_res):
                block = nn.ModuleList([ResBlock(ch, out, emb_dim, dropout), Attention(out) if res in attn_resolutions else None])
                self.down.append(block)
                ch = out
                chans.append(ch)
            if i != len(mults) - 1:
                self.down.append(nn.ModuleList([Downsample(ch), None]))
                chans.append(ch)
                res //= 2

        self.mid1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = Attention(ch)
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout)

        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            out = base * m
            for _ in range(num_res + 1):
                block = nn.ModuleList([ResBlock(ch + chans.pop(), out, emb_dim, dropout), Attention(out) if res in attn_resolutions else None])
                self.up.append(block)
                ch = out
            if i:
                self.up.append(nn.ModuleList([Upsample(ch), None]))
                res *= 2

        self.out = nn.Sequential(nn.GroupNorm(min(8, ch), ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1, padding_mode='circular'))

    def forward(self, x, t, y, y_mask):
        """x (B,1,64,64) in [-1,1]; t (B,) int; y (B,y_dim); y_mask (B,) in {0,1}."""
        emb = self.time_mlp(timestep_embedding(t, self.base))
        cond = torch.cat([y * y_mask[:, None], y_mask[:, None]], dim=-1)
        emb = emb + self.y_mlp(cond)

        h = self.stem(x)
        skips = [h]
        for block in self.down:
            layer, attn = block
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            if attn is not None:
                h = attn(h)
            skips.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)

        for block in self.up:
            layer, attn = block
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, skips.pop()], dim=1), emb)
            else:
                h = layer(h)
            if attn is not None:
                h = attn(h)
        return self.out(h)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
