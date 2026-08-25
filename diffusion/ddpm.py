"""
DDPM training and sampling for the conditional unit-cell model.

Cosine beta schedule: at 64x64 a linear schedule destroys the signal too early, and
these fields are mostly low-frequency structure that survives noise well.

Conditioning is classifier-free. `y` is dropped with probability `cfg_dropout` during
training so one set of weights learns both the conditional and unconditional score; at
sampling time the two predictions are extrapolated with a guidance weight.

`y` is standardized on the train split only. Fitting the normalizer on everything would leak
val/test statistics into training. `K_radial` is log-transformed first because it spans two decades.
"""

import json
import math
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
from diffusion import dataset

Y_KEYS = dataset.Y_KEYS
LOG_KEYS = dataset.LOG_KEYS

def cosine_betas(timesteps, s=0.008, max_beta=0.999):
    """Nichol & Dhariwal cosine schedule."""
    steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
    alphas_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(max=max_beta).float()

class Normalizer:
    """Standardize `y`, with a log transform on the components that span decades."""

    def __init__(self, mean, std, keys=Y_KEYS, log_keys=LOG_KEYS):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.keys = tuple(keys)
        self.log_keys = tuple(log_keys)

    @classmethod
    def fit(cls, frame, keys=Y_KEYS, log_keys=LOG_KEYS):
        raw = cls._raw(frame, keys, log_keys)
        std = raw.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(raw.mean(axis=0), std, keys, log_keys)

    @staticmethod
    def _raw(frame, keys, log_keys):
        cols = []
        for k in keys:
            v = frame[k].to_numpy(dtype=np.float64)
            if k in log_keys:
                v = np.log10(np.maximum(v, 1e-12))
            cols.append(v)
        return np.stack(cols, axis=1)

    def transform(self, frame):
        return (self._raw(frame, self.keys, self.log_keys) - self.mean) / self.std

    def transform_dict(self, values):
        """Normalize a single {key: value} target, for sampling at a chosen y*."""
        row = []
        for k in self.keys:
            v = float(values[k])
            if k in self.log_keys:
                v = math.log10(max(v, 1e-12))
            row.append(v)
        return (np.array(row) - self.mean) / self.std

    def to_dict(self):
        return {'mean': self.mean.tolist(), 'std': self.std.tolist(),
                'keys': list(self.keys), 'log_keys': list(self.log_keys)}

    @classmethod
    def from_dict(cls, d):
        return cls(d['mean'], d['std'], d['keys'], d['log_keys'])

class DDPM:
    """Forward noising, the training objective, and CFG sampling."""

    def __init__(self, model, timesteps=1000, device='cuda'):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        betas = cosine_betas(timesteps).to(device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_ab = self.alphas_bar.sqrt()
        self.sqrt_1mab = (1.0 - self.alphas_bar).sqrt()

    def q_sample(self, x0, t, noise):
        return self.sqrt_ab[t][:, None, None, None] * x0 + \
            self.sqrt_1mab[t][:, None, None, None] * noise

    def loss(self, x0, y, cfg_dropout=0.1):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        mask = (torch.rand(b, device=x0.device) >= cfg_dropout).float()
        pred = self.model(xt, t, y, mask)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, n, y, guidance=2.0, shape=(1, 64, 64), progress=False):
        """
        Ancestral sampling with classifier-free guidance.

        guidance = 0 is unconditional, 1 is plain conditional, >1 extrapolates away from
        the unconditional prediction and sharpens adherence to `y`.
        """
        x = torch.randn(n, *shape, device=self.device)
        y = y.to(self.device)
        ones = torch.ones(n, device=self.device)
        zeros = torch.zeros(n, device=self.device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            # Only run the branches the guidance weight actually uses. guidance=0 is
            # purely unconditional and guidance=1 purely conditional, so evaluating both
            # there doubles sampling cost for a term that gets multiplied by zero.
            if guidance == 0.0:
                eps = self.model(x, t, y, zeros)
            elif guidance == 1.0:
                eps = self.model(x, t, y, ones)
            else:
                eps_c = self.model(x, t, y, ones)
                eps_u = self.model(x, t, y, zeros)
                eps = eps_u + guidance * (eps_c - eps_u)

            alpha, ab = self.alphas[i], self.alphas_bar[i]
            mean = (x - (1 - alpha) / (1 - ab).sqrt() * eps) / alpha.sqrt()
            if i:
                x = mean + self.betas[i].sqrt() * torch.randn_like(x)
            else:
                x = mean
            if progress and i % 200 == 0:
                print(f'    sampling t={i}', flush=True)
        return x

    @torch.no_grad()
    def ddim_sample(self, n, y, guidance=2.0, steps=50, eta=0.0, shape=(1, 64, 64), clamp=True, trajectory=False):
        """
        DDIM sampling on a subsequence of the training timesteps.

        DDIM walks a subsequence with a deterministic update (eta=0), so 50 steps is 20x 
        cheaper.

        `clamp` bounds the predicted x0 to [-1, 1]. The data is binary, so an x0 estimate
        outside that range is known to be wrong and letting it through only injects error.

        `trajectory` additionally returns the per-step record as
        `(timesteps, x_t, x0_hat)`, for visualizing how structure emerges. It is off by
        default because it holds every step in memory.
        """
        seq = np.linspace(0, self.timesteps - 1, steps).round().astype(int)[::-1]
        traj_t, traj_x, traj_x0 = [], [], []
        x = torch.randn(n, *shape, device=self.device)
        y = y.to(self.device)
        ones = torch.ones(n, device=self.device)
        zeros = torch.zeros(n, device=self.device)

        for i, t_cur in enumerate(seq):
            t = torch.full((n,), int(t_cur), device=self.device, dtype=torch.long)
            if guidance == 0.0:
                eps = self.model(x, t, y, zeros)
            elif guidance == 1.0:
                eps = self.model(x, t, y, ones)
            else:
                eps_c = self.model(x, t, y, ones)
                eps_u = self.model(x, t, y, zeros)
                eps = eps_u + guidance * (eps_c - eps_u)

            ab_t = self.alphas_bar[int(t_cur)]
            t_prev = int(seq[i + 1]) if i + 1 < len(seq) else -1
            ab_prev = self.alphas_bar[t_prev] if t_prev >= 0 else torch.tensor(
                1.0, device=self.device)

            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            if clamp:
                x0 = x0.clamp(-1.0, 1.0)
                eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
            if trajectory:
                traj_t.append(int(t_cur))
                traj_x.append(x.detach().cpu().clone())
                traj_x0.append(x0.detach().cpu().clone())

            sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
            direction = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            x = ab_prev.sqrt() * x0 + direction
            if eta > 0 and t_prev >= 0:
                x = x + sigma * torch.randn_like(x)
        if trajectory:
            return x, (np.array(traj_t), torch.stack(traj_x), torch.stack(traj_x0))
        return x

class EMA:
    """Exponential moving average of weights; DDPM samples are much better from these."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def copy_to(self, model):
        state = model.state_dict()
        for k, v in self.shadow.items():
            state[k].copy_(v)

def load_tensors(stem='dataset', out_dir=None):
    """Arrays scaled to [-1, 1], plus normalized `y`."""
    arrs, frame = dataset.load(out_dir=out_dir, stem=stem)
    if 'split' not in frame.columns:
        raise RuntimeError('no split column, run `python -m diffusion.splits` first')
    norm = Normalizer.fit(frame[frame['split'] == 'train'])
    x = torch.from_numpy(arrs.astype(np.float32) * 2.0 - 1.0).unsqueeze(1)
    y = torch.from_numpy(norm.transform(frame).astype(np.float32))
    out = {}
    for name in ('train', 'val', 'test'):
        # np.array(...) copies: pandas hands back a read-only view and torch warns on it.
        m = torch.from_numpy(np.array(frame['split'] == name, dtype=bool))
        out[name] = (x[m], y[m])
    return out, norm, frame
