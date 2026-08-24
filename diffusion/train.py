"""
Train the conditional DDPM.

This script keeps the pieces that make DDPM training behave: EMA weights, bf16 autocast,
gradient clipping, and a val loss computed on fixed noise so the curve is comparable between
epochs rather than jittering with whatever noise was drawn.

Usage: python -m diffusion.train --epochs 60 --batch 128
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch

import config
from diffusion.ddpm import DDPM, EMA, Normalizer, load_tensors
from diffusion.model import UNet

CKPT_DIR = config.PROJECT_ROOT / 'diffusion' / 'checkpoints'

def evaluate(ddpm, x, y, batch=256, seed=0, device='cuda'):
    """
    Val loss at fixed timesteps and noise.

    Sampling fresh t and noise each epoch makes the val curve jump around for reasons that
    have nothing to do with the model, which hides whether it is still improving.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    total, count = 0.0, 0
    ddpm.model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(device)
            yb = y[i:i + batch].to(device)
            n = xb.shape[0]
            t = torch.randint(0, ddpm.timesteps, (n,), generator=g).to(device)
            noise = torch.randn(xb.shape, generator=g).to(device)
            xt = ddpm.q_sample(xb, t, noise)
            mask = torch.ones(n, device=device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                pred = ddpm.model(xt, t, yb, mask)
                loss = torch.nn.functional.mse_loss(pred.float(), noise)
            total += loss.item() * n
            count += n
    ddpm.model.train()
    return total / max(count, 1)

def train(epochs=60, batch=128, lr=2e-4, timesteps=1000, cfg_dropout=0.1, base=64,
          ema_decay=0.999, seed=0, device=None, stem='dataset', out_dir=None,
          limit=None, checkpoint_every=10):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    data, norm, frame = load_tensors(stem=stem, out_dir=out_dir)
    xtr, ytr = data['train']
    xva, yva = data['val']
    if limit:
        xtr, ytr = xtr[:limit], ytr[:limit]
        xva, yva = xva[:max(1, limit // 8)], yva[:max(1, limit // 8)]
    print(f'train {len(xtr):,}  val {len(xva):,}  device {device}', flush=True)

    model = UNet(y_dim=ytr.shape[1], base=base).to(device)
    print(f'model {model.parameter_count() / 1e6:.2f} M parameters', flush=True)
    ddpm = DDPM(model, timesteps=timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ema = EMA(model, decay=ema_decay)

    steps_per_epoch = max(1, len(xtr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.05)

    history = []
    t0 = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(len(xtr))
        running, seen = 0.0, 0
        for step in range(steps_per_epoch):
            idx = perm[step * batch:(step + 1) * batch]
            xb = xtr[idx].to(device, non_blocking=True)
            yb = ytr[idx].to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                loss = ddpm.loss(xb, yb, cfg_dropout=cfg_dropout)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(model)
            running += loss.item() * len(idx)
            seen += len(idx)

        val = evaluate(ddpm, xva, yva, device=device)
        row = {'epoch': epoch + 1, 'train_loss': running / seen, 'val_loss': val,
               'lr': sched.get_last_lr()[0], 'minutes': (time.time() - t0) / 60}
        history.append(row)
        print(f"  epoch {row['epoch']:3d}  train {row['train_loss']:.5f}  "
              f"val {row['val_loss']:.5f}  {row['minutes']:.1f} min", flush=True)

        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            snapshot = UNet(y_dim=ytr.shape[1], base=base).to(device)
            snapshot.load_state_dict(model.state_dict())
            ema.copy_to(snapshot)
            torch.save({'model': snapshot.state_dict(),
                        'normalizer': norm.to_dict(),
                        'epoch': epoch + 1,
                        'config': {'base': base, 'timesteps': timesteps,
                                   'y_dim': ytr.shape[1]}},
                       CKPT_DIR / f'{stem}_ddpm_epoch{epoch + 1:03d}.pt')
            (CKPT_DIR / f'{stem}_history.json').write_text(
                json.dumps(history, indent=2), encoding='utf-8')
            del snapshot

    ema.copy_to(model)
    torch.save({'model': model.state_dict(),
                'normalizer': norm.to_dict(),
                'config': {'base': base, 'timesteps': timesteps, 'y_dim': ytr.shape[1],
                           'cfg_dropout': cfg_dropout, 'epochs': epochs, 'batch': batch,
                           'lr': lr, 'ema_decay': ema_decay, 'seed': seed}},
               CKPT_DIR / f'{stem}_ddpm.pt')
    (CKPT_DIR / f'{stem}_history.json').write_text(json.dumps(history, indent=2),
                                                   encoding='utf-8')
    print(f'saved {CKPT_DIR / (stem + "_ddpm.pt")}')
    return history

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Conditional DDPM training')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--timesteps', type=int, default=1000)
    ap.add_argument('--base', type=int, default=64)
    ap.add_argument('--cfg-dropout', type=float, default=0.1)
    ap.add_argument('--checkpoint-every', type=int, default=10)
    ap.add_argument('--limit', type=int, default=None,
                    help='truncate the training set')
    ap.add_argument('--stem', default='dataset')
    args = ap.parse_args()
    train(epochs=args.epochs, batch=args.batch, lr=args.lr, timesteps=args.timesteps,
          base=args.base, cfg_dropout=args.cfg_dropout, limit=args.limit,
          checkpoint_every=args.checkpoint_every, stem=args.stem)
