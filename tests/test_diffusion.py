"""The conditional U-Net, the noise schedule, and y normalization."""

import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')

pytestmark = pytest.mark.torch

from diffusion.ddpm import DDPM, EMA, Normalizer, cosine_betas
from diffusion.model import UNet

@pytest.fixture(scope='module')
def net():
    torch.manual_seed(0)
    return UNet(base=16, mults=(1, 2), num_res=1, attn_resolutions=()).eval()

def _inputs(n=2):
    return (torch.randn(n, 1, 64, 64), torch.randint(0, 1000, (n,)),
            torch.randn(n, 4), torch.ones(n))

def test_unet_preserves_shape(net):
    x, t, y, m = _inputs()
    assert net(x, t, y, m).shape == x.shape

def test_circular_padding_gives_exact_shift_equivariance_at_the_stride():
    """
    The cell is a torus. Shifts by a multiple of the total downsampling factor must map
    exactly; odd shifts can't, because stride-2 sampling picks different pixels.
    """
    torch.manual_seed(1)
    model = UNet(base=16, mults=(1, 2), num_res=1, attn_resolutions=()).eval()
    x, t, y, m = _inputs(1)
    with torch.no_grad():
        a = model(x, t, y, m)
        s = 2 ** (2 - 1)                       # one downsample for mults=(1,2)
        shifted = torch.roll(x, shifts=(s, s), dims=(2, 3))
        b = torch.roll(model(shifted, t, y, m), shifts=(-s, -s), dims=(2, 3))
    assert (a - b).abs().max().item() < 1e-4

def test_zero_padding_would_break_the_torus():
    """padding_mode='circular' everywhere in the model."""
    torch.manual_seed(2)
    x = torch.randn(1, 1, 64, 64)
    circ = torch.nn.Conv2d(1, 4, 3, padding=1, padding_mode='circular')
    zero = torch.nn.Conv2d(1, 4, 3, padding=1)
    with torch.no_grad():
        for conv, exact in ((circ, True), (zero, False)):
            a = conv(x)
            b = torch.roll(conv(torch.roll(x, (7, 7), (2, 3))), (-7, -7), (2, 3))
            err = (a - b).abs().max().item()
            assert (err < 1e-6) is exact

def test_conditioning_mask_changes_the_prediction(net):
    """If the mask did nothing, classifier-free guidance would be a no-op."""
    x, t, y, _ = _inputs()
    with torch.no_grad():
        cond = net(x, t, y, torch.ones(len(x)))
        uncond = net(x, t, y, torch.zeros(len(x)))
    assert (cond - uncond).abs().max().item() > 1e-4

def test_dropped_conditioning_ignores_the_y_values(net):
    """A dropped y can't leak through: two different y with mask 0 have to agree."""
    x, t, _, _ = _inputs()
    zeros = torch.zeros(len(x))
    with torch.no_grad():
        a = net(x, t, torch.randn(len(x), 4), zeros)
        b = net(x, t, torch.randn(len(x), 4), zeros)
    assert torch.allclose(a, b, atol=1e-6)

def test_cosine_schedule_is_monotone_and_bounded():
    betas = cosine_betas(1000)
    assert betas.shape == (1000,)
    assert (betas > 0).all() and (betas <= 0.999).all()
    alphas_bar = torch.cumprod(1 - betas, dim=0)
    assert alphas_bar[0] > 0.99          # almost no noise at t=0
    assert alphas_bar[-1] < 0.01         # almost pure noise at T

def test_q_sample_is_nearly_the_clean_image_at_t_zero():
    model = UNet(base=16, mults=(1, 2), num_res=1, attn_resolutions=())
    ddpm = DDPM(model, timesteps=1000, device='cpu')
    x0 = torch.randn(3, 1, 64, 64)
    t = torch.zeros(3, dtype=torch.long)
    out = ddpm.q_sample(x0, t, torch.randn_like(x0))
    assert (out - x0).abs().mean().item() < 0.2

def _frame(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'K_radial': 10 ** rng.uniform(0, 2, n),
        'eps_a_max': rng.uniform(0.02, 0.12, n),
        'A_over_lim': rng.uniform(0.05, 0.9, n),
        'f_metal': rng.uniform(0.2, 0.5, n),
    })

def test_normalizer_standardizes_each_component():
    frame = _frame(500)
    z = Normalizer.fit(frame).transform(frame)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(z.std(axis=0), 1, atol=1e-6)

def test_normalizer_log_transforms_k_radial():
    """Standardizing the raw value would squash the low end."""
    frame = _frame(200)
    norm = Normalizer.fit(frame)
    small = norm.transform_dict({'K_radial': 1.0, 'eps_a_max': 0.05,
                                 'A_over_lim': 0.4, 'f_metal': 0.3})
    large = norm.transform_dict({'K_radial': 100.0, 'eps_a_max': 0.05,
                                 'A_over_lim': 0.4, 'f_metal': 0.3})
    # A 100x change in K_radial spans a few standard deviations, not dozens.
    assert 1.0 < abs(large[0] - small[0]) < 10.0

def test_normalizer_roundtrips_through_a_dict():
    norm = Normalizer.fit(_frame())
    back = Normalizer.from_dict(norm.to_dict())
    assert np.allclose(norm.mean, back.mean)
    assert np.allclose(norm.std, back.std)
    assert back.keys == norm.keys

def test_normalizer_survives_a_constant_column():
    frame = _frame()
    frame['f_metal'] = 0.3
    z = Normalizer.fit(frame).transform(frame)
    assert np.isfinite(z).all()

def test_ema_tracks_but_lags_the_weights():
    model = UNet(base=16, mults=(1, 2), num_res=1, attn_resolutions=())
    ema = EMA(model, decay=0.9)
    before = {k: v.clone() for k, v in ema.shadow.items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    key = next(iter(before))
    moved = (ema.shadow[key] - before[key]).abs().max().item()
    assert moved > 0.0
    assert moved < 1.0
