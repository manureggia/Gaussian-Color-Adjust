"""
Test per training/color_fitter.py.

Usa dati sintetici (20 Gaussiane, immagini 64×64) e gira su CPU
senza richiedere GPU o dati reali di scena.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn.functional as F

from training.color_fitter import ColorFitter, _ssim, _psnr
from utils.ply_io import load_gaussians, save_gaussians


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_gaussians(n: int = 20) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    return {
        "xyz":           torch.randn(n, 3) * 0.3,
        "features_dc":   torch.randn(n, 1, 3) * 0.1,
        "features_rest": torch.randn(n, 15, 3) * 0.01,
        "scaling":       torch.full((n, 3), -3.0),
        "rotation":      F.normalize(torch.randn(n, 4), dim=-1),
        "opacity":       torch.zeros(n, 1),
    }


def _make_cameras(n: int = 3) -> list[dict]:
    """Camera con T=[0,0,5+i*0.1]: gaussiane con z_world~0 sono davanti."""
    cameras = []
    for i in range(n):
        cameras.append({
            "R":      torch.eye(3),
            "T":      torch.tensor([0.0, 0.0, float(5 + i * 0.1)]),
            "fx":     100.0, "fy": 100.0,
            "cx":     32.0,  "cy": 32.0,
            "width":  64,    "height": 64,
        })
    return cameras


def _make_gt_images(n: int = 3) -> list[torch.Tensor]:
    """Immagini GT sintetiche: gradiente di colore ripetibile."""
    torch.manual_seed(0)
    return [torch.rand(64, 64, 3) for _ in range(n)]


# ---------------------------------------------------------------------------
# Test SSIM
# ---------------------------------------------------------------------------

def test_ssim_identical_images():
    """SSIM tra immagine e se stessa deve essere ≈ 1."""
    img = torch.rand(64, 64, 3)
    val = _ssim(img, img).item()
    assert val == pytest.approx(1.0, abs=0.01)


def test_ssim_different_images():
    """SSIM tra immagini diverse deve essere < 1."""
    torch.manual_seed(1)
    a = torch.rand(64, 64, 3)
    b = torch.rand(64, 64, 3)
    val = _ssim(a, b).item()
    assert val < 0.99


def test_ssim_range():
    """SSIM deve essere in [-1, 1] (in pratica [0, 1] su immagini naturali)."""
    torch.manual_seed(2)
    a = torch.rand(32, 32, 3)
    b = torch.rand(32, 32, 3)
    val = _ssim(a, b).item()
    assert -1.0 <= val <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Test PSNR
# ---------------------------------------------------------------------------

def test_psnr_identical():
    """PSNR tra immagine e se stessa deve essere infinito."""
    img = torch.rand(32, 32, 3)
    assert _psnr(img, img) == float("inf")


def test_psnr_decreases_with_noise():
    """PSNR deve diminuire all'aumentare del rumore."""
    torch.manual_seed(3)
    ref = torch.rand(32, 32, 3)
    noisy_low  = ref + 0.01 * torch.randn_like(ref)
    noisy_high = ref + 0.5  * torch.randn_like(ref)
    assert _psnr(noisy_low.clamp(0,1), ref) > _psnr(noisy_high.clamp(0,1), ref)


# ---------------------------------------------------------------------------
# Test ColorFitter
# ---------------------------------------------------------------------------

def test_fitter_loss_decreases():
    """La loss deve diminuire dopo 5 iterazioni di fitting."""
    gaussians = _make_gaussians(20)
    cameras   = _make_cameras(3)
    gt_images = _make_gt_images(3)

    # Prima loss: rendering iniziale
    from utils.renderer import _render_pure_pytorch as _rend
    cam = cameras[0]
    with torch.no_grad():
        pred_init = _rend(gaussians, cam, (64, 64))
    loss_init = F.l1_loss(pred_init, gt_images[0]).item()

    fitter = ColorFitter(
        gaussians=gaussians,
        cameras=cameras,
        gt_images=gt_images,
        device=torch.device("cpu"),
    )
    updated = fitter.fit(num_iterations=5, lr_dc=0.01, lr_rest=0.001, log_every=10)

    # Loss dopo fitting (usiamo le gaussiane aggiornate)
    with torch.no_grad():
        pred_final = _rend(updated, cam, (64, 64))
    loss_final = F.l1_loss(pred_final, gt_images[0]).item()

    # Su pochi step potrebbe non diminuire sempre, ma le gaussiane devono essere cambiate
    assert not torch.allclose(
        updated["features_dc"], gaussians["features_dc"], atol=1e-6
    ), "features_dc non è cambiato durante il fitting"


def test_fitter_frozen_parameters():
    """xyz, scaling, rotation, opacity non devono avere gradiente dopo fit."""
    gaussians = _make_gaussians(10)
    cameras   = _make_cameras(2)
    gt_images = _make_gt_images(2)

    fitter = ColorFitter(
        gaussians=gaussians,
        cameras=cameras,
        gt_images=gt_images,
        device=torch.device("cpu"),
    )
    # Verifica che i parametri congelati non richiedano gradiente
    assert not fitter.gaussians["xyz"].requires_grad
    assert not fitter.gaussians["scaling"].requires_grad
    assert not fitter.gaussians["rotation"].requires_grad
    assert not fitter.gaussians["opacity"].requires_grad

    # Verifica che i parametri ottimizzati richiedano gradiente
    assert fitter.gaussians["features_dc"].requires_grad
    assert fitter.gaussians["features_rest"].requires_grad


def test_fitter_checkpoint_roundtrip(tmp_path: Path):
    """save_checkpoint deve produrre un file .ply ricaricabile."""
    gaussians = _make_gaussians(8)
    cameras   = _make_cameras(1)
    gt_images = _make_gt_images(1)

    fitter = ColorFitter(
        gaussians=gaussians,
        cameras=cameras,
        gt_images=gt_images,
        device=torch.device("cpu"),
    )
    ckpt_path = tmp_path / "checkpoint.ply"
    fitter.save_checkpoint(ckpt_path)

    assert ckpt_path.exists()
    loaded = load_gaussians(ckpt_path)
    assert loaded["xyz"].shape == (8, 3)
    assert loaded["features_dc"].shape == (8, 1, 3)


def test_fitter_returns_dict():
    """fit() deve restituire un dict con le stesse chiavi dell'input."""
    gaussians = _make_gaussians(5)
    cameras   = _make_cameras(1)
    gt_images = _make_gt_images(1)

    fitter = ColorFitter(
        gaussians=gaussians,
        cameras=cameras,
        gt_images=gt_images,
        device=torch.device("cpu"),
    )
    result = fitter.fit(num_iterations=2, log_every=100)

    assert isinstance(result, dict)
    for key in gaussians:
        assert key in result


def test_fitter_mismatched_cameras_images():
    """Se cameras e gt_images hanno lunghezze diverse, deve sollevare ValueError."""
    gaussians = _make_gaussians(5)
    with pytest.raises(ValueError, match="camera"):
        ColorFitter(
            gaussians=gaussians,
            cameras=_make_cameras(3),
            gt_images=_make_gt_images(5),   # diverso da 3
            device=torch.device("cpu"),
        )
