"""
Test per utils/renderer.py — entrambi i backend (gsplat e PyTorch puro).

Il backend PyTorch puro viene sempre testato (nessuna GPU richiesta).
Il backend gsplat viene testato solo se get_gsplat_available() è True;
altrimenti i relativi test vengono saltati automaticamente.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from utils.device import get_gsplat_available
from utils.renderer import (
    _render_pure_pytorch,
    camera_to_K,
    camera_to_viewmat,
    get_active_backend,
    render,
)


# ---------------------------------------------------------------------------
# Fixture locali (ridondanti rispetto a conftest ma self-contained)
# ---------------------------------------------------------------------------

@pytest.fixture
def gaussians():
    import torch.nn.functional as F
    torch.manual_seed(7)
    N = 20
    return {
        "xyz":           torch.randn(N, 3) * 0.5,
        "features_dc":   torch.randn(N, 1, 3) * 0.1,
        "features_rest": torch.randn(N, 15, 3) * 0.01,
        "scaling":       torch.full((N, 3), -3.0),   # piccole
        "rotation":      F.normalize(torch.randn(N, 4), dim=-1),
        "opacity":       torch.zeros(N, 1),
    }


@pytest.fixture
def camera():
    """Camera con T=[0,0,5]: punti con z_world~0 hanno z_cam~5 (davanti)."""
    return {
        "R":      torch.eye(3),
        "T":      torch.tensor([0.0, 0.0, 5.0]),
        "fx":     100.0, "fy": 100.0,
        "cx":     32.0,  "cy": 32.0,
        "width":  64,    "height": 64,
    }


# ---------------------------------------------------------------------------
# Helper per muovere gaussiane su CPU
# ---------------------------------------------------------------------------

def _to_cpu(g: dict) -> dict:
    return {k: v.cpu() for k, v in g.items()}


# ---------------------------------------------------------------------------
# Test helper: camera_to_viewmat / camera_to_K
# ---------------------------------------------------------------------------

def test_camera_to_viewmat_shape(camera):
    vm = camera_to_viewmat(camera)
    assert vm.shape == (1, 4, 4)


def test_camera_to_viewmat_identity(camera):
    """Con R=I e T=0, il 3×3 superiore è identità."""
    cam_zero = dict(camera)
    cam_zero["T"] = torch.zeros(3)
    # Forza CPU per confronto deterministico
    vm = camera_to_viewmat(cam_zero, device=torch.device("cpu"))
    assert torch.allclose(vm[0, :3, :3], torch.eye(3), atol=1e-6)


def test_camera_to_K_shape(camera):
    K = camera_to_K(camera)
    assert K.shape == (1, 3, 3)


def test_camera_to_K_values(camera):
    K = camera_to_K(camera)
    assert K[0, 0, 0].item() == pytest.approx(camera["fx"])
    assert K[0, 1, 1].item() == pytest.approx(camera["fy"])
    assert K[0, 0, 2].item() == pytest.approx(camera["cx"])
    assert K[0, 1, 2].item() == pytest.approx(camera["cy"])


# ---------------------------------------------------------------------------
# Test backend PyTorch puro
# ---------------------------------------------------------------------------

def test_pure_pytorch_output_shape(gaussians, camera):
    """Il renderer PyTorch puro deve produrre un tensore (H, W, 3)."""
    out = _render_pure_pytorch(gaussians, camera, (64, 64))
    assert out.shape == (64, 64, 3)


def test_pure_pytorch_output_range(gaussians, camera):
    """I valori devono essere in [0, 1]."""
    out = _render_pure_pytorch(gaussians, camera, (64, 64))
    assert out.min().item() >= 0.0 - 1e-6
    assert out.max().item() <= 1.0 + 1e-6


def test_pure_pytorch_gradient_flows(gaussians, camera):
    """Il gradiente deve fluire fino a features_dc dopo backward()."""
    g = _to_cpu(gaussians)
    g["features_dc"] = g["features_dc"].requires_grad_(True)
    g["features_rest"] = g["features_rest"].requires_grad_(True)

    out = _render_pure_pytorch(g, camera, (64, 64))
    loss = out.mean()
    loss.backward()

    assert g["features_dc"].grad is not None, "Nessun gradiente su features_dc"
    assert g["features_rest"].grad is not None, "Nessun gradiente su features_rest"


def test_pure_pytorch_no_grad_on_geometry(gaussians, camera):
    """xyz e scaling non devono ricevere gradiente."""
    g = _to_cpu(gaussians)
    g["features_dc"] = g["features_dc"].requires_grad_(True)

    out = _render_pure_pytorch(g, camera, (64, 64))
    out.mean().backward()

    assert g["xyz"].grad is None
    assert g["scaling"].grad is None


def test_pure_pytorch_empty_scene(camera):
    """Una scena vuota (0 gaussiane) deve restituire un'immagine nera."""
    import torch.nn.functional as F
    g = {
        "xyz":           torch.zeros(0, 3),
        "features_dc":   torch.zeros(0, 1, 3),
        "features_rest": torch.zeros(0, 15, 3),
        "scaling":       torch.zeros(0, 3),
        "rotation":      torch.zeros(0, 4),
        "opacity":       torch.zeros(0, 1),
    }
    out = _render_pure_pytorch(g, camera, (32, 32))
    assert out.shape == (32, 32, 3)
    assert out.sum().item() == 0.0


# ---------------------------------------------------------------------------
# Test backend gsplat (saltato se non disponibile)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not get_gsplat_available(),
    reason="gsplat non disponibile su questo sistema",
)
def test_gsplat_output_shape(gaussians, camera):
    from utils.renderer import _render_with_gsplat
    out = _render_with_gsplat(gaussians, camera, (64, 64))
    assert out.shape == (64, 64, 3)


@pytest.mark.skipif(
    not get_gsplat_available(),
    reason="gsplat non disponibile su questo sistema",
)
def test_gsplat_output_range(gaussians, camera):
    from utils.renderer import _render_with_gsplat
    out = _render_with_gsplat(gaussians, camera, (64, 64))
    assert out.min().item() >= -1e-6
    assert out.max().item() <= 1.0 + 1e-6


@pytest.mark.skipif(
    not get_gsplat_available(),
    reason="gsplat non disponibile su questo sistema",
)
def test_gsplat_gradient_flows(gaussians, camera):
    from utils.renderer import _render_with_gsplat
    g = dict(gaussians)
    g["features_dc"] = g["features_dc"].requires_grad_(True)
    out = _render_with_gsplat(g, camera, (64, 64))
    out.mean().backward()
    assert g["features_dc"].grad is not None


# ---------------------------------------------------------------------------
# Test funzione pubblica render()
# ---------------------------------------------------------------------------

def test_render_public_api(gaussians, camera):
    """La funzione render() pubblica deve produrre un tensore (H,W,3)."""
    out = render(gaussians, camera, (64, 64))
    assert out.shape == (64, 64, 3)


def test_render_backend_name():
    """get_active_backend() deve restituire 'gsplat' o 'pure_pytorch'."""
    backend = get_active_backend()
    assert backend in ("gsplat", "pure_pytorch")
