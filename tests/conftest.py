"""
Fixture condivise per i test del progetto Gaussian Color Adjust.

Tutte le fixture usano dati sintetici di piccole dimensioni (20 Gaussiane,
immagini 64×64) e non richiedono GPU reale né file di scena.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Permette import dalla root del progetto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def synthetic_gaussians() -> dict[str, torch.Tensor]:
    """Dict di 20 gaussiane sintetiche con geometria piccola e opacità media."""
    N = 20
    torch.manual_seed(42)
    return {
        "xyz":           torch.randn(N, 3),
        "features_dc":   torch.randn(N, 1, 3) * 0.1,
        "features_rest": torch.randn(N, 15, 3) * 0.01,
        "scaling":       torch.full((N, 3), -2.0),       # scale piccola: exp(-2) ≈ 0.135
        "rotation":      F.normalize(torch.randn(N, 4), dim=-1),
        "opacity":       torch.zeros(N, 1),               # sigmoid(0) = 0.5
    }


@pytest.fixture
def synthetic_camera() -> dict:
    """Camera frontale semplice con gaussiane davanti alla camera.

    Convenzione world-to-camera: xyz_cam = R @ xyz + T.
    Con R=I e T=[0,0,5] i punti con xyz~0 hanno z_cam~5 > 0 (davanti alla camera).
    """
    return {
        "R":      torch.eye(3),
        "T":      torch.tensor([0.0, 0.0, 5.0]),
        "fx":     100.0,
        "fy":     100.0,
        "cx":     32.0,
        "cy":     32.0,
        "width":  64,
        "height": 64,
    }


@pytest.fixture
def image_size() -> tuple[int, int]:
    """Dimensione immagine di test."""
    return (64, 64)


@pytest.fixture
def tmp_ply(tmp_path: Path) -> Path:
    """Percorso temporaneo per file PLY di test."""
    return tmp_path / "test_gaussians.ply"
