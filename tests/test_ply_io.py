"""Test per utils/ply_io.py — caricamento e salvataggio file PLY."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from utils.ply_io import load_gaussians, save_gaussians


def _make_gaussians(n: int = 10) -> dict[str, torch.Tensor]:
    """Crea un dict di gaussiane sintetiche deterministico."""
    torch.manual_seed(0)
    return {
        "xyz":           torch.randn(n, 3),
        "features_dc":   torch.randn(n, 1, 3),
        "features_rest": torch.randn(n, 15, 3),
        "scaling":       torch.randn(n, 3),
        "rotation":      F.normalize(torch.randn(n, 4), dim=-1),
        "opacity":       torch.randn(n, 1),
    }


def test_save_and_load_roundtrip(tmp_path: Path):
    """Salva e ricarica gaussiane: i valori devono essere identici."""
    ply_path = tmp_path / "test.ply"
    original = _make_gaussians(15)
    save_gaussians(ply_path, original)

    loaded = load_gaussians(ply_path)

    for key in original:
        assert key in loaded, f"Chiave mancante dopo reload: {key}"
        torch.testing.assert_close(
            original[key], loaded[key], atol=1e-5, rtol=1e-5,
            msg=f"Mismatch su {key}",
        )


def test_load_returns_correct_shapes(tmp_path: Path):
    """Le forme dei tensori caricate devono corrispondere allo schema atteso."""
    N = 12
    ply_path = tmp_path / "shapes.ply"
    save_gaussians(ply_path, _make_gaussians(N))
    g = load_gaussians(ply_path)

    assert g["xyz"].shape           == (N, 3),   f"xyz: {g['xyz'].shape}"
    assert g["features_dc"].shape   == (N, 1, 3), f"features_dc: {g['features_dc'].shape}"
    assert g["features_rest"].shape == (N, 15, 3), f"features_rest: {g['features_rest'].shape}"
    assert g["scaling"].shape       == (N, 3),   f"scaling: {g['scaling'].shape}"
    assert g["rotation"].shape      == (N, 4),   f"rotation: {g['rotation'].shape}"
    assert g["opacity"].shape       == (N, 1),   f"opacity: {g['opacity'].shape}"


def test_load_returns_float32(tmp_path: Path):
    """Tutti i tensori caricati devono essere float32."""
    ply_path = tmp_path / "dtype.ply"
    save_gaussians(ply_path, _make_gaussians(5))
    g = load_gaussians(ply_path)
    for key, tensor in g.items():
        assert tensor.dtype == torch.float32, f"{key} non è float32: {tensor.dtype}"


def test_file_not_found_raises():
    """Caricare un file inesistente deve sollevare FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_gaussians("/nonexistent/path/scene.ply")


def test_save_creates_parent_dirs(tmp_path: Path):
    """save_gaussians deve creare le directory intermedie se assenti."""
    ply_path = tmp_path / "nested" / "dir" / "scene.ply"
    save_gaussians(ply_path, _make_gaussians(3))
    assert ply_path.exists()
