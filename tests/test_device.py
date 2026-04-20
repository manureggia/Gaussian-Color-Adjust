"""Test per utils/device.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest
from utils.device import get_device, get_device_name, get_gsplat_available


def test_get_device_returns_torch_device():
    """get_device() deve restituire un oggetto torch.device."""
    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_type_is_valid():
    """Il tipo di device deve essere uno dei backend supportati."""
    device = get_device()
    assert device.type in ("cuda", "mps", "cpu")


def test_get_device_memoized():
    """Chiamate successive restituiscono lo stesso oggetto."""
    d1 = get_device()
    d2 = get_device()
    assert d1 == d2


def test_get_device_name_returns_string():
    """get_device_name() deve restituire una stringa non vuota."""
    name = get_device_name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_get_device_name_contains_backend():
    """Il nome del device deve contenere uno dei backend attesi."""
    name = get_device_name()
    expected_keywords = ["CUDA", "MPS", "ROCm", "CPU", "Apple"]
    assert any(kw in name for kw in expected_keywords), (
        f"Nome device inatteso: '{name}'"
    )


def test_get_gsplat_available_returns_bool():
    """get_gsplat_available() deve restituire un bool."""
    result = get_gsplat_available()
    assert isinstance(result, bool)


def test_mps_fallback_env_set():
    """Su MPS, PYTORCH_ENABLE_MPS_FALLBACK deve essere impostato a '1'."""
    import os
    device = get_device()
    if device.type == "mps":
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
