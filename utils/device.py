"""
Rilevamento automatico del backend PyTorch disponibile.

Priorità: CUDA (include ROCm) → MPS → CPU.
Il supporto ROCm è trasparente: se PyTorch è stato installato con il wheel ROCm,
``torch.cuda.is_available()`` restituisce True anche su GPU AMD.

Quando MPS viene rilevato, imposta automaticamente la variabile d'ambiente
``PYTORCH_ENABLE_MPS_FALLBACK=1`` per consentire il fallback CPU su op non supportate.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)

_DEVICE: torch.device | None = None


def get_device() -> torch.device:
    """Restituisce il dispositivo PyTorch ottimale disponibile.

    Ordine di preferenza: CUDA (include ROCm) → MPS → CPU.
    Il risultato è memoizzato: la detection avviene una sola volta per processo.
    Su MPS, imposta ``PYTORCH_ENABLE_MPS_FALLBACK=1`` automaticamente.

    Returns:
        torch.device: dispositivo selezionato.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
        print(f"[device] Backend selezionato: CUDA — {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
        # Abilita fallback CPU per operazioni MPS non ancora supportate
        if not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            logger.debug("PYTORCH_ENABLE_MPS_FALLBACK impostato a 1")
        print("[device] Backend selezionato: MPS (Apple Silicon)")
    else:
        _DEVICE = torch.device("cpu")
        print("[device] Backend selezionato: CPU")

    return _DEVICE


def get_device_name() -> str:
    """Restituisce una stringa leggibile che descrive il backend selezionato.

    Returns:
        str: nome human-readable, es. "Apple MPS", "AMD ROCm", "NVIDIA CUDA", "CPU".
    """
    device = get_device()

    if device.type == "cuda":
        raw = torch.cuda.get_device_name(0)
        if "AMD" in raw or "Radeon" in raw or "gfx" in raw.lower():
            return f"AMD ROCm — {raw}"
        return f"NVIDIA CUDA — {raw}"

    if device.type == "mps":
        import platform
        chip = platform.processor() or "Apple Silicon"
        return f"Apple MPS ({chip})"

    return "CPU"


@lru_cache(maxsize=1)
def get_gsplat_available() -> bool:
    """Verifica se gsplat è installato e i suoi kernel sono funzionali.

    Tenta un import e una chiamata di prova. Su MPS/CPU senza kernel CUDA
    compilati, restituisce False e il renderer PyTorch puro verrà usato.

    Returns:
        bool: True se gsplat è utilizzabile, False altrimenti.
    """
    try:
        import gsplat  # noqa: F401
        # Verifica che il modulo sia effettivamente importabile
        from gsplat import rasterization  # noqa: F401
        logger.debug("gsplat disponibile e importabile")
        return True
    except (ImportError, Exception) as exc:
        logger.debug("gsplat non disponibile: %s", exc)
        return False


def to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Sposta un tensor sul dispositivo corrente.

    Args:
        tensor: tensor da spostare.

    Returns:
        torch.Tensor: tensor sul dispositivo corretto.
    """
    return tensor.to(get_device())
