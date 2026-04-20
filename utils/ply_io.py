"""
Caricamento e salvataggio di file .ply nel formato 3D Gaussian Splatting.

Convenzioni di naming degli attributi (formato originale 3DGS):
  Posizione  : x, y, z
  Normali    : nx, ny, nz  (solitamente zero, presenti per compatibilità)
  SH DC      : f_dc_0, f_dc_1, f_dc_2
  SH rest    : f_rest_0 … f_rest_44  (15 bande × 3 canali = 45 valori)
  Opacità    : opacity
  Scala      : scale_0, scale_1, scale_2  (log-scala)
  Rotazione  : rot_0, rot_1, rot_2, rot_3  (quaternione wxyz)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

try:
    from plyfile import PlyData, PlyElement
except ImportError as exc:
    raise ImportError(
        "plyfile non trovato. Installa con: pip install plyfile"
    ) from exc

logger = logging.getLogger(__name__)

# Numero di coefficienti SH rest per grado 3 (escludendo DC): 15 bande × 3 canali
_N_REST = 45


def load_gaussians(path: str | Path) -> dict[str, torch.Tensor]:
    """Carica un file .ply 3DGS e restituisce un dict di tensori float32.

    Args:
        path: percorso al file .ply.

    Returns:
        dict con chiavi:
          - ``xyz``           (N, 3)
          - ``features_dc``   (N, 1, 3)  — coefficienti SH DC
          - ``features_rest`` (N, 15, 3) — coefficienti SH gradi 1-3
          - ``scaling``       (N, 3)     — log-scala
          - ``rotation``      (N, 4)     — quaternione wxyz (normalizzato)
          - ``opacity``       (N, 1)     — logit-opacità

    Raises:
        FileNotFoundError: se il file non esiste.
        KeyError: se mancano attributi obbligatori.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File PLY non trovato: {path}")

    logger.info("Caricamento gaussiane da %s", path)
    ply_data = PlyData.read(str(path))
    vertex = ply_data["vertex"]

    def _get(name: str) -> np.ndarray:
        try:
            return np.asarray(vertex[name], dtype=np.float32)
        except ValueError as exc:
            raise KeyError(
                f"Attributo '{name}' mancante nel file PLY: {path}"
            ) from exc

    # ---- Posizioni ----
    xyz = np.stack([_get("x"), _get("y"), _get("z")], axis=1)  # (N,3)
    n = xyz.shape[0]
    logger.info("Caricate %d gaussiane", n)

    # ---- Spherical Harmonics DC ----
    dc = np.stack([_get("f_dc_0"), _get("f_dc_1"), _get("f_dc_2")], axis=1)  # (N,3)
    features_dc = dc[:, np.newaxis, :]  # (N,1,3)

    # ---- Spherical Harmonics rest ----
    rest_cols = [_get(f"f_rest_{i}") for i in range(_N_REST)]  # lista di (N,)
    rest_flat = np.stack(rest_cols, axis=1)  # (N, 45)
    # Layout 3DGS: [R0,R1,...,R14, G0,...,G14, B0,...,B14]
    features_rest = rest_flat.reshape(n, 3, 15).transpose(0, 2, 1)  # (N,15,3)

    # ---- Scala, rotazione, opacità ----
    scaling = np.stack([_get(f"scale_{i}") for i in range(3)], axis=1)  # (N,3)
    rotation = np.stack([_get(f"rot_{i}") for i in range(4)], axis=1)   # (N,4)
    opacity = _get("opacity")[:, np.newaxis]                              # (N,1)

    return {
        "xyz": torch.from_numpy(xyz),
        "features_dc": torch.from_numpy(features_dc),
        "features_rest": torch.from_numpy(features_rest),
        "scaling": torch.from_numpy(scaling),
        "rotation": torch.from_numpy(rotation),
        "opacity": torch.from_numpy(opacity),
    }


def save_gaussians(path: str | Path, gaussians: dict[str, torch.Tensor]) -> None:
    """Salva un dict di gaussiane su file .ply nel formato standard 3DGS.

    Args:
        path: percorso di output (viene creata la directory se necessario).
        gaussians: dict nel formato restituito da :func:`load_gaussians`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _np(key: str) -> np.ndarray:
        return gaussians[key].detach().cpu().float().numpy()

    xyz = _np("xyz")                    # (N,3)
    features_dc = _np("features_dc")   # (N,1,3)
    features_rest = _np("features_rest")  # (N,15,3)
    scaling = _np("scaling")            # (N,3)
    rotation = _np("rotation")          # (N,4)
    opacity = _np("opacity")            # (N,1)

    n = xyz.shape[0]

    # Ricostruisce il layout flat per f_rest: (N,3,15) → (N,45)
    rest_transposed = features_rest.transpose(0, 2, 1)  # (N,3,15)
    rest_flat = rest_transposed.reshape(n, _N_REST)

    # Costruisce la lista di attributi nell'ordine standard 3DGS
    attrs = []
    attrs += [("x", xyz[:, 0]), ("y", xyz[:, 1]), ("z", xyz[:, 2])]
    attrs += [("nx", np.zeros(n)), ("ny", np.zeros(n)), ("nz", np.zeros(n))]
    for i in range(3):
        attrs.append((f"f_dc_{i}", features_dc[:, 0, i]))
    for i in range(_N_REST):
        attrs.append((f"f_rest_{i}", rest_flat[:, i]))
    attrs.append(("opacity", opacity[:, 0]))
    for i in range(3):
        attrs.append((f"scale_{i}", scaling[:, i]))
    for i in range(4):
        attrs.append((f"rot_{i}", rotation[:, i]))

    dtype = [(name, "f4") for name, _ in attrs]
    arr = np.empty(n, dtype=dtype)
    for name, data in attrs:
        arr[name] = data

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(str(path))
    logger.info("Gaussiane salvate in %s (%d punti)", path, n)
