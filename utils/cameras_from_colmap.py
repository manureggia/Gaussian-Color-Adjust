"""
Conversione delle pose COLMAP in formato cameras.json.

Supporta due modalità di lettura:
  1. Libreria ``pycolmap`` (se installata).
  2. Parsing manuale dei file binari COLMAP ``cameras.bin`` / ``images.bin``.

Formato di output cameras.json (lista di oggetti):
  {
    "id":         <int>,
    "image_name": <str>,
    "R":          [[...], [...], [...]]  (matrice 3×3, row-major),
    "T":          [tx, ty, tz],
    "fx": <float>, "fy": <float>,
    "cx": <float>, "cy": <float>,
    "width": <int>, "height": <int>
  }

Note COLMAP → 3DGS:
  - COLMAP memorizza la rotazione come quaternione (qw, qx, qy, qz) e la
    traslazione T come la posizione del centro del mondo in camera space.
  - La matrice R ricavata è R_world2cam; T è la traslazione world2cam.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing manuale dei file binari COLMAP
# ---------------------------------------------------------------------------

def _read_cameras_bin(path: Path) -> dict[int, dict]:
    """Legge ``cameras.bin`` e restituisce dict id→camera_params."""
    cameras: dict[int, dict] = {}
    with open(path, "rb") as f:
        n_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_cameras):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<I", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            # Numero di parametri per modello
            # 0=SIMPLE_PINHOLE(1), 1=PINHOLE(2), 2=SIMPLE_RADIAL(2),
            # 3=RADIAL(3), 4=OPENCV(4), ...
            n_params = {0: 1, 1: 4, 2: 2, 3: 3, 4: 4, 5: 8, 6: 1, 7: 2}.get(model_id, 4)
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[cam_id] = {
                "model_id": model_id,
                "width": int(width),
                "height": int(height),
                "params": list(params),
            }
    return cameras


def _read_images_bin(path: Path) -> dict[int, dict]:
    """Legge ``images.bin`` e restituisce dict id→image_data."""
    images: dict[int, dict] = {}
    with open(path, "rb") as f:
        n_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_images):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))   # qw, qx, qy, qz
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            # Leggi i 2D points (non ci servono ma bisogna consumare i byte)
            n_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * n_points2d)   # x, y, point3d_id (8+8+8 byte)
            images[img_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "cam_id": cam_id,
                "name": name.decode("utf-8"),
            }
    return images


def _qvec_to_rotation_matrix(qvec) -> np.ndarray:
    """Converte quaternione COLMAP (qw,qx,qy,qz) in matrice 3×3."""
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz,  2*qx*qy - 2*qw*qz,    2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz,       1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy,       2*qy*qz + 2*qw*qx,     1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)
    return R


def _extract_intrinsics(cam: dict) -> tuple[float, float, float, float]:
    """Estrae (fx, fy, cx, cy) dai parametri del modello di camera COLMAP."""
    params = cam["params"]
    model_id = cam["model_id"]
    w, h = cam["width"], cam["height"]
    if model_id == 0:   # SIMPLE_PINHOLE: f, cx, cy
        return params[0], params[0], params[1], params[2]
    if model_id == 1:   # PINHOLE: fx, fy, cx, cy
        return params[0], params[1], params[2], params[3]
    if model_id in (2, 3):  # SIMPLE_RADIAL / RADIAL: f, cx, cy, [k1, ...]
        return params[0], params[0], params[1], params[2]
    if model_id in (4, 5):  # OPENCV / FULL_OPENCV: fx, fy, cx, cy, ...
        return params[0], params[1], params[2], params[3]
    # Fallback: usa focal = max(w,h) * 0.7, centro immagine
    logger.warning("Modello camera %d non riconosciuto, uso valori default", model_id)
    f = max(w, h) * 0.7
    return f, f, w / 2.0, h / 2.0


# ---------------------------------------------------------------------------
# API pubblica
# ---------------------------------------------------------------------------

def convert_colmap_to_json(
    colmap_sparse_dir: str | Path,
    output_json: str | Path,
) -> None:
    """Converte le pose COLMAP sparse in ``cameras.json`` per il progetto.

    Tenta prima di usare ``pycolmap``; se non disponibile, esegue il parsing
    manuale dei file binari.

    Args:
        colmap_sparse_dir: cartella con ``cameras.bin`` e ``images.bin``
                           (tipicamente ``sparse/0/``).
        output_json: percorso del file JSON di output.

    Raises:
        FileNotFoundError: se i file binari COLMAP non esistono.
    """
    colmap_dir = Path(colmap_sparse_dir)
    output_json = Path(output_json)

    cam_bin = colmap_dir / "cameras.bin"
    img_bin = colmap_dir / "images.bin"

    if not cam_bin.exists() or not img_bin.exists():
        raise FileNotFoundError(
            f"File COLMAP non trovati in {colmap_dir}. "
            "Assicurati che cameras.bin e images.bin esistano."
        )

    # Prova pycolmap
    try:
        import pycolmap
        reconstruction = pycolmap.Reconstruction(str(colmap_dir))
        cameras_raw = reconstruction.cameras
        images_raw = reconstruction.images
        logger.info("Usando pycolmap per leggere la ricostruzione COLMAP")
        _convert_pycolmap(cameras_raw, images_raw, output_json)
        return
    except ImportError:
        logger.info("pycolmap non disponibile, uso parsing manuale dei binari")
    except Exception as exc:
        logger.warning("pycolmap ha fallito (%s), uso parsing manuale", exc)

    # Parsing manuale
    cameras_raw = _read_cameras_bin(cam_bin)
    images_raw = _read_images_bin(img_bin)
    _convert_manual(cameras_raw, images_raw, output_json)


def _convert_manual(cameras_raw: dict, images_raw: dict, output_json: Path) -> None:
    """Converte i dati parsati manualmente in cameras.json."""
    result = []
    for img_id, img in images_raw.items():
        cam = cameras_raw[img["cam_id"]]
        R = _qvec_to_rotation_matrix(img["qvec"])
        T = np.array(img["tvec"], dtype=np.float64)
        fx, fy, cx, cy = _extract_intrinsics(cam)
        result.append({
            "id": img_id,
            "image_name": img["name"],
            "R": R.tolist(),
            "T": T.tolist(),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": cam["width"],
            "height": cam["height"],
        })
    result.sort(key=lambda x: x["image_name"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Scritte %d pose camera in %s", len(result), output_json)


def _convert_pycolmap(cameras_raw, images_raw, output_json: Path) -> None:
    """Converte i dati pycolmap in cameras.json."""
    result = []
    for img_id, img in images_raw.items():
        cam = cameras_raw[img.camera_id]
        # pycolmap < 0.6: img.rotation_matrix(), img.tvec
        # pycolmap >= 0.6: img.cam_from_world.rotation.matrix(), img.cam_from_world.translation
        if hasattr(img, "rotation_matrix"):
            R = img.rotation_matrix()
            T = np.array(img.tvec, dtype=np.float64)
        else:
            R = img.cam_from_world.rotation.matrix()
            T = np.array(img.cam_from_world.translation, dtype=np.float64)
        params = cam.params
        model = cam.model.name
        if "PINHOLE" in model and "SIMPLE" not in model:
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        else:
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        result.append({
            "id": img_id,
            "image_name": img.name,
            "R": np.array(R).tolist(),
            "T": T.tolist(),
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "width": int(cam.width),
            "height": int(cam.height),
        })
    result.sort(key=lambda x: x["image_name"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Scritte %d pose camera in %s (via pycolmap)", len(result), output_json)
