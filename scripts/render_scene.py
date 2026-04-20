#!/usr/bin/env python
"""
CLI: renderizza una scena Gaussian Splatting da un dato punto di vista.

Esempio::

    python scripts/render_scene.py \\
        --ply data/garden/point_cloud_edited.ply \\
        --cameras_json data/garden/cameras.json \\
        --camera_id 0 \\
        --output renders/frame_000.png
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.device import get_device, get_device_name
from utils.ply_io import load_gaussians
from utils.renderer import render, get_active_backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("render_scene")


def parse_args() -> argparse.Namespace:
    """Parsa gli argomenti da riga di comando."""
    parser = argparse.ArgumentParser(
        description="Renderizza una scena 3DGS da un dato punto di vista",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ply", type=Path, required=True,
        help="File .ply delle gaussiane",
    )
    parser.add_argument(
        "--cameras_json", type=Path, required=True,
        help="File cameras.json con le pose camera",
    )
    parser.add_argument(
        "--camera_id", type=int, default=0,
        help="Indice della camera in cameras.json (0-based)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="File PNG di output",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Larghezza di output (default: dalla camera)",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Altezza di output (default: dalla camera)",
    )
    return parser.parse_args()


def main() -> None:
    """Punto di ingresso principale."""
    args = parse_args()

    device = get_device()
    logger.info("Device: %s  |  Renderer: %s", get_device_name(), get_active_backend())

    # ---- Validazione ----
    for p, name in [(args.ply, "PLY"), (args.cameras_json, "cameras_json")]:
        if not p.exists():
            logger.error("File non trovato: %s (%s)", p, name)
            sys.exit(1)

    # ---- Carica gaussiane ----
    logger.info("Caricamento gaussiane da %s", args.ply)
    gaussians = load_gaussians(args.ply)
    gaussians = {k: v.to(device) for k, v in gaussians.items()}
    logger.info("Caricate %d gaussiane", gaussians["xyz"].shape[0])

    # ---- Carica camera ----
    with open(args.cameras_json) as f:
        cameras = json.load(f)

    if args.camera_id >= len(cameras):
        logger.error(
            "camera_id=%d fuori range (cameras.json contiene %d entry)",
            args.camera_id, len(cameras),
        )
        sys.exit(1)

    cam_raw = cameras[args.camera_id]
    camera = {
        "R": torch.tensor(cam_raw["R"], dtype=torch.float32, device=device),
        "T": torch.tensor(cam_raw["T"], dtype=torch.float32, device=device),
        "fx": float(cam_raw["fx"]),
        "fy": float(cam_raw["fy"]),
        "cx": float(cam_raw["cx"]),
        "cy": float(cam_raw["cy"]),
        "width":  int(cam_raw.get("width",  args.width  or 800)),
        "height": int(cam_raw.get("height", args.height or 600)),
    }

    W = args.width  or camera["width"]
    H = args.height or camera["height"]
    logger.info(
        "Rendering camera %d (%s) — %dx%d",
        args.camera_id, cam_raw.get("image_name", "?"), W, H,
    )

    # ---- Rendering ----
    with torch.no_grad():
        image_tensor = render(gaussians, camera, (H, W))   # (H, W, 3)

    # ---- Salva immagine ----
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        logger.error("Pillow non trovato: pip install Pillow")
        sys.exit(1)

    img_np = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype("uint8")
    pil_img = Image.fromarray(img_np)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(args.output)
    logger.info("Rendering salvato in %s", args.output)


if __name__ == "__main__":
    main()
