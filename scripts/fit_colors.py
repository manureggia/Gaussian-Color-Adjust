#!/usr/bin/env python
"""
CLI: re-fitting dei colori SH delle Gaussiane su immagini editate.

Le immagini vengono abbinate alle camera tramite il campo ``image_name``
in cameras.json. Se un'immagine non viene trovata nella cartella editata,
la camera corrispondente viene saltata con un avviso.

Esempio::

    python scripts/fit_colors.py \\
        --ply data/garden/point_cloud.ply \\
        --images_dir data/garden/input_edited \\
        --cameras_json data/garden/cameras.json \\
        --output_ply data/garden/point_cloud_edited.ply \\
        --num_iterations 2000 \\
        --lr_dc 0.005 \\
        --lr_rest 0.001 \\
        --log_every 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from training.color_fitter import ColorFitter
from utils.device import get_device, get_device_name
from utils.ply_io import load_gaussians, save_gaussians
from utils.renderer import get_active_backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fit_colors")


def parse_args() -> argparse.Namespace:
    """Parsa gli argomenti da riga di comando."""
    parser = argparse.ArgumentParser(
        description="Re-fitting dei colori Gaussiani su immagini editate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ply", type=Path, required=True,
        help="File .ply delle gaussiane originali",
    )
    parser.add_argument(
        "--images_dir", type=Path, required=True,
        help="Cartella con le immagini editate",
    )
    parser.add_argument(
        "--cameras_json", type=Path, required=True,
        help="File cameras.json con le pose camera",
    )
    parser.add_argument(
        "--output_ply", type=Path, required=True,
        help="File .ply di output con i colori aggiornati",
    )
    parser.add_argument(
        "--num_iterations", type=int, default=2000,
        help="Numero di iterazioni di ottimizzazione",
    )
    parser.add_argument(
        "--lr_dc", type=float, default=0.005,
        help="Learning rate per features_dc (SH grado 0)",
    )
    parser.add_argument(
        "--lr_rest", type=float, default=0.001,
        help="Learning rate per features_rest (SH gradi 1-3)",
    )
    parser.add_argument(
        "--lambda_dssim", type=float, default=0.2,
        help="Peso della loss SSIM nella loss combinata",
    )
    parser.add_argument(
        "--log_every", type=int, default=100,
        help="Frequenza di stampa dei log (iterazioni)",
    )
    return parser.parse_args()


def _load_cameras_json(path: Path) -> list[dict]:
    """Carica cameras.json e converte R/T in tensori PyTorch."""
    with open(path) as f:
        raw = json.load(f)
    cameras = []
    for cam in raw:
        c = dict(cam)
        c["R"] = torch.tensor(cam["R"], dtype=torch.float32)
        c["T"] = torch.tensor(cam["T"], dtype=torch.float32)
        cameras.append(c)
    return cameras


def main() -> None:
    """Punto di ingresso principale."""
    args = parse_args()

    device = get_device()
    logger.info("Device: %s  |  Renderer: %s", get_device_name(), get_active_backend())

    # ---- Validazione input ----
    for p, name in [(args.ply, "PLY"), (args.images_dir, "images_dir"), (args.cameras_json, "cameras_json")]:
        if not p.exists():
            logger.error("File/cartella non trovato: %s (%s)", p, name)
            sys.exit(1)

    # ---- Carica dati ----
    logger.info("Caricamento gaussiane da %s", args.ply)
    gaussians = load_gaussians(args.ply)
    logger.info("Caricate %d gaussiane", gaussians["xyz"].shape[0])

    logger.info("Caricamento camera da %s", args.cameras_json)
    all_cameras = _load_cameras_json(args.cameras_json)
    logger.info("Caricate %d camera", len(all_cameras))

    # ---- Abbinamento immagini ↔ camera per image_name ----
    matched_cameras: list[dict] = []
    matched_images: list[Path] = []

    for cam in all_cameras:
        img_name = cam.get("image_name", "")
        img_path = args.images_dir / img_name
        # Prova anche senza estensione o con estensione alternativa
        if not img_path.exists():
            stem = Path(img_name).stem
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
                alt = args.images_dir / (stem + ext)
                if alt.exists():
                    img_path = alt
                    break
        if not img_path.exists():
            logger.warning("Immagine non trovata per camera '%s', salto", img_name)
            continue
        matched_cameras.append(cam)
        matched_images.append(img_path)

    if not matched_cameras:
        logger.error(
            "Nessuna immagine abbinata alle camera. "
            "Verifica che image_name in cameras.json corrisponda ai file in %s",
            args.images_dir,
        )
        sys.exit(1)

    logger.info(
        "Abbinate %d/%d coppie camera-immagine",
        len(matched_cameras), len(all_cameras),
    )

    # ---- Fitting ----
    fitter = ColorFitter(
        gaussians=gaussians,
        cameras=matched_cameras,
        gt_images=[str(p) for p in matched_images],
        device=device,
    )
    updated = fitter.fit(
        num_iterations=args.num_iterations,
        lr_dc=args.lr_dc,
        lr_rest=args.lr_rest,
        lambda_dssim=args.lambda_dssim,
        log_every=args.log_every,
    )

    # ---- Salva risultato ----
    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    save_gaussians(args.output_ply, updated)
    logger.info("Gaussiane aggiornate salvate in %s", args.output_ply)


if __name__ == "__main__":
    main()
