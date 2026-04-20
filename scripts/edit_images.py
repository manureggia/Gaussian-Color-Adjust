#!/usr/bin/env python
"""
CLI: modifica tutte le immagini di una cartella con InstructPix2Pix.

Esempio::

    python scripts/edit_images.py \\
        --input_dir data/garden/input \\
        --output_dir data/garden/input_edited \\
        --prompt "make it look like autumn with orange and red leaves" \\
        --num_steps 25 \\
        --guidance_scale 7.5 \\
        --image_guidance_scale 1.5
"""

import argparse
import logging
import sys
from pathlib import Path

# Permette import assoluti dalla root del progetto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion.editor import ImageEditor
from utils.device import get_device, get_device_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("edit_images")


def parse_args() -> argparse.Namespace:
    """Parsa gli argomenti da riga di comando."""
    parser = argparse.ArgumentParser(
        description="Modifica immagini con InstructPix2Pix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir", type=Path, required=True,
        help="Cartella con le immagini originali (.jpg/.png)",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Cartella di output per le immagini modificate",
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Istruzione di modifica (es. 'make it look like autumn')",
    )
    parser.add_argument(
        "--model_id", type=str,
        default="timbrooks/instruct-pix2pix",
        help="Identificatore HuggingFace del modello",
    )
    parser.add_argument(
        "--num_steps", type=int, default=20,
        help="Passi di denoising",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5,
        help="Scala di guida del testo (CFG scale)",
    )
    parser.add_argument(
        "--image_guidance_scale", type=float, default=1.5,
        help="Scala di ancoraggio all'immagine originale",
    )
    parser.add_argument(
        "--no_skip_existing", action="store_true",
        help="Rielabora le immagini già presenti in output_dir",
    )
    return parser.parse_args()


def main() -> None:
    """Punto di ingresso principale."""
    args = parse_args()

    # Rilevamento device
    device = get_device()
    logger.info("Device attivo: %s", get_device_name())

    if not args.input_dir.exists():
        logger.error("La cartella di input non esiste: %s", args.input_dir)
        sys.exit(1)

    logger.info("Caricamento modello di diffusione...")
    editor = ImageEditor(model_id=args.model_id, device=device)

    logger.info(
        "Avvio editing: input=%s  output=%s  prompt='%s'",
        args.input_dir, args.output_dir, args.prompt,
    )
    editor.edit_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        skip_existing=not args.no_skip_existing,
    )
    logger.info("Editing completato. Output salvato in: %s", args.output_dir)


if __name__ == "__main__":
    main()
