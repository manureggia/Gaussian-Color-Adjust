#!/usr/bin/env python
"""
gaussadj — CLI unificato per Gaussian Color Adjust.

Flusso di lavoro tipico:

  1. gaussadj colmap  <sparse_dir> <output_json>
  2. gaussadj edit    --input_dir ... --output_dir ... --prompt "..."
  3. gaussadj fit     --ply ... --images_dir ... --cameras_json ... --output_ply ...
  4. gaussadj render  --ply ... --cameras_json ... --output ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gaussadj")


# ---------------------------------------------------------------------------
# Sottocomando: colmap
# ---------------------------------------------------------------------------

def _parser_colmap(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "colmap",
        help="Converte le pose COLMAP sparse in cameras.json",
        description=(
            "Legge i file binari COLMAP (cameras.bin + images.bin) nella cartella\n"
            "sparse e genera un file cameras.json compatibile con gli altri comandi.\n\n"
            "Prova automaticamente a usare la libreria pycolmap se installata;\n"
            "in caso contrario esegue il parsing manuale dei file binari.\n\n"
            "Esempio:\n"
            "  gaussadj colmap data/garden/sparse/0 data/garden/cameras.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "sparse_dir",
        type=Path,
        metavar="SPARSE_DIR",
        help=(
            "Cartella COLMAP sparse contenente cameras.bin e images.bin "
            "(tipicamente sparse/0/ dentro la cartella della scena)"
        ),
    )
    p.add_argument(
        "output_json",
        type=Path,
        metavar="OUTPUT_JSON",
        help="Percorso del file cameras.json da creare (es. data/garden/cameras.json)",
    )
    p.set_defaults(func=_run_colmap)


def _run_colmap(args: argparse.Namespace) -> None:
    from utils.cameras_from_colmap import convert_colmap_to_json

    if not args.sparse_dir.exists():
        logger.error("Cartella COLMAP non trovata: %s", args.sparse_dir)
        sys.exit(1)

    logger.info("Conversione pose COLMAP da %s", args.sparse_dir)
    convert_colmap_to_json(args.sparse_dir, args.output_json)
    logger.info("cameras.json salvato in %s", args.output_json)


# ---------------------------------------------------------------------------
# Sottocomando: edit
# ---------------------------------------------------------------------------

def _parser_edit(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "edit",
        help="Modifica immagini con InstructPix2Pix",
        description=(
            "Applica una modifica testuale a tutte le immagini di una cartella\n"
            "usando il modello di diffusione InstructPix2Pix.\n\n"
            "Le immagini già presenti in output_dir vengono saltate per default\n"
            "(usa --no_skip_existing per rielaborarle).\n\n"
            "Esempio:\n"
            '  gaussadj edit \\\n'
            "      --input_dir  data/garden/input \\\n"
            "      --output_dir data/garden/input_edited \\\n"
            '      --prompt "make it look like autumn with orange leaves"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input_dir", type=Path, required=True, metavar="DIR",
        help="Cartella con le immagini originali (.jpg / .png)",
    )
    p.add_argument(
        "--output_dir", type=Path, required=True, metavar="DIR",
        help="Cartella di destinazione per le immagini modificate",
    )
    p.add_argument(
        "--prompt", type=str, required=True, metavar="TESTO",
        help='Istruzione di editing in inglese (es. "make it look like winter")',
    )
    p.add_argument(
        "--model_id", type=str, default="timbrooks/instruct-pix2pix", metavar="ID",
        help="Identificatore HuggingFace del modello di diffusione (default: %(default)s)",
    )
    p.add_argument(
        "--num_steps", type=int, default=20, metavar="N",
        help="Passi di denoising — più passi = qualità maggiore ma più lento (default: %(default)s)",
    )
    p.add_argument(
        "--guidance_scale", type=float, default=7.5, metavar="F",
        help=(
            "CFG scale: peso della guida testuale. "
            "Valori alti seguono di più il prompt ma riducono varietà (default: %(default)s)"
        ),
    )
    p.add_argument(
        "--image_guidance_scale", type=float, default=1.5, metavar="F",
        help=(
            "Scala di ancoraggio all'immagine originale. "
            "Valori alti preservano di più la struttura (default: %(default)s)"
        ),
    )
    p.add_argument(
        "--max_size", type=int, default=None, metavar="N",
        help=(
            "Se impostato, ridimensiona il lato maggiore a N pixel prima di "
            "passare l'immagine al modello (multiplo di 8, es. 512 o 768). "
            "Utile per limitare VRAM su GPU piccole (default: nessun resize)"
        ),
    )
    p.add_argument(
        "--cpu_offload", action="store_true",
        help=(
            "Abilita CPU offload: tiene il modello in RAM e sposta i layer su "
            "GPU uno alla volta. Riduce drasticamente la VRAM ma rallenta molto. "
            "Da usare solo se full-res non entra in VRAM nemmeno con --max_size."
        ),
    )
    p.add_argument(
        "--no_skip_existing", action="store_true",
        help="Rielabora le immagini già presenti in output_dir (default: salta)",
    )
    p.set_defaults(func=_run_edit)


def _run_edit(args: argparse.Namespace) -> None:
    from utils.device import get_device, get_device_name
    from diffusion.editor import ImageEditor

    if not args.input_dir.exists():
        logger.error("Cartella input non trovata: %s", args.input_dir)
        sys.exit(1)

    device = get_device()
    logger.info("Device: %s", get_device_name())

    logger.info("Caricamento modello di diffusione '%s'...", args.model_id)
    editor = ImageEditor(model_id=args.model_id, device=device, cpu_offload=args.cpu_offload)

    logger.info(
        "Editing: input=%s  output=%s  prompt='%s'",
        args.input_dir, args.output_dir, args.prompt,
    )
    editor.edit_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        max_size=args.max_size,
        skip_existing=not args.no_skip_existing,
    )
    logger.info("Editing completato. Output in: %s", args.output_dir)


# ---------------------------------------------------------------------------
# Sottocomando: fit
# ---------------------------------------------------------------------------

def _parser_fit(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "fit",
        help="Ottimizza i colori SH delle Gaussiane su immagini editate",
        description=(
            "Congela la geometria (posizione, scala, rotazione, opacità) e\n"
            "ottimizza solo i coefficienti Spherical Harmonics del colore\n"
            "minimizzando la differenza tra il rendering e le immagini editate.\n\n"
            "Le immagini vengono abbinate alle camera tramite il campo image_name\n"
            "in cameras.json. Quelle non trovate vengono saltate con un avviso.\n\n"
            "Esempio:\n"
            "  gaussadj fit \\\n"
            "      --ply          data/garden/point_cloud.ply \\\n"
            "      --images_dir   data/garden/input_edited \\\n"
            "      --cameras_json data/garden/cameras.json \\\n"
            "      --output_ply   data/garden/point_cloud_edited.ply"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Input obbligatori
    p.add_argument(
        "--ply", type=Path, required=True, metavar="FILE",
        help="File .ply della scena originale (Gaussian Splatting)",
    )
    p.add_argument(
        "--images_dir", type=Path, required=True, metavar="DIR",
        help="Cartella con le immagini editate (prodotte da 'gaussadj edit')",
    )
    p.add_argument(
        "--cameras_json", type=Path, required=True, metavar="FILE",
        help="File cameras.json con le pose camera (prodotto da 'gaussadj colmap')",
    )
    p.add_argument(
        "--output_ply", type=Path, required=True, metavar="FILE",
        help="File .ply di output con i colori aggiornati",
    )
    # Parametri di ottimizzazione
    optim = p.add_argument_group("parametri ottimizzazione")
    optim.add_argument(
        "--num_iterations", type=int, default=2000, metavar="N",
        help="Numero di iterazioni di ottimizzazione (default: %(default)s)",
    )
    optim.add_argument(
        "--lr_dc", type=float, default=0.005, metavar="F",
        help=(
            "Learning rate per i coefficienti DC (banda 0, colore base). "
            "Valori tipici: 0.001–0.01 (default: %(default)s)"
        ),
    )
    optim.add_argument(
        "--lr_rest", type=float, default=0.001, metavar="F",
        help=(
            "Learning rate per le bande SH 1–3 (effetti direzionali). "
            "Di solito più basso di lr_dc (default: %(default)s)"
        ),
    )
    optim.add_argument(
        "--lambda_dssim", type=float, default=0.2, metavar="F",
        help=(
            "Peso della loss SSIM nella loss combinata L = (1-λ)·L1 + λ·(1-SSIM). "
            "Range [0, 1] (default: %(default)s)"
        ),
    )
    optim.add_argument(
        "--log_every", type=int, default=100, metavar="N",
        help="Stampa il log di avanzamento ogni N iterazioni (default: %(default)s)",
    )
    p.set_defaults(func=_run_fit)


def _run_fit(args: argparse.Namespace) -> None:
    import torch
    from utils.device import get_device, get_device_name
    from utils.ply_io import load_gaussians, save_gaussians
    from utils.renderer import get_active_backend
    from training.color_fitter import ColorFitter

    device = get_device()
    logger.info("Device: %s  |  Renderer: %s", get_device_name(), get_active_backend())

    for p, name in [
        (args.ply, "--ply"),
        (args.images_dir, "--images_dir"),
        (args.cameras_json, "--cameras_json"),
    ]:
        if not p.exists():
            logger.error("File/cartella non trovato: %s (%s)", p, name)
            sys.exit(1)

    logger.info("Caricamento gaussiane da %s", args.ply)
    gaussians = load_gaussians(args.ply)
    logger.info("Caricate %d gaussiane", gaussians["xyz"].shape[0])

    with open(args.cameras_json) as f:
        raw_cameras = json.load(f)
    cameras = []
    for cam in raw_cameras:
        c = dict(cam)
        # Formato Inria (gaussian-splatting originale): rotation=R_c2w, position=centro camera mondo,
        # img_name invece di image_name, nessun cx/cy (principal point al centro).
        if "R" not in cam and "rotation" in cam:
            import numpy as _np
            rot_c2w = _np.array(cam["rotation"], dtype=_np.float32)
            pos_world = _np.array(cam["position"], dtype=_np.float32)
            R_w2c = rot_c2w.T
            T_w2c = -R_w2c @ pos_world
            c["R"] = torch.tensor(R_w2c, dtype=torch.float32)
            c["T"] = torch.tensor(T_w2c, dtype=torch.float32)
            c.setdefault("image_name", cam.get("img_name", ""))
            c.setdefault("cx", cam["width"] / 2.0)
            c.setdefault("cy", cam["height"] / 2.0)
        else:
            c["R"] = torch.tensor(cam["R"], dtype=torch.float32)
            c["T"] = torch.tensor(cam["T"], dtype=torch.float32)
        cameras.append(c)
    logger.info("Caricate %d camera da %s", len(cameras), args.cameras_json)

    matched_cameras: list[dict] = []
    matched_images: list[Path] = []
    for cam in cameras:
        img_name = cam.get("image_name", "")
        img_path = args.images_dir / img_name
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
            "Controlla che image_name in cameras.json corrisponda ai file in %s",
            args.images_dir,
        )
        sys.exit(1)

    logger.info("Abbinate %d/%d coppie camera-immagine", len(matched_cameras), len(cameras))

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

    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    save_gaussians(args.output_ply, updated)
    logger.info("Gaussiane aggiornate salvate in %s", args.output_ply)


# ---------------------------------------------------------------------------
# Sottocomando: render
# ---------------------------------------------------------------------------

def _parser_render(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "render",
        help="Renderizza una scena 3DGS da un dato punto di vista",
        description=(
            "Carica le Gaussiane da un file .ply e produce un'immagine PNG\n"
            "dal punto di vista della camera selezionata in cameras.json.\n\n"
            "Esempio:\n"
            "  gaussadj render \\\n"
            "      --ply          data/garden/point_cloud_edited.ply \\\n"
            "      --cameras_json data/garden/cameras.json \\\n"
            "      --camera_id    0 \\\n"
            "      --output       renders/frame_000.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ply", type=Path, required=True, metavar="FILE",
        help="File .ply della scena (originale o editata)",
    )
    p.add_argument(
        "--cameras_json", type=Path, required=True, metavar="FILE",
        help="File cameras.json con le pose camera",
    )
    p.add_argument(
        "--camera_id", type=int, default=0, metavar="N",
        help="Indice (0-based) della camera in cameras.json da usare (default: %(default)s)",
    )
    p.add_argument(
        "--output", type=Path, required=True, metavar="FILE",
        help="File PNG di output (la cartella viene creata se non esiste)",
    )
    p.add_argument(
        "--width", type=int, default=None, metavar="N",
        help="Larghezza immagine in pixel (default: valore dalla camera)",
    )
    p.add_argument(
        "--height", type=int, default=None, metavar="N",
        help="Altezza immagine in pixel (default: valore dalla camera)",
    )
    p.set_defaults(func=_run_render)


def _run_render(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image
    import numpy as np
    from utils.device import get_device, get_device_name
    from utils.ply_io import load_gaussians
    from utils.renderer import render, get_active_backend

    device = get_device()
    logger.info("Device: %s  |  Renderer: %s", get_device_name(), get_active_backend())

    for p, name in [(args.ply, "--ply"), (args.cameras_json, "--cameras_json")]:
        if not p.exists():
            logger.error("File non trovato: %s (%s)", p, name)
            sys.exit(1)

    logger.info("Caricamento gaussiane da %s", args.ply)
    gaussians = load_gaussians(args.ply)
    gaussians = {k: v.to(device) for k, v in gaussians.items()}
    logger.info("Caricate %d gaussiane", gaussians["xyz"].shape[0])

    with open(args.cameras_json) as f:
        cameras = json.load(f)

    if args.camera_id >= len(cameras):
        logger.error(
            "camera_id=%d fuori range (cameras.json ha %d entry, indici 0–%d)",
            args.camera_id, len(cameras), len(cameras) - 1,
        )
        sys.exit(1)

    cam_raw = cameras[args.camera_id]
    camera = {
        "R":  torch.tensor(cam_raw["R"], dtype=torch.float32, device=device),
        "T":  torch.tensor(cam_raw["T"], dtype=torch.float32, device=device),
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

    with torch.no_grad():
        image_tensor = render(gaussians, camera, (H, W))

    img_np = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype("uint8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_np).save(args.output)
    logger.info("Immagine salvata in %s", args.output)


# ---------------------------------------------------------------------------
# Entry point principale
# ---------------------------------------------------------------------------

def main() -> None:
    root = argparse.ArgumentParser(
        prog="gaussadj",
        description=(
            "Gaussian Color Adjust — modifica i colori di scene 3D Gaussian Splatting\n"
            "usando modelli di diffusione, mantenendo invariata la geometria.\n\n"
            "Flusso di lavoro tipico:\n"
            "  1. gaussadj colmap  sparse/0  cameras.json\n"
            "  2. gaussadj edit    --input_dir orig/ --output_dir edited/ --prompt \"...\"\n"
            "  3. gaussadj fit     --ply scene.ply --images_dir edited/ \\\n"
            "                       --cameras_json cameras.json --output_ply out.ply\n"
            "  4. gaussadj render  --ply out.ply --cameras_json cameras.json --output img.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = root.add_subparsers(
        dest="command",
        metavar="COMANDO",
        title="comandi disponibili",
    )
    sub.required = True

    _parser_colmap(sub)
    _parser_edit(sub)
    _parser_fit(sub)
    _parser_render(sub)

    args = root.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
