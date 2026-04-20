# Gaussian Color Adjust

Modifica i colori di una scena **3D Gaussian Splatting** tramite modelli di diffusione,
poi ri-adatta solo i coefficienti di colore (Spherical Harmonics) delle Gaussiane
per corrispondere alle immagini modificate.

**Posizione, scala, rotazione e opacità rimangono invariati. La densificazione è disabilitata.**

## Caratteristiche

- **Backend di rendering adattivo**: usa `gsplat` (CUDA/ROCm) se disponibile,
  cade automaticamente su un renderer PyTorch puro (MPS/CPU).
- **Editing con InstructPix2Pix**: modifica tutte le viste della scena con un prompt testuale.
- **Color fitting differenziabile**: ottimizza solo i coefficienti SH con loss L1+SSIM.
- **Portabile**: macOS Apple Silicon (MPS) e Linux/Windows con AMD ROCm o NVIDIA CUDA.

## Avvio rapido

```bash
# 1. Installa (vedi HOWTO.md per istruzioni dettagliate)
pip install -r requirements.txt

# 2. Verifica il backend
python -c "from utils.device import get_device; get_device()"

# 3. Converti le pose COLMAP
python -c "from utils.cameras_from_colmap import convert_colmap_to_json; \
           convert_colmap_to_json('data/garden/sparse/0', 'data/garden/cameras.json')"

# 4. Modifica le immagini
python scripts/edit_images.py \
    --input_dir data/garden/input \
    --output_dir data/garden/input_edited \
    --prompt "make it look like autumn"

# 5. Ri-adatta i colori
python scripts/fit_colors.py \
    --ply data/garden/point_cloud.ply \
    --images_dir data/garden/input_edited \
    --cameras_json data/garden/cameras.json \
    --output_ply data/garden/point_cloud_edited.ply

# 6. Renderizza il risultato
python scripts/render_scene.py \
    --ply data/garden/point_cloud_edited.ply \
    --cameras_json data/garden/cameras.json \
    --camera_id 0 \
    --output renders/result.png
```

## Test

```bash
pytest tests/ -v
```

## Documentazione

- **`GUIDE.md`** — approfondimento tecnico (in italiano): matematica del rendering,
  architettura, loss function, formato `.ply`.
- **`HOWTO.md`** — guida operativa (in italiano): installazione, esempi, troubleshooting.
