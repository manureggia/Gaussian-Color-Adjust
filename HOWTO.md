# HOWTO — Come usare Gaussian Color Adjust

## Indice

1. [Installazione dell'ambiente](#1-installazione)
2. [Verifica del backend attivo](#2-verifica-backend)
3. [Download scene di esempio](#3-scene-di-esempio)
4. [Conversione pose COLMAP → cameras.json](#4-conversione-colmap)
5. [Modifica immagini con il diffusion model](#5-editing-immagini)
6. [Lancio del color fitting](#6-color-fitting)
7. [Rendering finale e confronto before/after](#7-rendering)
8. [Troubleshooting](#8-troubleshooting)
9. [Tabella parametri CLI](#9-parametri-cli)

---

## 1. Installazione

### 1a. macOS Apple Silicon (M1/M2/M3) — MPS

```bash
# Crea ambiente virtuale
python3 -m venv .venv
source .venv/bin/activate

# PyTorch con supporto MPS (incluso nel wheel standard da PyTorch 2.0+)
pip install torch torchvision

# Installa le dipendenze del progetto
pip install -r requirements.txt

# gsplat viene installato ma non funzionerà senza kernel CUDA:
# il progetto usa automaticamente il renderer PyTorch puro su MPS.
# Per verificare: python -c "from utils.device import get_gsplat_available; print(get_gsplat_available())"
# → False   (è normale e corretto su Mac)
```

**Nota**: Il pacchetto `pycolmap` potrebbe non avere wheel precompilato per arm64.
Se l'installazione fallisce:
```bash
# Installa senza pycolmap (userai il parser binario manuale)
pip install -r requirements.txt --ignore-requires-python
# oppure
pip install torch torchvision diffusers transformers accelerate Pillow numpy plyfile tqdm pytest pytest-mock
```

### 1b. PC con AMD RX 9070 XT — ROCm (Linux)

```bash
# STEP 1: Installa PyTorch con ROCm
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1

# Verifica che la GPU sia rilevata
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → True   AMD Radeon RX 9070 XT

# STEP 2: Installa gsplat per ROCm
# Opzione A: PyPI (se disponibile con ROCm)
pip install gsplat
# Opzione B: build da sorgente
pip install git+https://github.com/ROCm/gsplat.git
# Opzione C: senza gsplat (usa il renderer PyTorch puro)
# Non installare nulla — il fallback è automatico

# STEP 3: Installa le dipendenze rimanenti
pip install diffusers transformers accelerate Pillow numpy plyfile tqdm pytest pytest-mock pycolmap
```

### 1c. PC con NVIDIA GPU — CUDA

```bash
# Installa PyTorch con CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Installa gsplat (si compila automaticamente con CUDA)
pip install gsplat

# Installa il resto
pip install -r requirements.txt
```

### Struttura directory di lavoro

Prima di procedere, crea la struttura dei dati:
```bash
mkdir -p data/garden/input data/garden/input_edited renders
```

---

## 2. Verifica del backend attivo

Dopo l'installazione, verifica che il backend corretto sia stato rilevato:

```bash
cd gaussian_color_adjust

# Verifica device
python -c "
from utils.device import get_device, get_device_name, get_gsplat_available
device = get_device()
print('Device:', get_device_name())
print('gsplat disponibile:', get_gsplat_available())
"
```

**Output atteso su Mac M1**:
```
[device] Backend selezionato: MPS (Apple Silicon)
Device: Apple MPS (arm)
gsplat disponibile: False
```

**Output atteso su PC ROCm**:
```
[device] Backend selezionato: CUDA — AMD Radeon RX 9070 XT
Device: AMD ROCm — AMD Radeon RX 9070 XT
gsplat disponibile: True   (se installato)
```

**Verifica del renderer**:
```bash
python -c "
from utils.renderer import get_active_backend
print('Renderer attivo:', get_active_backend())
"
# → 'gsplat' oppure 'pure_pytorch'
```

---

## 3. Scene di esempio

### Dataset ufficiale 3DGS

Il sito ufficiale del progetto 3DGS mette a disposizione scene pre-ricostruite:

```bash
# Scene disponibili: garden, bicycle, bonsai, counter, kitchen, room, stump, treehill

# Scarica con wget (esempio per 'garden')
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/garden.zip
unzip garden.zip -d data/

# Struttura risultante:
# data/garden/
#   input/          ← immagini originali (JPEG)
#   sparse/0/       ← ricostruzione COLMAP (cameras.bin, images.bin, points3D.bin)
#   point_cloud/    ← gaussiane pre-addestrate (point_cloud.ply)
```

### Struttura attesa per il progetto

```
data/garden/
├── input/
│   ├── IMG_0001.JPG
│   ├── IMG_0002.JPG
│   └── ...
├── sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── point_cloud/iteration_7000/
    └── point_cloud.ply
```

---

## 4. Conversione pose COLMAP → cameras.json

Le pose camera COLMAP devono essere convertite nel formato JSON del progetto:

```bash
python -c "
from utils.cameras_from_colmap import convert_colmap_to_json
convert_colmap_to_json(
    colmap_sparse_dir='data/garden/sparse/0',
    output_json='data/garden/cameras.json',
)
print('Conversione completata!')
"
```

**oppure** direttamente da script:
```bash
python -c "
import sys; sys.path.insert(0, '.')
from utils.cameras_from_colmap import convert_colmap_to_json
convert_colmap_to_json('data/garden/sparse/0', 'data/garden/cameras.json')
"
```

**Verifica del file generato**:
```bash
python -c "
import json
with open('data/garden/cameras.json') as f:
    cams = json.load(f)
print(f'Camera caricate: {len(cams)}')
print('Prima camera:', cams[0]['image_name'], f'{cams[0][\"width\"]}x{cams[0][\"height\"]}')
"
```

**Nota**: se `pycolmap` non è installato, il parser binario manuale viene usato
automaticamente. Se anche quello fallisce, il formato dei file binari potrebbe
essere di una versione COLMAP non supportata (raro con versioni >= 3.x).

---

## 5. Modifica immagini con il diffusion model

### Sintassi base

```bash
python scripts/edit_images.py \
    --input_dir data/garden/input \
    --output_dir data/garden/input_edited \
    --prompt "TESTO_DEL_PROMPT" \
    --num_steps 25
```

### Esempi di prompt

**Effetto autunno** (foglie arancioni e rosse):
```bash
python scripts/edit_images.py \
    --input_dir data/garden/input \
    --output_dir data/garden/input_autumn \
    --prompt "make it look like autumn with orange and red leaves falling" \
    --num_steps 30 \
    --guidance_scale 8.0 \
    --image_guidance_scale 1.5
```

**Tramonto dorato**:
```bash
python scripts/edit_images.py \
    --input_dir data/garden/input \
    --output_dir data/garden/input_sunset \
    --prompt "make it a golden hour sunset with warm orange and pink sky" \
    --num_steps 25 \
    --guidance_scale 7.5
```

**Effetto invernale / neve**:
```bash
python scripts/edit_images.py \
    --input_dir data/garden/input \
    --output_dir data/garden/input_winter \
    --prompt "make it look like winter with snow covering the ground and plants" \
    --num_steps 30 \
    --image_guidance_scale 1.2
```

**Stile notturno**:
```bash
python scripts/edit_images.py \
    --input_dir data/garden/input \
    --output_dir data/garden/input_night \
    --prompt "change to nighttime with moonlight and artificial lights" \
    --num_steps 35 \
    --guidance_scale 9.0
```

**Effetto nebbia / mattino**:
```bash
python scripts/edit_images.py \
    --input_dir data/garden/input \
    --output_dir data/garden/input_foggy \
    --prompt "add morning fog and mist to the scene" \
    --num_steps 20 \
    --image_guidance_scale 1.8
```

### Note sull'editing

- Il primo avvio scarica il modello da HuggingFace (~5GB). Impostare
  `HF_HOME=/percorso/personalizzato` per cambiare la cache.
- Su Mac M1 con 8GB RAM unificata, usa `--num_steps 15` per evitare OOM.
- Le immagini già presenti in `output_dir` vengono saltate automaticamente
  (riprendi da dove eri arrivato se interrotto). Usa `--no_skip_existing`
  per forzare la rielaborazione.

---

## 6. Lancio del color fitting

### Comando base

```bash
python scripts/fit_colors.py \
    --ply data/garden/point_cloud/iteration_7000/point_cloud.ply \
    --images_dir data/garden/input_autumn \
    --cameras_json data/garden/cameras.json \
    --output_ply data/garden/point_cloud_autumn.ply \
    --num_iterations 2000 \
    --log_every 100
```

### Parametri raccomandati per piattaforma

| Piattaforma    | `--num_iterations` | `--lr_dc` | Note                              |
|----------------|--------------------|-----------|-----------------------------------|
| NVIDIA CUDA    | 2000-5000          | 0.005     | gsplat veloce, può fare più iter  |
| AMD ROCm       | 2000-5000          | 0.005     | gsplat se compilato, altrimenti PT|
| Mac M1 (MPS)   | 500-1000           | 0.005     | Renderer PT puro, più lento       |
| CPU            | 100-300            | 0.01      | Solo test, molto lento            |

### Lettura dei log

```
[   1/2000] loss=0.2341  psnr=6.31dB    ← loss alta all'inizio, PSNR basso
[ 100/2000] loss=0.1823  psnr=14.72dB   ← miglioramento rapido all'inizio
[ 500/2000] loss=0.1205  psnr=18.38dB   ← rallenta la convergenza
[1000/2000] loss=0.0912  psnr=20.85dB   ← zona di convergenza
[2000/2000] loss=0.0734  psnr=22.61dB   ← risultato finale
```

**Quando interrompere**: se la loss non migliora tra due log consecutivi,
il training ha converso. Premi `Ctrl+C` — un checkpoint verrà salvato
automaticamente in `checkpoint_interrupt.ply`.

### Fitting con parametri personalizzati

```bash
# Fitting aggressivo (più iterazioni, LR più alto)
python scripts/fit_colors.py \
    --ply data/garden/point_cloud/iteration_7000/point_cloud.ply \
    --images_dir data/garden/input_autumn \
    --cameras_json data/garden/cameras.json \
    --output_ply data/garden/point_cloud_autumn.ply \
    --num_iterations 5000 \
    --lr_dc 0.008 \
    --lr_rest 0.002 \
    --lambda_dssim 0.3 \
    --log_every 200
```

---

## 7. Rendering finale e confronto before/after

### Renderizza la scena originale

```bash
python scripts/render_scene.py \
    --ply data/garden/point_cloud/iteration_7000/point_cloud.ply \
    --cameras_json data/garden/cameras.json \
    --camera_id 0 \
    --output renders/before.png
```

### Renderizza la scena editata

```bash
python scripts/render_scene.py \
    --ply data/garden/point_cloud_autumn.ply \
    --cameras_json data/garden/cameras.json \
    --camera_id 0 \
    --output renders/after.png
```

### Confronto before/after in Python

```python
from PIL import Image, ImageDraw, ImageFont
import numpy as np

before = Image.open("renders/before.png")
after  = Image.open("renders/after.png")

# Affianca le due immagini
W, H = before.size
combined = Image.new("RGB", (W * 2, H))
combined.paste(before, (0, 0))
combined.paste(after, (W, 0))
combined.save("renders/comparison.png")
print("Confronto salvato in renders/comparison.png")
```

### Renderizza più punti di vista in batch

```bash
for i in $(seq 0 9); do
    python scripts/render_scene.py \
        --ply data/garden/point_cloud_autumn.ply \
        --cameras_json data/garden/cameras.json \
        --camera_id $i \
        --output renders/frame_$(printf "%03d" $i).png
done
```

---

## 8. Troubleshooting

### gsplat non disponibile su Mac (MPS)

**Sintomo**: `get_gsplat_available()` → `False`

**Soluzione**: è il comportamento corretto su Mac. gsplat richiede kernel CUDA
non disponibili su MPS. Il renderer PyTorch puro viene usato automaticamente.
Non è necessaria nessuna azione.

```bash
# Conferma che il fallback funziona
python -c "
from utils.renderer import get_active_backend
print('Backend:', get_active_backend())  # → 'pure_pytorch'
"
```

---

### OOM (Out of Memory) su Mac M1 durante editing

**Sintomo**: `RuntimeError: MPS backend out of memory`

**Soluzioni**:
```bash
# 1. Riduci i passi di denoising
python scripts/edit_images.py ... --num_steps 10

# 2. Abilita il garbage collection MPS tra immagini (aggiungi al codice o via env)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# 3. Usa float32 invece di float16 su MPS (più stabile, più memoria)
# Modifica diffusion/editor.py: cambia 'mps' da float16 a float32
```

---

### ROCm non rilevato su RX 9070 XT

**Sintomo**: `torch.cuda.is_available()` → `False`

**Diagnosi**:
```bash
# Verifica che il driver ROCm sia installato
rocm-smi
# → mostra la GPU AMD

# Verifica la versione PyTorch
python -c "import torch; print(torch.__version__)"
# Deve contenere '+rocm': es. '2.1.0+rocm6.0'

# Se il PyTorch installato è la versione standard (senza +rocm), reinstalla:
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
```

---

### PYTORCH_ENABLE_MPS_FALLBACK

**Sintomo**: `NotImplementedError: The operator 'X' is not currently implemented for MPS`

**Soluzione**: il progetto imposta questa variabile automaticamente, ma se
lanci script in modo non standard:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/fit_colors.py ...
```

---

### Il matching immagini-camera fallisce

**Sintomo**: `Nessuna immagine abbinata alle camera`

**Causa**: i nomi file in `cameras.json` non corrispondono ai file in `input_edited/`.

**Diagnosi**:
```bash
# Visualizza i nomi in cameras.json
python -c "
import json
cams = json.load(open('data/garden/cameras.json'))
for c in cams[:5]:
    print(c['image_name'])
"
# Confronta con i file nella cartella editata
ls data/garden/input_autumn/ | head -5
```

**Soluzione**: assicurati che i nomi file in `input_edited/` corrispondano
esattamente a `image_name` in `cameras.json`. Il progetto prova anche varianti
con estensione diversa (`.jpg` ↔ `.png`).

---

### gsplat dà errori di shape su ROCm

**Sintomo**: `RuntimeError: expected scalar type Float but found Half`

**Soluzione**:
```bash
# Forza float32 nel renderer gsplat modificando utils/renderer.py:
# cambia `dtype = torch.float32` (già il default per gsplat)
# oppure usa il renderer PyTorch puro:
export FORCE_PURE_PYTORCH=1   # (se implementi questa env var)
```

---

## 9. Tabella parametri CLI

### `scripts/edit_images.py`

| Parametro                | Tipo    | Default                         | Descrizione                            |
|--------------------------|---------|---------------------------------|----------------------------------------|
| `--input_dir`            | Path    | **obbligatorio**                | Cartella immagini originali            |
| `--output_dir`           | Path    | **obbligatorio**                | Cartella immagini editate (output)     |
| `--prompt`               | str     | **obbligatorio**                | Istruzione di modifica                 |
| `--model_id`             | str     | `timbrooks/instruct-pix2pix`    | Identificatore HuggingFace modello     |
| `--num_steps`            | int     | `20`                            | Passi di denoising                     |
| `--guidance_scale`       | float   | `7.5`                           | Intensità del prompt testuale          |
| `--image_guidance_scale` | float   | `1.5`                           | Ancoraggio all'immagine originale      |
| `--no_skip_existing`     | flag    | `False`                         | Se presente, rielabora file esistenti  |

### `scripts/fit_colors.py`

| Parametro           | Tipo    | Default          | Descrizione                                    |
|---------------------|---------|------------------|------------------------------------------------|
| `--ply`             | Path    | **obbligatorio** | File .ply delle gaussiane originali            |
| `--images_dir`      | Path    | **obbligatorio** | Cartella immagini editate                      |
| `--cameras_json`    | Path    | **obbligatorio** | File cameras.json con le pose                  |
| `--output_ply`      | Path    | **obbligatorio** | File .ply output con colori aggiornati         |
| `--num_iterations`  | int     | `2000`           | Numero di iterazioni di ottimizzazione         |
| `--lr_dc`           | float   | `0.005`          | Learning rate per features_dc (grado 0)        |
| `--lr_rest`         | float   | `0.001`          | Learning rate per features_rest (gradi 1-3)    |
| `--lambda_dssim`    | float   | `0.2`            | Peso della loss SSIM (0=solo L1, 1=solo SSIM)  |
| `--log_every`       | int     | `100`            | Frequenza di stampa dei log                    |

### `scripts/render_scene.py`

| Parametro         | Tipo    | Default          | Descrizione                                  |
|-------------------|---------|------------------|----------------------------------------------|
| `--ply`           | Path    | **obbligatorio** | File .ply delle gaussiane                    |
| `--cameras_json`  | Path    | **obbligatorio** | File cameras.json                            |
| `--camera_id`     | int     | `0`              | Indice della camera (0-based)                |
| `--output`        | Path    | **obbligatorio** | File PNG di output                           |
| `--width`         | int     | dalla camera     | Override larghezza immagine                  |
| `--height`        | int     | dalla camera     | Override altezza immagine                    |
