# Guida Tecnica — Gaussian Color Adjust

## Indice

1. [Introduzione a Gaussian Splatting e Spherical Harmonics](#1-introduzione)
2. [Architettura del progetto](#2-architettura)
3. [I due backend di rendering](#3-backend-rendering)
4. [Come funziona `gsplat.rasterization()`](#4-gsplat)
5. [Il renderer PyTorch puro: EWA splatting](#5-pytorch-puro)
6. [Color fitting: loss e parametri congelati](#6-color-fitting)
7. [Diffusion Models: InstructPix2Pix](#7-diffusion)
8. [Gestione del device: MPS vs ROCm vs CUDA vs CPU](#8-device)
9. [Formato `.ply` per le Gaussiane 3D](#9-formato-ply)
10. [Limitazioni note e estensioni future](#10-limitazioni)

---

## 1. Introduzione

### Gaussian Splatting

Il **3D Gaussian Splatting** (3DGS, Kerbl et al. 2023) è una tecnica di rappresentazione
di scene 3D che modella la radiance field come una miscela di gaussiane 3D anisotropiche.
Ogni primitiva è definita da:

| Parametro      | Simbolo    | Dimensione | Significato                                |
|----------------|------------|------------|--------------------------------------------|
| Posizione      | **μ**      | (3,)       | Centro della gaussiana in world space      |
| Scala          | **s**      | (3,)       | Log-scala degli assi principali             |
| Rotazione      | **q**      | (4,)       | Quaternione wxyz                            |
| Opacità        | α          | (1,)       | Logit-opacità (σ applicata a runtime)       |
| Colore SH      | **sh**     | (16, 3)    | Coefficienti SH per direzione dipendenza    |

La matrice di covarianza 3D si ricostruisce come:

```
Σ = R · S · Sᵀ · Rᵀ
```

dove `R` è la matrice di rotazione dal quaternione e `S = diag(exp(s))`.

### Spherical Harmonics (SH)

Le **Spherical Harmonics** codificano la dipendenza direzionale del colore
(effetti speculari, illuminazione variabile). 3DGS usa fino al grado 3 (16 bande):

- **Grado 0** (coefficiente DC): colore base indipendente dalla direzione.
  `C₀ = 1/(2√π) ≈ 0.2821`
- **Grado 1** (3 coefficienti): dipendenza lineare dalla direzione.
  `C₁ = √(3/4π) ≈ 0.4886`
- **Gradi 2–3**: effetti speculari più complessi.

La conversione SH → RGB per direzione di vista **d** è:

```
color = C₀·sh[0] + C₁·(sh[1]·y + sh[2]·z + sh[3]·x) + ... + 0.5
color = clamp(color, 0, 1)
```

Il termine `+0.5` è una convenzione del codice 3DGS originale per centrare
i valori attorno a grigio neutro.

---

## 2. Architettura

### Flusso dati completo

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GAUSSIAN COLOR ADJUST                           │
└─────────────────────────────────────────────────────────────────────┘

  FASE 1: EDITING IMMAGINI
  ┌─────────────┐     prompt testuale     ┌──────────────────────┐
  │  Input      │ ──────────────────────► │  InstructPix2Pix     │
  │  immagini   │                         │  (diffusion/editor)  │
  │  (scene     │ ◄─── immagini editate ─ │                      │
  │  captures)  │                         └──────────────────────┘
  └─────────────┘
        │ edited images
        ▼

  FASE 2: LOADING GAUSSIANE
  ┌─────────────────────────┐
  │   point_cloud.ply       │
  │   (utils/ply_io.py)     │
  │                         │
  │  xyz ────────── frozen  │
  │  scaling ─────── frozen │
  │  rotation ────── frozen │
  │  opacity ─────── frozen │
  │  features_dc ─► grad ✓  │
  │  features_rest ► grad ✓ │
  └──────────┬──────────────┘
             │
             ▼

  FASE 3: COLOR FITTING LOOP
  ┌────────────────────────────────────────────────────────────────┐
  │  training/color_fitter.py                                      │
  │                                                                │
  │  for iter in range(N):                                         │
  │    cam = random_camera()                                       │
  │    ┌──────────────────────────────────────────────┐           │
  │    │  utils/renderer.py                           │           │
  │    │  ┌──────────────┐   ┌──────────────────────┐ │           │
  │    │  │   gsplat     │   │  PyTorch puro (EWA)  │ │           │
  │    │  │  (CUDA/ROCm) │ OR│  (MPS/CPU fallback)  │ │           │
  │    │  └──────────────┘   └──────────────────────┘ │           │
  │    └────────────────────────────────────────────── ┘           │
  │         │  pred (H,W,3)                                        │
  │         ▼                                                      │
  │    loss = (1-λ)·L1(pred, gt) + λ·(1-SSIM(pred, gt))          │
  │    loss.backward()  →  grad su features_dc, features_rest     │
  │    Adam.step()                                                 │
  └────────────────────────────────────────────────────────────────┘
             │
             ▼

  FASE 4: OUTPUT
  ┌──────────────────────────────┐
  │   point_cloud_edited.ply     │
  │   (utils/ply_io.py)          │
  │                              │
  │   xyz, scaling, rot, op      │  ← identici all'input
  │   features_dc ←── aggiornati │
  │   features_rest ← aggiornati │
  └──────────────────────────────┘
             │
             ▼
  ┌──────────────────────────────┐
  │   scripts/render_scene.py    │
  │   → render_out.png           │
  └──────────────────────────────┘
```

### Struttura moduli

```
gaussian_color_adjust/
├── utils/
│   ├── device.py           ← detection backend (CUDA/MPS/CPU)
│   ├── renderer.py         ← gsplat + PyTorch puro fallback
│   ├── ply_io.py           ← I/O file .ply
│   └── cameras_from_colmap.py ← conversione pose COLMAP
├── diffusion/
│   └── editor.py           ← InstructPix2Pix wrapper
├── training/
│   └── color_fitter.py     ← loop ottimizzazione SH
└── scripts/
    ├── edit_images.py       ← CLI editing
    ├── fit_colors.py        ← CLI fitting
    └── render_scene.py      ← CLI rendering
```

---

## 3. I due backend di rendering

### Perché due backend?

`gsplat` è la libreria standard per il rasterizzatore di Gaussian Splatting ad alta
velocità. I suoi kernel critici sono scritti in **C++/CUDA** (o ROCm tramite HIP):
non funzionano su Apple MPS senza ricompilazione specifica.

Il progetto implementa quindi una **catena di fallback**:

```
         gsplat disponibile?
              │
         ╔════╧════╗
         ║  gsplat ║  → velocissimo (GPU nativa)
         ╚═════════╝
              │ no
         ╔══════════════════╗
         ║  PyTorch puro    ║  → portabile (MPS, CPU, qualsiasi device)
         ╚══════════════════╝
```

La selezione avviene **una sola volta** all'import di `utils.renderer`:

```python
_render_fn, _backend_name = get_renderer()
```

---

## 4. Come funziona `gsplat.rasterization()`

### Interfaccia principale

```python
from gsplat import rasterization

colors_out, alphas, meta = rasterization(
    means    = xyz,          # (N, 3)  posizioni world
    quats    = quats,        # (N, 4)  quaternioni normalizzati wxyz
    scales   = scales,       # (N, 3)  scale DOPO exp()
    opacities= opacities,    # (N,)    opacità DOPO sigmoid()
    colors   = sh_colors,    # (N, 3)  colori SH grado 0
    viewmats = viewmat,      # (1, 4, 4) world-to-camera
    Ks       = K,            # (1, 3, 3) matrice intrinseci
    width    = W,
    height   = H,
    sh_degree= 0,
)
# colors_out: (1, H, W, 3) in [0, 1]
```

### Preparazione dei parametri

| Parametro gsplat | Da dove viene              | Trasformazione               |
|------------------|----------------------------|------------------------------|
| `means`          | `gaussians['xyz']`         | nessuna                      |
| `quats`          | `gaussians['rotation']`    | `F.normalize(..., dim=-1)`   |
| `scales`         | `gaussians['scaling']`     | `torch.exp(...)`             |
| `opacities`      | `gaussians['opacity']`     | `torch.sigmoid(...).squeeze(-1)` |
| `colors`         | `gaussians['features_dc']` | `torch.sigmoid(...).squeeze(1)` |

### Matrice viewmat (world-to-camera)

```
viewmat = [R | T]   con forma (1, 4, 4)
          [0 | 1]
```

Il progetto la costruisce con `camera_to_viewmat(camera)` in `utils/renderer.py`.

---

## 5. Il renderer PyTorch puro: EWA splatting

Il renderer PyTorch puro implementa **EWA (Elliptical Weighted Average) Splatting**,
la stessa tecnica matematica alla base di gsplat ma interamente in operazioni PyTorch
differenziabili.

### Passo 1: Trasformazione in camera space

```
xyz_cam = R · xyz + T       # (N, 3)
```

Filtra gaussiane con `z_cam < 0.1` (dietro la camera).

### Passo 2: Proiezione prospettica

```
u = (x_cam / z_cam) · fx + cx
v = (y_cam / z_cam) · fy + cy
```

### Passo 3: Covarianza 2D con Jacobiano EWA

La covarianza 3D viene prima portata in camera space:
```
Σ_cam = R · Σ_world · Rᵀ
```

Poi proiettata in 2D tramite il Jacobiano della proiezione prospettica:
```
J = [fx/z,  0,    -fx·x/z²]
    [0,     fy/z, -fy·y/z²]

Σ_2D = J · Σ_cam · Jᵀ
```

Si aggiunge `ε·I` per stabilità numerica (`_COV_EPS = 1e-4`).

### Passo 4: Raggio di splat

Il raggio in pixel viene calcolato dall'autovalore massimo della covarianza 2D:
```
λ_max = (tr(Σ)/2) + sqrt((tr(Σ)/2)² - det(Σ))
radius = ceil(3 · sqrt(λ_max))
```

### Passo 5: Colori SH

Per grado 0: `color = C₀ · features_dc + 0.5`

Per grado 1 (con direzione di vista **d** = normalize(cam_pos - xyz)):
```
color += C₁ · (-sh[0]·dy + sh[1]·dz - sh[2]·dx)
```

### Passo 6: Alpha compositing front-to-back

Le gaussiane vengono ordinate per profondità crescente. Per ogni gaussiana,
si calcola il peso gaussiano su tutti i pixel nel suo bounding box:

```
maha(x,y) = [dx,dy] · Σ_2D⁻¹ · [dx,dy]ᵀ
α(x,y) = opacity · exp(-0.5 · maha)

color_out += T · α · color_gaussian
T *= (1 - α)
```

dove `T` è la trasmittanza residua (inizia a 1).

### Differenziabilità

Tutte le operazioni sono PyTorch puro → `autograd` propaga i gradienti fino a
`features_dc` e `features_rest`. Il loop `for i in range(nv)` è il collo di bottiglia
in velocità rispetto a gsplat (kernel C++), ma mantiene la portabilità completa su MPS.

---

## 6. Color fitting: loss e parametri congelati

### Parametri congelati

```python
# In ColorFitter.__init__():
for key in {"xyz", "scaling", "rotation", "opacity"}:
    gaussians[key].requires_grad_(False)
```

**Perché congelare la geometria?**
- La forma 3D della scena (posizione, scala, rotazione delle gaussiane) cattura
  la struttura geometrica della scena originale. Modificarla causerebbe artefatti
  e distorsioni spaziali incoerenti con il video di input.
- L'opacità è congelata per non alterare la struttura di trasparenza/occlusione.
- Solo il colore (SH) deve cambiare per riflettere lo stile editato.

### Loss combinata L1 + SSIM

```
L = (1 - λ)·L1(pred, gt) + λ·(1 - SSIM(pred, gt))
```

- **L1** penalizza le differenze di intensità pixel-by-pixel. Robusta al rumore,
  favorisce immagini nitide.
- **SSIM** (Structural Similarity Index) cattura differenze percettive in
  contrasto locale, luminanza e struttura. Penalizza le sfocature che L1 non cattura.
- Default: `λ = 0.2` (80% L1 + 20% SSIM).

### SSIM in PyTorch puro

```python
def _ssim(pred, target, window_size=11, sigma=1.5, c1=0.01², c2=0.03²):
    # Kernel gaussiano 2D separabile applicato con conv2d per gruppo
    # Calcola: μ_p, μ_t, σ_p², σ_t², σ_pt
    # SSIM = (2μ_pμ_t + c1)(2σ_pt + c2) / ((μ_p² + μ_t² + c1)(σ_p² + σ_t² + c2))
```

Implementazione `groups=C` per efficienza: un solo `conv2d` su tutti i canali.

### Optimizer Adam con learning rate separati

```python
optimizer = torch.optim.Adam([
    {"params": [features_dc],   "lr": lr_dc},    # default: 5e-3
    {"params": [features_rest], "lr": lr_rest},  # default: 1e-3
])
```

Il coefficiente DC (grado 0) ha LR maggiore perché determina il colore base;
i coefficienti rest (gradi 1-3) hanno effetti più fini e beneficiano di LR
inferiore per la stabilità.

### PSNR come metrica di monitoraggio

```
PSNR = 10 · log₁₀(1 / MSE)   [dB]
```

Valori tipici: `< 20 dB` (scarso), `25-30 dB` (buono), `> 35 dB` (ottimo).

---

## 7. Diffusion Models: InstructPix2Pix

### Architettura del modello

**InstructPix2Pix** (Brooks et al. 2023) è un modello di diffusione condizionale
basato su Stable Diffusion 1.5. Riceve:

1. L'immagine originale `I` (condizionamento visivo).
2. Il prompt testuale `p` (condizionamento linguistico).

E produce l'immagine modificata `I'` secondo le istruzioni in `p`.

### Parametri chiave

| Parametro              | Default | Descrizione                                           |
|------------------------|---------|-------------------------------------------------------|
| `num_steps`            | 20      | Passi di denoising. Più passi = migliore qualità      |
| `guidance_scale`       | 7.5     | Forza del testo (CFG). Più alto = più fedele al prompt|
| `image_guidance_scale` | 1.5     | Forza dell'immagine originale. `1.0` = max fedeltà    |

### Pipeline di editing di una directory

```
input_dir/              output_dir/
  img_001.jpg  ─────►     img_001.jpg  (editata)
  img_002.jpg  ─────►     img_002.jpg  (editata)
  img_003.png  ─────►     img_003.png  (editata)
```

Per ogni immagine della scena multi-view, il modello applica la **stessa
istruzione testuale**, producendo un insieme di viste coerenti nello stile.

### Dtype e dispositivo

| Device | dtype     | Motivo                                  |
|--------|-----------|-----------------------------------------|
| CUDA   | float16   | Risparmia ~50% VRAM, CUDA ha TF32/FP16 |
| MPS    | float16   | Apple GPU supporta float16 nativamente  |
| CPU    | float32   | CPU non ottimizzata per float16          |

---

## 8. Gestione del device: MPS vs ROCm vs CUDA vs CPU

### Detection automatica

```python
# In utils/device.py
if torch.cuda.is_available():
    device = "cuda"          # NVIDIA o AMD ROCm (stessa API)
elif torch.backends.mps.is_available():
    device = "mps"           # Apple Silicon
else:
    device = "cpu"
```

### Perché ROCm appare come "cuda"

PyTorch su ROCm usa lo stesso stack software (API CUDA-compatibile tramite HIP).
`torch.cuda.is_available()` restituisce `True` su GPU AMD con PyTorch ROCm.
Il nome si distingue con `torch.cuda.get_device_name(0)` che mostra "AMD Radeon...".

### MPS e operazioni non supportate

Non tutte le operazioni PyTorch hanno kernel MPS nativi. La variabile d'ambiente
`PYTORCH_ENABLE_MPS_FALLBACK=1` abilita il fallback automatico su CPU per le
operazioni mancanti. Il progetto la imposta automaticamente quando MPS viene rilevato.

### gsplat su MPS

`gsplat` richiede kernel CUDA/ROCm compilati. Su MPS:
- `get_gsplat_available()` → `False`
- Il renderer PyTorch puro viene usato automaticamente
- Non è necessaria alcuna configurazione manuale

---

## 9. Formato `.ply` per le Gaussiane 3D

Il formato `.ply` 3DGS è binario (little-endian) con i seguenti attributi per vertice:

```
element vertex N
property float x           # posizione
property float y
property float z
property float nx          # normali (sempre 0)
property float ny
property float nz
property float f_dc_0      # SH DC: canale R
property float f_dc_1      # SH DC: canale G
property float f_dc_2      # SH DC: canale B
property float f_rest_0    # SH rest: banda 0, canale R
property float f_rest_1    # SH rest: banda 1, canale R
...
property float f_rest_14   # SH rest: banda 14, canale R
property float f_rest_15   # SH rest: banda 0, canale G
...
property float f_rest_44   # SH rest: banda 14, canale B
property float opacity     # logit-opacità
property float scale_0     # log-scala asse x
property float scale_1     # log-scala asse y
property float scale_2     # log-scala asse z
property float rot_0       # quaternione w
property float rot_1       # quaternione x
property float rot_2       # quaternione y
property float rot_3       # quaternione z
```

### Layout dei coefficienti SH rest

Il layout flat `f_rest_0..44` è ordinato **per canale, poi per banda**:
```
f_rest[0..14]  → bande 0-14 del canale R
f_rest[15..29] → bande 0-14 del canale G
f_rest[30..44] → bande 0-14 del canale B
```

In memoria il tensore `features_rest` ha shape `(N, 15, 3)` (banda × canale).
La conversione avviene in `ply_io.py`:

```python
rest_flat = np.stack(rest_cols, axis=1)            # (N, 45)
features_rest = rest_flat.reshape(n, 3, 15)        # (N, 3, 15)
features_rest = features_rest.transpose(0, 2, 1)   # (N, 15, 3)
```

---

## 10. Limitazioni note e possibili estensioni future

### Limitazioni attuali

| Limitazione                    | Causa                                              |
|--------------------------------|----------------------------------------------------|
| Renderer PyTorch lento su CPU  | Loop Python su N gaussiane (O(N) iterazioni)       |
| SH grado 0 nel renderer puro   | Grado 1 richiede direzione di vista per pixel      |
| No supporto multi-GPU          | Un solo device per sessione                        |
| Match immagini per nome file   | Richiede che i nomi in cameras.json siano esatti   |
| No normalizzazione SH fitting  | I valori SH possono esplodere con LR elevato       |

### Estensioni future

1. **SH fino a grado 3**: supporto completo nel renderer PyTorch puro, con
   tabella completa delle costanti SH.
2. **Renderer PyTorch vettorizzato**: sostituire il loop Python con operazioni
   batch su `scatter_add` / tile rendering parallelo per MPS.
3. **Regolarizzazione SH**: aggiungere `λ_reg · ||features_rest||²` per evitare
   overfitting nelle bande di ordine superiore.
4. **Multi-prompt editing**: editing differenziato per regione della scena.
5. **Depth-aware editing**: usare la mappa di profondità per limitare l'editing
   a oggetti specifici (foreground/background).
6. **Video temporale**: applicare il fitting su sequenze temporali mantenendo
   coerenza inter-frame.
7. **Esportazione Gaussian formato .splat**: formato compresso per viewer web.
8. **Supporto gsplat-MPS**: integrare il fork sperimentale `gsplat-mps` quando
   stabile per accelerare il training su Apple Silicon.
