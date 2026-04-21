"""
Renderer differenziabile per Gaussian Splatting.

Strategia di selezione del backend (in ordine di priorità):
  1. **gsplat** — rasterizzatore C++/CUDA/ROCm ad alte prestazioni.
     Disponibile su NVIDIA (CUDA) e AMD (ROCm). NON funziona su MPS/CPU
     senza kernel compilati.
  2. **PyTorch puro** — renderer EWA tile-based interamente in PyTorch.
     Portabile su MPS, CPU e qualsiasi dispositivo. Più lento ma pienamente
     differenziabile.

Il backend viene selezionato automaticamente all'import del modulo.
Entrambi espongono la stessa interfaccia::

    render(gaussians, camera, image_size) -> Tensor(H, W, 3)
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Tuple

import torch
import torch.nn.functional as F

from utils.device import get_device, get_gsplat_available

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti Spherical Harmonics
# ---------------------------------------------------------------------------
_SH_C0 = 0.28209479177387814   # 1 / (2 * sqrt(pi))
_SH_C1 = 0.4886025119029199    # sqrt(3 / (4*pi))
_COV_EPS = 1e-4                 # regolarizzazione covarianza 2D


# ===========================================================================
# Helper: conversione parametri camera
# ===========================================================================

def camera_to_viewmat(
    camera: dict,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Converte i parametri camera in matrice world-to-camera 4×4.

    Args:
        camera: dict con ``R`` (3,3) e ``T`` (3,).
        device: dispositivo target; se None usa il device di R/T (o get_device()).

    Returns:
        torch.Tensor: (1, 4, 4) float32 matrice world-to-camera.
    """
    dtype = torch.float32

    R = camera["R"]
    T = camera["T"]
    if not isinstance(R, torch.Tensor):
        R = torch.tensor(R, dtype=dtype)
    if not isinstance(T, torch.Tensor):
        T = torch.tensor(T, dtype=dtype)

    # Usa il device degli input se non specificato esplicitamente
    target_device = device or (R.device if R.device.type != "cpu" else get_device())
    # Se R è già su CPU e device non è forzato, mantieni CPU
    if device is None and R.device.type == "cpu" and T.device.type == "cpu":
        target_device = torch.device("cpu")

    R = R.to(device=target_device, dtype=dtype)
    T = T.to(device=target_device, dtype=dtype)

    viewmat = torch.eye(4, device=target_device, dtype=dtype)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = T
    return viewmat.unsqueeze(0)   # (1, 4, 4)


def camera_to_K(
    camera: dict,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Costruisce la matrice di intrinseci 3×3.

    Args:
        camera: dict con ``fx``, ``fy``, ``cx``, ``cy``.
        device: dispositivo target; se None usa get_device().

    Returns:
        torch.Tensor: (1, 3, 3) float32 matrice intrinseci.
    """
    target_device = device or get_device()
    dtype = torch.float32
    fx, fy = float(camera["fx"]), float(camera["fy"])
    cx, cy = float(camera["cx"]), float(camera["cy"])
    K = torch.tensor(
        [[fx, 0., cx],
         [0., fy, cy],
         [0., 0., 1.]],
        device=target_device, dtype=dtype,
    )
    return K.unsqueeze(0)   # (1, 3, 3)


# ===========================================================================
# Backend 1: gsplat
# ===========================================================================

def _render_with_gsplat(
    gaussians: dict[str, torch.Tensor],
    camera: dict,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """Renderizza usando ``gsplat.rasterization()``.

    Args:
        gaussians: dict con xyz, features_dc, features_rest, scaling, rotation, opacity.
        camera: dict con R, T, fx, fy, cx, cy, width, height.
        image_size: (H, W).

    Returns:
        torch.Tensor: (H, W, 3) float32 in [0, 1], differenziabile.
    """
    from gsplat import rasterization

    # gsplat richiede il device GPU (CUDA o ROCm)
    device = get_device()
    dtype = torch.float32
    H, W = image_size

    def _t(key: str) -> torch.Tensor:
        return gaussians[key].to(device=device, dtype=dtype)

    xyz = _t("xyz")                          # (N,3)
    rotation = _t("rotation")               # (N,4) wxyz
    scaling_log = _t("scaling")             # (N,3) log-scale
    opacity_logit = _t("opacity")           # (N,1)
    features_dc = _t("features_dc")         # (N,1,3)

    # gsplat vuole: scales dopo exp, opacities dopo sigmoid, quats normalizzati
    scales = torch.exp(scaling_log)                         # (N,3)
    opacities = torch.sigmoid(opacity_logit).squeeze(-1)    # (N,)
    quats = F.normalize(rotation, dim=-1)                   # (N,4)

    # Coefficienti SH grado 0: gsplat li valuta internamente quando sh_degree è
    # passato, quindi NON applichiamo sigmoid qui. La shape deve essere
    # (N, K, 3) con K = (sh_degree+1)^2 = 1.
    sh_coeffs = features_dc                                  # (N,1,3)

    viewmat = camera_to_viewmat(camera, device=device)   # (1,4,4)
    K = camera_to_K(camera, device=device)              # (1,3,3)

    colors_out, _alphas, _meta = rasterization(
        means=xyz,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=sh_coeffs,
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        sh_degree=0,
    )
    # colors_out: (1, H, W, 3)
    return colors_out[0].clamp(0.0, 1.0)


# ===========================================================================
# Backend 2: PyTorch puro (MPS / CPU fallback)
# ===========================================================================

def _quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Quaternioni (N,4) wxyz → matrici di rotazione (N,3,3)."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def _build_3d_covariance(
    scaling: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Σ = R S S^T R^T  con S = diag(exp(scaling))."""
    S = torch.diag_embed(torch.exp(scaling))    # (N,3,3)
    R = _quaternion_to_matrix(rotation)          # (N,3,3)
    M = R @ S
    return M @ M.transpose(-1, -2)              # (N,3,3)


def _project_covariance_2d(
    cov3d_cam: torch.Tensor,
    xyz_cam: torch.Tensor,
    fx: float,
    fy: float,
) -> torch.Tensor:
    """Proiezione EWA: Σ_2D = J Σ_cam J^T."""
    tx, ty, tz = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    tz = tz.clamp(min=1e-4)
    zero = torch.zeros_like(tx)
    J = torch.stack([
        fx / tz,  zero,     -fx * tx / (tz * tz),
        zero,     fy / tz,  -fy * ty / (tz * tz),
    ], dim=-1).reshape(-1, 2, 3)

    cov2d = J @ cov3d_cam @ J.transpose(-1, -2)    # (N,2,2)
    eye2 = torch.eye(2, device=cov3d_cam.device, dtype=cov3d_cam.dtype).unsqueeze(0)
    return cov2d + _COV_EPS * eye2


def _sh_to_rgb(
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    view_dirs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Valuta SH gradi 0-1 → colori (N,3) in [0, 1]."""
    color = _SH_C0 * features_dc[:, 0, :]   # (N,3)

    if view_dirs is not None and features_rest.shape[1] >= 3:
        x, y, z = view_dirs[:, 0:1], view_dirs[:, 1:2], view_dirs[:, 2:3]
        color = color + _SH_C1 * (
            -features_rest[:, 0, :] * y
            + features_rest[:, 1, :] * z
            - features_rest[:, 2, :] * x
        )

    return (color + 0.5).clamp(0.0, 1.0)


def _render_pure_pytorch(
    gaussians: dict[str, torch.Tensor],
    camera: dict,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """Renderizza con EWA splatting in PyTorch puro (MPS/CPU).

    Algoritmo:
      1. Trasforma posizioni in camera space.
      2. Proietta in pixel space.
      3. Calcola covarianze 2D via EWA.
      4. Valuta colori SH.
      5. Ordina per profondità (front-to-back).
      6. Alpha-compositing per gaussian.

    Args:
        gaussians: dict nel formato standard del progetto.
        camera: dict con R, T, fx, fy, cx, cy.
        image_size: (H, W).

    Returns:
        torch.Tensor: (H, W, 3) differenziabile rispetto a features_dc/rest.
    """
    H, W = image_size
    dtype = torch.float32

    # Usa il device delle gaussiane in input (supporta CPU, MPS, CUDA)
    device = gaussians["xyz"].device

    R = camera["R"]
    T = camera["T"]
    if not isinstance(R, torch.Tensor):
        R = torch.tensor(R, dtype=dtype)
    if not isinstance(T, torch.Tensor):
        T = torch.tensor(T, dtype=dtype)
    R = R.to(device=device, dtype=dtype)
    T = T.to(device=device, dtype=dtype)

    fx, fy = float(camera["fx"]), float(camera["fy"])
    cx, cy = float(camera["cx"]), float(camera["cy"])

    def _g(key: str) -> torch.Tensor:
        return gaussians[key].to(device=device, dtype=dtype)

    xyz = _g("xyz")
    features_dc = _g("features_dc")
    features_rest = _g("features_rest")
    scaling = _g("scaling")
    rotation = _g("rotation")
    opacity_logit = _g("opacity")

    # ---- 1. Camera space ----
    xyz_cam = (xyz @ R.T) + T.unsqueeze(0)   # (N,3)

    # Filtra gaussiane dietro la camera
    valid = xyz_cam[:, 2] > 0.1
    if valid.sum() == 0:
        return torch.zeros(H, W, 3, device=device, dtype=dtype)

    xyz_cam = xyz_cam[valid]
    features_dc = features_dc[valid]
    features_rest = features_rest[valid]
    scaling = scaling[valid]
    rotation = rotation[valid]
    opacity_logit = opacity_logit[valid]
    nv = xyz_cam.shape[0]

    # ---- 2. Proiezione pixel ----
    tz = xyz_cam[:, 2]
    u = (xyz_cam[:, 0] / tz) * fx + cx
    v = (xyz_cam[:, 1] / tz) * fy + cy

    # ---- 3. Covarianza 2D ----
    cov3d = _build_3d_covariance(scaling, rotation)    # (N,3,3)
    cov3d_cam = R @ cov3d @ R.T                         # (N,3,3) — broadcast
    cov2d = _project_covariance_2d(cov3d_cam, xyz_cam, fx, fy)  # (N,2,2)

    # Raggio di splat (3 sigma)
    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    d = cov2d[:, 1, 1]
    mid = 0.5 * (a + d)
    disc = (mid * mid - (a * d - b * b)).clamp(min=0.0)
    lambda_max = mid + torch.sqrt(disc)
    radius = (3.0 * torch.sqrt(lambda_max)).ceil().int().clamp(max=max(H, W))

    # ---- 4. Colori SH ----
    cam_pos = -(R.T @ T)   # posizione camera in world
    view_dirs = F.normalize(cam_pos.unsqueeze(0) - xyz[valid], dim=-1)
    colors = _sh_to_rgb(features_dc, features_rest, view_dirs)   # (N,3)
    opacity = torch.sigmoid(opacity_logit[:, 0])                  # (N,)

    # ---- 5. Ordina front-to-back ----
    order = torch.argsort(tz)
    u_s = u[order]; v_s = v[order]
    cov2d_s = cov2d[order]
    colors_s = colors[order]
    opacity_s = opacity[order]
    radius_s = radius[order]

    # ---- 6. Rasterizzazione differenziabile con F.pad ----
    # Per ogni gaussiana calcoliamo l'alpha map locale e la paddiamo a dimensione
    # intera usando F.pad (operazione out-of-place → grafo autograd preservato).
    # Accumuliamo contributi con operazioni out-of-place per preservare la
    # differenziabilità rispetto a features_dc e features_rest.
    ys = torch.arange(H, device=device, dtype=dtype)
    xs = torch.arange(W, device=device, dtype=dtype)

    # Lista dei contributi (T_i * alpha_i * c_i) e dei "1-alpha" per trasmittanza
    color_contribs: list[torch.Tensor] = []   # ciascuno (H, W, 3)
    one_minus_alphas: list[torch.Tensor] = [] # ciascuno (H, W)

    for i in range(nv):
        ri = int(radius_s[i].item())
        ui = u_s[i].item()
        vi = v_s[i].item()

        x0 = max(0, int(math.floor(ui - ri)))
        x1 = min(W, int(math.ceil(ui + ri)) + 1)
        y0 = max(0, int(math.floor(vi - ri)))
        y1 = min(H, int(math.ceil(vi + ri)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        dx = xs[x0:x1].unsqueeze(0) - ui     # (1, bW)
        dy = ys[y0:y1].unsqueeze(1) - vi     # (bH, 1)

        cov_i = cov2d_s[i]
        det = (cov_i[0, 0] * cov_i[1, 1] - cov_i[0, 1] * cov_i[1, 0]).clamp(min=1e-8)
        inv_a = cov_i[1, 1] / det
        inv_b = -cov_i[0, 1] / det
        inv_d = cov_i[0, 0] / det

        maha = inv_a * dx * dx + 2.0 * inv_b * dx * dy + inv_d * dy * dy
        gauss_w = torch.exp(-0.5 * maha)                          # (bH, bW)
        alpha_local = (opacity_s[i] * gauss_w).clamp(max=0.99)   # (bH, bW)

        # Pad a (H, W) — out-of-place, differenziabile
        # F.pad ordine: (left, right, top, bottom)
        pad_left = x0
        pad_right = W - x1
        pad_top = y0
        pad_bottom = H - y1
        alpha_full = F.pad(
            alpha_local.unsqueeze(0).unsqueeze(0),   # (1,1,bH,bW)
            (pad_left, pad_right, pad_top, pad_bottom),
        ).squeeze(0).squeeze(0)                       # (H, W)

        color_contribs.append(alpha_full)
        one_minus_alphas.append(1.0 - alpha_full)    # (H, W)

    if not color_contribs:
        return torch.zeros(H, W, 3, device=device, dtype=dtype)

    # Calcola trasmittanza cumulativa T_i = prod_{j<i}(1 - alpha_j)
    # e accumula: image = sum_i T_i * alpha_i * c_i
    image_rgb = torch.zeros(H, W, 3, device=device, dtype=dtype)
    T = torch.ones(H, W, device=device, dtype=dtype)

    for idx, (alpha_full, one_minus_a) in enumerate(
        zip(color_contribs, one_minus_alphas)
    ):
        contrib = (T * alpha_full).unsqueeze(-1) * colors_s[idx]  # (H,W,3)
        # Accumulo out-of-place: crea nuovo tensore
        image_rgb = image_rgb + contrib
        T = T * one_minus_a

    return image_rgb.clamp(0.0, 1.0)


# ===========================================================================
# Selezione automatica del backend
# ===========================================================================

def get_renderer() -> Tuple[Callable, str]:
    """Restituisce la funzione di rendering e il nome del backend selezionato.

    Priorità: gsplat → PyTorch puro.

    Returns:
        Tuple[Callable, str]: (render_fn, backend_name).

    Example::

        render_fn, backend = get_renderer()
        image = render_fn(gaussians, camera, (H, W))
    """
    if get_gsplat_available():
        logger.info("Renderer: gsplat (CUDA/ROCm)")
        return _render_with_gsplat, "gsplat"
    else:
        logger.info("Renderer: PyTorch puro (MPS/CPU fallback)")
        return _render_pure_pytorch, "pure_pytorch"


# Funzione pubblica unificata
_render_fn, _backend_name = get_renderer()
logger.debug("Backend renderer attivo: %s", _backend_name)


def render(
    gaussians: dict[str, torch.Tensor],
    camera: dict,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """Renderizza le gaussiane nella vista specificata.

    Usa automaticamente il backend ottimale (gsplat o PyTorch puro).

    Args:
        gaussians: dict con chiavi xyz, features_dc, features_rest,
                   scaling, rotation, opacity.
        camera: dict con R (3,3), T (3,), fx, fy, cx, cy, width, height.
        image_size: (H, W) dimensione output.

    Returns:
        torch.Tensor: (H, W, 3) float32 in [0, 1], differenziabile
                      rispetto a gaussians['features_dc'] e ['features_rest'].
    """
    return _render_fn(gaussians, camera, image_size)


def get_active_backend() -> str:
    """Restituisce il nome del backend renderer attivo.

    Returns:
        str: "gsplat" o "pure_pytorch".
    """
    return _backend_name
