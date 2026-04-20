"""
Re-fitting dei coefficienti Spherical Harmonics (colori) delle Gaussiane 3D.

Strategia di ottimizzazione:
  - Parametri **congelati**: xyz, scaling, rotation, opacity.
  - Parametri **ottimizzati**: features_dc (lr_dc), features_rest (lr_rest).
  - Loss: (1 - λ) · L1  +  λ · (1 - SSIM)
  - Optimizer: Adam con learning rate separato per DC e rest.
  - A ogni iterazione viene scelta casualmente una camera dal training set.
  - La densificazione è completamente disabilitata.

La classe gestisce KeyboardInterrupt salvando un checkpoint prima di uscire.
"""

from __future__ import annotations

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from utils.device import get_device
from utils.ply_io import save_gaussians
from utils.renderer import render

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSIM puro PyTorch
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Crea un kernel gaussiano 1D normalizzato."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    return kernel / kernel.sum()


def _ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Calcola l'Structural Similarity Index (SSIM) in PyTorch puro.

    Args:
        pred:   (H, W, 3) o (B, 3, H, W) immagine predetta, valori in [0,1].
        target: stessa forma di pred.
        window_size: dimensione del kernel gaussiano (default 11).
        sigma:  deviazione standard del kernel.
        c1, c2: costanti di stabilità numerica.

    Returns:
        torch.Tensor: scalare SSIM medio su tutti i canali e pixel.
    """
    device = pred.device

    # Porta in formato (B, C, H, W)
    if pred.dim() == 3:
        pred = pred.permute(2, 0, 1).unsqueeze(0)    # (1,3,H,W)
        target = target.permute(2, 0, 1).unsqueeze(0)

    # Costruisce kernel 2D separabile
    k1d = _gaussian_kernel_1d(window_size, sigma, device)
    k2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)        # (ws, ws)
    k2d = k2d.unsqueeze(0).unsqueeze(0)               # (1,1,ws,ws)

    C = pred.shape[1]
    kernel = k2d.repeat(C, 1, 1, 1)                   # (C,1,ws,ws)
    pad = window_size // 2

    def conv(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu_pred   = conv(pred)
    mu_target = conv(target)

    mu_p2    = mu_pred * mu_pred
    mu_t2    = mu_target * mu_target
    mu_pt    = mu_pred * mu_target

    sigma_p2  = conv(pred * pred) - mu_p2
    sigma_t2  = conv(target * target) - mu_t2
    sigma_pt  = conv(pred * target) - mu_pt

    numerator   = (2 * mu_pt + c1) * (2 * sigma_pt + c2)
    denominator = (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2)

    ssim_map = numerator / denominator.clamp(min=1e-8)
    return ssim_map.mean()


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calcola il PSNR in dB."""
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# ColorFitter
# ---------------------------------------------------------------------------

class ColorFitter:
    """Re-fitting dei colori SH delle Gaussiane mantenendo la geometria congelata.

    Args:
        gaussians: dict nel formato di :func:`utils.ply_io.load_gaussians`.
        cameras:   lista di dict camera nel formato ``{R, T, fx, fy, cx, cy, width, height}``.
        gt_images: lista di tensori (H,W,3) oppure lista di path alle immagini editerate.
                   Viene abbinata 1-a-1 con ``cameras`` per indice.
        device:    dispositivo di training; None → auto-detect.
    """

    def __init__(
        self,
        gaussians: dict[str, torch.Tensor],
        cameras: list[dict],
        gt_images: list[torch.Tensor | str | Path],
        device: Any | None = None,
    ) -> None:
        self.device = device if device is not None else get_device()
        self.cameras = cameras

        # ---- Carica le immagini ground-truth ----
        self.gt_images: list[torch.Tensor] = []
        for img in gt_images:
            if isinstance(img, (str, Path)):
                self.gt_images.append(self._load_image(img))
            else:
                self.gt_images.append(img.to(self.device, dtype=torch.float32))

        if len(self.cameras) != len(self.gt_images):
            raise ValueError(
                f"Numero di camera ({len(self.cameras)}) != "
                f"numero di immagini ({len(self.gt_images)})"
            )

        # ---- Sposta gaussiane sul device ----
        self.gaussians: dict[str, torch.Tensor] = {}
        _frozen = {"xyz", "scaling", "rotation", "opacity"}
        for key, tensor in gaussians.items():
            # .clone() garantisce memoria separata: le modifiche in-place di Adam
            # non modificano il dict originale passato dall'utente.
            t = tensor.to(self.device, dtype=torch.float32).clone()
            if key in _frozen:
                t = t.detach().requires_grad_(False)
            else:
                t = t.detach().requires_grad_(True)
            self.gaussians[key] = t

        logger.info(
            "ColorFitter inizializzato: %d gaussiane, %d viste, device=%s",
            self.gaussians["xyz"].shape[0], len(self.cameras), self.device,
        )

    def _load_image(self, path: str | Path) -> torch.Tensor:
        """Carica un'immagine PNG/JPEG come tensore (H,W,3) float32 in [0,1]."""
        from PIL import Image
        import torchvision.transforms.functional as TF
        img = Image.open(path).convert("RGB")
        return TF.to_tensor(img).permute(1, 2, 0).to(self.device, dtype=torch.float32)

    def fit(
        self,
        num_iterations: int = 2000,
        lr_dc: float = 0.005,
        lr_rest: float = 0.001,
        lambda_dssim: float = 0.2,
        log_every: int = 100,
    ) -> dict[str, torch.Tensor]:
        """Esegue il loop di ottimizzazione dei colori SH.

        A ogni iterazione viene scelta una camera casuale dal training set,
        l'immagine viene renderizzata e la loss calcolata rispetto al GT editato.

        Args:
            num_iterations: numero totale di iterazioni.
            lr_dc:          learning rate per features_dc.
            lr_rest:        learning rate per features_rest.
            lambda_dssim:   peso della loss SSIM (0 → solo L1, 1 → solo SSIM).
            log_every:      frequenza di stampa dei log.

        Returns:
            dict: gaussiane aggiornate (stesso formato dell'input, su CPU).
        """
        features_dc = self.gaussians["features_dc"]
        features_rest = self.gaussians["features_rest"]

        optimizer = torch.optim.Adam(
            [
                {"params": [features_dc],   "lr": lr_dc},
                {"params": [features_rest], "lr": lr_rest},
            ]
        )

        n_views = len(self.cameras)
        logger.info(
            "Avvio color fitting: %d iter, lr_dc=%.4f, lr_rest=%.4f, λ_dssim=%.2f",
            num_iterations, lr_dc, lr_rest, lambda_dssim,
        )

        try:
            for it in range(1, num_iterations + 1):
                # Sceglie una camera casuale
                idx = random.randint(0, n_views - 1)
                cam = self.cameras[idx]
                gt = self.gt_images[idx]        # (H, W, 3)

                H = int(cam.get("height", gt.shape[0]))
                W = int(cam.get("width",  gt.shape[1]))

                # Forward pass
                pred = render(self.gaussians, cam, (H, W))  # (H, W, 3)

                # Adatta il ground truth alla dimensione di rendering
                if gt.shape[:2] != (H, W):
                    gt_r = gt.permute(2, 0, 1).unsqueeze(0)
                    gt_r = F.interpolate(gt_r, size=(H, W), mode="bilinear", align_corners=False)
                    gt_r = gt_r.squeeze(0).permute(1, 2, 0)
                else:
                    gt_r = gt

                # Loss L1 + SSIM
                l1_loss = F.l1_loss(pred, gt_r)
                ssim_val = _ssim(pred, gt_r)
                loss = (1.0 - lambda_dssim) * l1_loss + lambda_dssim * (1.0 - ssim_val)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if it % log_every == 0 or it == 1:
                    psnr = _psnr(pred.detach(), gt_r)
                    logger.info(
                        "[%4d/%d] loss=%.4f  psnr=%.2fdB",
                        it, num_iterations, loss.item(), psnr,
                    )
                    print(
                        f"[{it:4d}/{num_iterations}] "
                        f"loss={loss.item():.4f}  psnr={psnr:.2f}dB"
                    )

        except KeyboardInterrupt:
            logger.warning("Interruzione da tastiera! Salvo checkpoint di emergenza...")
            self.save_checkpoint("checkpoint_interrupt.ply")
            print("\n[!] Checkpoint salvato in checkpoint_interrupt.ply")

        # Restituisce le gaussiane aggiornate (detach, su CPU)
        result = {
            k: v.detach().cpu()
            for k, v in self.gaussians.items()
        }
        return result

    def save_checkpoint(self, path: str | Path) -> None:
        """Salva lo stato corrente delle gaussiane come file .ply.

        Args:
            path: percorso del file di output.
        """
        path = Path(path)
        gaussians_cpu = {k: v.detach().cpu() for k, v in self.gaussians.items()}
        save_gaussians(path, gaussians_cpu)
        logger.info("Checkpoint salvato in %s", path)
