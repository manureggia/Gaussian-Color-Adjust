"""
Test minimo di gsplat su GPU: verifica che rasterization() funzioni
con pochi Gaussian sintetici. Serve a isolare eventuali crash di
amd_gsplat / gsplat dal resto del codice del progetto.

Uso:
    HIP_LAUNCH_BLOCKING=1 python scripts/test_gsplat_minimal.py

Se crasha qui, il problema è nel runtime gsplat/ROCm, non nel progetto.
"""

import torch
from gsplat import rasterization

dev = "cuda"
N = 1000

means   = torch.randn(N, 3, device=dev)
quats   = torch.zeros(N, 4, device=dev); quats[:, 0] = 1
scales  = torch.ones(N, 3, device=dev) * 0.05
opac    = torch.ones(N, device=dev) * 0.5
colors  = torch.rand(N, 1, 3, device=dev)

viewmat = torch.eye(4, device=dev).unsqueeze(0)
viewmat[0, 2, 3] = 5
K = torch.tensor([[[300., 0, 128],
                   [0, 300, 128],
                   [0,   0,   1]]], device=dev)

img, alpha, meta = rasterization(
    means=means, quats=quats, scales=scales,
    opacities=opac, colors=colors,
    viewmats=viewmat, Ks=K,
    width=256, height=256, sh_degree=0,
)
print("OK shape:", img.shape)
