"""
Modifica di immagini tramite InstructPix2Pix (HuggingFace diffusers).

Il modello predefinito è ``timbrooks/instruct-pix2pix``, che riceve
un'immagine originale e un testo di istruzione e restituisce l'immagine
modificata secondo il prompt (es. "rendi le foglie autunnali").

Gestione del dtype:
  - CUDA NVIDIA → float16  (risparmio memoria)
  - CUDA ROCm   → float32  (float16 causa GPU page fault con diffusers su AMD)
  - MPS         → float16
  - CPU         → float32
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from utils.device import get_device

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


class ImageEditor:
    """Editor di immagini basato su InstructPix2Pix.

    Args:
        model_id: identificatore HuggingFace del modello (default: instruct-pix2pix).
        device: dispositivo PyTorch; se None usa :func:`utils.device.get_device`.

    Raises:
        ImportError: se ``diffusers`` non è installato.
    """

    def __init__(
        self,
        model_id: str = "timbrooks/instruct-pix2pix",
        device: Any | None = None,
    ) -> None:
        try:
            import torch
            from diffusers import StableDiffusionInstructPix2PixPipeline
        except ImportError as exc:
            raise ImportError(
                "Il pacchetto 'diffusers' non è installato.\n"
                "Installa con: pip install diffusers transformers accelerate"
            ) from exc

        import torch

        self._torch = torch
        self.device = device if device is not None else get_device()
        self.model_id = model_id

        # ROCm espone la GPU come "cuda" ma float16 causa GPU page fault con diffusers;
        # si distingue dal vero CUDA controllando il nome del device.
        is_rocm = self.device.type == "cuda" and "amd" in torch.cuda.get_device_name(self.device).lower()
        dtype = torch.float32 if (is_rocm or self.device.type == "cpu") else torch.float16
        logger.info(
            "Caricamento modello %s su %s (dtype=%s)",
            model_id, self.device, dtype,
        )
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        logger.info("Modello caricato correttamente")

    def edit_image(
        self,
        image: Any,
        prompt: str,
        num_steps: int = 20,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 7.5,
        max_size: int = 512,
    ) -> Any:
        """Applica una modifica all'immagine in base al prompt testuale.

        Args:
            image: ``PIL.Image.Image`` in input.
            prompt: istruzione di modifica (es. "make it look like autumn").
            num_steps: passi di denoising (più passi → qualità maggiore, più lento).
            image_guidance_scale: forza dell'ancoraggio all'immagine originale.
            guidance_scale: forza del seguire il prompt testuale (CFG scale).
            max_size: lato massimo in pixel prima di passare al modello (default: 512).
                      Ridimensiona mantenendo le proporzioni; deve essere multiplo di 8.

        Returns:
            ``PIL.Image.Image``: immagine modificata, ridimensionata a max_size.
        """
        # InstructPix2Pix è addestrato a 512px; immagini più grandi
        # consumano VRAM in modo quadratico senza migliorare la qualità.
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w = (int(w * scale) // 8) * 8
            new_h = (int(h * scale) // 8) * 8
            image = image.resize((new_w, new_h), resample=3)  # BICUBIC

        logger.debug("Editing immagine %dx%d con prompt: '%s'", image.width, image.height, prompt)
        result = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=num_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
        )
        return result.images[0]

    def edit_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        prompt: str,
        extensions: tuple[str, ...] = _SUPPORTED_EXTENSIONS,
        skip_existing: bool = True,
        **kwargs: Any,
    ) -> None:
        """Applica il prompt a tutte le immagini in ``input_dir``.

        Salva i risultati in ``output_dir`` con lo stesso nome file.
        Mostra una barra di progresso tqdm.

        Args:
            input_dir: cartella con le immagini originali.
            output_dir: cartella di destinazione (creata se assente).
            prompt: istruzione di modifica da applicare a ogni immagine.
            extensions: estensioni dei file da processare.
            skip_existing: se True, salta le immagini già presenti in output_dir.
            **kwargs: argomenti aggiuntivi passati a :meth:`edit_image`.
        """
        try:
            from PIL import Image
            from tqdm import tqdm
        except ImportError as exc:
            raise ImportError("Installa: pip install Pillow tqdm") from exc

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = sorted([
            p for p in input_dir.iterdir()
            if p.suffix in extensions
        ])
        if not images:
            logger.warning("Nessuna immagine trovata in %s", input_dir)
            return

        logger.info(
            "Editing di %d immagini con prompt: '%s'", len(images), prompt
        )
        for img_path in tqdm(images, desc="Editing immagini", unit="img"):
            out_path = output_dir / img_path.name
            if skip_existing and out_path.exists():
                logger.debug("Salto (già esistente): %s", out_path.name)
                continue
            try:
                pil_img = Image.open(img_path).convert("RGB")
                edited = self.edit_image(pil_img, prompt, **kwargs)
                edited.save(out_path, quality=95)
                logger.debug("Salvata: %s", out_path.name)
            except Exception as exc:
                logger.error("Errore su %s: %s", img_path.name, exc)
            finally:
                # Libera la memoria GPU frammentata dopo ogni immagine
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
