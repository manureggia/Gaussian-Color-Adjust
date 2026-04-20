"""
Test per diffusion/editor.py — mock del modello di diffusione.

I test non scaricano il modello reale: la pipeline diffusers viene
interamente sostituita da un mock che restituisce immagini sintetiche.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from PIL import Image
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixture: immagine PIL sintetica
# ---------------------------------------------------------------------------

@pytest.fixture
def pil_image_64():
    """Immagine RGB 64×64 sintetica."""
    import numpy as np
    arr = (255 * np.random.rand(64, 64, 3)).astype("uint8")
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Helper: mock della pipeline diffusers
# ---------------------------------------------------------------------------

def _make_mock_pipeline(output_image: Image.Image):
    """Crea un mock che imita StableDiffusionInstructPix2PixPipeline."""
    mock_result = MagicMock()
    mock_result.images = [output_image]

    mock_pipe = MagicMock()
    mock_pipe.return_value = mock_result
    mock_pipe.to.return_value = mock_pipe
    mock_pipe.set_progress_bar_config = MagicMock()
    return mock_pipe


# ---------------------------------------------------------------------------
# Test: edit_image
# ---------------------------------------------------------------------------

def test_edit_image_returns_pil(pil_image_64):
    """edit_image() deve restituire un oggetto PIL.Image."""
    import numpy as np
    output_img = Image.fromarray((255 * np.random.rand(64, 64, 3)).astype("uint8"))
    mock_pipe_cls = MagicMock(return_value=_make_mock_pipeline(output_img))

    with patch("diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained", mock_pipe_cls):
        from diffusion.editor import ImageEditor
        editor = ImageEditor(model_id="mock/model")
        result = editor.edit_image(pil_image_64, prompt="make it red")

    assert isinstance(result, Image.Image)


def test_edit_image_prompt_passed(pil_image_64):
    """Il prompt deve essere passato correttamente alla pipeline."""
    import numpy as np
    output_img = Image.fromarray((255 * np.random.rand(64, 64, 3)).astype("uint8"))
    mock_pipe = _make_mock_pipeline(output_img)
    mock_pipe_cls = MagicMock(return_value=mock_pipe)

    with patch("diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained", mock_pipe_cls):
        from diffusion.editor import ImageEditor
        editor = ImageEditor(model_id="mock/model")
        editor.edit_image(pil_image_64, prompt="turn leaves orange", num_steps=5)

    # Verifica che la pipeline sia stata chiamata con il prompt giusto
    call_kwargs = mock_pipe.call_args
    assert call_kwargs is not None
    assert "turn leaves orange" in str(call_kwargs)


# ---------------------------------------------------------------------------
# Test: edit_directory
# ---------------------------------------------------------------------------

def test_edit_directory_creates_output_files(tmp_path, pil_image_64):
    """edit_directory() deve creare i file di output."""
    import numpy as np

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    # Crea 3 immagini di test
    for i in range(3):
        pil_image_64.save(input_dir / f"frame_{i:03d}.png")

    output_img = Image.fromarray((255 * np.random.rand(64, 64, 3)).astype("uint8"))
    mock_pipe = _make_mock_pipeline(output_img)
    mock_pipe_cls = MagicMock(return_value=mock_pipe)

    with patch("diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained", mock_pipe_cls):
        from diffusion.editor import ImageEditor
        editor = ImageEditor(model_id="mock/model")
        editor.edit_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            prompt="make it look like winter",
        )

    output_files = list(output_dir.iterdir())
    assert len(output_files) == 3


def test_edit_directory_skip_existing(tmp_path, pil_image_64):
    """Con skip_existing=True, le immagini già presenti non vengono rielaborate."""
    import numpy as np

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    pil_image_64.save(input_dir / "frame_001.png")
    # Pre-popola output
    pil_image_64.save(output_dir / "frame_001.png")

    output_img = Image.fromarray((255 * np.random.rand(64, 64, 3)).astype("uint8"))
    mock_pipe = _make_mock_pipeline(output_img)
    mock_pipe_cls = MagicMock(return_value=mock_pipe)

    with patch("diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained", mock_pipe_cls):
        from diffusion.editor import ImageEditor
        editor = ImageEditor(model_id="mock/model")
        editor.edit_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            prompt="test",
            skip_existing=True,
        )

    # La pipeline non deve essere stata chiamata (file già esistente)
    assert mock_pipe.call_count == 0


# ---------------------------------------------------------------------------
# Test: ImportError senza diffusers
# ---------------------------------------------------------------------------

def test_import_error_without_diffusers():
    """Se diffusers non è installato, deve sollevare ImportError con messaggio chiaro."""
    with patch.dict("sys.modules", {"diffusers": None}):
        with pytest.raises((ImportError, TypeError)):
            # L'import del modulo può fallire in modi diversi a seconda
            # di come Python gestisce il None nel sys.modules
            import importlib
            import diffusion.editor as ed
            importlib.reload(ed)
            ed.ImageEditor(model_id="mock/model")
