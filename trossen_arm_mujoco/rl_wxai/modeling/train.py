"""Code to train models.

See wxai_gadgetarm_tinker/train_gadgetarm.py and
wxai_gadgetarm_tinker/train_stage2_gripping.py for full training scripts.
This module exposes reusable training utilities.
"""

from pathlib import Path

from rl_wxai.config import MODELS_DIR


def save_model(model, name: str) -> Path:
    """Save a PyTorch model to the models/ directory.

    Args:
        model: PyTorch model or state dict.
        name: Filename (without extension).

    Returns:
        Path to the saved file.
    """
    import torch

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pt"
    torch.save(model, path)
    return path
