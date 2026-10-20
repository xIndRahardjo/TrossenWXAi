"""Code to run model inference with trained models.

See wxai_gadgetarm_tinker/evaluate_gadgetarm.py and
wxai_gadgetarm_tinker/evaluate_stage2_gripping.py for full evaluation scripts.
"""

from pathlib import Path

import torch

from rl_wxai.config import MODELS_DIR


def load_model(name: str):
    """Load a trained model from the models/ directory.

    Args:
        name: Filename without extension, e.g. 'best_gadgetarm_model'.

    Returns:
        Loaded PyTorch model / state dict.
    """
    path = MODELS_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return torch.load(path, map_location="cpu")
