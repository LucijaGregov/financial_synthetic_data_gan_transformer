import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str,
    filename: str = 'checkpoint.pth.tar'
) -> None:
    """
    Save model checkpoint.
    
    Args:
        state: State dictionary to save
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoint
        filename: Name of checkpoint file
    """
    filepath = Path(checkpoint_dir) / filename
    torch.save(state, filepath)
    if is_best:
        best_filepath = Path(checkpoint_dir) / 'model_best.pth.tar'
        torch.save(state, best_filepath)

def load_checkpoint(
    checkpoint_path: str,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: Optional[torch.optim.Optimizer] = None,
    d_optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[nn.Module, nn.Module, int, float]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        generator: Generator model to load state into
        discriminator: Discriminator model to load state into
        g_optimizer: Optional generator optimizer
        d_optimizer: Optional discriminator optimizer
        
    Returns:
        Tuple of (generator, discriminator, epoch, validation_loss)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    if g_optimizer is not None:
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    if d_optimizer is not None:
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

    return generator, discriminator, checkpoint['epoch'], checkpoint['val_loss']
