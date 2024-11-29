"""Utility functions for data encoding and processing."""

import torch
import numpy as np
from typing import Tuple

def generate_day_encoding(
    sequence_length: int,
    batch_size: int,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """Generate cyclical encoding for days of the week.
    
    Args:
        sequence_length (int): Length of time series sequence
        batch_size (int): Batch size for generation
        device (torch.device): Device to place tensor on
        
    Returns:
        torch.Tensor: Tensor of shape (batch_size, sequence_length, 2) containing
                     sin/cos encoding of days
    """
    days = np.arange(sequence_length) % 7  # weekly cycle
    days_rad = days * (2 * np.pi / 7)
    sin_day = np.sin(days_rad)
    cos_day = np.cos(days_rad)
    
    # Expand for batch size
    sin_day_batch = np.tile(sin_day, (batch_size, 1)).reshape(batch_size, sequence_length, 1)
    cos_day_batch = np.tile(cos_day, (batch_size, 1)).reshape(batch_size, sequence_length, 1)
    
    day_encoding = np.concatenate([sin_day_batch, cos_day_batch], axis=2)
    return torch.tensor(day_encoding, dtype=torch.float32, device=device)