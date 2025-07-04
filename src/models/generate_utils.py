"""
Utility functions for generating synthetic financial data.
"""

from typing import Dict, Tuple
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.preprocessing import (
    denormalize_data, encode_dates,
    normalize_data
)

logger = logging.getLogger(__name__)


def generate_synthetic_data(
    generator: nn.Module,
    df_ticker: pd.DataFrame,
    sequence_length: int,
    device: torch.device,
    cyclic_config: Dict[str, bool],
    ticker_to_idx: Dict[str, int]
) -> Tuple[np.ndarray, pd.Series]:
    """
    Generate synthetic financial data using trained generator.

    Args:
        generator: Trained generator model
        df_ticker: DataFrame for specific ticker
        sequence_length: Length of input sequences
        device: PyTorch device
        cyclic_config: Which cyclic date features to use
        ticker_to_idx: Mapping from ticker to integer index

    Returns:
        Tuple of generated data and dates
    """
    if len(df_ticker) <= sequence_length:
        raise ValueError("Input data length must be greater than sequence_length")

    ticker = df_ticker['ticker'].iloc[0]
    ticker_idx = ticker_to_idx[ticker]

    normalized_data = normalize_data(df_ticker, [ticker], cyclic_config)
    ticker_data = normalized_data[ticker]

    sequences = []
    volume_data = ticker_data['norm_volume'].reshape(-1, 1)
    cyclic_features = encode_dates(df_ticker['date'], cyclic_config)

    full_data = np.hstack([
        ticker_data['norm_ohlc'],
        volume_data,
        cyclic_features
    ])

    for i in range(len(full_data) - sequence_length):
        sequences.append(full_data[i:i+sequence_length])

    sequences_tensor = torch.tensor(np.array(sequences), dtype=torch.float32).to(device)

    # Create ticker_idx tensor with shape (batch_size,)
    ticker_indices_tensor = torch.tensor(
        [ticker_idx] * sequences_tensor.size(0),
        dtype=torch.long
    ).to(device)

    generator.eval()
    with torch.no_grad():
        synthetic_normalized = generator(sequences_tensor, ticker_indices_tensor).cpu().numpy()

    synthetic_ohlc = synthetic_normalized[:, :4]
    synthetic_volume = synthetic_normalized[:, 4:5]
    synthetic_cyclic = synthetic_normalized[:, 5:]

    denorm_ohlc, denorm_volume = denormalize_data(
        synthetic_ohlc, 
        synthetic_volume,
        ticker_data['norm_params']
    )

    if denorm_volume.ndim == 1:
        denorm_volume = denorm_volume.reshape(-1, 1)

    synthetic_data = np.hstack([denorm_ohlc, denorm_volume, synthetic_cyclic])

    return synthetic_data, df_ticker['date'].iloc[sequence_length:].reset_index(drop=True)
