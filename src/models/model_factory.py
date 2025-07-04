"""
Module for creating GAN models.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn

from src.models.generator import TransformerGenerator
from src.models.discriminator import Discriminator

def create_models(
    input_size: int,
    output_size: int,
    config: Dict[str, Any],
    device: torch.device,
    num_tickers,
    embedding_dim
) -> Tuple[nn.Module, nn.Module]:
    """
    Create generator and discriminator models.

    Args:
        input_size: Number of input features
        output_size: Number of output features
        config: Configuration dictionary
        device: PyTorch device

    Returns:
        Tuple of (generator, discriminator)
    """
    # Create generator
    generator = TransformerGenerator(
        input_size=input_size,
        output_size=output_size,
        num_layers=config['model_params']['generator']['num_layers'],
        d_model=config['model_params']['generator']['d_model'],
        nhead=config['model_params']['generator']['nhead'],
        dim_feedforward=config['model_params']['generator']['dim_feedforward'],
        dropout=config['model_params']['generator']['dropout'],
        num_tickers=num_tickers,
        ticker_embedding_dim=embedding_dim
    ).to(device)

    # Create discriminator
    discriminator = Discriminator(
        input_size=output_size,
        hidden_sizes=config['model_params']['discriminator']['hidden_sizes'],
        dropout=config['model_params']['discriminator']['dropout'],
        leaky_relu_slope=config['model_params']['discriminator']['leaky_relu_slope']
    ).to(device)

    return generator, discriminator
