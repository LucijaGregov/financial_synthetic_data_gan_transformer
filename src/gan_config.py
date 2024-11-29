"""Configuration for Financial GAN models and training."""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class GANConfig:
    """Configuration for Financial GAN architecture and training.
    
    Attributes:
        batch_size (int): Size of training batches
        sequence_length (int): Length of time series sequences
        feature_dim (int): Total dimension of features
        noise_dim (int): Dimension of noise input
        hidden_dim1 (int): First hidden layer dimension
        hidden_dim2 (int): Second hidden layer dimension
        learning_rate_gen (float): Generator learning rate
        learning_rate_disc (float): Discriminator learning rate
        n_critic (int): Number of critic updates per generator update
        lambda_gp (float): Gradient penalty coefficient
        data_dir (Path): Directory for data storage
        checkpoint_dir (Path): Directory for model checkpoints
    """
    # Model dimensions
    batch_size: int = 64
    sequence_length: int = 5
    feature_dim: int = 501
    noise_dim: int = 200
    hidden_dim1: int = 256
    hidden_dim2: int = 512
    hidden_dim1_disc: int = 256
    hidden_dim2_disc: int = 128
    hidden_dim_lstm_gen: int = 128
    
    # Training hyperparameters
    learning_rate_gen: float = 0.00015
    learning_rate_disc: float = 0.00002
    lambda_gp: float = 7
    n_critic: int = 2
    
    # Loss weights
    temporal_weight: float = 0.2
    feature_matching_weight: float = 0.3
    volatility_weight: float = 0.35
    diversity_loss_weight: float = 0.2
    
    # Data dimensions
    ohlcv_dim: int = 5
    ticker_dim: int = 494
    cyclical_day_dim: int = 2
    
    # Paths
    data_dir: Path = Path('data')
    checkpoint_dir: Path = Path('checkpoints')
    
    def __post_init__(self):
        """Ensure directories exist after initialization."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)