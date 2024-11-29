"""Configuration settings for the Financial GAN project."""

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )

@dataclass
class Config:
    """Configuration class for Financial GAN project.
    
    Args:
        data_dir (Path): Directory for storing data
        start_date (str): Start date for historical data
        end_date (str): End date for historical data
        min_window (int): Minimum number of trading days required
        sequence_length (int): Length of sequences to generate
        
    Attributes:
        DATA_DIR (Path): Path to data directory
        start_date (str): Start date for data collection
        end_date (str): End date for data collection
        min_window (int): Minimum window size for valid data
        sequence_length (int): Length of sequences
    """

    def __init__(
        self,
        data_info: Dict = None,
        batch_size: int = 264,
        sequence_length: int = 5,
        noise_dim: int = 100
    ):
        # Set data-dependent dimensions if data_info provided
        if data_info is not None:
            self.feature_dim = data_info['feature_dim']
            self.ohlcv_dim = data_info['ohlcv_dim']
            self.ticker_dim = data_info['ticker_dim']
            self.cyclical_day_dim = data_info['cyclical_day_dim']
            
            # Adjust batch size if needed for small datasets
            total_sequences = data_info['total_sequences']
            self.batch_size = min(batch_size, total_sequences)
            logger.info(f"Adjusted batch size to {self.batch_size} for {total_sequences} sequences")
        else:
            # Default values
            self.feature_dim = 501
            self.ohlcv_dim = 5
            self.ticker_dim = 494
            self.cyclical_day_dim = 2
            self.batch_size = batch_size

        # Fixed dimensions
        self.sequence_length = sequence_length
        self.noise_dim = noise_dim

        # Model dimensions - could be adjusted based on feature_dim
        self.hidden_dim1 = 128
        self.hidden_dim2 = 256
        self.hidden_dim1_disc = 256
        self.hidden_dim2_disc = 128
        self.hidden_dim_lstm_gen = 128

        # Training hyperparameters
        self.learning_rate_gen = 0.00015
        self.learning_rate_disc = 0.00002
        self.lambda_gp = 7
        self.n_critic = 1

        # Loss weights
        self.temporal_weight = 0.2
        self.feature_matching_weight = 0.3
        self.volatility_weight = 0.35
        self.diversity_loss_weight = 0.2

        # Training settings
        self.print_interval = 10
        # self.print_interval = 1
        self.num_epochs = 200

        # Paths
        self.DATA_DIR = Path('data')
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def __post_init__(self):
        """Log configuration details."""
        logger.info("GAN Configuration:")
        logger.info(f"Feature dimension: {self.feature_dim}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Sequence length: {self.sequence_length}")
