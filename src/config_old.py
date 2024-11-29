# import torch
# import numpy as np
# from pathlib import Path
# import logging

# def setup_logging(config: 'Config') -> logging.Logger:
#     """Setup logging configuration."""
#     logger = logging.getLogger('GAN_Training')
#     logger.setLevel(logging.INFO)

#     # Check if handlers are already set to avoid duplicates
#     if not logger.handlers:
#         # Create directories if they don't exist
#         config.LOG_DIR.mkdir(exist_ok=True)

#         # File handler
#         fh = logging.FileHandler(config.LOG_DIR / 'training.log')
#         fh.setLevel(logging.INFO)

#         # Console handler
#         ch = logging.StreamHandler()
#         ch.setLevel(logging.INFO)

#         # Formatter
#         formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#         fh.setFormatter(formatter)
#         ch.setFormatter(formatter)

#         # Add handlers only if they are not already added
#         logger.addHandler(fh)
#         logger.addHandler(ch)

#     return logger

# class Config:
#     """Configuration class for model and training parameters."""
#     def __init__(self):
#         # Set up directories
#         self.LOG_DIR = Path('logs')
#         self.CHECKPOINT_DIR = Path('checkpoints')
#         self.DATA_DIR = Path('data')
#         # self.RESULTS_DIR = Path.home() / 'results'
#         # self.LOG_DIR = self.RESULTS_DIR / 'logs'
#         # self.CHECKPOINT_DIR = self.RESULTS_DIR / 'checkpoints'

#         # Create directories
#         # self.RESULTS_DIR.mkdir(exist_ok=True)
#         self.LOG_DIR.mkdir(exist_ok=True)
#         self.CHECKPOINT_DIR.mkdir(exist_ok=True)
#         self.DATA_DIR.mkdir(exist_ok=True)

#         # Model architecture
#         # self.NOISE_DIM = 100    # Dimension of noise input
#         # self.HIDDEN_SIZE = 512  # Size of hidden layers
#         # self.N_LAYERS = 3       # Number of transformer layers
#         # self.N_HEADS = 8        # Number of attention heads
#         # self.DROPOUT = 0.2      # Dropout rate
        # self.FEATURE_SIZE = 5   # OHLCV features

#         # # Training parameters
#         # self.BATCH_SIZE = 256   # Training batch size
#         # self.N_EPOCHS = 200     # Number of training epochs
#         # self.BASE_LR = 0.0001   # Learning rate
        # self.PATIENCE = 15      # Early stopping patience
#         # self.MIN_DELTA = 0.0001 # Minimum improvement for early stopping
        
#         self.NOISE_DIM = 100    # Dimension of noise input
#         self.HIDDEN_SIZE = 264  # Size of hidden layers (must be divisible by N_HEADS and larger than the embedding dimension)
#         self.N_LAYERS = 4       # Number of transformer layers
#         self.N_HEADS = 4        # Number of attention heads
#         self.DROPOUT = 0.1      # Dropout rate
#         self.FEATURE_SIZE = 5   # OHLCV features
#         self.SEQUENCE_LENGTH = 200

#         # Training parameters
#         self.BATCH_SIZE = 64   # Training batch size
#         self.N_EPOCHS = 10     # Number of training epochs
#         self.GEN_LR = 0.0002   # Learning rate
#         self.DISC_LR = 0.0001
#         self.PATIENCE = 15      # Early stopping patience
#         self.MIN_DELTA = 0.0001 # Minimum improvement for early stopping

#         # Optimizer parameters
#         self.BETA1 = 0.5
#         self.BETA2 = 0.999
#         self.WEIGHT_DECAY = 1e-5
#         self.GRADIENT_CLIP = 1.0
        
#         # self.NUM_STOCKS = 10
#         self.EMBEDDING_DIM = 100

#         self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         self.CHECKPOINT_FREQ = 5
#         self.FEEDFORWARD_DIM = 300

#         self.CPU_OPTIMIZATIONS = {
#             'num_threads': 4,
#             'pin_memory': False,
#             'persistent_workers': False,
#             'prefetch_factor': 2,
#         }

#         if self.DEVICE.type == 'cpu':
#             # Set number of threads for PyTorch
#             torch.set_num_threads(self.CPU_OPTIMIZATIONS['num_threads'])

#             # Enable Intel MKL optimizations if available
#             try:
#                 import mkl
#                 mkl.set_num_threads(self.CPU_OPTIMIZATIONS['num_threads'])
#             except ImportError:
#                 pass

#         # System setup
#         self.setup_system()

#     def setup_system(self):
#         """Setup system-specific parameters."""
#         # Determine device (CPU or GPU)
#         self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # Set up CPU/GPU specific parameters
#         if self.DEVICE.type == 'cuda':
#             self.NUM_GPUS = torch.cuda.device_count()
#             # If multiple GPUs, adjust batch size and learning rate
#             if self.NUM_GPUS > 1:
#                 self.BATCH_SIZE *= self.NUM_GPUS
#                 self.GEN_LR *= np.sqrt(self.NUM_GPUS)
#                 self.DISC_LR *= np.sqrt(self.NUM_GPUS)
#             self.NUM_WORKERS = 4  # Number of data loading workers for GPU
#             self.PIN_MEMORY = True
#         else:
#             self.NUM_GPUS = 0
#             torch.set_num_threads(4)  # Optimize CPU threads
#             self.NUM_WORKERS = 0  # No multiprocessing for CPU
#             self.PIN_MEMORY = False

import torch
import numpy as np
from pathlib import Path
import logging

def setup_logging(config: 'Config') -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('GAN_Training')
    logger.setLevel(logging.INFO)

    # Check if handlers are already set to avoid duplicates
    if not logger.handlers:
        # Create directories if they don't exist
        config.LOG_DIR.mkdir(exist_ok=True)

        # File handler
        fh = logging.FileHandler(config.LOG_DIR / 'training.log')
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers only if they are not already added
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

class Config:
    """Configuration class for model and training parameters."""
    def __init__(self):
        # Set up directories
        self.LOG_DIR = Path('logs')
        self.CHECKPOINT_DIR = Path('checkpoints')
        self.DATA_DIR = Path('data')
        # self.RESULTS_DIR = Path.home() / 'results'
        # self.LOG_DIR = self.RESULTS_DIR / 'logs'
        # self.CHECKPOINT_DIR = self.RESULTS_DIR / 'checkpoints'

        # Create directories
        # self.RESULTS_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)

        # Model architecture - optimized for financial time series
        self.NOISE_DIM = 128
        self.HIDDEN_SIZE = 256
        self.N_LAYERS = 4
        self.N_HEADS = 8
        self.DROPOUT = 0.2
        self.EMBEDDING_DIM = 64
        self.FEEDFORWARD_DIM = 512
        self.SEQUENCE_LENGTH = 50
        self.FEATURE_SIZE = 5
        
        # Training parameters
        self.GEN_LR = 0.0004     # 4x higher than discriminator
        self.DISC_LR = 0.0001    # Very low discriminator learning rate
        self.BATCH_SIZE = 16     # Smaller batch size
        self.GRADIENT_CLIP = 0.1    # Smaller batch size
        self.N_EPOCHS = 200     # More epochs
        self.PATIENCE = 30      # More patience

        self.BETA1 = 0.5
        self.BETA2 = 0.999
        self.WEIGHT_DECAY = 1e-6
        
        # Early stopping
        self.MIN_DELTA = 0.01
        self.FEATURE_SIZE = 5   # OHLCV features
        # self.PATIENCE = 15      # Early stopping patience

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_system()

    def setup_system(self):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.DEVICE.type == 'cuda':
            self.NUM_GPUS = torch.cuda.device_count()
            if self.NUM_GPUS > 1:
                self.BATCH_SIZE *= self.NUM_GPUS
                self.GEN_LR *= np.sqrt(self.NUM_GPUS)
                self.DISC_LR *= np.sqrt(self.NUM_GPUS)
            self.NUM_WORKERS = 4
            self.PIN_MEMORY = True
        else:
            self.NUM_GPUS = 0
            torch.set_num_threads(4)
            self.NUM_WORKERS = 0
            self.PIN_MEMORY = False

