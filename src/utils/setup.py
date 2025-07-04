import logging
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist 
import os

logger = logging.getLogger(__name__)

def set_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Integer seed for random number generators
        
    Note:
        Sets seeds for Python's random, NumPy, and PyTorch (CPU and CUDA)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_environment(config: Dict[str, Any]) -> None:
    """
    Setup training environment.

    Args:
        config: Configuration dictionary containing paths and seed
    """
    try:
        set_seeds(config["seed"])
        for directory in config['paths'].values():
            Path(directory).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error setting up environment: {str(e)}")
        raise RuntimeError(f"Environment setup failed: {str(e)}")
