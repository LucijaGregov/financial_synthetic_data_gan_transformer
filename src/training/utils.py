from typing import Any, Dict

import torch


def warmup_learning_rate(
    epoch: int, 
    warmup_epochs: int, 
    base_lr: float
) -> float:
    """
    Calculate learning rate during warmup period.
    
    Args:
        epoch: Current epoch number
        warmup_epochs: Number of warmup epochs
        base_lr: Base learning rate to reach after warmup
        
    Returns:
        Calculated learning rate for current epoch
    """
    if epoch >= warmup_epochs:
        return base_lr
    return base_lr * (epoch + 1) / warmup_epochs

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer, 
    config: Dict[str, Any]
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """
    Create learning rate scheduler with plateau detection.
    
    Args:
        optimizer: PyTorch optimizer instance
        config: Configuration dictionary containing scheduler parameters
        
    Returns:
        ReduceLROnPlateau scheduler
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config['training_params']['scheduler_patience'],
        factor=config['training_params']['scheduler_factor']
    )
