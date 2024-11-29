"""Performance optimization utilities for Financial GAN."""

import torch
import torch.cuda.amp as amp
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Utilities for optimizing model performance.
    
    Attributes:
        model: The GAN model
        device: Computing device
        use_amp (bool): Whether to use automatic mixed precision
    """
    
    def __init__(
        self,
        model: Any,
        device: torch.device,
        use_amp: bool = True
    ):
        """Initialize optimizer.
        
        Args:
            model: The GAN model
            device: Computing device
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        
        if self.use_amp:
            self.scaler = amp.GradScaler()
            logger.info("Using automatic mixed precision training")
    
    def optimize_memory(self):
        """Optimize memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Enable memory pinning
            torch.cuda.set_device(self.device)
            
            # Enable cudnn benchmarking
            torch.backends.cudnn.benchmark = True
    
    def compile_model(self):
        """Compile model using torch.compile."""
        if hasattr(torch, 'compile'):
            self.model.generator = torch.compile(self.model.generator)
            self.model.discriminator = torch.compile(self.model.discriminator)
            logger.info("Model compiled using torch.compile")
    
    def optimize_training_step(
        self,
        data: torch.Tensor
    ) -> Dict[str, float]:
        """Optimized training step with mixed precision.
        
        Args:
            data: Input data tensor
            
        Returns:
            Dict[str, float]: Training metrics
        """
        if self.use_amp:
            with amp.autocast():
                metrics = self.model.train_step(data)
            
            self.scaler.step(self.model.g_optimizer)
            self.scaler.step(self.model.d_optimizer)
            self.scaler.update()
        else:
            metrics = self.model.train_step(data)
        
        return metrics