"""Discriminator model for financial time series GAN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import List, Tuple
from .attention import TransformerBlock

class Discriminator(nn.Module):
    """Transformer-based discriminator with spectral normalization.
    
    Attributes:
        feature_dim (int): Dimension of input features
        sequence_length (int): Length of input sequences
        hidden_dims (List[int]): Dimensions of hidden layers
    """
    
    def __init__(
        self,
        feature_dim: int,
        sequence_length: int,
        hidden_dims: List[int] = [64, 128],
        num_heads: int = 1
    ):
        """Initialize discriminator.
        
        Args:
            feature_dim (int): Dimension of input features
            sequence_length (int): Length of input sequences
            hidden_dims (List[int]): Dimensions of hidden layers
            num_heads (int): Number of attention heads
        """
        super().__init__()
        
        # Initial projection with spectral normalization
        self.proj = spectral_norm(nn.Linear(feature_dim, hidden_dims[0]))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dims[0], heads=num_heads)
            # for _ in range(3)
        ])
        
        # Progressive discriminator layers
        self.prog_layers = nn.ModuleList([
            spectral_norm(nn.Linear(in_dim, out_dim))
            for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:])
        ])
        
        # Critic head
        self.critic = spectral_norm(nn.Linear(hidden_dims[-1], 1))
        
        # Feature matching layers
        self.feature_layers = nn.ModuleList([
            nn.Linear(hidden_dims[-1], dim)
            for dim in [64, 32]
        ])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize network weights.
        
        Args:
            module (nn.Module): Module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process input through discriminator.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Critic values and extracted features
        """
        # Initial projection
        x = F.leaky_relu(self.proj(x), 0.2)
        
        # Apply transformer blocks
        features = []
        for block in self.transformer_blocks:
            x = block(x)
            features.append(x)
        
        # Progressive discrimination
        for layer in self.prog_layers:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        
        # Feature matching
        matched_features = []
        for layer in self.feature_layers:
            matched_features.append(layer(x))
        
        # Final critic output
        critic_value = self.critic(x)
        
        return critic_value, torch.cat(matched_features, dim=-1)