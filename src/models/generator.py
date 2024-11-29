"""Generator model for financial time series GAN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from ..utils.encoding import generate_day_encoding
from .attention import TransformerBlock

class Generator(nn.Module):
    """Transformer-based generator for financial time series.
    
    Attributes:
        noise_dim (int): Dimension of input noise
        feature_dim (int): Dimension of output features
        sequence_length (int): Length of generated sequences
    """
    
    def __init__(
        self,
        noise_dim: int,
        feature_dim: int,
        sequence_length: int,
        hidden_dims: List[int] = [64, 64],
        num_heads: int = 1
    ):
        """Initialize generator.
        
        Args:
            noise_dim (int): Dimension of input noise
            feature_dim (int): Dimension of output features
            sequence_length (int): Length of generated sequences
            hidden_dims (List[int]): Dimensions of hidden layers
            num_heads (int): Number of attention heads
        """
        super().__init__()
        
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        
        # Register day encoding
        self.register_buffer(
            'day_encoding',
            generate_day_encoding(sequence_length, 1)
        )
        
        # Initial projection
        self.proj = nn.Linear(noise_dim + 2, hidden_dims[0])

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, sequence_length, hidden_dims[0])
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dims[0], heads=num_heads)
            # for _ in range(3)
        ])
        
        # Progressive growth layers
        self.prog_layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:])
        ])
        
        # Output heads
        self.ohlc_head = nn.Linear(hidden_dims[-1], 4)
        self.volume_head = nn.Linear(hidden_dims[-1], 1)
        self.ticker_head = nn.Linear(hidden_dims[-1], feature_dim - 7)
        self.day_head = nn.Linear(hidden_dims[-1], 2)
        
        self.norm = nn.LayerNorm(hidden_dims[-1])
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
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Generate financial time series from noise.
        
        Args:
            noise (torch.Tensor): Input noise of shape (batch_size, seq_len, noise_dim)
            
        Returns:
            torch.Tensor: Generated sequences of shape (batch_size, seq_len, feature_dim)
        """
        batch_size = noise.shape[0]
        
        # Expand day encoding for batch
        day_encoding = self.day_encoding.expand(batch_size, -1, -1).to(noise.device)
        
        # Concatenate noise and day encoding
        x = torch.cat([noise, day_encoding], dim=-1)
        
        # Project and transform
        x = self.proj(x)
        x = x + self.pos_encoding
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Apply progressive layers
        for layer in self.prog_layers:
            x = F.leaky_relu(layer(x), 0.2)
        
        x = self.norm(x)
        
        # Generate components
        ohlc = 3.5 * torch.tanh(self.ohlc_head(x))
        volume = 10 * torch.tanh(self.volume_head(x))
        ticker = torch.sigmoid(self.ticker_head(x))
        day = torch.tanh(self.day_head(x))
        
        return torch.cat([ohlc, volume, ticker, day], dim=-1)
