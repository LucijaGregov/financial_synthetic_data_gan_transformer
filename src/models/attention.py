"""Attention and transformer modules for the GAN architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class Attention(nn.Module):
    """Self-attention mechanism for sequence processing.
    
    Attributes:
        scale (float): Scaling factor for attention scores
        q (nn.Linear): Query transformation
        k (nn.Linear): Key transformation
        v (nn.Linear): Value transformation
    """
    
    def __init__(self, dim: int):
        """Initialize attention module.
        
        Args:
            dim (int): Dimension of input features
        """
        super().__init__()
        self.scale = dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute self-attention over input sequence.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            torch.Tensor: Attended tensor of same shape as input
        """
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        return attn @ v

class TransformerBlock(nn.Module):
    """Transformer block combining self-attention and feed-forward layers.
    
    Attributes:
        norm1 (nn.LayerNorm): First normalization layer
        attn (Attention): Self-attention module
        norm2 (nn.LayerNorm): Second normalization layer
        mlp (nn.Sequential): Feed-forward network
    """
    
    def __init__(self, dim: int, heads: int = 1, mlp_dim: Optional[int] = None):
        """Initialize transformer block.
        
        Args:
            dim (int): Input dimension
            heads (int): Number of attention heads
            mlp_dim (Optional[int]): Dimension of feed-forward network
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = mlp_dim or dim * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            torch.Tensor: Transformed tensor of same shape as input
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x