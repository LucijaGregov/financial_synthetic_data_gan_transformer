"""
Generator module implementing a Transformer-based architecture for synthetic financial data generation.
"""

from typing import Optional
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class TransformerGenerator(nn.Module):
    """
    Transformer-based generator for financial time series data.
    
    Uses a transformer encoder architecture to generate synthetic financial data,
    processing OHLC prices, volume, and cyclic date features.
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        output_size: int,
        dropout: float,
        num_tickers: int,
        ticker_embedding_dim: int,
        noise_dim: int = 8
    ) -> None:
        """
        Initialize the generator.

        Args:
            input_size: Number of input features (OHLC + volume + enabled cyclic features)
            num_layers: Number of transformer encoder layers
            d_model: Dimension of the transformer model
            nhead: Number of attention heads
            dim_feedforward: Dimension of feedforward network in transformer
            output_size: Number of output features (same as input_size)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.noise_dim = noise_dim
        self.noise_projection = nn.Linear(noise_dim, d_model)
        
        self.input_size = input_size
        self.embedding_dim = ticker_embedding_dim
        self.ticker_embedding = nn.Embedding(num_tickers, ticker_embedding_dim)

        self.positional_encoding = nn.Linear(input_size + ticker_embedding_dim, d_model)
        self.price_linear = nn.Linear(d_model, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        self.feature_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_size)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        src: torch.Tensor, 
        ticker_idx: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        embed = self.ticker_embedding(ticker_idx).unsqueeze(1).repeat(1, src.shape[1], 1)
        x = torch.cat([src, embed], dim=-1)
        x = self.positional_encoding(x)
        x = self.norm1(x)

        if noise is None:
            noise = torch.randn(src.size(0), self.noise_dim, device=src.device)

        noise_proj = self.noise_projection(noise)  # Shape: (batch_size, d_model)
        noise_proj = noise_proj.unsqueeze(1).repeat(1, src.size(1), 1)
        x = x + noise_proj

        price_features = self.price_linear(x)
        x = x + price_features
        x = self.norm2(x)

        output = self.transformer_encoder(x)
        output = self.feature_decoder(output[:, -1, :])
        output = self._apply_financial_constraints(output)
        return output


    def _apply_financial_constraints(self, output: torch.Tensor) -> torch.Tensor:
        """
        Apply financial constraints to generator output.
        
        Args:
            output: Raw generator output of shape (batch_size, output_size)
            
        Returns:
            Constrained output respecting financial relationships
        """
        # Split output into components
        open_price = output[:, 0:1]   # Open
        high_price = output[:, 1:2]   # High
        low_price = output[:, 2:3]    # Low
        close_price = output[:, 3:4]  # Close
        volume = torch.relu(output[:, 4:5])  # Volume must be positive
        
        if output.size(1) > 5:  # If we have cyclic features
            cyclic_features = output[:, 5:]
        else:
            cyclic_features = torch.empty(output.size(0), 0, device=output.device)

        # Apply OHLC constraints
        constrained_ohlc = self._enforce_ohlc_constraints(open_price, high_price, low_price, close_price)

        return torch.cat([constrained_ohlc, volume, cyclic_features], dim=1)

    def _enforce_ohlc_constraints(self, open_p: torch.Tensor, high_p: torch.Tensor,
                                 low_p: torch.Tensor, close_p: torch.Tensor) -> torch.Tensor:
        """
        Enforce OHLC financial constraints:
        - High >= max(Open, Close)
        - Low <= min(Open, Close)
        
        Args:
            open_p, high_p, low_p, close_p: Individual OHLC price tensors
            
        Returns:
            Constrained OHLC tensor of shape (batch_size, 4)
        """
        # High >= max(Open, Close)
        min_high = torch.max(open_p, close_p)
        constrained_high = torch.max(high_p, min_high)

        # Low <= min(Open, Close)  
        max_low = torch.min(open_p, close_p)
        constrained_low = torch.min(low_p, max_low)

        return torch.cat([open_p, constrained_high, constrained_low, close_p], dim=1)
