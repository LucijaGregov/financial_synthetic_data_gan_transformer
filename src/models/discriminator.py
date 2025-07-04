"""
Discriminator module for GAN architecture that evaluates whether financial data is real or generated.
"""

from typing import List
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class Discriminator(nn.Module):
    """
    Discriminator network for GAN architecture.

    Processes financial data including OHLC, volume, and cyclic features to
    determine if the input is real or generated. Architecture consists of
    fully connected layers with batch normalization, LeakyReLU,
    and dropout, ending with a sigmoid activation for binary classification.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        dropout: float,
        leaky_relu_slope: float
    ) -> None:
        """
        Initialize the discriminator.

        Args:
            input_size: Number of input features (OHLC + volume + enabled cyclic features)
            hidden_sizes: List of sizes for hidden layers
            dropout: Dropout rate for regularization
            leaky_relu_slope: Negative slope coefficient for LeakyReLU activation
        """
        super().__init__()

        layers = []
        current_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(leaky_relu_slope),
                nn.Dropout(dropout)
            ])
            current_size = hidden_size

        layers.extend([
            nn.Linear(hidden_sizes[-1], 1),
            nn.Sigmoid()
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.

        Args:
            x: Input tensor of shape (batch_size, input_size) containing
               normalized financial data features

        Returns:
            Tensor of shape (batch_size, 1) containing probabilities
            that each input sample is real (1) vs generated (0)
        """
        return self.model(x)
