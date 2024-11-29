
"""
Models module containing GAN architectures for financial time series generation.

This module defines the Generator and Discriminator neural network architectures
using PyTorch. The Generator uses a Transformer-based architecture for generating
synthetic financial data, while the Discriminator uses a feed-forward network
for distinguishing real from generated data.

Classes:
    TransformerGenerator: Transformer-based generator network
    Discriminator: Feed-forward discriminator network
"""

import torch
import torch.nn as nn
from typing import Optional
from config import Config

# class TransformerGenerator(nn.Module):
#     """
#     Simplified Transformer-based Generator for financial time series data.
    
#     This generator uses a transformer architecture to generate synthetic financial data
#     without separate trend and volatility layers.
#     """

#     def __init__(self, config: Config):
#         super().__init__()
#         self.noise_dim = config.NOISE_DIM
#         self.hidden_size = config.HIDDEN_SIZE
#         self.n_layers = config.N_LAYERS
#         self.n_heads = config.N_HEADS
#         self.feature_size = config.FEATURE_SIZE

#         # Embedding layers
#         print(config.NUM_STOCKS)
#         self.stock_embedding = nn.Embedding(config.NUM_STOCKS, config.EMBEDDING_DIM)
#         self.noise_embedding = nn.Linear(self.noise_dim, self.hidden_size - config.EMBEDDING_DIM)

#         # Transformer blocks
#         self.transformer_blocks = nn.ModuleList([
#             nn.TransformerEncoderLayer(
#                 d_model=self.hidden_size,
#                 nhead=self.n_heads,
#                 dim_feedforward=self.hidden_size * 4,
#                 dropout=config.DROPOUT,
#                 activation='gelu',
#                 batch_first=True,
#                 norm_first=True
#             ) for _ in range(self.n_layers)
#         ])

#         # Output layer
#         self.output_layer = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.LayerNorm(self.hidden_size),
#             nn.GELU(),
#             nn.Dropout(config.DROPOUT),
#             nn.Linear(self.hidden_size, self.feature_size)
#         )

#     def forward(self, noise: torch.Tensor, stock_id: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the simplified generator.
        
#         Args:
#             noise (torch.Tensor): Input noise tensor of shape (batch_size, noise_dim)
#             stock_id (torch.Tensor): Stock ID tensor of shape (batch_size,)
            
#         Returns:
#             torch.Tensor: Generated financial data of shape (batch_size, feature_size)
#         """
#         # Stock embedding
#         stock_embed = self.stock_embedding(stock_id)  # Shape: (batch_size, embedding_dim)

#         # Noise projection and concatenation
#         noise_proj = self.noise_embedding(noise)  # Shape: (batch_size, hidden_size - embedding_dim)
#         x = torch.cat([noise_proj, stock_embed], dim=1)  # Shape: (batch_size, hidden_size)
        
#         # Add sequence dimension for transformer
#         x = x.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)

#         # Apply transformer blocks
#         for block in self.transformer_blocks:
#             x = block(x)

#         # Remove sequence dimension
#         x = x.squeeze(1)  # Shape: (batch_size, hidden_size)

#         # Output layer
#         return self.output_layer(x)

# class Discriminator(nn.Module):
#     """
#     Discriminator network for the GAN architecture.
    
#     Feed-forward neural network that determines whether input data is real or generated.
#     Uses layer normalization and dropout for stable training.
    
#     Attributes:
#         feature_size (int): Number of input features (OHLCV = 5)
#         hidden_size (int): Size of hidden layers
#     """

#     def __init__(self, config: Config):
#         """
#         Initialize the discriminator network.
        
#         Args:
#             config (Config): Configuration object containing model parameters
#         """
#         super().__init__()
#         self.feature_size = config.FEATURE_SIZE
#         self.hidden_size = config.HIDDEN_SIZE

#         self.model = nn.Sequential(
#             # First layer
#             nn.Linear(self.feature_size, self.hidden_size),
#             nn.LayerNorm(self.hidden_size),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(config.DROPOUT),

#             # Second layer
#             nn.Linear(self.hidden_size, self.hidden_size // 2),
#             nn.LayerNorm(self.hidden_size // 2),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(config.DROPOUT),

#             # Output layer
#             nn.Linear(self.hidden_size // 2, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the discriminator.
        
#         Args:
#             x (torch.Tensor): Input data of shape (batch_size, feature_size)
            
#         Returns:
#             torch.Tensor: Probability of input being real, shape (batch_size, 1)
#         """
#         return self.model(x)

import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math

# class TransformerGenerator(nn.Module):
#     """
#     Simplified Transformer-based Generator with positional encoding.
#     """
#     def __init__(self, config):
#         super().__init__()
#         self.noise_dim = config.NOISE_DIM
#         self.hidden_size = config.HIDDEN_SIZE
#         self.n_layers = config.N_LAYERS
#         self.n_heads = config.N_HEADS
#         self.feature_size = config.FEATURE_SIZE

#         # Embedding layers
#         self.stock_embedding = nn.Embedding(config.NUM_STOCKS, config.EMBEDDING_DIM)
#         self.noise_embedding = nn.Linear(self.noise_dim, self.hidden_size - config.EMBEDDING_DIM)

#         # Positional encoding layer
#         self.positional_encoding = PositionalEncoding(self.hidden_size)

#         # Transformer blocks
#         self.transformer_blocks = nn.ModuleList([
#             nn.TransformerEncoderLayer(
#                 d_model=self.hidden_size,
#                 nhead=self.n_heads,
#                 dim_feedforward=self.hidden_size * 4,
#                 dropout=config.DROPOUT,
#                 activation='gelu'
#             ) for _ in range(self.n_layers)
#         ])

#         # Output layer
#         self.output_layer = nn.Linear(self.hidden_size, self.feature_size)

#     def forward(self, noise: torch.Tensor, stock_id: torch.Tensor) -> torch.Tensor:
#         # Embed stock ID and noise, then concatenate
#         stock_embed = self.stock_embedding(stock_id)  # Shape: (batch_size, embedding_dim)
#         noise_proj = self.noise_embedding(noise)      # Shape: (batch_size, hidden_size - embedding_dim)
#         x = torch.cat([noise_proj, stock_embed], dim=1)  # Shape: (batch_size, hidden_size)

#         # Add sequence dimension for transformer and apply positional encoding
#         x = x.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
#         x = self.positional_encoding(x)

#         # Apply transformer blocks
#         for block in self.transformer_blocks:
#             x = block(x)

#         # Remove sequence dimension and apply output layer
#         x = x.squeeze(1)  # Shape: (batch_size, hidden_size)
#         return self.output_layer(x)  # Shape: (batch_size, feature_size)

import torch
import torch.nn as nn
from typing import Tuple

import torch
import torch.nn as nn
from typing import Tuple

import torch
import torch.nn as nn
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import math

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int = 5000, scale: float = 1.0):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size * scale))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)  # Shape: (1, max_len, hidden_size)
        self.register_buffer('positional_encoding_buffer', encoding)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.positional_encoding_buffer[:, :seq_len, :]

import torch
import torch.nn as nn

class TransformerGenerator(nn.Module):
    def __init__(self, config, num_stocks: int):
        super().__init__()
        
        self.num_stocks = num_stocks
        self.noise_dim = config.NOISE_DIM
        self.hidden_size = config.HIDDEN_SIZE
        self.sequence_length = config.SEQUENCE_LENGTH
        
        # Price level encoder
        self.price_level_encoder = nn.Sequential(
            nn.Linear(self.noise_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU()
        )
        
        # Stock-specific features
        self.stock_embedding = nn.Sequential(
            nn.Embedding(num_stocks, config.EMBEDDING_DIM),
            nn.Linear(config.EMBEDDING_DIM, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU()
        )
        
        # Movement pattern generator
        self.movement_generator = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=3,
            dropout=0.1,
            bidirectional=True,
            batch_first=True
        )
        
        # Transformer for temporal dependencies
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size * 2,  # *2 for bidirectional LSTM
                nhead=config.N_HEADS,
                dim_feedforward=config.FEEDFORWARD_DIM,
                dropout=0.1,
                batch_first=True
            ) for _ in range(config.N_LAYERS)
        ])
        
        # OHLC price generator
        self.price_generator = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 4),  # OHLC
            nn.Tanh()  # Keep relative price movements in [-1, 1]
        )
        
        # Volume generator
        self.volume_generator = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Softplus()  # Ensure positive volume
        )
        
        # Price scaler to match real data range
        self.price_scaler = nn.Parameter(torch.ones(4))  # OHLC scaling
        self.volume_scaler = nn.Parameter(torch.ones(1))  # Volume scaling
        
    def forward(self, noise: torch.Tensor, stock_id: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = noise.size(0)
        
        # Generate base features
        price_features = self.price_level_encoder(noise)
        stock_features = self.stock_embedding(stock_id).unsqueeze(1).expand(-1, self.sequence_length, -1)
        
        # Combine features
        combined_features = price_features + stock_features
        
        # Generate movement patterns
        movement_patterns, _ = self.movement_generator(combined_features)
        
        # Apply transformer layers
        temporal_features = movement_patterns
        for transformer_layer in self.transformer:
            temporal_features = transformer_layer(temporal_features, src_mask=attention_mask)
        
        # Generate OHLC prices
        prices = self.price_generator(temporal_features)
        prices = prices * self.price_scaler.exp()  # Learned scaling
        
        # Generate volume
        volume = self.volume_generator(temporal_features)
        volume = volume * self.volume_scaler.exp()  # Learned scaling
        
        # Ensure OHLC relationship constraints
        open_price = prices[:, :, 0:1]
        high_price = prices[:, :, 1:2].abs() + open_price  # High > Open
        low_price = open_price - prices[:, :, 2:3].abs()   # Low < Open
        close_price = prices[:, :, 3:4] + open_price      # Close varies around Open
        
        # Combine all features
        output = torch.cat([
            open_price,
            high_price,
            low_price,
            close_price,
            volume
        ], dim=-1)
        
        return output
    
class Discriminator(nn.Module):
    def __init__(self, config, num_stocks: int):
        super().__init__()
        
        self.feature_size = config.FEATURE_SIZE
        self.hidden_size = config.HIDDEN_SIZE
        self.embedding_dim = config.EMBEDDING_DIM
        
        # Stock embedding
        self.stock_embedding = nn.Sequential(
            nn.Embedding(num_stocks, self.embedding_dim),
            nn.Linear(self.embedding_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU()
        )
        
        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Conv1d(self.feature_size, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Discrimination layers
        self.discriminator_layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def get_features(self, x, stock_id):
        """Extract features for feature matching loss"""
        stock_features = self.stock_embedding(stock_id)
        x = x.transpose(1, 2)
        return self.feature_layers(x)
    
    def forward(self, x, stock_id):
        features = self.get_features(x, stock_id)
        return self.discriminator_layers(features)
    
# Configuration setup
# class Config:
#     NOISE_DIM = 50  # Dimension of input noise
#     HIDDEN_SIZE = 128  # Hidden layer size for generator and discriminator
#     FEATURE_SIZE = 5  # Number of features in the output (OHLCV)
#     SEQUENCE_LENGTH = 30  # Number of time steps in each sequence
#     NUM_HEADS = 4  # Transformer heads
#     NUM_LAYERS = 2  # Number of transformer layers
#     NUM_STOCKS = 10  # Number of unique stocks in the dataset
#     EMBEDDING_DIM = 16  # Dimensionality of stock embedding
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Example instantiation
# config = Config()
# generator = RecurrentTransformerGenerator(config).to(config.DEVICE)
# discriminator = CNNDiscriminator(config).to(config.DEVICE)


def init_weights(model: nn.Module) -> None:
    """
    Initialize model weights for better training.
    
    Args:
        model (nn.Module): PyTorch model to initialize
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='leaky_relu')
            else:
                nn.init.normal_(param, mean=0.0, std=0.02)
        elif 'bias' in name:
            nn.init.zeros_(param)

def get_models(config: Config) -> tuple[nn.Module, nn.Module]:
    """
    Create and initialize generator and discriminator models.
    
    Args:
        config (Config): Configuration object containing model parameters
        
    Returns:
        tuple[nn.Module, nn.Module]: Initialized generator and discriminator models
    """
    generator = TransformerGenerator(config)
    discriminator = Discriminator(config)

    # Initialize weights
    init_weights(generator)
    init_weights(discriminator)

    # Move to appropriate device
    generator = generator.to(config.DEVICE)
    discriminator = discriminator.to(config.DEVICE)

    # Handle multi-GPU if available
    if config.NUM_GPUS > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    return generator, discriminator

# # src/pipeline/model.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from typing import Optional
# from pipeline.config import Config  # assuming Config is in config.py

# class PositionalEncoding(nn.Module):
#     """
#     Adds positional encoding to input embeddings to retain positional information in the model.

#     Args:
#         d_model (int): The dimensionality of the embeddings.
#         dropout (float): The dropout rate applied to the positional encoding.
#         max_len (int): The maximum length of sequences for which to compute positional encodings.
#     """
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pe = torch.zeros(1, max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[0, :, 0::2] = torch.sin(position * div_term)
#         pe[0, :, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass to add positional encoding to the input tensor.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

#         Returns:
#             torch.Tensor: Output tensor with positional encoding added.
#         """
#         x = x + self.pe[:, :x.size(1)]
#         return self.dropout(x)


# class Generator(nn.Module):
#     """
#     Transformer-based generator for GAN, which generates synthetic stock sequences.

#     Args:
#         config (Config): Configuration object containing model parameters.
#     """
#     def __init__(self, config: Config):
#         super().__init__()
#         self.config = config
#         self.embedding = nn.Embedding(config.num_bins, config.input_dim)
#         self.pos_encoder = PositionalEncoding(config.input_dim, dropout=config.dropout)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=config.input_dim,
#             nhead=config.attention_heads,
#             dim_feedforward=4 * config.input_dim,
#             dropout=config.dropout,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
#         self.fc_out = nn.Linear(config.input_dim, config.num_bins)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the generator.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

#         Returns:
#             torch.Tensor: Output tensor representing generated data logits of shape 
#                           (batch_size, sequence_length, num_bins).
#         """
#         x = self.embedding(x) * math.sqrt(self.config.input_dim)
#         x = self.pos_encoder(x)
#         x = self.transformer(x)
#         return self.fc_out(x)


# class Discriminator(nn.Module):
#     """
#     Transformer-based discriminator for GAN, which distinguishes real from synthetic stock sequences.

#     Args:
#         config (Config): Configuration object containing model parameters.
#     """
#     def __init__(self, config: Config):
#         super().__init__()
#         self.config = config
#         self.embedding = nn.Embedding(config.num_bins, config.input_dim)
#         self.pos_encoder = PositionalEncoding(config.input_dim, dropout=config.dropout)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=config.input_dim,
#             nhead=config.attention_heads,
#             dim_feedforward=4 * config.input_dim,
#             dropout=config.dropout,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
#         self.fc1 = nn.Linear(config.input_dim, config.input_dim // 2)
#         self.fc2 = nn.Linear(config.input_dim // 2, 1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the discriminator.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

#         Returns:
#             torch.Tensor: Output tensor representing real/fake probability, of shape (batch_size, 1).
#         """
#         x = self.embedding(x) * math.sqrt(self.config.input_dim)
#         x = self.pos_encoder(x)
#         x = self.transformer(x)
#         x = torch.mean(x, dim=1)
#         x = F.relu(self.fc1(x))
#         return torch.sigmoid(self.fc2(x))
