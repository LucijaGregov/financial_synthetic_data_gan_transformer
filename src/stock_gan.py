"""
Stock Market GAN Implementation
"""
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_gan.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class StockDataset(Dataset):
    """Dataset for stock market data."""
    def __init__(self, data: Dict[str, pd.DataFrame], config):
        super().__init__()
        self.sequences = []
        self.targets = []
        self.config = config
        
        logger.info(f"Processing {len(data)} stocks...")
        
        for ticker, df in data.items():
            try:
                sequences = self.prepare_sequences(df)
                
                for i, seq in enumerate(sequences):
                    if i < len(sequences) - 1:  # Ensure we have a target
                        # Convert to tensor and quantize
                        seq_tensor = torch.tensor(seq, dtype=torch.float32)
                        target_tensor = torch.tensor(
                            sequences[i + 1][-1],  # Next day's values
                            dtype=torch.float32
                        )
                        
                        # Quantize data
                        seq_quantized = torch.bucketize(
                            seq_tensor,
                            torch.linspace(-3, 3, self.config.num_bins)
                        )
                        target_quantized = torch.bucketize(
                            target_tensor,
                            torch.linspace(-3, 3, self.config.num_bins)
                        )
                        
                        self.sequences.append(seq_quantized)
                        self.targets.append(target_quantized)
                
            except Exception as e:
                logger.warning(f"Error processing {ticker}: {str(e)}")
                continue
        
        logger.info(f"Total sequences created: {len(self.sequences)}")

    def prepare_sequences(self, df: pd.DataFrame) -> List[np.ndarray]:
        """Create sequences from DataFrame."""
        # Select OHLCV data
        price_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
        
        # Normalize
        normalized_data = (price_data - np.mean(price_data, axis=0)) / (np.std(price_data, axis=0) + 1e-8)
        
        # Create sequences
        sequences = []
        for i in range(len(normalized_data) - self.config.min_window):
            seq = normalized_data[i:i + self.config.min_window]
            sequences.append(seq)
        
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]

@dataclass
class Config:
    """Configuration for the GAN model."""
    # Training Parameters
    num_epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-4
    early_stopping_patience: int = 10
    validation_interval: int = 1
    gradient_clip_val: float = 1.0
    warmup_steps: int = 100
    
    # Model Architecture
    input_dim: int = 64
    output_dim: int = 5  # OHLCV features
    min_window: int = 10
    max_window: int = 20
    num_bins: int = 256
    transformer_layers: int = 3
    attention_heads: int = 4
    dropout: float = 0.2
    
    # Data Parameters
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    data_dir: Path = Path('data')
    
    # Hardware
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers: int = 0
    seed: int = 42
    
    def __post_init__(self):
        """Initialize directories and set random seeds."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.data_dir / 'models'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.data_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

class PositionalEncoding(nn.Module):
    """Add positional encoding to input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Generator(nn.Module):
    """Transformer-based generator."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.num_bins, config.input_dim)
        self.pos_encoder = PositionalEncoding(config.input_dim, dropout=config.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.input_dim,
            nhead=config.attention_heads,
            dim_feedforward=4*config.input_dim,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers
        )
        
        self.fc_out = nn.Linear(config.input_dim, config.num_bins)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.config.input_dim)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.fc_out(x)

class Discriminator(nn.Module):
    """Transformer-based discriminator."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.num_bins, config.input_dim)
        self.pos_encoder = PositionalEncoding(config.input_dim, dropout=config.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.input_dim,
            nhead=config.attention_heads,
            dim_feedforward=4*config.input_dim,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers
        )
        
        self.fc1 = nn.Linear(config.input_dim, config.input_dim // 2)
        self.fc2 = nn.Linear(config.input_dim // 2, 1)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.to(torch.long)
        x = self.embedding(x) * math.sqrt(self.config.input_dim)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = torch.mean(x, dim=1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            delta = self.best_score - score
        else:
            delta = score - self.best_score

        if delta > self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

def create_data_loaders(dataset: Dataset, config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(config.train_ratio * total_size)
    val_size = int(config.val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

class Trainer:
    """Handles model training and evaluation."""
    def __init__(
        self,
        config: Config,
        generator: nn.Module,
        discriminator: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        self.config = config
        self.generator = generator.to(config.device)
        self.discriminator = discriminator.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizers
        self.g_optimizer = optim.Adam(generator.parameters(), lr=config.learning_rate)
        self.d_optimizer = optim.Adam(discriminator.parameters(), lr=config.learning_rate)
        
        # Schedulers
        self.g_scheduler = ReduceLROnPlateau(self.g_optimizer, mode='min', factor=0.5, patience=5)
        self.d_scheduler = ReduceLROnPlateau(self.d_optimizer, mode='min', factor=0.5, patience=5)
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        
        # Loss tracking
        self.losses = {'g_loss': [], 'd_loss': [], 'val_g_loss': [], 'val_d_loss': []}

    def train(self):
        """Training loop."""
        for epoch in range(self.config.num_epochs):
            # Training
            train_g_loss, train_d_loss = self._train_epoch()
            self.losses['g_loss'].append(train_g_loss)
            self.losses['d_loss'].append(train_d_loss)
            
            # Validation
            val_g_loss, val_d_loss = self._validate()
            self.losses['val_g_loss'].append(val_g_loss)
            self.losses['val_d_loss'].append(val_d_loss)
            
            # Update schedulers
            self.g_scheduler.step(val_g_loss)
            self.d_scheduler.step(val_d_loss)
            
            # Early stopping
            if self.early_stopping(val_g_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Logging
            logger.info(
                f"Epoch {epoch}: "
                f"G_loss={train_g_loss:.4f}, "
                f"D_loss={train_d_loss:.4f}, "
                f"Val_G_loss={val_g_loss:.4f}, "
                f"Val_D_loss={val_d_loss:.4f}"
            )
        
        return self.losses

    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        g_losses = []
        d_losses = []
        
        pbar = tqdm(self.train_loader)
        for real_seq, _ in pbar:
            batch_size = real_seq.size(0)
            real_seq = real_seq.to(self.config.device)
            
            # Train Discriminator
            self.d_optimizer.zero_grad()
            noise = torch.randint(
                0, self.config.num_bins,
                (batch_size, real_seq.size(1)),
                device=self.config.device
            )
            
            # Generate fake sequences
            fake_seq = self.generator(noise)
            fake_seq = torch.argmax(fake_seq, dim=-1)
            
            # Discriminator predictions
            d_real = self.discriminator(real_seq)
            d_fake = self.discriminator(fake_seq.detach())
            
            # Discriminator loss
            d_loss_real = F.binary_cross_entropy(
                d_real,
                torch.ones_like(d_real)
            )
            d_loss_fake = F.binary_cross_entropy(
                d_fake,
                torch.zeros_like(d_fake)
            )
            d_loss = (d_loss_real + d_loss_fake) / 2
            
            d_loss.backward()
            clip_grad_norm_(
                self.discriminator.parameters(),
                self.config.gradient_clip_val
            )
            self.d_optimizer.step()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            d_fake = self.discriminator(fake_seq)
            g_loss = F.binary_cross_entropy(
                d_fake,
                torch.ones_like(d_fake)
            )
            
            g_loss.backward()
            clip_grad_norm_(
                self.generator.parameters(),
                self.config.gradient_clip_val
            )
            self.g_optimizer.step()
            
            # Record losses
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'G_Loss': f'{g_loss.item():.4f}',
                'D_Loss': f'{d_loss.item():.4f}'
            })
        
        return np.mean(g_losses), np.mean(d_losses)

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        """Validate models."""
        self.generator.eval()
        self.discriminator.eval()
        
        g_losses = []
        d_losses = []
        
        for real_seq, _ in self.val_loader:
            batch_size = real_seq.size(0)
            real_seq = real_seq.to(self.config.device)
            
            # Generate fake sequences
            noise = torch.randint(
                0,
                self.config.num_bins,
                (batch_size, real_seq.size(1)),
                device=self.config.device
            )
            fake_seq = self.generator(noise)
            fake_seq = torch.argmax(fake_seq, dim=-1)
            
            # Discriminator predictions
            d_real = self.discriminator(real_seq)
            d_fake = self.discriminator(fake_seq)
            
            # Losses
            d_loss = (
                F.binary_cross_entropy(d_real, torch.ones_like(d_real)) +
                F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
            ) / 2
            
            g_loss = F.binary_cross_entropy(
                d_fake,
                torch.ones_like(d_fake)
            )
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
        
        return np.mean(g_losses), np.mean(d_losses)

    def generate_samples(self, num_samples: int, seq_length: int) -> torch.Tensor:
        """Generate synthetic stock sequences."""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randint(
                0,
                self.config.num_bins,
                (num_samples, seq_length),
                device=self.config.device
            )
            fake_seq = self.generator(noise)
            return torch.argmax(fake_seq, dim=-1)

def plot_training_history(losses: Dict[str, List[float]], save_path: str = 'training_history.png'):
    """Plot training and validation losses."""
    plt.figure(figsize=(12, 6))
    
    # Plot generator losses
    plt.subplot(1, 2, 1)
    plt.plot(losses['g_loss'], label='Train')
    plt.plot(losses['val_g_loss'], label='Validation')
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot discriminator losses
    plt.subplot(1, 2, 2)
    plt.plot(losses['d_loss'], label='Train')
    plt.plot(losses['val_d_loss'], label='Validation')
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(stock_data: Dict[str, pd.DataFrame]):
    """Main execution function."""
    try:
        # Initialize configuration
        config = Config()
        logger.info("Configuration initialized")
        
        # Create dataset
        dataset = StockDataset(stock_data, config)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(dataset, config)
        
        # Initialize models
        generator = Generator(config)
        discriminator = Discriminator(config)
        
        # Create trainer
        trainer = Trainer(
            config=config,
            generator=generator,
            discriminator=discriminator,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Train model
        logger.info("Starting training...")
        history = trainer.train()
        
        # Plot training history
        plot_training_history(history)
        
        # Generate samples
        samples = trainer.generate_samples(
            num_samples=5,
            seq_length=config.min_window
        )
        
        logger.info("Training completed successfully!")
        return trainer, history, samples
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Load data from pickle file
        with open('stock_data.pkl', 'rb') as f:
            stock_data = pickle.load(f)
            
        trainer, history, samples = main(stock_data)
        print("Success! Check the logs and output directory for results.")
    except Exception as e:
        print(f"Error: {str(e)}")
