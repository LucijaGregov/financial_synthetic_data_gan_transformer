"""Trainer module for Financial GAN."""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from src.models.gan import FinancialGAN
from src.utils.paths import DIRS
from config import Config
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class for Financial GAN model.
    
    Args:
        normalized_data (np.ndarray): Normalized training data of shape (n_samples, seq_length, n_features)
        config (Config): Configuration object containing model parameters
        device (str): Computing device ('cuda' or 'cpu')
        
    Attributes:
        config (Config): Configuration object
        device (str): Computing device
        model (FinancialGAN): The GAN model
        train_data (torch.Tensor): Training data tensor
        history (Dict[str, List]): Training history
        checkpoint_dir (Path): Directory for checkpoints
        plots_dir (Path): Directory for plots
    """
    
    def __init__(
        self,
        normalized_data: np.ndarray,
        config: Config,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        self.config = config
        self.device = device
        self.checkpoint_dir = DIRS['checkpoints']
        self.plots_dir = DIRS['plots']

        logger.info(f"Initializing model with:")
        logger.info(f"- Feature dimension: {self.config.feature_dim}")
        logger.info(f"- Sequence length: {self.config.sequence_length}")
        logger.info(f"- Batch size: {self.config.batch_size}")

        self.model = FinancialGAN(
            feature_dim=self.config.feature_dim,
            sequence_length=self.config.sequence_length,
            noise_dim=self.config.noise_dim,
            device=device
        )

        self.train_data = torch.FloatTensor(normalized_data).to(device)
        self.train_loader = DataLoader(
            TensorDataset(self.train_data),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if device=='cuda' else False,
            prefetch_factor=2,
            persistent_workers=True,  # Keep workers alive between epochs
            drop_last=True  # Skip incomplete final batch
        )

        self.history: Dict[str, List[Any]] = {
            'd_loss': [],
            'g_loss': [],
            'consistency_loss': [],
            'diversity_loss': [],
            'temporal_loss': [],
            'generated_samples': []
        }

    def quick_profile(self, num_batches: int = 5) -> None:
        """Quick profiling of first few batches to identify bottlenecks.
        
        Args:
            num_batches (int): Number of batches to profile
        """
        import time
        print("\nQuick Training Profile:")
        
        batch_times = []
        d_times = []
        g_times = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx >= num_batches:
                break
                
            batch_start = time.time()
            real_batch = batch[0]
            
            # Time discriminator updates
            d_start = time.time()
            for _ in range(self.config.n_critic):
                self.model.d_optimizer.zero_grad()
                noise = torch.randn(len(real_batch), self.config.sequence_length, 
                                  self.config.noise_dim).to(self.device)
                fake_data = self.model.generator(noise)
                real_critic, real_features = self.model.discriminator(real_batch)
                fake_critic, fake_features = self.model.discriminator(fake_data.detach())
                d_loss = fake_critic.mean() - real_critic.mean()
                d_loss.backward()
                self.model.d_optimizer.step()
            d_time = time.time() - d_start
            d_times.append(d_time)
            
            # Time generator update
            g_start = time.time()
            self.model.g_optimizer.zero_grad()
            noise = torch.randn(len(real_batch), self.config.sequence_length, 
                              self.config.noise_dim).to(self.device)
            fake_data = self.model.generator(noise)
            fake_critic, fake_features = self.model.discriminator(fake_data)
            g_loss = -fake_critic.mean()
            g_loss.backward()
            self.model.g_optimizer.step()
            g_time = time.time() - g_start
            g_times.append(g_time)
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            print(f"\nBatch {batch_idx+1}:")
            print(f"Total batch time: {batch_time:.4f}s")
            print(f"Discriminator time: {d_time:.4f}s")
            print(f"Generator time: {g_time:.4f}s")
        
        avg_batch_time = sum(batch_times) / len(batch_times)
        print(f"\nAverage times over {num_batches} batches:")
        print(f"Batch: {avg_batch_time:.4f}s")
        print(f"Discriminator: {sum(d_times)/len(d_times):.4f}s")
        print(f"Generator: {sum(g_times)/len(g_times):.4f}s")
        
        total_batches = len(self.train_loader)
        projected_epoch_time = avg_batch_time * total_batches
        print(f"\nProjected full epoch time: {projected_epoch_time:.2f}s ({projected_epoch_time/60:.2f}min)")

    def generate_sequences(
        self,
        num_sequences: int = 1,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate new sequences.
        
        Args:
            num_sequences (int): Number of sequences to generate
            temperature (float): Sampling temperature
            
        Returns:
            torch.Tensor: Generated sequences
        """
        self.model.generator.eval()

        with torch.no_grad():
            noise = torch.randn(
                num_sequences, 
                self.config.sequence_length,
                self.config.noise_dim,
                device=self.device
            ) * temperature

            sequences = self.model.generator(noise)

        return sequences

    def train(self) -> None:
        """Train the model.
        
        Records training metrics in history and prints progress at specified intervals.
        Saves checkpoints and generated samples during training.
        """
        logger.info("\nStarting training...")
        logger.info(f"Training on device: {self.device}")

        for epoch in range(self.config.num_epochs):
            # print(epoch)
            epoch_metrics = self._train_epoch()

            # Update history
            for key, value in epoch_metrics.items():
                self.history[key].append(value)

            # Generate and save sample
            if (epoch + 1) % self.config.print_interval == 0:
                self._print_progress(epoch, epoch_metrics)
                self._save_sample(epoch)
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dict[str, float]: Dictionary containing average metrics for the epoch
        """
        total_metrics: Dict[str, float] = {
            'd_loss': 0.0,
            'g_loss': 0.0,
            'consistency_loss': 0.0,
            'diversity_loss': 0.0,
            'temporal_loss': 0.0
        }
        batch_count = 0

        for batch in self.train_loader:
            real_batch = batch[0]  # DataLoader returns tuple
            metrics = self.model.train_step(real_batch)

            for key in total_metrics:
                total_metrics[key] += metrics[key]
            batch_count += 1

        return {k: v / (batch_count + 1e-6) for k, v in total_metrics.items()}

    def _print_progress(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Print training progress.
        
        Args:
            epoch (int): Current epoch number
            metrics (Dict[str, float]): Dictionary of training metrics
        """
        logger.info(f"\nEpoch [{epoch + 1}/{self.config.num_epochs}]")
        logger.info(f"D Loss: {metrics['d_loss']:.4f}")
        logger.info(f"G Loss: {metrics['g_loss']:.4f}")
        logger.info(f"Consistency: {metrics['consistency_loss']:.10f}")
        logger.info(f"Diversity: {metrics['diversity_loss']:.10f}")
        logger.info(f"Temporal: {metrics['temporal_loss']:.10f}")

    def _save_sample(self, epoch: int) -> None:
        """Generate and save a sample sequence.
        
        Args:
            epoch (int): Current epoch number
        """
        with torch.no_grad():
            sample = self.generate_sequences(num_sequences=1)
            self.history['generated_samples'].append(sample[0].cpu().numpy())
            logger.info("\nGenerated sample (OHLCV):")
            logger.info(sample[0, :, :5].cpu().numpy())

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint and training history.
        
        Args:
            filename (str): Name of the checkpoint file
        """
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'generator_state': self.model.generator.state_dict(),
            'discriminator_state': self.model.discriminator.state_dict(),
            'g_optimizer': self.model.g_optimizer.state_dict(),
            'd_optimizer': self.model.d_optimizer.state_dict(),
            'history': self.history
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint and training history.
        
        Args:
            filename (str): Name of the checkpoint file
        """
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path)
        
        self.model.generator.load_state_dict(checkpoint['generator_state'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.model.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.model.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        self.history = checkpoint['history']
        logger.info(f"Loaded checkpoint: {checkpoint_path}")

    def plot_training_history(self) -> None:
        """Plot and save training history visualizations."""
        plt.figure(figsize=(15, 10))

        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(self.history['d_loss'], label='Discriminator')
        plt.plot(self.history['g_loss'], label='Generator')
        plt.title('Losses')
        plt.legend()

        # Plot consistency loss
        plt.subplot(2, 2, 2)
        plt.plot(self.history['consistency_loss'])
        plt.title('Consistency Loss')

        # Plot diversity loss
        plt.subplot(2, 2, 3)
        plt.plot(self.history['diversity_loss'])
        plt.title('Diversity Loss')

        # Plot temporal loss
        plt.subplot(2, 2, 4)
        plt.plot(self.history['temporal_loss'])
        plt.title('Temporal Loss')

        plt.tight_layout()
        save_path = self.plots_dir / 'training_history.png'
        plt.savefig(save_path)
        logger.info(f"Saved training history plot: {save_path}")
        plt.close()
