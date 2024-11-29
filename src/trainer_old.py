"""
Trainer module for financial time series GAN.

This module implements the training loop and evaluation metrics for the GAN.
Includes support for multi-GPU training, early stopping, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, Optional
from config import Config, setup_logging
import matplotlib.pyplot as plt
from pathlib import Path
import os
from torch.cuda.amp import autocast, GradScaler


def create_attention_mask(batch_size: int, sequence_length: int, num_heads: int, device: torch.device) -> torch.Tensor:
    """Create a causal attention mask for transformer."""
    mask = torch.triu(
        torch.ones(sequence_length, sequence_length), 
        diagonal=1
    ).bool()
    mask = mask.unsqueeze(0).expand(batch_size * num_heads, -1, -1)
    return mask.to(device)


class GANTrainer:
    def __init__(self, config: Config, generator, discriminator, dataloader, g_optimizer, d_optimizer, num_stocks: int):
        """Initialize GAN trainer with all components."""
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.device = config.DEVICE
        self.num_stocks = num_stocks
        
        # Initialize optimizers
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        
        # Initialize training state
        self.last_d_loss = 1.0
        
        # Gradient scalers for mixed precision
        self.g_scaler = GradScaler()
        self.d_scaler = GradScaler()
        
        # Setup schedulers
        self.g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.g_optimizer,
            mode='min',
            factor=0.5,
            patience=config.PATIENCE // 2,
            verbose=True
        )
        
        self.d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.d_optimizer,
            mode='min',
            factor=0.5,
            patience=config.PATIENCE // 2,
            verbose=True
        )
        
        # Setup logging
        self.logger = setup_logging(config)
        self.logger.info(f"Using device: {self.device}")

    def gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        """
        Calculate gradient penalty for WGAN-GP with sequence data.
        
        Args:
            real_data: Real sequence data of shape [batch_size, seq_len, features]
            fake_data: Generated sequence data of shape [batch_size, seq_len, features]
            stock_ids: Stock IDs of shape [batch_size]
            
        Returns:
            Gradient penalty tensor
        """
        batch_size = real_data.size(0)
        
        # Generate random weights for interpolation
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        alpha = alpha.expand_as(real_data)
        
        # Create interpolated samples
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Get discriminator output for interpolated samples
        d_interpolated = self.discriminator(interpolated, stock_ids)
        
        # Calculate gradients
        grad_outputs = torch.ones_like(d_interpolated).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Reshape gradients properly
        gradients = gradients.reshape(batch_size, -1)
        
        # Calculate gradient penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty

    def train_epoch(self) -> Dict[str, float]:
        """Run a single training epoch with improved error handling."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_losses = []
        epoch_d_losses = []
        
        pbar = tqdm(self.dataloader, desc='Training', leave=True)
        
        for batch in pbar:
            try:
                real_data = batch['sequence'].to(self.device)
                padding_mask = batch['padding_mask'].to(self.device)
                stock_ids = batch['stock_id'].to(self.device)
                
                batch_size = real_data.size(0)
                
                # Train discriminator
                if self.last_d_loss > 0.1:
                    self.d_optimizer.zero_grad()
                    
                    # Process real data
                    real_data_noisy = real_data + torch.randn_like(real_data) * 0.1
                    real_labels = torch.ones(batch_size, 1).to(self.device) * 0.9
                    fake_labels = torch.zeros(batch_size, 1).to(self.device)
                    
                    d_real = self.discriminator(real_data_noisy, stock_ids)
                    d_real_loss = F.binary_cross_entropy(d_real, real_labels)
                    
                    # Generate fake data
                    noise = torch.randn(
                        batch_size, 
                        self.config.SEQUENCE_LENGTH, 
                        self.config.NOISE_DIM
                    ).to(self.device)
                    
                    attention_mask = create_attention_mask(
                        batch_size,
                        self.config.SEQUENCE_LENGTH,
                        self.config.N_HEADS,
                        self.device
                    )
                    
                    with torch.no_grad():
                        fake_data = self.generator(noise, stock_ids, attention_mask)
                    
                    d_fake = self.discriminator(fake_data.detach(), stock_ids)
                    d_fake_loss = F.binary_cross_entropy(d_fake, fake_labels)
                    
                    # Calculate gradient penalty with error handling
                    try:
                        gp = self.gradient_penalty(real_data, fake_data, stock_ids)
                        d_loss = d_real_loss + d_fake_loss + 10.0 * gp
                    except RuntimeError as e:
                        self.logger.warning(f"Gradient penalty calculation failed: {e}")
                        d_loss = d_real_loss + d_fake_loss
                    
                    d_loss.backward()
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(),
                        self.config.GRADIENT_CLIP
                    )
                    self.d_optimizer.step()
                    
                    epoch_d_losses.append(d_loss.item())
                    self.last_d_loss = d_loss.item()
                
                # Train generator
                n_critic = 1 if self.last_d_loss > 0.5 else 2
                for _ in range(n_critic):
                    self.g_optimizer.zero_grad()
                    
                    noise = torch.randn(
                        batch_size, 
                        self.config.SEQUENCE_LENGTH, 
                        self.config.NOISE_DIM
                    ).to(self.device)
                    
                    fake_data = self.generator(noise, stock_ids, attention_mask)
                    g_fake = self.discriminator(fake_data, stock_ids)
                    
                    # Generator losses
                    g_fake_loss = F.binary_cross_entropy(g_fake, real_labels)
                    temporal_loss = torch.mean(torch.abs(torch.diff(fake_data, dim=1)))
                    
                    try:
                        real_features = self.discriminator.get_features(real_data, stock_ids)
                        fake_features = self.discriminator.get_features(fake_data, stock_ids)
                        feature_loss = F.mse_loss(fake_features, real_features.detach())
                        g_loss = g_fake_loss + 0.1 * temporal_loss + 0.1 * feature_loss
                    except:
                        g_loss = g_fake_loss + 0.1 * temporal_loss
                    
                    g_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(),
                        self.config.GRADIENT_CLIP
                    )
                    self.g_optimizer.step()
                    
                    epoch_g_losses.append(g_loss.item())
                
                pbar.set_description(
                    f'G_loss: {g_loss.item():.4f} D_loss: {self.last_d_loss:.4f}'
                )
                
            except RuntimeError as e:
                self.logger.warning(f"Error in batch: {e}")
                continue
        
        return {
            'g_loss': np.mean(epoch_g_losses),
            'd_loss': np.mean(epoch_d_losses) if epoch_d_losses else self.last_d_loss
        }

    def train(self) -> Dict[str, list]:
        """Train the GAN model."""
        history = {'g_losses': [], 'd_losses': []}
        best_loss = float('inf')
        patience_counter = 0
        
        self.logger.info("Starting GAN training with configuration:")
        self.logger.info(f"Number of stocks: {self.num_stocks}")
        
        for epoch in range(self.config.N_EPOCHS):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.N_EPOCHS} started.")
            epoch_loss = self.train_epoch()
            
            # Update schedulers
            self.g_scheduler.step(epoch_loss['g_loss'])
            self.d_scheduler.step(epoch_loss['d_loss'])
            
            # Save losses
            if not np.isnan(epoch_loss['g_loss']):
                history['g_losses'].append(epoch_loss['g_loss'])
                history['d_losses'].append(epoch_loss['d_loss'])
                
                # Check for improvement
                if epoch_loss['g_loss'] < best_loss - self.config.MIN_DELTA:
                    best_loss = epoch_loss['g_loss']
                    self.save_model_checkpoint(epoch + 1)
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            # Save periodic checkpoints
            if (epoch + 1) % 5 == 0:
                self.save_model_checkpoint(epoch + 1)
                self.logger.info(f"Checkpoint saved at epoch {epoch + 1}")
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch + 1} - "
                f"Generator Loss: {epoch_loss['g_loss']:.4f}, "
                f"Discriminator Loss: {epoch_loss['d_loss']:.4f}"
            )
            
            # Early stopping check
            if patience_counter >= self.config.PATIENCE:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        return history

    def save_model_checkpoint(self, epoch: int, final: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint_name = "final_model_checkpoint.pth" if final else f"model_checkpoint_epoch_{epoch}.pth"
        checkpoint_path = self.config.CHECKPOINT_DIR / checkpoint_name
        
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_scheduler_state_dict': self.g_scheduler.state_dict(),
            'd_scheduler_state_dict': self.d_scheduler.state_dict()
        }, checkpoint_path)
        
        self.logger.info(f"Saved {'final' if final else 'epoch'} checkpoint at epoch {epoch}")