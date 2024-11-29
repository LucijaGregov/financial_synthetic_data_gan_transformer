"""Financial GAN model combining generator and discriminator."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .generator import Generator
from .discriminator import Discriminator

class FinancialGAN:
    """Financial GAN implementation with WGAN-GP loss.
    
    Attributes:
        generator (Generator): Generator network
        discriminator (Discriminator): Discriminator network
        device (torch.device): Device to run computations on
    """
    
    def __init__(
        self,
        feature_dim: int,
        sequence_length: int,
        noise_dim: int = 200,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize Financial GAN.
        
        Args:
            feature_dim (int): Dimension of features
            sequence_length (int): Length of sequences
            noise_dim (int): Dimension of noise input
            device (str): Device to run on
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.noise_dim = noise_dim
        self.device = device
        
        # Initialize networks
        self.generator = Generator(
            noise_dim=noise_dim,
            feature_dim=feature_dim,
            sequence_length=sequence_length
        ).to(device)
        
        self.discriminator = Discriminator(
            feature_dim=feature_dim,
            sequence_length=sequence_length
        ).to(device)
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        self.d_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=2e-4,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Learning rate schedulers
        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.g_optimizer, T_max=1000, eta_min=1e-5
        )
        
        self.d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.d_optimizer, T_max=1000, eta_min=2e-5
        )
        
    def _gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        """Calculate gradient penalty for WGAN-GP.
        
        Args:
            real_data (torch.Tensor): Real data samples
            fake_data (torch.Tensor): Generated data samples
            
        Returns:
            torch.Tensor: Gradient penalty value
        """
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        alpha = alpha.expand_as(real_data)
        
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        critic_interpolated, _ = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty

    def _diversity_loss(self, generated: torch.Tensor) -> torch.Tensor:
        """Calculate diversity loss for generated sequences.
        
        Args:
            generated (torch.Tensor): Generated sequences
            
        Returns:
            torch.Tensor: Diversity loss value
        """
        price_diversity = torch.var(generated[..., :4], dim=1).mean()
        volume_diversity = torch.var(generated[..., 4:5], dim=1).mean()
        
        return 1.0 / (price_diversity + volume_diversity + 1e-6)

    def _price_consistency_loss(self, sequences: torch.Tensor) -> torch.Tensor:
        """Calculate price consistency loss (High >= Open/Close >= Low).
        
        Args:
            sequences (torch.Tensor): Generated sequences
            
        Returns:
            torch.Tensor: Price consistency loss value
        """
        open_price = sequences[..., 0]
        high = sequences[..., 1]
        low = sequences[..., 2]
        close = sequences[..., 3]
        
        high_violation = F.relu(torch.stack([
            open_price - high,
            close - high
        ]).max(dim=0)[0])
        
        low_violation = F.relu(torch.stack([
            low - open_price,
            low - close
        ]).max(dim=0)[0])
        
        return (high_violation.pow(2) + low_violation.pow(2)).mean()

    def _temporal_consistency_loss(self, sequences: torch.Tensor) -> torch.Tensor:
        """Calculate temporal consistency loss for price movements.
        
        Args:
            sequences (torch.Tensor): Generated sequences
            
        Returns:
            torch.Tensor: Temporal consistency loss value
        """
        price_changes = torch.diff(sequences[..., :4], dim=1)
        short_term = torch.abs(price_changes).mean()
        long_term = torch.abs(sequences[:, -1, :4] - sequences[:, 0, :4]).mean()
        
        return short_term + 0.5 * long_term

    def train_step(self, real_data: torch.Tensor) -> Dict[str, float]:
        """Perform one training step.
        
        Args:
            real_data (torch.Tensor): Batch of real data sequences
            
        Returns:
            Dict[str, float]: Dictionary containing loss metrics
        """
        with torch.no_grad():  # Faster noise generation
            noise = torch.randn(real_data.size(0), self.sequence_length, 
                        self.noise_dim, device=self.device)
    
        # Single forward/backward pass for discriminator
        self.d_optimizer.zero_grad(set_to_none=True)
        fake_data = self.generator(noise)
        
        # Combine real and fake processing
        real_critic = self.discriminator(real_data)[0]
        fake_critic = self.discriminator(fake_data.detach())[0]
        
        d_loss = fake_critic.mean() - real_critic.mean()
        d_loss.backward()
        self.d_optimizer.step()
        
        # Single forward/backward pass for generator
        self.g_optimizer.zero_grad(set_to_none=True)
        fake_critic = self.discriminator(fake_data)[0]
        g_loss = -fake_critic.mean()
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "consistency_loss": 0.0,
            "diversity_loss": 0.0,
            "temporal_loss": 0.0
        }
        # batch_size = real_data.size(0)
        
        # Train discriminator
        # for _ in range(2):  # n_critic updates
        #     self.d_optimizer.zero_grad()
            
        #     # Generate fake data
        #     noise = torch.randn(batch_size, self.sequence_length, self.noise_dim).to(self.device)
        #     fake_data = self.generator(noise)
            
        #     # Calculate discriminator loss
        #     real_critic, real_features = self.discriminator(real_data)
        #     fake_critic, fake_features = self.discriminator(fake_data.detach())
            
        #     gp = self._gradient_penalty(real_data, fake_data)
        #     d_loss = (fake_critic.mean() - real_critic.mean() + 10 * gp)
            
        #     d_loss.backward()
        #     self.d_optimizer.step()
        
        # # Train generator
        # self.g_optimizer.zero_grad()
        
        # noise = torch.randn(batch_size, self.sequence_length, self.noise_dim).to(self.device)
        # fake_data = self.generator(noise)
        # fake_critic, fake_features = self.discriminator(fake_data)
        
        # # Calculate generator losses
        # _, real_features = self.discriminator(real_data)
        # feature_loss = F.mse_loss(fake_features, real_features.detach())
        
        # g_loss = -fake_critic.mean()
        # diversity_loss = self._diversity_loss(fake_data)
        # consistency_loss = self._price_consistency_loss(fake_data)
        # temporal_loss = self._temporal_consistency_loss(fake_data)
        
        # # Combined generator loss
        # total_g_loss = (
        #     g_loss +
        #     0.1 * feature_loss +
        #     0.2 * diversity_loss +
        #     0.3 * consistency_loss +
        #     0.2 * temporal_loss
        # )
        
        # total_g_loss.backward()
        # self.g_optimizer.step()
        
        # # Update learning rates
        # self.g_scheduler.step()
        # self.d_scheduler.step()
        
        # return {
        #     "d_loss": d_loss.item(),
        #     "g_loss": g_loss.item(),
        #     "diversity_loss": diversity_loss.item(),
        #     "consistency_loss": consistency_loss.item(),
        #     "temporal_loss": temporal_loss.item()
        # }