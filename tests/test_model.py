"""Unit tests for Financial GAN model."""

import pytest
import torch
import numpy as np
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.models.gan import FinancialGAN
from src.config.gan_config import GANConfig

@pytest.fixture
def config():
    """Create test configuration."""
    return GANConfig()

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return torch.randn(32, 5, 501)  # batch_size, seq_length, features

class TestGenerator:
    def test_initialization(self, config):
        """Test generator initialization."""
        generator = Generator(
            noise_dim=config.noise_dim,
            feature_dim=config.feature_dim,
            sequence_length=config.sequence_length
        )
        assert isinstance(generator, Generator)
    
    def test_forward_pass(self, config):
        """Test generator forward pass."""
        generator = Generator(
            noise_dim=config.noise_dim,
            feature_dim=config.feature_dim,
            sequence_length=config.sequence_length
        )
        
        noise = torch.randn(32, config.sequence_length, config.noise_dim)
        output = generator(noise)
        
        assert output.shape == (32, config.sequence_length, config.feature_dim)
    
    def test_output_ranges(self, config):
        """Test generator output ranges."""
        generator = Generator(
            noise_dim=config.noise_dim,
            feature_dim=config.feature_dim,
            sequence_length=config.sequence_length
        )
        
        noise = torch.randn(32, config.sequence_length, config.noise_dim)
        output = generator(noise)
        
        # Check OHLC ranges
        assert torch.all(output[..., :4] >= -3.5)
        assert torch.all(output[..., :4] <= 3.5)
        
        # Check volume range
        assert torch.all(output[..., 4:5] >= -10)
        assert torch.all(output[..., 4:5] <= 10)
        
        # Check ticker one-hot encoding
        assert torch.all(output[..., 5:499] >= 0)
        assert torch.all(output[..., 5:499] <= 1)

class TestDiscriminator:
    def test_initialization(self, config):
        """Test discriminator initialization."""
        discriminator = Discriminator(
            feature_dim=config.feature_dim,
            sequence_length=config.sequence_length
        )
        assert isinstance(discriminator, Discriminator)
    
    def test_forward_pass(self, config, sample_data):
        """Test discriminator forward pass."""
        discriminator = Discriminator(
            feature_dim=config.feature_dim,
            sequence_length=config.sequence_length
        )
        
        critic_value, features = discriminator(sample_data)
        
        assert critic_value.shape[0] == sample_data.shape[0]
        assert len(features.shape) == 2

class TestFinancialGAN:
    def test_training_step(self, config, sample_data):
        """Test GAN training step."""
        model = FinancialGAN(
            feature_dim=config.feature_dim,
            sequence_length=config.sequence_length,
            noise_dim=config.noise_dim
        )
        
        metrics = model.train_step(sample_data)
        
        required_metrics = ['d_loss', 'g_loss', 'diversity_loss', 
                          'consistency_loss', 'temporal_loss']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)