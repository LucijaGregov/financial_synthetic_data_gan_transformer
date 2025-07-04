import os
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from src.data.preprocessing import create_sequences, normalize_data
from src.training.early_stopping import EarlyStopping
from src.training.utils import get_lr_scheduler, warmup_learning_rate
from src.utils.checkpoint import save_checkpoint
from src.models.model_factory import create_models

logger = logging.getLogger(__name__)


def setup_distributed() -> Tuple[torch.device, int, int]:
    """Setup for either CPU or multi-GPU training on a single machine or EC2 instance.
    Hence I am only using local_rank, without global rank."""
    if 'RANK' in os.environ:  # If distributed training
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        # Initialize process group
        dist.init_process_group(
            backend='gloo' if not torch.cuda.is_available() else 'nccl',
            rank=local_rank,
            world_size=world_size
        )

        # Choose device based on availability
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(local_rank)
        else:
            device = torch.device('cpu')
    else:  # Single process training
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        world_size = 1

    return device, local_rank, world_size


def compute_volume_weights(data: pd.DataFrame, sequences_dict: Dict, tickers: List[str]) -> np.ndarray:
    """
    Compute volume-aware weights for each training sample
    
    Args:
        data: Original data DataFrame
        sequences_dict: Dictionary of sequences per ticker
        tickers: List of ticker symbols
        
    Returns:
        Array of volume weights for each sample
    """
    # Calculate mean volume per ticker
    ticker_mean_volumes = {}
    for ticker in tickers:
        ticker_data = data[data['ticker'] == ticker]
        ticker_mean_volumes[ticker] = ticker_data['volume'].mean()

    # Create weights for each sequence
    weights = []
    for ticker in tickers:
        num_sequences = len(sequences_dict[ticker])
        mean_vol = ticker_mean_volumes[ticker]

        weight = np.log(mean_vol + 1e-8)
        weights.extend([weight] * num_sequences)
    
    weights = np.array(weights)
    weights = weights / np.mean(weights)

    return weights


def train_gan(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    sequence_length: int,
    epochs: int,
    batch_size: int,
    checkpoint_dir: str,
    patience: int,
    config: Dict[str, Any],
    input_size: int,
    output_size: int,
    ticker_to_idx
) -> Tuple[nn.Module, torch.device]:
    """
    Train the GAN model with DDP.
    
    Args:
        train_data: Training DataFrame with financial data
        validation_data: Validation DataFrame with financial data  
        sequence_length: Length of input sequences
        epochs: Number of training epochs
        batch_size: Size of training batches
        checkpoint_dir: Directory for saving model checkpoints
        patience: Number of epochs to wait before early stopping
        config: Configuration dictionary with model and training parameters
        input_size: Number of input features
        output_size: Number of output features
        
    Returns:
        Tuple of (trained_generator, device)
    """
    device, local_rank, world_size = setup_distributed()
    torch.autograd.set_detect_anomaly(True)

    if local_rank == 0:
        logger.info(f"Training on device: {device}, local_rank: {local_rank}, world_size: {world_size}")

    # Data preparation
    tickers = train_data['ticker'].unique()
    normalized_train = normalize_data(train_data, tickers, config['data_params']['cyclic_features'])
    normalized_val = normalize_data(
        validation_data, tickers, config['data_params']['cyclic_features']
    )

    train_sequences, train_targets = create_sequences(normalized_train, sequence_length)
    val_sequences, val_targets = create_sequences(normalized_val, sequence_length)

    X_train, y_train, ticker_indices_train = [], [], []
    for ticker in train_sequences:
        idx = ticker_to_idx[ticker]
        X_train.extend(train_sequences[ticker])
        y_train.extend(train_targets[ticker])
        ticker_indices_train.extend([idx] * len(train_sequences[ticker]))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    ticker_indices_train = np.array(ticker_indices_train)

    X_val, y_val, ticker_indices_val = [], [], []
    for ticker in val_sequences:
        idx = ticker_to_idx[ticker]  
        X_val.extend(val_sequences[ticker])
        y_val.extend(val_targets[ticker])
        ticker_indices_val.extend([idx] * len(val_sequences[ticker]))

    X_val = np.array(X_val)
    y_val = np.array(y_val)
    ticker_indices_val = np.array(ticker_indices_val)

    volume_weights_train = compute_volume_weights(train_data, train_sequences, tickers)
    volume_weights_val = compute_volume_weights(validation_data, val_sequences, tickers)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(volume_weights_train, dtype=torch.float32),
        torch.tensor(ticker_indices_train, dtype=torch.long)  
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(volume_weights_val, dtype=torch.float32),
        torch.tensor(ticker_indices_val, dtype=torch.long)  
    )

    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    unique_tickers = sorted(train_data['ticker'].unique())
    num_tickers = len(unique_tickers)

    generator, discriminator = create_models(
        input_size=input_size,
        output_size=output_size,
        config=config,
        device=device,
        num_tickers=num_tickers,
        embedding_dim=config['model_params']['ticker_embedding_dim']
    )

    if world_size > 1:
        if device.type == "cuda":
            generator = DDP(generator, device_ids=[local_rank], output_device=local_rank)
            discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank)
        else:
            generator = DDP(generator)
            discriminator = DDP(discriminator)

    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config['training_params']['g_learning_rate']
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config['training_params']['d_learning_rate']
    )

    g_scheduler = get_lr_scheduler(g_optimizer, config)
    d_scheduler = get_lr_scheduler(d_optimizer, config)

    adversarial_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    warmup_epochs = config['training_params'].get('warmup_epochs', 5)
    early_stopping = EarlyStopping(patience=patience)

    best_generator_state = None
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        if epoch < warmup_epochs:
            g_lr = warmup_learning_rate(
                epoch,
                warmup_epochs,
                config['training_params']['g_learning_rate']
            )
            d_lr = warmup_learning_rate(
                epoch,
                warmup_epochs,
                config['training_params']['d_learning_rate']
            )
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = g_lr
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = d_lr

        generator.train()
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        for batch_X, batch_y, batch_vol_weights, batch_ticker_idx in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_vol_weights = batch_vol_weights.to(device)
            current_batch_size = batch_X.size(0)

            # Train Discriminator
            d_optimizer.zero_grad()
            real_labels = torch.ones(current_batch_size, 1).to(device)
            fake_labels = torch.zeros(current_batch_size, 1).to(device)

            real_outputs = discriminator(batch_y)
            d_loss_real = adversarial_loss(real_outputs, real_labels)

            fake_data = generator(batch_X, batch_ticker_idx)
            fake_outputs = discriminator(fake_data.detach())
            d_loss_fake = adversarial_loss(fake_outputs, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            fake_outputs = discriminator(fake_data)

            g_loss_adv = adversarial_loss(fake_outputs, real_labels)

            volume_feature_idx = 4  # Volume is the 5th feature (index 4)

            # Regular MSE for OHLC and cyclic features
            non_volume_features = torch.cat([
                fake_data[:, :volume_feature_idx],      # OHLC features (0-3)
                fake_data[:, volume_feature_idx+1:]     # Cyclic features (5+)
            ], dim=1)

            non_volume_targets = torch.cat([
                batch_y[:, :volume_feature_idx],
                batch_y[:, volume_feature_idx+1:]
            ], dim=1)

            g_loss_regular = mse_loss(non_volume_features, non_volume_targets)
            
            # Volume-aware loss for volume feature
            g_loss_volume = volume_aware_loss(
                fake_data[:, volume_feature_idx], 
                batch_y[:, volume_feature_idx],
                batch_vol_weights,
                device
            )

            # Combine losses
            g_loss_mse = g_loss_regular + g_loss_volume
            adv_weight = config['training_params']['adv_weight']
            g_loss = adv_weight * g_loss_adv + (1 - adv_weight) * g_loss_mse

            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1

        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches

        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_X, val_y, val_vol_weights, val_ticker_idx in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_vol_weights = val_vol_weights.to(device)

                fake_data = generator(val_X, val_ticker_idx)

                volume_feature_idx = 4

                # Regular MSE for non-volume features
                non_volume_fake = torch.cat([
                    fake_data[:, :volume_feature_idx],
                    fake_data[:, volume_feature_idx+1:]
                ], dim=1)

                non_volume_real = torch.cat([
                    val_y[:, :volume_feature_idx],
                    val_y[:, volume_feature_idx+1:]
                ], dim=1)
                regular_loss = mse_loss(non_volume_fake, non_volume_real)
                
                # Volume-aware loss
                vol_loss = volume_aware_loss(
                    fake_data[:, volume_feature_idx],
                    val_y[:, volume_feature_idx],
                    val_vol_weights,
                    device
                )
                
                val_loss += (regular_loss + vol_loss).item()

        val_loss /= len(val_loader)

        if epoch >= warmup_epochs:
            g_scheduler.step(val_loss)
            d_scheduler.step(val_loss)

        logger.info(f"Epoch {epoch + 1}: Generator LR: {g_optimizer.param_groups[0]['lr']:.6f}")
        logger.info(f"Epoch {epoch + 1}: Discriminator LR: {d_optimizer.param_groups[0]['lr']:.6f}")
        logger.info(f"Validation Loss: {val_loss:.6f}")

        if local_rank == 0:
            if (epoch + 1) % 10 == 0 or epoch < warmup_epochs:
                logger.info(f"\nEpoch {epoch + 1} / {epochs}")
                logger.info(f"Generator Loss: {avg_g_loss:.4f}")
                logger.info(f"Discriminator Loss: {avg_d_loss:.4f}")
                logger.info(f"Validation Loss: {val_loss:.4f}")
                logger.info(f"Generator LR: {g_optimizer.param_groups[0]['lr']:.6f}")
                logger.info(f"Discriminator LR: {d_optimizer.param_groups[0]['lr']:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_generator_state = generator.state_dict().copy()
                save_checkpoint({
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'g_optimizer_state_dict': g_optimizer.state_dict(),
                    'd_optimizer_state_dict': d_optimizer.state_dict(),
                    'val_loss': val_loss,
                }, True, checkpoint_dir)

            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    if best_generator_state is not None:
        generator.load_state_dict(best_generator_state)

    return generator, device


def volume_aware_loss(pred_volume, real_volume, volume_weights, device):
    """
    Apply volume-scale aware weighting to balance training across different volume regimes
    
    Args:
        pred_volume: Predicted volume values for current batch
        real_volume: Real volume values for current batch  
        volume_weights: Pre-computed volume weights per sample
        device: Training device
    """
    base_loss = nn.MSELoss(reduction='none')(pred_volume, real_volume)
    weighted_loss = base_loss * volume_weights.to(device)

    return torch.mean(weighted_loss)
