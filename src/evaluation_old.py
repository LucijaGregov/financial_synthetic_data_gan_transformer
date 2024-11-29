"""
Evaluation module for financial time series GAN with proper normalization handling.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as ticker
from typing import Dict, Optional, Tuple
from scipy import stats
from config import Config

def load_model_checkpoint(model, checkpoint_path: str, device: torch.device):
    """Load a model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['generator_state_dict'])
    else:
        model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()
    return model

def plot_training_history(
    history: Dict[str, list],
    config: Config,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history and metrics.

    Args:
        history (Dict[str, list]): Dictionary containing training history.
        config (Config): Configuration object.
        save_path (Optional[str], optional): Path to save the plot. Defaults to None.
    """
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['g_losses'], label='Generator', alpha=0.7)
    plt.plot(history['d_losses'], label='Discriminator', alpha=0.7)
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss ratios
    plt.subplot(1, 2, 2)
    loss_ratios = np.array(history['g_losses']) / np.array(history['d_losses'])
    plt.plot(loss_ratios, label='G/D Ratio', color='green', alpha=0.7)
    plt.title('Generator/Discriminator Loss Ratio')
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_quality(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    config: Config
) -> Dict[str, float]:
    """Evaluate quality of generated data."""
    metrics = {}
    
    # Basic statistical metrics
    for column in real_data.columns:
        real_values = real_data[column].values
        synthetic_values = synthetic_data[column].values
        
        ks_statistic, _ = stats.ks_2samp(real_values, synthetic_values)
        metrics[f'ks_test_{column}'] = ks_statistic
        metrics[f'mae_{column}'] = np.mean(np.abs(real_values - synthetic_values))
        metrics[f'correlation_{column}'] = np.corrcoef(real_values, synthetic_values)[0, 1]
    
    # Return metrics
    real_returns = real_data.pct_change().dropna()
    synthetic_returns = synthetic_data.pct_change().dropna()
    
    # Volatility comparison
    real_vol = real_returns.std()
    synthetic_vol = synthetic_returns.std()
    metrics['volatility_diff'] = np.mean(np.abs(real_vol - synthetic_vol))
    
    # Higher moments
    for column in real_data.columns:
        real_skew = stats.skew(real_returns[column])
        synthetic_skew = stats.skew(synthetic_returns[column])
        metrics[f'skew_diff_{column}'] = abs(real_skew - synthetic_skew)
        
        real_kurt = stats.kurtosis(real_returns[column])
        synthetic_kurt = stats.kurtosis(synthetic_returns[column])
        metrics[f'kurt_diff_{column}'] = abs(real_kurt - synthetic_kurt)
    
    return metrics

def plot_results(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    stock_name: str,
    config: Config,
    future_days: int = 0,
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison between real and synthetic data with proper date alignment.
    """
    fig = plt.figure(figsize=(15, 12))
    
    # Ensure synthetic data uses the same date range as real data
    synthetic_data.index = real_data.index[:len(synthetic_data)]
    
    print("Debug info:")
    print(f"Real data shape: {real_data.shape}")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print(f"Real data date range: {real_data.index[0]} to {real_data.index[-1]}")
    print(f"Synthetic data date range: {synthetic_data.index[0]} to {synthetic_data.index[-1]}")
    
    for i, column in enumerate(real_data.columns, 1):
        plt.subplot(len(real_data.columns), 1, i)
        
        # Plot real data
        plt.plot(real_data.index, real_data[column], 
                label='Real', color='blue', alpha=0.7, linewidth=1)
        
        # Plot synthetic data
        plt.plot(synthetic_data.index, synthetic_data[column],
                label='Synthetic', color='orange', alpha=0.7, linewidth=1)
        
        plt.title(f'{column} - {stock_name}')
        plt.xlabel('Date')
        plt.ylabel(column)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format y-axis
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
        
        # Add vertical line to separate training/prediction if future_days > 0
        if future_days > 0:
            split_date = real_data.index[-future_days]
            plt.axvline(x=split_date, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_synthetic_data(
    generator,
    config: Config,
    stock_id: int,
    sequence_length: int,
    n_samples: int,
    normalization_params: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """Generate synthetic data with improved financial characteristics."""
    generator.eval()
    
    # Get normalization parameters
    median = np.array(normalization_params['median']).astype(np.float32)
    iqr = np.array(normalization_params['iqr']).astype(np.float32)
    
    # Calculate price ranges for better scaling
    price_range = iqr[:4]  # OHLC ranges
    volume_range = iqr[4]  # Volume range
    
    base_level = median[0]  # Use Open price as base level
    
    synthetic_sequences = []
    
    with torch.no_grad():
        num_batches = (n_samples + config.BATCH_SIZE - 1) // config.BATCH_SIZE
        
        for _ in range(num_batches):
            current_batch_size = min(config.BATCH_SIZE, n_samples - len(synthetic_sequences))
            if current_batch_size <= 0:
                break
            
            # Generate structured noise
            noise = torch.randn(
                current_batch_size, 
                sequence_length, 
                config.NOISE_DIM
            ).to(config.DEVICE) * 0.5  # Reduced noise
            
            # Add trend component to noise
            trend = torch.linspace(-1, 1, sequence_length).view(1, -1, 1).repeat(
                current_batch_size, 1, config.NOISE_DIM
            ).to(config.DEVICE) * 0.2
            
            noise = noise + trend
            
            stock_ids = torch.full(
                (current_batch_size,), 
                stock_id, 
                dtype=torch.long
            ).to(config.DEVICE)
            
            attention_mask = torch.triu(
                torch.ones(current_batch_size * config.N_HEADS, sequence_length, sequence_length),
                diagonal=1
            ).bool().to(config.DEVICE)
            
            # Generate data
            synthetic_batch = generator(noise, stock_ids, attention_mask)
            
            # Scale to match real data distribution
            synthetic_batch = synthetic_batch.cpu().numpy()
            synthetic_batch[:, :, :4] = synthetic_batch[:, :, :4] * price_range + base_level
            synthetic_batch[:, :, 4:] = synthetic_batch[:, :, 4:] * volume_range
            
            synthetic_sequences.append(synthetic_batch)
    
    # Combine sequences
    synthetic_data = np.concatenate(synthetic_sequences, axis=0)
    synthetic_data = synthetic_data.reshape(-1, synthetic_data.shape[-1])
    
    # Create DataFrame
    dates = pd.date_range(
        start=pd.Timestamp.now() - pd.Timedelta(days=len(synthetic_data)),
        periods=len(synthetic_data),
        freq='B'
    )
    
    synthetic_df = pd.DataFrame(
        synthetic_data,
        index=dates,
        columns=['Open', 'High', 'Low', 'Close', 'Volume']
    )
    
    return synthetic_df[:n_samples]

def evaluate_models(
    real_data: pd.DataFrame,
    config: Config,
    generator,
    stock_id: int,
    normalization_params: Dict[str, np.ndarray],
    checkpoint_path: str,
    save_dir: str = 'evaluation_results'
):
    """Evaluate the trained GAN models."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print("\nStarting evaluation:")
    print(f"Real data shape: {real_data.shape}")
    
    # Load model
    generator = load_model_checkpoint(generator, checkpoint_path, config.DEVICE)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(
        generator=generator,
        config=config,
        stock_id=stock_id,
        sequence_length=config.SEQUENCE_LENGTH,
        n_samples=len(real_data),
        normalization_params=normalization_params
    )
    
    print(f"Generated synthetic data shape: {synthetic_data.shape}")
    
    # Plot results
    plot_results(
        real_data=real_data,
        synthetic_data=synthetic_data,
        stock_name=f"Stock_{stock_id}",
        config=config,
        future_days=0,
        save_path=save_dir / 'real_vs_synthetic.png'
    )
    
    # Calculate metrics
    metrics = evaluate_quality(real_data, synthetic_data, config)
    
    return synthetic_data, metrics

def analyze_distributions(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    config: Config,
    save_path: Optional[str] = None
) -> None:
    """Analyze and plot distribution comparisons."""
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(real_data.columns, 1):
        plt.subplot(2, 3, i)
        sns.kdeplot(real_data[column], label='Real', color='blue')
        sns.kdeplot(synthetic_data[column], label='Synthetic', color='orange')
        plt.title(f'{column} Distribution')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()