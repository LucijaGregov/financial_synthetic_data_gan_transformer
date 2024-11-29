"""Main script for training and evaluating Financial GAN."""

import torch
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from src.data.data_loading import (
    get_sp500_tickers, download_stock_data
)
from src.data.data_processing import (
    create_sequences,
    normalize_sequences,
    denormalize_sequences
)
from src.training.trainer import Trainer
from src.evaluation.metrics import FinancialMetrics
from src.visualization.plots import SequenceVisualizer
from config import Config, setup_logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils.paths import DIRS

import cProfile
import pstats
from pstats import SortKey
import time
from pathlib import Path
import torch
import numpy as np

# Now you can use DIRS throughout your code
print("Project directories:")
for name, path in DIRS.items():
    print(f"- {name}: {path}")

setup_logging()
logger = logging.getLogger(__name__)

# def prepare_config(stock_data, sequences, normalized_data):
#     """Prepare configuration based on actual data dimensions.
    
#     Args:
#         stock_data (Dict[str, pd.DataFrame]): Dictionary of stock DataFrames
#             Example: {'AAPL': df1, 'MMM': df2}
#         sequences (np.ndarray): Created sequences before normalization
#             Shape: (6, 5, 9) in your case where:
#             - 6 sequences
#             - 5 days each
#             - 9 features (5 OHLCV + 2 stocks + 2 day encoding)
#         normalized_data (np.ndarray): Normalized sequences
#             Same shape as sequences but normalized values
    
#     Returns:
#         Config: Configuration object with dimensions matching your data
#     """
#     # Create data_info dictionary from your actual data
#     data_info = {
#         'total_sequences': len(normalized_data),  # 6 in your case
#         'feature_dim': normalized_data.shape[2],  # 9 in your case
#         'ohlcv_dim': 5,  # Always 5 (Open, High, Low, Close, Volume)
#         'ticker_dim': len(stock_data),  # 2 in your case (AAPL, MMM)
#         'cyclical_day_dim': 2  # Always 2 (sin and cos encoding)
#     }

#     print("Data dimensions:")
#     print(f"Number of stocks: {len(stock_data)}")  # 2
#     print(f"Number of sequences: {len(sequences)}")  # 6
#     print(f"Sequence shape: {sequences.shape}")  # (6, 5, 9)
#     print(f"Features breakdown:")
#     print(f"- OHLCV: {data_info['ohlcv_dim']}")  # 5
#     print(f"- Tickers: {data_info['ticker_dim']}")  # 2
#     print(f"- Day encoding: {data_info['cyclical_day_dim']}")  # 2

#     return Config(data_info=data_info)

def main():
    config = Config()
    # 1. Data Loading and Processing
    print("\n=== Loading Data ===")
    tickers = get_sp500_tickers(config)

    # Setup profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Time data loading
    data_start = time.time()
    # tickers = ['AAPL', 'MMM']
    stock_data = download_stock_data(tickers, config)
    # stock_data = {key: stock_data[key].head(7) for key in tickers}
    # print(stock_data)
    logger.info(f"Data loading time: {time.time() - data_start:.2f}s")

    # Time sequence creation
    seq_start = time.time()
    sequences, ticker_mapping = create_sequences(stock_data, test_env=False)
    logger.info(f"Sequence creation time: {time.time() - seq_start:.2f}s")

    # Time normalization
    norm_start = time.time()
    normalized_data, feature_info = normalize_sequences(sequences)
    logger.info(f"Normalization time: {time.time() - norm_start:.2f}s")

    # Update config with actual dimensions
    config.feature_dim = normalized_data.shape[2]
    config.ticker_dim = len(ticker_mapping)
    config.batch_size = min(config.batch_size, len(normalized_data))

    # Time model training
    train_start = time.time()
    trainer = Trainer(
        normalized_data=normalized_data,
        config=config
    )

    trainer.quick_profile(num_batches=5)

    response = input("Continue with full training? (y/n): ")
    if response.lower() != 'y':
        print("Exiting to adjust parameters...")
        exit()

    trainer.train()
    logger.info(f"Training time: {time.time() - train_start:.2f}s")

    # Stop profiling and save results
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.TIME)
    stats.dump_stats('profile_results.prof')

    # Print profiling summary
    logger.info("\nTop 20 time-consuming operations:")
    stats.strip_dirs()
    stats.sort_stats(SortKey.TIME)
    stats.print_stats(20)

    # This will print progress similar to your original code:
    # Epoch [300/2000], D Loss: X.XXX, G Loss: X.XXX, Consistency: X.XXX, Volatility: X.XXX
    trainer.train()

    # 4. Generate and evaluate samples
    print("\n=== Generating Samples ===")
    num_samples = 10
    generated_sequences = trainer.generate_sequences(
        num_sequences=num_samples,
        temperature=0.8
    ).cpu().detach().numpy()

    # Denormalize both real and generated sequences
    real_sequences = normalized_data[:num_samples]  # Take same number of real sequences
    
    denorm_real = denormalize_sequences(real_sequences, feature_info)
    denorm_generated = denormalize_sequences(generated_sequences, feature_info)
    
    # 5. Evaluate and compare
    print("\n=== Evaluation ===")
    metrics = FinancialMetrics(denorm_real, denorm_generated)
    results = metrics.calculate_all_metrics()
    
    print("\nEvaluation Results:")
    print("------------------")
    for metric_name, value in results.items():
        print(f"{metric_name}:")
        if isinstance(value, dict):
            for k, v in value.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"  {value:.4f}")
    
    # 6. Visualize comparisons
    print("\n=== Visualizing Results ===")
    visualizer = SequenceVisualizer(denorm_real, denorm_generated)
    
    # Plot sequence comparisons
    visualizer.plot_sequence_comparison(
        num_samples=3,
        save_path='results/sequence_comparison.png'
    )
    
    # Plot return distributions
    visualizer.plot_return_distributions(
        save_path='results/return_distributions.png'
    )
    
    # Plot training history
    visualizer.plot_training_history(
        trainer.history,
        save_path='results/training_history.png'
    )
    
    # 7. Save detailed results
    print("\n=== Saving Results ===")
    save_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('results') / save_time
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(results_dir / 'metrics.txt', 'w') as f:
        f.write("Evaluation Results:\n")
        f.write("------------------\n")
        for metric_name, value in results.items():
            f.write(f"{metric_name}:\n")
            if isinstance(value, dict):
                for k, v in value.items():
                    f.write(f"  {k}: {v:.4f}\n")
            else:
                f.write(f"  {value:.4f}\n")
    
    # Save sample sequences
    np.save(results_dir / 'real_sequences.npy', denorm_real)
    np.save(results_dir / 'generated_sequences.npy', denorm_generated)
    
    print(f"\nResults saved to: {results_dir}")

if __name__ == "__main__":
    main()

"""Main script for Financial GAN training."""

import logging
import cProfile
import pstats
from pstats import SortKey
import time
from pathlib import Path
from typing import Dict, Any
from config import Config
from data.data_loading import download_stock_data
from data.data_processing import create_sequences, normalize_sequences
from training.trainer import Trainer

def main() -> None:
    """
    Main training function with profiling.
    
    Profiles and times:
    - Data loading
    - Sequence creation
    - Model training
    - Overall execution
    
    Saves profiling results to profile_results.prof
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Setup profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Initialize base configuration
    config = Config()

    # Time data loading
    data_start = time.time()
    # tickers = ['AAPL', 'MMM']
    stock_data = download_stock_data(tickers, config)
    stock_data = {key: stock_data[key].head(7) for key in tickers}
    print(stock_data)
    logger.info(f"Data loading time: {time.time() - data_start:.2f}s")

    # Time sequence creation
    seq_start = time.time()
    sequences, ticker_mapping = create_sequences(stock_data, test_env=True)
    logger.info(f"Sequence creation time: {time.time() - seq_start:.2f}s")

    # Time normalization
    norm_start = time.time()
    normalized_data, feature_info = normalize_sequences(sequences)
    logger.info(f"Normalization time: {time.time() - norm_start:.2f}s")

    # Update config with actual dimensions
    config.feature_dim = normalized_data.shape[2]
    config.ticker_dim = len(ticker_mapping)
    config.batch_size = min(config.batch_size, len(normalized_data))

    # Time model training
    train_start = time.time()
    trainer = Trainer(
        normalized_data=normalized_data,
        config=config
    )
    trainer.train()
    logger.info(f"Training time: {time.time() - train_start:.2f}s")

    # Stop profiling and save results
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.TIME)
    stats.dump_stats('profile_results.prof')

    # Print profiling summary
    logger.info("\nTop 20 time-consuming operations:")
    stats.strip_dirs()
    stats.sort_stats(SortKey.TIME)
    stats.print_stats(20)

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    logging.info(f"Total execution time: {total_time:.2f}s")
