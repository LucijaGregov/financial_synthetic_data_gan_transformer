"""Example script for training the Financial GAN."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loading import download_stock_data
from src.data.data_processing import create_sequences, normalize_sequences
from src.training.trainer import Trainer
import matplotlib.pyplot as plt
import yfinance as yf

def main():
    # Download data
    print("Downloading stock data...")
    sp500 = yf.Tickers("^GSPC")
    constituents = sp500.tickers_list
    stock_data = download_stock_data(constituents[:50])  # Start with 50 stocks
    
    # Create sequences
    print("\nCreating sequences...")
    sequences, ticker_mapping = create_sequences(stock_data)
    
    # Normalize data
    print("\nNormalizing data...")
    normalized_data, feature_info = normalize_sequences(sequences)
    
    # Initialize trainer
    trainer = Trainer(
        normalized_data=normalized_data,
        batch_size=64,
        num_epochs=2000,
        print_interval=100
    )
    
    # Train model
    trainer.train()
    
    # Generate and plot results
    print("\nGenerating final samples...")
    samples = trainer.generate_sequences(num_sequences=5)
    
    plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.subplot(2, 3, i+1)
        ohlc = samples[i, :, :4].cpu().numpy()
        plt.plot(ohlc[:, 3], label='Close')
        plt.fill_between(range(len(ohlc)), ohlc[:, 1], ohlc[:, 2], alpha=0.3)
        plt.title(f'Generated Sample {i+1}')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('final_samples.png')
    plt.show()

if __name__ == "__main__":
    main()