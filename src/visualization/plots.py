"""Visualization utilities for financial sequences."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional
import pandas as pd

class SequenceVisualizer:
    """Visualization tools for financial sequences.
    
    Attributes:
        real_data (np.ndarray): Real financial sequences
        generated_data (np.ndarray): Generated financial sequences
    """

    def __init__(
        self,
        real_data: np.ndarray,
        generated_data: np.ndarray
    ):
        """Initialize visualizer.

        Args:
            real_data (np.ndarray): Real financial sequences
            generated_data (np.ndarray): Generated financial sequences
        """
        self.real_data = real_data
        self.generated_data = generated_data

    def plot_sequence_comparison(
        self,
        num_samples: int = 3,
        save_path: Optional[str] = None
    ):
        """Plot comparison of real and generated sequences.
        
        Args:
            num_samples (int): Number of samples to plot
            save_path (Optional[str]): Path to save plot
        """
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))

        for i in range(num_samples):
            # Plot real sequence
            self._plot_ohlc(self.real_data[i], axes[0, i], title=f'Real Sample {i+1}')

            # Plot generated sequence
            self._plot_ohlc(self.generated_data[i], axes[1, i], 
                           title=f'Generated Sample {i+1}')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def _plot_ohlc(self, sequence: np.ndarray, ax: plt.Axes, title: str):
        """Plot OHLC data.
        
        Args:
            sequence (np.ndarray): Single OHLC sequence
            ax (plt.Axes): Matplotlib axes
            title (str): Plot title
        """
        ax.plot(sequence[:, 3], label='Close', color='blue')
        ax.fill_between(range(len(sequence)), 
                       sequence[:, 1], sequence[:, 2],
                       alpha=0.3, color='blue')
        ax.set_title(title)
        ax.grid(True)

    def plot_return_distributions(
        self,
        save_path: Optional[str] = None
    ):
        """Plot return distributions comparison.

        Args:
            save_path (Optional[str]): Path to save plot
        """
        real_returns = np.diff(self.real_data[..., 3], axis=1).flatten()
        gen_returns = np.diff(self.generated_data[..., 3], axis=1).flatten()

        plt.figure(figsize=(10, 6))
        sns.kdeplot(real_returns, label='Real Returns', color='blue')
        sns.kdeplot(gen_returns, label='Generated Returns', color='red')
        plt.title('Return Distributions Comparison')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_training_history(
        self,
        history: dict,
        save_path: Optional[str] = None
    ):
        """Plot training history.

        Args:
            history (dict): Training history dictionary
            save_path (Optional[str]): Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot losses
        axes[0, 0].plot(history['d_loss'], label='Discriminator')
        axes[0, 0].plot(history['g_loss'], label='Generator')
        axes[0, 0].set_title('Adversarial Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()

        # Plot consistency loss
        axes[0, 1].plot(history['consistency_loss'])
        axes[0, 1].set_title('Price Consistency Loss')
        axes[0, 1].set_xlabel('Epoch')

        # Plot diversity loss
        axes[1, 0].plot(history['diversity_loss'])
        axes[1, 0].set_title('Diversity Loss')
        axes[1, 0].set_xlabel('Epoch')

        # Plot temporal loss
        axes[1, 1].plot(history['temporal_loss'])
        axes[1, 1].set_title('Temporal Consistency Loss')
        axes[1, 1].set_xlabel('Epoch')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
