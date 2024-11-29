"""Evaluation metrics for generated financial sequences."""

import numpy as np
import torch
from typing import Dict, Tuple, List
from scipy import stats

class FinancialMetrics:
    """Financial time series evaluation metrics.
    
    Attributes:
        real_data (np.ndarray): Real financial sequences
        generated_data (np.ndarray): Generated financial sequences
    """
    
    def __init__(
        self,
        real_data: np.ndarray,
        generated_data: np.ndarray
    ):
        """Initialize metrics calculator.
        
        Args:
            real_data (np.ndarray): Real financial sequences
            generated_data (np.ndarray): Generated financial sequences
        """
        self.real_data = real_data
        self.generated_data = generated_data
    
    def calculate_all_metrics(self) -> Dict[str, float]:
        """Calculate all evaluation metrics.
        
        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        return {
            'price_consistency': self.price_consistency(),
            'volatility_similarity': self.volatility_similarity(),
            'return_distribution': self.return_distribution_similarity(),
            'temporal_correlation': self.temporal_correlation(),
            'volume_correlation': self.volume_price_correlation()
        }
    
    def price_consistency(self) -> Dict[str, float]:
        """Check OHLC price consistency.
        
        Returns:
            Dict[str, float]: Price consistency metrics
        """
        def check_consistency(data):
            high = data[..., 1]
            low = data[..., 2]
            open_price = data[..., 0]
            close = data[..., 3]
            
            high_valid = np.all(high >= np.maximum(open_price, close))
            low_valid = np.all(low <= np.minimum(open_price, close))
            return (high_valid.mean() + low_valid.mean()) / 2
        
        real_consistent = check_consistency(self.real_data)
        gen_consistent = check_consistency(self.generated_data)
        
        return {
            'real': real_consistent,
            'generated': gen_consistent,
            'ratio': gen_consistent / real_consistent if real_consistent > 0 else 0
        }
    
    def volatility_similarity(self) -> Dict[str, float]:
        """Compare volatility patterns.
        
        Returns:
            Dict[str, float]: Volatility similarity metrics
        """
        def calculate_volatility(data):
            returns = np.diff(data[..., 3], axis=1)  # Using close prices
            return np.std(returns, axis=1)
        
        real_vol = calculate_volatility(self.real_data)
        gen_vol = calculate_volatility(self.generated_data)
        
        return {
            'real_mean': real_vol.mean(),
            'gen_mean': gen_vol.mean(),
            'similarity': 1 - np.abs(real_vol.mean() - gen_vol.mean()) / real_vol.mean()
        }
    
    def return_distribution_similarity(self) -> float:
        """Calculate similarity of return distributions.
        
        Returns:
            float: KS test statistic
        """
        real_returns = np.diff(self.real_data[..., 3], axis=1).flatten()
        gen_returns = np.diff(self.generated_data[..., 3], axis=1).flatten()
        
        ks_stat, _ = stats.ks_2samp(real_returns, gen_returns)
        return 1 - ks_stat  # Higher value means more similar
    
    def temporal_correlation(self) -> Dict[str, float]:
        """Calculate temporal correlation of price movements.
        
        Returns:
            Dict[str, float]: Temporal correlation metrics
        """
        def autocorr(data):
            returns = np.diff(data[..., 3], axis=1)
            return np.corrcoef(returns[:, :-1].flatten(), returns[:, 1:].flatten())[0, 1]
        
        real_corr = autocorr(self.real_data)
        gen_corr = autocorr(self.generated_data)
        
        return {
            'real': real_corr,
            'generated': gen_corr,
            'difference': abs(real_corr - gen_corr)
        }
    
    def volume_price_correlation(self) -> Dict[str, float]:
        """Calculate volume-price correlation.
        
        Returns:
            Dict[str, float]: Volume-price correlation metrics
        """
        def vol_price_corr(data):
            price_changes = np.abs(np.diff(data[..., 3], axis=1))
            volume_changes = np.diff(data[..., 4], axis=1)
            return np.corrcoef(price_changes.flatten(), volume_changes.flatten())[0, 1]
        
        real_corr = vol_price_corr(self.real_data)
        gen_corr = vol_price_corr(self.generated_data)
        
        return {
            'real': real_corr,
            'generated': gen_corr,
            'difference': abs(real_corr - gen_corr)
        }