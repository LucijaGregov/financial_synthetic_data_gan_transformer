"""
Financial GAN Inference Module.

Note: This module is currently experimental and under development.
Further testing and validation are required before production use.

This module provides methods to use a trained GAN model for various financial applications
including backtesting, risk management, scenario generation, and stress testing.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd


class FinancialGANInference:
    """
    Inference wrapper for using trained GAN model for multiple financial purposes.
    
    This class provides methods to generate synthetic financial data for:
    - Backtesting: Single-step predictions for trading strategy evaluation
    - Risk Management: VaR/CVaR calculations for portfolio risk assessment
    - Scenario Generation: Multiple possible future market paths
    - Stress Testing: Market behavior under adverse conditions
    
    Attributes:
        generator: Trained transformer generator model
        device: Torch device (CPU/GPU)
    """
    
    def __init__(self, generator: torch.nn.Module, device: torch.device, ticker: str, ticker_to_idx: Dict[str, int]) -> None:
        """
        Initialize the inference wrapper.
        
        Args:
            generator: Trained GAN generator model
            device: Torch device for computation
        """
        self.generator = generator
        self.generator.eval()
        self.device = device
        self.ticker_to_idx = ticker_to_idx
        self.ticker = ticker
    
    def predict_next_day(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Predict next trading day (single-step prediction).
        
        Used for backtesting trading strategies where you need to predict
        tomorrow's OHLCV values based on historical sequence.
        
        Args:
            sequence: Tensor of shape (batch_size, sequence_length, features)
                     containing normalized historical OHLCV data
        
        Returns:
            Tensor of shape (batch_size, features) containing next day's
            predicted OHLCV values (normalized)
            
        Example:
            >>> next_day = inference.predict_next_day(last_10_days)
            >>> # Use for trading signal generation
        """
        with torch.no_grad():
            return self.generator(sequence)
    
    def generate_scenarios(
        self,
        initial_sequence: torch.Tensor,
        n_days: int = 30,
        n_scenarios: int = 100,
        add_noise: bool = False,
        noise_scale: float = 0.01
    ) -> List[torch.Tensor]:
        """
        Generate multiple possible future market scenarios.
        
        A scenario is one possible path the market could take over the next n_days.
        This is used for:
        - Monte Carlo simulations
        - Option pricing
        - Portfolio optimization
        - Understanding range of possible outcomes
        
        Args:
            initial_sequence: Starting sequence tensor of shape (1, seq_length, features)
            n_days: Number of days to project into future
            n_scenarios: Number of different scenarios to generate
            add_noise: Whether to add small noise for scenario diversity
            noise_scale: Standard deviation of noise if add_noise=True
        
        Returns:
            List of tensors, each of shape (1, n_days, features) representing
            one possible future scenario
            
        Example:
            >>> # Generate 1000 possible 30-day futures for risk assessment
            >>> scenarios = inference.generate_scenarios(
            ...     current_market_state, 
            ...     n_days=30, 
            ...     n_scenarios=1000
            ... )
        """
        scenarios = []
        ticker_idx = self.ticker_to_idx[self.ticker]
        ticker_tensor = torch.tensor([ticker_idx], device=self.device)
        
        with torch.no_grad():
            for i in range(n_scenarios):
                seq = initial_sequence.clone()
                path = []
                
                # Use consistent but varied noise seed for each scenario
                scenario_noise_seed = torch.randn(1, self.generator.noise_dim).to(self.device)
                
                for day in range(n_days):
                    # Gradually decay the noise influence
                    noise = scenario_noise_seed * (1 - day / n_days) * 0.5
                    
                    if add_noise:
                        # Add small random walk noise
                        seq_noise = torch.randn_like(seq) * noise_scale
                        seq_input = seq + seq_noise
                    else:
                        seq_input = seq
                    
                    next_day = self.generator(seq_input, ticker_tensor, noise=noise)
                    path.append(next_day)
                    seq = torch.cat([seq[:, 1:], next_day.unsqueeze(1)], dim=1)
                
                scenarios.append(torch.stack(path, dim=1))
        
        return scenarios

    def calculate_risk_metrics(
        self,
        initial_sequence: torch.Tensor,
        n_days: int = 10, 
        n_scenarios: int = 1000,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Calculate risk management metrics (VaR and CVaR).
        
        Risk Management Metrics Explained:
        - VaR (Value at Risk): The maximum loss expected with a given confidence
          level. E.g., 95% VaR of -5% means "95% of the time, we won't lose 
          more than 5%"
        
        - CVaR (Conditional VaR): The average loss in the worst cases beyond VaR.
          Also called Expected Shortfall. E.g., if VaR is -5%, CVaR might be -7%,
          meaning "when things go bad (worst 5% of cases), we lose 7% on average"
        
        These metrics help:
        - Set position sizes
        - Determine stop losses
        - Allocate capital
        - Meet regulatory requirements
        
        Args:
            initial_sequence: Starting market state tensor
            n_days: Time horizon for risk calculation
            n_scenarios: Number of Monte Carlo scenarios
            confidence_levels: List of confidence levels for VaR/CVaR
        
        Returns:
            Dictionary containing:
            - 'VaR': Dict mapping confidence level to VaR value
            - 'CVaR': Dict mapping confidence level to CVaR value
            - 'expected_return': Mean return across all scenarios
            - 'volatility': Standard deviation of returns
            - 'worst_case': Minimum return observed
            - 'best_case': Maximum return observed
            
        Example:
            >>> risk = inference.calculate_risk_metrics(current_position, n_days=10)
            >>> print(f"10-day 95% VaR: {risk['VaR'][0.95]:.2%}")
            >>> # Output: "10-day 95% VaR: -4.23%"
        """
        scenarios = self.generate_scenarios(initial_sequence, n_days, n_scenarios, add_noise=True)
        
        # Calculate returns from close prices (index 3)
        returns = []
        for scenario in scenarios:
            start_price = initial_sequence[0, -1, 3]  # Last close price
            end_price = scenario[0, -1, 3]  # Final close price
            total_return = (end_price - start_price) / start_price
            returns.append(total_return.item())
        
        returns = np.array(returns)
        sorted_returns = np.sort(returns)
        
        var_dict = {}
        cvar_dict = {}
        
        for conf in confidence_levels:
            # VaR is the threshold at (1-confidence) percentile
            var_index = int((1 - conf) * len(sorted_returns))
            var_value = sorted_returns[var_index]
            var_dict[conf] = var_value
            
            # CVaR is the mean of returns worse than VaR
            cvar_value = np.mean(sorted_returns[:var_index])
            cvar_dict[conf] = cvar_value
        
        return {
            'VaR': var_dict,
            'CVaR': cvar_dict,
            'expected_return': np.mean(returns),
            'volatility': np.std(returns),
            'skewness': self._calculate_skewness(returns),
            'kurtosis': self._calculate_kurtosis(returns),
            'worst_case': np.min(returns),
            'best_case': np.max(returns),
            'percentiles': {
                1: np.percentile(returns, 1),
                5: np.percentile(returns, 5),
                25: np.percentile(returns, 25),
                50: np.percentile(returns, 50),
                75: np.percentile(returns, 75),
                95: np.percentile(returns, 95),
                99: np.percentile(returns, 99)
            }
        }
    
    def stress_test(
        self,
        initial_sequence: torch.Tensor,
        stress_type: str = 'market_crash',
        stress_magnitude: float = 0.2,
        n_days: int = 20,
        n_scenarios: int = 100
    ) -> Dict[str, Union[List[torch.Tensor], Dict[str, float]]]:
        """
        Perform stress testing by simulating adverse market conditions.
        
        Stress Testing Explained:
        Stress testing simulates how a portfolio/strategy would perform under
        extreme but plausible adverse conditions. Required by regulators and
        crucial for risk management.
        
        Types of stress tests:
        - Historical: Replay past crises (2008, COVID-19)
        - Hypothetical: Simulate potential future crises
        - Sensitivity: Test specific factor changes
        
        Args:
            initial_sequence: Current market state
            stress_type: Type of stress to apply:
                - 'market_crash': Sudden price drop
                - 'volatility_spike': Increased volatility
                - 'liquidity_crisis': Volume reduction
                - 'trend_reversal': Momentum shift
            stress_magnitude: Severity of stress (0.2 = 20%)
            n_days: Days to simulate after stress
            n_scenarios: Number of scenarios to generate
        
        Returns:
            Dictionary containing:
            - 'stressed_scenarios': List of scenarios under stress
            - 'normal_scenarios': List of scenarios without stress
            - 'impact_metrics': Comparison metrics
            
        Example:
            >>> # Test portfolio under 20% market crash
            >>> stress_results = inference.stress_test(
            ...     current_state,
            ...     stress_type='market_crash',
            ...     stress_magnitude=0.2
            ... )
            >>> print(f"Expected loss under stress: {stress_results['impact_metrics']['mean_return_change']:.2%}")
        """
        # Apply stress to the input sequence
        stressed_sequence = self._apply_stress(
            initial_sequence.clone(),
            stress_type, 
            stress_magnitude
        )
        
        # Generate scenarios from both normal and stressed conditions
        stressed_scenarios = self.generate_scenarios(
            stressed_sequence, n_days, n_scenarios, add_noise=True
        )
        
        normal_scenarios = self.generate_scenarios(
            initial_sequence, n_days, n_scenarios, add_noise=True
        )
        
        # Calculate impact metrics
        impact_metrics = self._calculate_stress_impact(
            normal_scenarios, 
            stressed_scenarios
        )
        
        return {
            'stressed_scenarios': stressed_scenarios,
            'normal_scenarios': normal_scenarios,
            'impact_metrics': impact_metrics,
            'stress_parameters': {
                'type': stress_type,
                'magnitude': stress_magnitude
            }
        }
    
    def _apply_stress(
        self, 
        sequence: torch.Tensor, 
        stress_type: str, 
        magnitude: float
    ) -> torch.Tensor:
        """Apply specific stress to market sequence."""
        if stress_type == 'market_crash':
            # Sudden drop in all prices
            sequence[:, :, :4] *= (1 - magnitude)  # OHLC columns
            
        elif stress_type == 'volatility_spike':
            # Increase variation in prices
            returns = sequence[:, 1:] - sequence[:, :-1]
            amplified_returns = returns * (1 + magnitude * 2)
            for i in range(1, sequence.shape[1]):
                sequence[:, i] = sequence[:, i-1] + amplified_returns[:, i-1]
                
        elif stress_type == 'liquidity_crisis':
            # Reduce volume significantly
            if sequence.shape[2] > 4:  # If volume column exists
                sequence[:, :, 4] *= (1 - magnitude * 0.7)

        elif stress_type == 'trend_reversal':
            # Reverse recent trend
            trend = sequence[:, -1] - sequence[:, 0]
            reversal = trend * magnitude
            for i in range(sequence.shape[1]):
                weight = i / sequence.shape[1]
                sequence[:, i] -= reversal * weight
        
        return sequence
    
    def _calculate_stress_impact(
        self, 
        normal_scenarios: List[torch.Tensor], 
        stressed_scenarios: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Calculate the impact of stress on returns."""
        normal_returns = []
        stressed_returns = []
        
        for normal, stressed in zip(normal_scenarios, stressed_scenarios):
            # Calculate total returns
            normal_ret = (normal[0, -1, 3] - normal[0, 0, 3]) / normal[0, 0, 3]
            stressed_ret = (stressed[0, -1, 3] - stressed[0, 0, 3]) / stressed[0, 0, 3]
            
            normal_returns.append(normal_ret.item())
            stressed_returns.append(stressed_ret.item())
        
        normal_returns = np.array(normal_returns)
        stressed_returns = np.array(stressed_returns)
        
        return {
            'mean_return_change': np.mean(stressed_returns) - np.mean(normal_returns),
            'volatility_change': np.std(stressed_returns) - np.std(normal_returns),
            'worst_case_change': np.min(stressed_returns) - np.min(normal_returns),
            'var_95_change': np.percentile(stressed_returns, 5) - np.percentile(normal_returns, 5),
            'stressed_mean_return': np.mean(stressed_returns),
            'normal_mean_return': np.mean(normal_returns)
        }
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns distribution."""
        mean = np.mean(returns)
        std = np.std(returns)
        return np.mean(((returns - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate excess kurtosis of returns distribution."""
        mean = np.mean(returns)
        std = np.std(returns)
        return np.mean(((returns - mean) / std) ** 4) - 3
