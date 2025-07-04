import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

def plot_comparisons_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.Series,
    features: List[str],
    ticker: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 15)
) -> None:
    """
    Plot comparisons between real and synthetic data.
    
    Args:
        y_true: True values array
        y_pred: Predicted values array
        dates: Series of dates
        features: List of feature names
        ticker: Ticker symbol
        save_path: Optional path to save plot
        figsize: Figure size
    """
    if len(features) == 0:
        raise ValueError("No features provided for plotting")

    rows = (len(features) + 2) // 3
    fig, axs = plt.subplots(rows, 3, figsize=figsize)
    fig.suptitle(f"{ticker} - Real vs Synthetic Data", fontsize=16)

    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for idx, feature in enumerate(features):
        ax = axs[idx]
        if feature.lower() == 'volume':
            ax.set_yscale('log')

        ax.plot(dates, y_true[:, idx], label='Real', color='blue', alpha=0.7)
        ax.plot(dates, y_pred[:, idx], label='Synthetic', color='red', alpha=0.7, linestyle='--')
        ax.set_title(f"{feature.capitalize()}")
        ax.set_xlabel('Date')
        ax.set_ylabel(feature.capitalize())
        ax.legend()
        ax.grid(True)

        plt.setp(ax.get_xticklabels(), rotation=45)

    for idx in range(len(features), len(axs)):
        fig.delaxes(axs[idx])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        except OSError as e:
            logger.error(f"Failed to save the plot to {save_path}: {e}")
            raise

def plot_distributions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    features: List[str],
    ticker: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 15)
) -> None:
    """
    Plot distributions for real and synthetic data.

    Args:
        y_true: True values array
        y_pred: Predicted values array
        features: List of feature names
        ticker: Ticker symbol
        save_path: Optional path to save plot
        figsize: Figure size
    """
    if len(features) == 0:
        raise ValueError("No features provided for plotting")

    rows = (len(features) + 2) // 3
    fig, axs = plt.subplots(rows, 3, figsize=figsize)
    fig.suptitle(f"{ticker} - Real vs Synthetic Data Distributions", fontsize=16)

    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for idx, feature in enumerate(features):
        ax = axs[idx]
        ax.hist(y_true[:, idx], bins=30, alpha=0.6, label='Real', color='blue', density=True)
        ax.hist(y_pred[:, idx], bins=30, alpha=0.6, label='Synthetic', color='red', density=True)
        ax.set_title(f"{feature.capitalize()}")
        ax.set_xlabel(feature.capitalize())
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True)

    for idx in range(len(features), len(axs)):
        fig.delaxes(axs[idx])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        except OSError as e:
            logger.error(f"Failed to save the distribution plot to {save_path}: {e}")
            raise

def pca_tsne_visualization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ticker: str,
    save_dir: str,
    random_state: int = 42
) -> None:
    """
    Create PCA and t-SNE visualizations.

    Args:
        y_true: True values array
        y_pred: Predicted values array
        ticker: Ticker symbol
        save_dir: Directory to save visualizations
        random_state: Random seed
    """
    pca = PCA(n_components=2, random_state=random_state)
    try:
        true_pca = pca.fit_transform(y_true)
        pred_pca = pca.transform(y_pred)
    except ValueError as e:
        logger.error(f"Error performing PCA: {e}")
        raise

    plt.figure(figsize=(12, 6))
    plt.scatter(true_pca[:, 0], true_pca[:, 1], label='Real', alpha=0.5, color='blue')
    plt.scatter(pred_pca[:, 0], pred_pca[:, 1], label='Synthetic', alpha=0.5, color='red')
    plt.title(f'PCA: Real vs Synthetic Data for {ticker}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(f"{save_dir}/{ticker}_pca.png", bbox_inches='tight', dpi=300)
    except OSError as e:
        logger.error(f"Failed to save PCA plot for {ticker}: {e}")
        raise
    plt.close()

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=300,
        random_state=random_state
    )
    try:
        combined_data = np.vstack((y_true, y_pred))
        tsne_result = tsne.fit_transform(combined_data)
    except ValueError as e:
        logger.error(f"Error performing t-SNE: {e}")
        raise

    true_tsne = tsne_result[:y_true.shape[0]]
    pred_tsne = tsne_result[y_true.shape[0]:]

    plt.figure(figsize=(12, 6))
    plt.scatter(true_tsne[:, 0], true_tsne[:, 1], label='Real', alpha=0.5, color='blue')
    plt.scatter(pred_tsne[:, 0], pred_tsne[:, 1], label='Synthetic', alpha=0.5, color='red')
    plt.title(f't-SNE: Real vs Synthetic Data for {ticker}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(f"{save_dir}/{ticker}_tsne.png", bbox_inches='tight', dpi=300)
    except OSError as e:
        logger.error(f"Failed to save t-SNE plot for {ticker}: {e}")
        raise
    plt.close()


def plot_inference_results(
    scenarios: List[torch.Tensor], 
    risk_metrics: Dict,
    stress_results: Dict,
    ticker: str,
    last_real_prices: np.ndarray,
    output_dir: Path
) -> None:
    """Create visualizations for inference results."""
    import matplotlib.pyplot as plt
    
    # Convert scenarios to numpy for plotting
    scenario_prices = []
    for scenario in scenarios:
        close_prices = scenario[0, :, 3].cpu().numpy()  # Extract close prices
        scenario_prices.append(close_prices)
    scenario_prices = np.array(scenario_prices)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{ticker} - Future Scenario Analysis', fontsize=16)
    
    # 1. Scenario Fan Chart
    ax1 = axes[0, 0]
    n_days = scenario_prices.shape[1]
    days = np.arange(n_days)

    # Plot individual scenarios
    for i in range(min(100, len(scenarios))):  # Plot up to 100 scenarios
        ax1.plot(days, scenario_prices[i], alpha=0.1, color='blue', linewidth=0.5)

    # Plot statistics
    mean_scenario = np.mean(scenario_prices, axis=0)
    percentile_95 = np.percentile(scenario_prices, 95, axis=0)
    percentile_5 = np.percentile(scenario_prices, 5, axis=0)

    ax1.plot(days, mean_scenario, 'r-', linewidth=2, label='Mean')
    ax1.fill_between(days, percentile_5, percentile_95, alpha=0.3, color='blue', label='90% Confidence')
    ax1.axhline(y=last_real_prices[-1], color='black', linestyle='--', label='Last Real Price')
    ax1.set_title('30-Day Price Scenarios')
    ax1.set_xlabel('Days Forward')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Return Distribution
    ax2 = axes[0, 1]
    final_returns = (scenario_prices[:, -1] - last_real_prices[-1]) / last_real_prices[-1]
    ax2.hist(final_returns, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=risk_metrics['VaR'][0.95], color='red', linestyle='--', 
                linewidth=2, label=f"VaR 95%: {risk_metrics['VaR'][0.95]:.2%}")
    ax2.axvline(x=risk_metrics['CVaR'][0.95], color='darkred', linestyle='--', 
                linewidth=2, label=f"CVaR 95%: {risk_metrics['CVaR'][0.95]:.2%}")
    ax2.axvline(x=risk_metrics['expected_return'], color='blue', linestyle='-', 
                linewidth=2, label=f"Expected: {risk_metrics['expected_return']:.2%}")
    ax2.set_title('30-Day Return Distribution')
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Risk Metrics by Time Horizon
    ax3 = axes[1, 0]
    time_horizons = [5, 10, 20, 30]
    vars_95 = []
    cvars_95 = []

    for horizon in time_horizons:
        # Quick calculation for different horizons
        horizon_returns = (scenario_prices[:, min(horizon-1, n_days-1)] - last_real_prices[-1]) / last_real_prices[-1]
        sorted_returns = np.sort(horizon_returns)
        var_idx = int(0.05 * len(sorted_returns))
        vars_95.append(sorted_returns[var_idx])
        cvars_95.append(np.mean(sorted_returns[:var_idx]))

    x = np.arange(len(time_horizons))
    width = 0.35
    ax3.bar(x - width/2, vars_95, width, label='VaR 95%', color='orange')
    ax3.bar(x + width/2, cvars_95, width, label='CVaR 95%', color='red')
    ax3.set_xlabel('Time Horizon (days)')
    ax3.set_ylabel('Risk Metric')
    ax3.set_title('Risk Metrics by Time Horizon')
    ax3.set_xticks(x)
    ax3.set_xticklabels(time_horizons)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Stress Test Comparison
    ax4 = axes[1, 1]
    categories = ['Expected\nReturn', 'Volatility', 'VaR 95%', 'Worst Case']
    normal_values = [
        risk_metrics['expected_return'],
        risk_metrics['volatility'],
        risk_metrics['VaR'][0.95],
        risk_metrics['worst_case']
    ]
    stressed_values = [
        stress_results['impact_metrics']['stressed_mean_return'],
        risk_metrics['volatility'] + stress_results['impact_metrics']['volatility_change'],
        risk_metrics['VaR'][0.95] + stress_results['impact_metrics']['var_95_change'],
        risk_metrics['worst_case'] + stress_results['impact_metrics']['worst_case_change']
    ]

    x = np.arange(len(categories))
    width = 0.35
    ax4.bar(x - width/2, normal_values, width, label='Normal', color='green', alpha=0.7)
    ax4.bar(x + width/2, stressed_values, width, label='Stressed (20% crash)', color='red', alpha=0.7)
    ax4.set_ylabel('Return')
    ax4.set_title('Normal vs Stress Test Scenarios')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Format percentages
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    plt.tight_layout()
    save_path = output_dir / f"{ticker}_inference_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"   Saved inference visualization to {save_path}")
