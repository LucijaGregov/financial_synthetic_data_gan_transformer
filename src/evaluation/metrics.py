"""Evaluation metrics for generated financial data"""

import logging
from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import spearmanr, entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)

def calculate_distribution_similarity(true_vals: np.ndarray, pred_vals: np.ndarray) -> Dict[str, float]:
    """
    Calculate distribution similarity metrics between true and predicted values.

    Args:
        true_vals: True values array
        pred_vals: Predicted values array

    Returns:
        Dictionary containing distribution similarity metrics
    """
    true_hist, _ = np.histogram(true_vals, bins=50, density=True)
    pred_hist, _ = np.histogram(pred_vals, bins=50, density=True)

    # Add small constant to avoid zero probabilities
    true_hist += 1e-10
    pred_hist += 1e-10

    kl_divergence = entropy(true_hist, pred_hist)
    js_divergence = jensenshannon(true_hist, pred_hist)
    wasserstein_dist = wasserstein_distance(true_vals, pred_vals)

    return {
        "kl_divergence": kl_divergence,
        "js_divergence": js_divergence,
        "wasserstein_distance": wasserstein_dist
    }

def evaluate_feature_metrics(true_vals: np.ndarray, pred_vals: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics for a single feature.

    Args:
        true_vals: True values array
        pred_vals: Predicted values array

    Returns:
        Dictionary containing metrics for the feature
    """
    mae = np.mean(np.abs(true_vals - pred_vals))
    mape = np.mean(np.abs((true_vals - pred_vals) / (true_vals + 1e-8))) * 100
    correlation, _ = spearmanr(true_vals, pred_vals)
    distribution_metrics = calculate_distribution_similarity(true_vals, pred_vals)

    return {
        "mae": mae,
        "mape": mape,
        "correlation": correlation,
        **distribution_metrics
    }

def evaluate_generated_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    features: List[str],
    ticker: str,
    cyclic_config: Dict[str, bool]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate generated data against real data using standard metrics.

    Args:
        y_true: True values array
        y_pred: Predicted values array
        features: List of feature names
        ticker: Ticker symbol
        cyclic_config: Dictionary configuring which cyclic features are used

    Returns:
        Dictionary containing evaluation metrics for each feature and cyclic component:
            Regular features:
                - mae: Mean Absolute Error
                - mape: Mean Absolute Percentage Error
                - correlation: Spearman correlation
                - kl_divergence: Kullback-Leibler divergence
                - js_divergence: Jensen-Shannon divergence
                - wasserstein_distance: Wasserstein distance
    """
    metrics = {}
    logger.info(f"\nEvaluation Metrics for {ticker}:")

    # Evaluate base features (OHLC + volume)
    for i, feature in enumerate(features):
        try:
            feature_metrics = evaluate_feature_metrics(
                y_true[:, i],
                y_pred[:, i]
            )
            metrics[feature] = feature_metrics

            logger.info(f"Feature: {feature}")
            logger.info(f"  MAE: {feature_metrics['mae']:.4f}")
            logger.info(f"  MAPE: {feature_metrics['mape']:.2f}%")
            logger.info(f"  Correlation: {feature_metrics['correlation']:.4f}")
            logger.info(f"  KL Divergence: {feature_metrics['kl_divergence']:.4f}")
            logger.info(f"  JS Divergence: {feature_metrics['js_divergence']:.4f}")
            logger.info(f"  Wasserstein Distance: {feature_metrics['wasserstein_distance']:.4f}\n")

        except Exception as e:
            logger.error(f"Error evaluating feature {feature}: {str(e)}")
            metrics[feature] = {
                "mae": float('nan'),
                "mape": float('nan'),
                "correlation": float('nan'),
                "kl_divergence": float('nan'),
                "js_divergence": float('nan'),
                "wasserstein_distance": float('nan')
            }

    return metrics
