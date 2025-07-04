import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

def split_time_series(data: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data temporally per ticker.
    
    Args:
        data: Input DataFrame with time series data
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
                  Note: test_ratio will be 1 - train_ratio - val_ratio

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for ticker in data['ticker'].unique():
        ticker_data = data[data['ticker'] == ticker].sort_values('date').reset_index(drop=True)

        n = len(ticker_data)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))

        train_data = pd.concat([train_data, ticker_data.iloc[:train_idx]])
        val_data = pd.concat([val_data, ticker_data.iloc[train_idx:val_idx]])
        test_data = pd.concat([test_data, ticker_data.iloc[val_idx:]])

    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    logger.info(f"Data split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return train_data, val_data, test_data
