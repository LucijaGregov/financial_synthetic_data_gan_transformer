import numpy as np
import pandas as pd

from src.data.preprocessing import normalize_data
from src.utils.config import load_or_create_config
from pathlib import Path


def test_normalize_data():
    config_path = "config/config.json"

    config_path = Path(config_path).resolve()
    config = load_or_create_config(config_path)

    data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'open': np.random.rand(10) * 100,
        'close': np.random.rand(10) * 100,
        'low': np.random.rand(10) * 100,
        'high': np.random.rand(10) * 100,
        'volume': np.random.randint(100, 1000, size=10),
        'ticker': ['AAPL'] * 10
    })
    tickers = ['AAPL']

    normalized = normalize_data(data, tickers, config['data_params']['cyclic_features'])

    assert 'AAPL' in normalized
    assert 'norm_ohlc' in normalized['AAPL']
    assert 'norm_volume' in normalized['AAPL']
    assert len(normalized['AAPL']['norm_ohlc']) == 10
    assert len(normalized['AAPL']['norm_volume']) == 10
