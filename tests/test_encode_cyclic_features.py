import numpy as np
import pandas as pd

from src.data.preprocessing import encode_dates
from src.utils.config import load_or_create_config
from pathlib import Path


def test_encode_cyclic_features():
    config_path = "config/config.json"

    config_path = Path(config_path).resolve()
    config = load_or_create_config(config_path)

    dates = pd.Series(pd.date_range(start='2023-01-01', periods=10, freq='D'))
    cyclic_features = encode_dates(dates, config['data_params']['cyclic_features'])

    assert cyclic_features.shape == (10, 6)  # 6 cyclic dates
    assert not np.any(np.isnan(cyclic_features))  # No NaN values
