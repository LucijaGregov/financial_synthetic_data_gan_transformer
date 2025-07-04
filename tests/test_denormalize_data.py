import numpy as np
import pandas as pd

from src.data.preprocessing import denormalize_data


def test_denormalize_data():
    norm_ohlc = np.random.rand(10, 4)
    norm_volume = np.random.rand(10)

    volume_ma = pd.Series(np.random.rand(10))
    volume_std = pd.Series(np.random.rand(10))

    norm_params = {
        'ohlc_scaler': MockScaler(),
        'volume_scaler': MockScaler(),
        'volume_ma': volume_ma,
        'volume_std': volume_std
    }

    denorm_ohlc, denorm_volume = denormalize_data(norm_ohlc, norm_volume, norm_params)

    assert denorm_ohlc.shape == (10, 4)
    assert denorm_volume.shape == (10, 1)

class MockScaler:
    def inverse_transform(self, data):
        return data * 100
