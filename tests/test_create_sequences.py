import numpy as np

from src.data.preprocessing import create_sequences


def test_create_sequences():
    normalized_data = {
        'AAPL': {
            'norm_ohlc': np.random.rand(10, 4),
            'norm_volume': np.random.rand(10),
            'cyclic_dates': np.random.rand(10, 6)
        }
    }
    sequence_length = 5

    sequences, targets = create_sequences(normalized_data, sequence_length)

    assert 'AAPL' in sequences
    assert 'AAPL' in targets
    assert sequences['AAPL'].shape == (10 - sequence_length, sequence_length, 11)  # 4+1+6 features
    assert targets['AAPL'].shape == (10 - sequence_length, 11)
