from torch.utils.data import DataLoader
from pipeline.data_loader import StockDataset
from typing import Tuple
import torch

def create_data_loaders(dataset: StockDataset, config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns data loaders for training, validation, and test sets based on a temporal split.
    
    Args:
        dataset (StockDataset): The dataset to be split.
        config: Configuration object with batch size and split ratios.
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for train, validation, and test sets.
    """
    total_size = len(dataset)

    # Ensure minimum sizes for validation and test sets
    min_val_size = 1
    min_test_size = 1  # You can set a minimum size for the test set if desired

    # Calculate sizes for each split
    train_size = int(total_size * config.train_ratio)
    val_size = int(total_size * config.val_ratio)
    test_size = total_size - train_size - val_size

    # Adjust sizes to ensure no negative values and that minimum sizes are met
    if train_size < 0: train_size = 0
    if val_size < min_val_size: val_size = min_val_size
    if test_size < min_test_size: test_size = min_test_size

    # Adjust train and test sizes if the total exceeds the dataset
    while train_size + val_size + test_size > total_size:
        if val_size > min_val_size:
            val_size -= 1
        elif train_size > 0:
            train_size -= 1
        elif test_size > min_test_size:
            test_size -= 1

    # Validate the sizes
    if train_size + val_size + test_size > total_size:
        raise ValueError("Total split sizes exceed available dataset size.")

    # Debugging print statements
    print(f"Total size: {total_size}")
    print(f"Calculated sizes -> Train: {train_size}, Validation: {val_size}, Test: {test_size}")

    # Create temporal splits
    train_dataset = dataset[:train_size]            # First part for training
    val_dataset = dataset[train_size:train_size + val_size]  # Next part for validation
    test_dataset = dataset[train_size + val_size:]  # Last part for testing

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create DataLoaders without shuffling
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
