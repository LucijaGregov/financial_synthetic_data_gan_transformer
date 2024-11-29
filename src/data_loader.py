# """
# Dataset module for financial time series GAN.

# This module handles data loading, preprocessing, and batching for training the GAN.
# Includes robust normalization and proper handling of missing values.

# Classes:
#     StockDataset: Custom PyTorch dataset for financial time series
# Functions:
#     prepare_data: Prepare and normalize stock data
#     create_dataloader: Create optimized DataLoader for training
# """

import os
import pandas as pd
import numpy as np
import logging
import yfinance as yf
from tqdm import tqdm
from config import Config
import requests
from io import StringIO
import certifi
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)

# class StockDataset(Dataset):
#     def __init__(self, aligned_data_dict, mask_dict, stock_to_id):
#         self.data = []
#         self.masks = []
#         self.stock_ids = []

#         for stock, stock_id in stock_to_id.items():
#             stock_data = aligned_data_dict[stock].to_numpy().astype(np.float32)
#             stock_mask = mask_dict[stock].to_numpy().astype(np.float32)

#             valid_data = stock_data[~np.isnan(stock_data).any(axis=1)]
#             valid_masks = stock_mask[~np.isnan(stock_mask).any(axis=1)]

#             self.data.append(valid_data)
#             self.masks.append(valid_masks)
#             self.stock_ids.extend([stock_id] * len(valid_data))

#         self.data = np.concatenate(self.data)
#         self.masks = np.concatenate(self.masks)
#         self.stock_ids = np.array(self.stock_ids, dtype=np.int64)  # Ensure stock IDs are integers

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         data = torch.tensor(self.data[idx])
#         mask = torch.tensor(self.masks[idx])
#         stock_id = torch.tensor(self.stock_ids[idx])
#         return data, mask, stock_id
    
    
# # import torch
# # from torch.utils.data import Dataset, DataLoader

# # class StockDataset(Dataset):
# #     def __init__(self, data_dict, mask_dict, stock_to_id):
# #         self.data = []
# #         self.masks = []
# #         self.stock_ids = []

# #         for stock, stock_id in stock_to_id.items():
# #             stock_data = data_dict[stock].to_numpy().astype(np.float32)
# #             stock_mask = mask_dict[stock].to_numpy().astype(np.float32)
# #             self.data.append(stock_data)
# #             self.masks.append(stock_mask)
# #             self.stock_ids.extend([stock_id] * len(stock_data))

# #         self.data = np.concatenate(self.data)
# #         self.masks = np.concatenate(self.masks)
# #         self.stock_ids = np.array(self.stock_ids, dtype=np.int64)

# #     def __len__(self):
# #         return len(self.data)

# #     def __getitem__(self, idx):
# #         data = torch.tensor(self.data[idx])
# #         mask = torch.tensor(self.masks[idx])
# #         stock_id = torch.tensor(self.stock_ids[idx])
# #         return data, mask, stock_id

# # # Create a mapping for stock names to IDs
# # stock_to_id = {stock: i for i, stock in enumerate(normalized_data_dict.keys())}

# # # Initialize dataset and dataloader
# # dataset = StockDataset(normalized_data_dict, mask_dict, stock_to_id)
# # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# # # Display a batch for inspection
# # for data, mask, stock_id in dataloader:
# #     print("Batch data:", data)
# #     print("Batch mask:", mask)
# #     print("Batch stock IDs:", stock_id)
# #     break


# def robust_normalize(data: np.ndarray) -> np.ndarray:
#     """
#     Normalize the input data using the median and IQR method.

#     Args:
#         data (np.ndarray): The data to normalize, expected to be 2D.

#     Returns:
#         np.ndarray: Normalized data.
#     """
#     median = np.nanmedian(data, axis=0)  # Ignore NaNs
#     q75, q25 = np.nanpercentile(data, [75, 25], axis=0)
#     iqr = q75 - q25
#     normalized_data = (data - median) / (iqr + 1e-8)  # Avoid division by zero
#     return normalized_data

# def prepare_data(stock_data, num_stocks=None, rows_per_stock=None):
#     """
#     Prepare stock data by normalizing and creating masks for missing values.
#     Also assigns a unique ID to each stock for embedding purposes.
    
#     Returns:
#         Tuple of dictionaries for aligned data, masks, normalization params, and stock IDs
#     """
#     # Select stocks if num_stocks is specified
#     if num_stocks is not None:
#         selected_stocks = np.random.choice(list(stock_data.keys()), num_stocks, replace=False)
#     else:
#         selected_stocks = list(stock_data.keys())
#     print("Selected stocks:", selected_stocks)

#     # Create a mapping of stock names to unique IDs
#     stock_to_id = {stock: idx for idx, stock in enumerate(selected_stocks)}

#     # Get the full date range from the stock data
#     min_date = min(df.index.min() for df in stock_data.values())
#     max_date = max(df.index.max() for df in stock_data.values())
#     all_dates = pd.date_range(start=min_date, end=max_date, freq='B')

#     # Initialize dictionaries
#     aligned_data_dict = {}
#     mask_dict = {}
#     normalization_params = {}

#     for stock in selected_stocks:
#         # Align stock data with the full date range
#         df = stock_data[stock]
#         df_aligned = df.reindex(all_dates)
#         mask = df_aligned.notna().astype(int)

#         # Normalize data
#         normalized_data = robust_normalize(df_aligned.values)
#         normalized_df = pd.DataFrame(normalized_data, index=all_dates, columns=df.columns)
#         normalized_df[mask == 0] = np.nan  # Keep NaNs where the original data is missing

#         # Now, select only the specified number of non-NaN rows
#         non_nan_rows = normalized_df.dropna().iloc[:rows_per_stock]

#         # Store only the selected rows in the final dictionaries
#         aligned_data_dict[stock] = non_nan_rows
#         mask_dict[stock] = mask.loc[non_nan_rows.index]
#         normalization_params[stock] = (
#             np.nanmedian(df_aligned.values, axis=0),
#             np.nanpercentile(df_aligned.values, 75, axis=0) - np.nanpercentile(df_aligned.values, 25, axis=0)
#         )

#     return aligned_data_dict, mask_dict, normalization_params, stock_to_id

# def create_dataloader(config: Config, aligned_data_dict: Dict[str, pd.DataFrame],
#                      mask_dict: Dict[str, np.ndarray], stock_to_id: Dict[str, int]) -> DataLoader:
#     """
#     Create optimized DataLoader for training.

#     Args:
#         config: Configuration object
#         aligned_data_dict: Dictionary of aligned stock DataFrames
#         mask_dict: Dictionary of masks for valid data points
#         stock_to_id: Dictionary mapping each stock to a unique ID

#     Returns:
#         DataLoader for training
#     """
#     dataset = StockDataset(aligned_data_dict, mask_dict, stock_to_id)

#     return DataLoader(
#         dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=False,
#         num_workers=config.NUM_WORKERS if config.DEVICE.type == 'cuda' else 0,
#         pin_memory=config.PIN_MEMORY if config.DEVICE.type == 'cuda' else False,
#         drop_last=True
#     )

# def get_sp500_tickers(config: Config) -> List[str]:
#     """
#     Fetches S&P 500 tickers from Wikipedia or loads them from a local file.

#     Args:
#         config: Config object containing settings such as file paths.

#     Returns:
#         List[str]: List of ticker symbols for S&P 500 companies.
#     """
#     file_path = config.DATA_DIR / 'sp500_tickers.txt'

#     if os.path.exists(file_path):
#         with open(file_path, 'r') as f:
#             tickers = f.read().splitlines()
#     else:
#         url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
#         response = requests.get(url, verify=certifi.where())
#         if response.status_code != 200:
#             raise Exception(f"Failed to fetch data: {response.status_code}")

#         table = pd.read_html(StringIO(response.text))
#         tickers = table[0]['Symbol'].tolist()

#         # Save the tickers to a file for future use
#         with open(file_path, 'w') as f:
#             f.write('\n'.join(tickers))

#     return tickers

# def download_stock_data(tickers: List[str], config: Config) -> Dict[str, pd.DataFrame]:
#     """
#     Downloads historical stock data for specified tickers or loads from file.

#     Args:
#         tickers: List of stock ticker symbols
#         config: Configuration object containing parameters

#     Returns:
#         Dictionary of DataFrames with stock data for each ticker
#     """
#     file_path = config.DATA_DIR / 'stock_data.pkl'
#     print(file_path)
#     if file_path.exists():
#         stock_data = pd.read_pickle(file_path)
#     else:
#         stock_data = {}
#         for ticker in tqdm(tickers, desc="Downloading stock data"):
#             try:
#                 df = yf.download(
#                     ticker,
#                     start=config.start_date,
#                     end=config.end_date,
#                     progress=False
#                 )
#                 if not df.empty and len(df) >= config.min_window:
#                     stock_data[ticker] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
#             except Exception as e:
#                 logger.warning(f"Failed to download data for {ticker}: {e}")

#         if not stock_data:
#             raise ValueError("No valid stock data downloaded")

#         os.makedirs(config.DATA_DIR, exist_ok=True)
#         pd.to_pickle(stock_data, file_path)

#     logger.info(f"Loaded data for {len(stock_data)} stocks")
#     return stock_data
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from config import Config

def create_dataloader(config: Config, 
                     aligned_data_dict: Dict[str, pd.DataFrame],
                     mask_dict: Dict[str, pd.DataFrame], 
                     stock_to_id: Dict[str, int]) -> DataLoader:
    """
    Create optimized DataLoader with proper batching and collation.
    
    Args:
        config: Configuration object
        aligned_data_dict: Dictionary of aligned stock DataFrames
        mask_dict: Dictionary of masks for valid data points
        stock_to_id: Dictionary mapping each stock to a unique ID
    
    Returns:
        DataLoader: Configured data loader for training
    """
    def collate_fn(batch):
        """Custom collate function to handle the dictionary-based samples"""
        return {
            'sequence': torch.stack([item['sequence'] for item in batch]),
            'padding_mask': torch.stack([item['padding_mask'] for item in batch]),
            'stock_id': torch.stack([item['stock_id'] for item in batch])
        }
    
    dataset = StockDataset(
        aligned_data_dict=aligned_data_dict,
        mask_dict=mask_dict,
        stock_to_id=stock_to_id,
        sequence_length=config.SEQUENCE_LENGTH
    )
    
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS if config.DEVICE.type == 'cuda' else 0,
        pin_memory=config.PIN_MEMORY if config.DEVICE.type == 'cuda' else False,
        collate_fn=collate_fn,
        drop_last=True
    )

class StockDataset(Dataset):
    """Dataset class for stock data with sequence handling"""
    def __init__(self, 
                 aligned_data_dict: Dict[str, pd.DataFrame],
                 mask_dict: Dict[str, pd.DataFrame],
                 stock_to_id: Dict[str, int],
                 sequence_length: int = 50):
        """
        Initialize the dataset.
        
        Args:
            aligned_data_dict: Dictionary of aligned stock DataFrames
            mask_dict: Dictionary of masks for valid data points
            stock_to_id: Dictionary mapping each stock to a unique ID
            sequence_length: Length of sequences to generate
        """
        self.sequence_length = sequence_length
        self.sequences = []
        self.padding_masks = []
        self.stock_ids = []
        
        for stock, stock_id in stock_to_id.items():
            stock_data = aligned_data_dict[stock].to_numpy().astype(np.float32)
            stock_mask = mask_dict[stock].to_numpy().astype(np.float32)
            
            # Create sequences with sliding window of 1
            for i in range(len(stock_data) - sequence_length + 1):
                seq = stock_data[i:i + sequence_length]
                mask = stock_mask[i:i + sequence_length]
                
                # Only include sequences with no NaN values
                if not np.isnan(seq).any():
                    self.sequences.append(seq)
                    self.padding_masks.append(mask)
                    self.stock_ids.append(stock_id)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.padding_masks = np.array(self.padding_masks, dtype=np.float32)
        self.stock_ids = np.array(self.stock_ids, dtype=np.int64)
        
        print(f"Created dataset with {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': torch.tensor(self.sequences[idx]),
            'padding_mask': torch.tensor(self.padding_masks[idx]),
            'stock_id': torch.tensor(self.stock_ids[idx])
        }

def prepare_data(stock_data: Dict[str, pd.DataFrame], 
                num_stocks: Optional[int] = None, 
                rows_per_stock: Optional[int] = None) -> Tuple[Dict, Dict, Dict, Dict]:
    """Prepare stock data with improved sequence handling"""
    # Select stocks
    selected_stocks = (np.random.choice(list(stock_data.keys()), num_stocks, replace=False) 
                      if num_stocks is not None else list(stock_data.keys()))
    
    # Create stock ID mapping
    stock_to_id = {stock: idx for idx, stock in enumerate(selected_stocks)}
    
    # Get full date range
    all_dates = pd.date_range(
        start=min(df.index.min() for df in stock_data.values()),
        end=max(df.index.max() for df in stock_data.values()),
        freq='B'
    )
    
    aligned_data_dict = {}
    mask_dict = {}
    normalization_params = {}
    
    for stock in selected_stocks:
        # Align and normalize data
        df = stock_data[stock]
        df_aligned = df.reindex(all_dates)
        mask = df_aligned.notna().astype(int)
        
        # Store normalization parameters
        normalization_params[stock] = {
            'median': np.nanmedian(df_aligned.values, axis=0),
            'iqr': (np.nanpercentile(df_aligned.values, 75, axis=0) - 
                   np.nanpercentile(df_aligned.values, 25, axis=0))
        }
        
        # Normalize data
        normalized_data = robust_normalize(df_aligned.values)
        normalized_df = pd.DataFrame(normalized_data, index=all_dates, columns=df.columns)
        normalized_df[mask == 0] = np.nan
        
        # Select specified number of rows
        if rows_per_stock is not None:
            valid_data = normalized_df.dropna()
            normalized_df = valid_data.iloc[:rows_per_stock]
            mask = mask.loc[normalized_df.index]
        
        aligned_data_dict[stock] = normalized_df
        mask_dict[stock] = mask
    
    return aligned_data_dict, mask_dict, normalization_params, stock_to_id

def robust_normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize data using robust scaling with median and IQR.
    Handles NaN values properly.
    """
    median = np.nanmedian(data, axis=0)
    q75, q25 = np.nanpercentile(data, [75, 25], axis=0)
    iqr = q75 - q25
    # Add small epsilon to avoid division by zero
    iqr = np.where(iqr == 0, 1e-8, iqr)
    normalized_data = (data - median) / iqr
    return normalized_data

def get_sp500_tickers(config: Config) -> List[str]:
    """
    Fetches S&P 500 tickers from Wikipedia or loads them from a local file.

    Args:
        config: Config object containing settings such as file paths.

    Returns:
        List[str]: List of ticker symbols for S&P 500 companies.
    """
    file_path = config.DATA_DIR / 'sp500_tickers.txt'

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            tickers = f.read().splitlines()
    else:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url, verify=certifi.where())
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data: {response.status_code}")

        table = pd.read_html(StringIO(response.text))
        tickers = table[0]['Symbol'].tolist()

        # Save the tickers to a file for future use
        with open(file_path, 'w') as f:
            f.write('\n'.join(tickers))

    return tickers

def download_stock_data(tickers: List[str], config: Config) -> Dict[str, pd.DataFrame]:
    """
    Downloads historical stock data for specified tickers or loads from file.

    Args:
        tickers: List of stock ticker symbols
        config: Configuration object containing parameters

    Returns:
        Dictionary of DataFrames with stock data for each ticker
    """
    file_path = config.DATA_DIR / 'stock_data.pkl'
    print(file_path)
    if file_path.exists():
        stock_data = pd.read_pickle(file_path)
    else:
        stock_data = {}
        for ticker in tqdm(tickers, desc="Downloading stock data"):
            try:
                df = yf.download(
                    ticker,
                    start=config.start_date,
                    end=config.end_date,
                    progress=False
                )
                if not df.empty and len(df) >= config.min_window:
                    stock_data[ticker] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            except Exception as e:
                logger.warning(f"Failed to download data for {ticker}: {e}")

        if not stock_data:
            raise ValueError("No valid stock data downloaded")

        os.makedirs(config.DATA_DIR, exist_ok=True)
        pd.to_pickle(stock_data, file_path)

    logger.info(f"Loaded data for {len(stock_data)} stocks")
    return stock_data