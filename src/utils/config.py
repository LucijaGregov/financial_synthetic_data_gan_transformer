import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that all required configuration fields exist and are properly formatted.

    Args:
        config: Configuration dictionary.

    Raises:
        ValueError: If the configuration is missing required fields or has invalid values.
    """
    if "paths" not in config:
        raise ValueError("Missing paths configuration")

    required_paths = ["checkpoint_dir", "output_dir", "data_dir"]
    for path in required_paths:
        if path not in config["paths"]:
            raise ValueError(f"Missing required path: {path}")

    required_fields = [
        "tickers", "features", "start_date", "end_date", "sequence_length",
        "epochs", "batch_size", "seed", "model_params", "training_params", 
        "patience", "data_params"
    ]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")

    # Validate tickers and features are non-empty lists
    if not isinstance(config["tickers"], list) or not config["tickers"]:
        raise ValueError("tickers must be a non-empty list")

    if not isinstance(config["features"], list) or not config["features"]:
        raise ValueError("features must be a non-empty list")

    # Validate date formats
    try:
        datetime.strptime(config["start_date"], "%Y-%m-%d")
        datetime.strptime(config["end_date"], "%Y-%m-%d")
    except ValueError:
        raise ValueError("start_date and end_date must be valid dates in YYYY-MM-DD format")

    # Validate data_params
    if "data_params" not in config:
        raise ValueError("Missing data_params in config")

    required_data_params = [
        "n_volume_features",
        "cyclic_features"
    ]
    for param in required_data_params:
        if param not in config["data_params"]:
            raise ValueError(f"Missing required data parameter: {param}")

    # Validate cyclic features config
    required_cyclic_config = ["day", "month", "quarter"]
    for param in required_cyclic_config:
        if param not in config["data_params"]["cyclic_features"]:
            raise ValueError(f"Missing cyclic feature configuration: {param}")
        if not isinstance(config["data_params"]["cyclic_features"][param], bool):
            raise ValueError(f"Cyclic feature {param} must be a boolean")

    # Validate model architecture params
    if "generator" not in config["model_params"] or "discriminator" not in config["model_params"]:
        raise ValueError("model_params must include 'generator' and 'discriminator' sections")

    # Validate generator params
    required_generator_params = [
        "num_layers", "d_model", "nhead", "dim_feedforward", "dropout"
    ]
    for param in required_generator_params:
        if param not in config["model_params"]["generator"]:
            raise ValueError(f"Missing required generator parameter: {param}")

    # Validate discriminator params
    required_discriminator_params = [
        "hidden_sizes", "dropout", "leaky_relu_slope"
    ]
    for param in required_discriminator_params:
        if param not in config["model_params"]["discriminator"]:
            raise ValueError(f"Missing required discriminator parameter: {param}")

    # Validate training params
    required_training_params = [
        "g_learning_rate", "d_learning_rate", "adv_weight", "warmup_epochs", "load_pretrained"
    ]
    for param in required_training_params:
        if param not in config["training_params"]:
            raise ValueError(f"Missing required training parameter: {param}")

    # Validate numeric values are positive
    if config["sequence_length"] <= 0:
        raise ValueError("sequence_length must be positive")
    if config["epochs"] <= 0:
        raise ValueError("epochs must be positive")
    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    if config["patience"] <= 0:
        raise ValueError("patience must be positive")

    # Validate learning rates and other params are in valid ranges
    if not (0 < config["training_params"]["g_learning_rate"] < 1):
        raise ValueError("Generator learning rate must be between 0 and 1")
    if not (0 < config["training_params"]["d_learning_rate"] < 1):
        raise ValueError("Discriminator learning rate must be between 0 and 1")
    if not (0 <= config["training_params"]["adv_weight"] <= 1):
        raise ValueError("Adversarial weight must be between 0 and 1")

def load_or_create_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file (absolute or relative)
        
    Returns:
        Configuration dictionary
        
    Raises:
        RuntimeError: If configuration file is missing or invalid
    """
    try:
        config_path = os.path.abspath(config_path)

        if not os.path.exists(config_path):
            parent_config = os.path.join(os.path.dirname(os.path.dirname(config_path)), 'config.json')
            if os.path.exists(parent_config):
                config_path = parent_config
                logger.info(f"Found configuration file in parent directory.")
            else:
                raise FileNotFoundError(
                    f"Configuration file not found in current or in parent directory. "
                    f"Please ensure config.json exists in the correct location."
                )

        with open(config_path, 'r') as f:
            config = json.load(f)

        validate_config(config)
        logger.info(f"Successfully loaded configuration.")
        return config

    except FileNotFoundError as e:
        logger.error(f"Configuration file error: {str(e)}")
        raise RuntimeError(str(e))

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {str(e)}")
        raise RuntimeError(f"Invalid JSON in: {str(e)}")

    except ValueError as e:
        logger.error(f"Configuration validation error: {str(e)}")
        raise RuntimeError(f"Invalid configuration in: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {str(e)}")
        raise RuntimeError(f"Error loading configuration: {str(e)}")
