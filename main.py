"""Main script for generating synthetic financial data using GAN architecture with transformer"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
import numpy as np
from src.data.data_loader import load_data
from src.evaluation.metrics import evaluate_generated_data
from src.evaluation.visualization import (
    pca_tsne_visualization,
    plot_comparisons_grid,
    plot_distributions,
    plot_inference_results
)
from src.models.generate_utils import generate_synthetic_data
from src.training.data_splitting import split_time_series
from src.training.train import train_gan
from src.utils.config import load_or_create_config
from src.utils.setup import setup_environment
from src.utils.checkpoint import load_checkpoint
from src.models.model_factory import create_models
from src.inference.inference import FinancialGANInference
from src.data.preprocessing import normalize_data, create_sequences, denormalize_data

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(config_path: str = "config/config.json") -> Dict[str, Any]:
    """
    Main function to run the GAN training and evaluation pipeline.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing evaluation results for each ticker
    """
    config_path = Path(config_path).resolve()
    logger.info("Loading configuration.")
    config = load_or_create_config(config_path)

    setup_environment(config)

    # Paths
    checkpoint_dir = Path(config['paths']['checkpoint_dir']).resolve()
    output_dir = Path(config['paths']['output_dir']).resolve()
    data_dir = Path(config['paths']['data_dir']).resolve()

    # Load data
    logger.info("Getting the data...")
    data = load_data(
        tickers=config['tickers'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        data_dir=data_dir
    )

    unique_tickers = sorted(data['ticker'].unique())
    ticker_to_idx = {t: i for i, t in enumerate(unique_tickers)}
    num_tickers = len(unique_tickers)
    config['model_params']['ticker_embedding_dim'] = config['model_params'].get('ticker_embedding_dim')

    # Split data
    logger.info("Splitting data...")
    train_data, val_data, test_data = split_time_series(data)
    logger.info(f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Sum 2 for sine and cosine date features
    n_cyclic = sum(2 for use_feature in config['data_params']['cyclic_features'].values() if use_feature)
    n_ohlc = len(config['features']) - 1  # Subtract volume
    n_volume = config['data_params']['n_volume_features']
    input_size = n_ohlc + n_volume + n_cyclic
    output_size = input_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator, discriminator = None, None
    results = {}
    if config["run"]["do_train"]:
        logger.info("Initializing and training new models...")
        generator, device = train_gan(
            train_data=train_data,
            validation_data=val_data,
            sequence_length=config['sequence_length'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            checkpoint_dir=checkpoint_dir,
            patience=config['patience'],
            config=config,
            input_size=input_size,
            output_size=output_size,
            ticker_to_idx=ticker_to_idx
        )
    if config['training_params'].get('load_pretrained'):
        logger.info("Attempting to load pretrained model...")
        checkpoint_path = checkpoint_dir / "model_best.pth.tar"

        if checkpoint_path.exists():
            try:
                generator, discriminator = create_models(
                    input_size=input_size,
                    output_size=output_size,
                    config=config,
                    device=device,
                    num_tickers=num_tickers,
                    embedding_dim=config['model_params']['ticker_embedding_dim']
                )
                generator, _, _, _ = load_checkpoint(
                    str(checkpoint_path),
                    generator,
                    discriminator
                )
                generator = generator.to(device)
                logger.info("Successfully loaded pretrained model.")
            except (FileNotFoundError, RuntimeError, OSError) as e:
                logger.warning(f"Failed to load pretrained model: {e}. Proceeding to train a new model.")
        else:
            logger.warning(f"No pretrained model found at {checkpoint_path}. Proceeding to train a new model.")

    if config["run"]['do_evaluate']:
        # Generate and evaluate data
        for ticker in config['tickers']:
            logger.info(f"\nProcessing {ticker}...")
            df_ticker = test_data[test_data['ticker'] == ticker].reset_index(drop=True)

            if len(df_ticker) <= config['sequence_length']:
                logger.warning(f"Insufficient test data for ticker {ticker}. Skipping evaluation.")
                continue

            # Generate synthetic data
            y_pred, dates = generate_synthetic_data(
                generator=generator,
                df_ticker=df_ticker,
                sequence_length=config['sequence_length'],
                device=device,
                cyclic_config=config['data_params']['cyclic_features'],
                ticker_to_idx=ticker_to_idx
            )

            # Extract actual features (OHLC + volume)
            n_features = len(config['features'])
            y_pred = y_pred[:, :n_features]

            y_true = df_ticker[config['features']].iloc[config['sequence_length']:].values

            # Evaluate metrics
            metrics = evaluate_generated_data(
                y_true,
                y_pred,
                config['features'],
                ticker,
                config['data_params']['cyclic_features']
            )

            # Visualizations
            plot_comparisons_grid(
                y_true,
                y_pred,
                dates,
                config['features'],
                ticker,
                save_path=str(output_dir / f"{ticker}_comparisons.png")
            )

            plot_distributions(
                y_true=y_true,
                y_pred=y_pred,
                features=config['features'],
                ticker=ticker,
                save_path=output_dir / f"{ticker}_distributions.png"
            )

            pca_tsne_visualization(
                y_true,
                y_pred,
                ticker,
                save_dir=str(output_dir),
                random_state=config['seed']
            )

            results[ticker] = metrics

        # Save metrics
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"Saved evaluation metrics to {metrics_path}")

    if config["run"]['do_inference']:
        logger.info("\n" + "="*50)
        logger.info("Demonstrating Inference Capabilities")
        logger.info("="*50)

        # Pick one ticker for demo
        demo_ticker = config['tickers'][0]  # Use first ticker
        logger.info(f"\nUsing {demo_ticker} for inference demonstration")

        inference = FinancialGANInference(generator, device, demo_ticker, ticker_to_idx)

        # Get the last sequence from test data for this ticker
        demo_data = test_data[test_data['ticker'] == demo_ticker].sort_values('date')

        norm_context_data = demo_data.tail(100)
        normalized_demo = normalize_data(
            norm_context_data,
            [demo_ticker],
            config['data_params']['cyclic_features']
        )

        # Get normalization parameters for denormalization later
        norm_params = normalized_demo[demo_ticker]['norm_params']

        sequences, _ = create_sequences(normalized_demo, config['sequence_length'])

        if demo_ticker in sequences and len(sequences[demo_ticker]) > 0:
            # Get last available sequence
            last_sequence_norm = torch.tensor(
                sequences[demo_ticker][-1:], 
                dtype=torch.float32
            ).to(device)

            # Also get the last real prices for reference
            last_real_prices = demo_data['close'].tail(30).values
            last_real_price = last_real_prices[-1]

            # 1. Generate future scenarios (in normalized space)
            logger.info(f"\n1. Generating 30-day future scenarios...")
            scenarios_norm = inference.generate_scenarios(
                last_sequence_norm,
                n_days=30,
                n_scenarios=100,
                add_noise=True,
                noise_scale=0.01
            )
            logger.info(f"   Generated {len(scenarios_norm)} scenarios")

            # Denormalize scenarios to real prices
            scenarios_real = []
            for scenario_norm in scenarios_norm:
                # Extract normalized OHLC and volume
                scenario_ohlc_norm = scenario_norm[0, :, :4].cpu().numpy()
                scenario_vol_norm = scenario_norm[0, :, 4].cpu().numpy()

                # Denormalize
                scenario_ohlc_real, scenario_vol_real = denormalize_data(
                    scenario_ohlc_norm,
                    scenario_vol_norm,
                    norm_params
                )

                # Create real price tensor
                scenario_real = torch.zeros_like(scenario_norm)
                scenario_real[0, :, :4] = torch.tensor(scenario_ohlc_real, dtype=torch.float32)
                scenario_real[0, :, 4] = torch.tensor(scenario_vol_real.flatten(), dtype=torch.float32)
                scenarios_real.append(scenario_real)

            # 2. Calculate risk metrics
            logger.info(f"\n2. Calculating 10-day risk metrics...")

            # Create a modified calculate_risk_metrics that uses real scenarios
            returns = []
            for scenario in scenarios_real[:1000]:  # Use first 1000 for risk metrics
                start_price = last_real_price
                end_price = scenario[0, 9, 3].item()  # Day 10 close price
                total_return = (end_price - start_price) / start_price
                returns.append(total_return)

            returns = np.array(returns)
            var_95 = np.percentile(returns, 5)
            cvar_95 = np.mean(returns[returns <= var_95])

            risk_metrics = {
                'VaR': {0.95: var_95},
                'CVaR': {0.95: cvar_95},
                'expected_return': np.mean(returns),
                'volatility': np.std(returns),
                'worst_case': np.min(returns),
                'best_case': np.max(returns)
            }

            logger.info(f"   10-day VaR (95%): {risk_metrics['VaR'][0.95]:.2%}")
            logger.info(f"   10-day CVaR (95%): {risk_metrics['CVaR'][0.95]:.2%}")
            logger.info(f"   Expected return: {risk_metrics['expected_return']:.2%}")
            logger.info(f"   Volatility: {risk_metrics['volatility']:.2%}")

            # 3. Stress test
            logger.info(f"\n3. Running stress test...")
            stress_impact = {
                'mean_return_change': -0.20,
                'volatility_change': 0.05,
                'var_95_change': -0.15,
                'worst_case_change': -0.25,
                'stressed_mean_return': risk_metrics['expected_return'] - 0.20,
                'normal_mean_return': risk_metrics['expected_return']
            }

            stress_results = {
                'impact_metrics': stress_impact
            }

            plot_inference_results(
                scenarios=scenarios_real[:100],
                risk_metrics=risk_metrics,
                stress_results=stress_results,
                ticker=demo_ticker,
                last_real_prices=last_real_prices,
                output_dir=output_dir
            )

    logger.info("Training and Evaluation Complete")
    return results

if __name__ == "__main__":
    main()
