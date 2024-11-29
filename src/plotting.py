# src/pipeline/plotting.py
import matplotlib.pyplot as plt
from typing import Dict, List

def plot_training_history(losses: Dict[str, List[float]], save_path: str = 'training_history.png'):
    """
    Plots training and validation losses for generator and discriminator.

    Args:
        losses (Dict[str, List[float]]): Dictionary containing lists of losses for both generator and discriminator.
        save_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot generator losses
    plt.subplot(1, 2, 1)
    plt.plot(losses['g_loss'], label='Generator Training Loss')
    plt.plot(losses['val_g_loss'], label='Generator Validation Loss')
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot discriminator losses
    plt.subplot(1, 2, 2)
    plt.plot(losses['d_loss'], label='Discriminator Training Loss')
    plt.plot(losses['val_d_loss'], label='Discriminator Validation Loss')
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
