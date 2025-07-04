from typing import Optional


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    
    Attributes:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change in monitored value
        counter: Number of epochs with no improvement
        best_loss: Best loss achieved
        early_stop: Whether to stop training
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001) -> None:
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change in monitored value
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            Boolean indicating whether to stop training
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop
