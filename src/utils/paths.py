"""Setup project directories and paths."""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for project.
    
    Returns:
        dict: Dictionary containing Path objects for each directory
    """
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # Define directory structure
    dirs = {
        'data': project_root / 'data',
        'results': project_root / 'results',
        'checkpoints': project_root / 'checkpoints',
        'plots': project_root / 'results/plots',
        'models': project_root / 'results/models',
        'logs': project_root / 'logs'
    }
    
    # Create directories
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
    
    return dirs

# Create and export directory paths
DIRS = setup_directories()