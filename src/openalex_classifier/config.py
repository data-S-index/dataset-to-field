"""Configuration for OpenAlex Topic Classifier."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


def _find_models_dir() -> Path:
    """
    Find models directory by checking multiple locations.
    
    Search order:
    1. OPENALEX_MODELS_DIR environment variable
    2. ./models (current working directory)
    3. Package installation directory
    4. ~/.openalex_classifier/models (user home)
    """
    # 1. Environment variable
    if env_dir := os.environ.get("OPENALEX_MODELS_DIR"):
        path = Path(env_dir)
        if path.exists():
            return path
    
    # 2. Current working directory
    cwd_models = Path.cwd() / "models"
    if (cwd_models / "topics.csv").exists():
        return cwd_models
    
    # 3. Package directory (for development installs)
    pkg_models = Path(__file__).parent.parent.parent / "models"
    if (pkg_models / "topics.csv").exists():
        return pkg_models
    
    # 4. User home directory (for pip installs)
    home_models = Path.home() / ".openalex_classifier" / "models"
    home_models.mkdir(parents=True, exist_ok=True)
    return home_models


@dataclass
class Config:
    """Configuration settings for the topic classifier."""
    
    # Model settings
    model_name: str = "BAAI/bge-small-en-v1.5"
    pca_dims: int = 256  # Dimensions for distilled model
    
    # Classification settings
    min_score: float = 0.40  # Minimum confidence threshold
    batch_size: int = 256  # Records per batch
    
    # Paths (auto-detected if not specified)
    models_dir: Optional[Path] = None
    topics_path: Optional[Path] = None
    
    def __post_init__(self):
        """Set default paths, searching multiple locations."""
        if self.models_dir is None:
            self.models_dir = _find_models_dir()
        if self.topics_path is None:
            self.topics_path = self.models_dir / "topics.csv"
        
        # Ensure Path types
        self.models_dir = Path(self.models_dir)
        self.topics_path = Path(self.topics_path)
    
    @property
    def distilled_model_path(self) -> Path:
        """Path to distilled embedding model."""
        return self.models_dir / "bge-small-distilled"
    
    @property
    def topic_embeddings_path(self) -> Path:
        """Path to pre-computed topic embeddings."""
        return self.models_dir / "topic_embeddings.npy"

