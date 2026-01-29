"""Configuration for OpenAlex Topic Classifier."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
        """Set default paths relative to package location."""
        if self.models_dir is None:
            self.models_dir = Path(__file__).parent.parent.parent / "models"
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

