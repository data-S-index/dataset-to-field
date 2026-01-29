"""
OpenAlex Topic Classifier
=========================

High-throughput CPU-based topic classification using distilled embeddings.

Performance: ~3,000 records/second on 32-core CPU.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import faiss

from .config import Config
from .text import prepare_record_text, get_dataset_id

logger = logging.getLogger(__name__)


class TopicClassifier:
    """
    OpenAlex topic classifier using distilled BGE-small embeddings.
    
    This classifier maps dataset metadata to the 4,516 topics in the
    OpenAlex taxonomy using semantic embedding similarity.
    
    Performance:
        - ~3,000 records/second on CPU
        - 94.6% of records above 0.50 confidence threshold
        - No GPU required
    
    Usage:
        classifier = TopicClassifier()
        classifier.initialize()
        result = classifier.classify(record)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize classifier.
        
        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or Config()
        self.model = None
        self.topic_index = None
        self.topic_embeddings = None
        self.topics_df = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Load model and topic data.
        
        This must be called before classification. On first run,
        it will download/create the distilled model (~10 seconds).
        
        Returns:
            True if initialization succeeded.
        """
        if self._initialized:
            return True
        
        logger.info("Initializing OpenAlex Topic Classifier...")
        start = time.time()
        
        try:
            # Load or create distilled model
            self._load_model()
            
            # Load topics
            self._load_topics()
            
            # Build search index
            self._build_index()
            
            self._initialized = True
            logger.info(f"Classifier initialized in {time.time()-start:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    def _load_model(self):
        """Load or create distilled embedding model."""
        from model2vec import StaticModel
        
        model_path = self.config.distilled_model_path
        
        if model_path.exists():
            logger.info(f"Loading distilled model from {model_path}")
            self.model = StaticModel.from_pretrained(str(model_path))
        else:
            logger.info("Creating distilled model (one-time operation)...")
            self._create_distilled_model()
    
    def _create_distilled_model(self):
        """Create distilled model from BGE-small."""
        from model2vec import StaticModel
        from model2vec.distill import distill
        
        logger.info(f"Distilling {self.config.model_name}...")
        self.model = distill(
            model_name=self.config.model_name,
            pca_dims=self.config.pca_dims,
        )
        
        # Save for future use
        model_path = self.config.distilled_model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(model_path))
        logger.info(f"Distilled model saved to {model_path}")
    
    def _load_topics(self):
        """Load topic taxonomy, downloading if necessary."""
        topics_path = self.config.topics_path
        
        if not topics_path.exists():
            logger.info("Topics file not found, downloading...")
            self._download_topics(topics_path)
        
        logger.info(f"Loading topics from {topics_path}")
        self.topics_df = pd.read_csv(topics_path)
        logger.info(f"Loaded {len(self.topics_df)} topics")
    
    def _download_topics(self, dest_path: Path):
        """Download topics.csv from GitHub repository."""
        import urllib.request
        
        url = "https://raw.githubusercontent.com/jimnoneill/openalex-topic-classifier/main/models/topics.csv"
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading topics from {url}")
        try:
            urllib.request.urlretrieve(url, dest_path)
            logger.info(f"Downloaded to {dest_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download topics.csv: {e}\n"
                "Please download manually from:\n"
                "https://github.com/jimnoneill/openalex-topic-classifier/blob/main/models/topics.csv"
            )
    
    def _build_index(self):
        """Build FAISS index for topic search."""
        embeddings_path = self.config.topic_embeddings_path
        
        if embeddings_path.exists():
            logger.info("Loading cached topic embeddings...")
            self.topic_embeddings = np.load(embeddings_path)
        else:
            logger.info("Embedding topics with distilled model...")
            self._embed_topics()
        
        # Build FAISS index
        logger.info("Building search index...")
        dim = self.topic_embeddings.shape[1]
        self.topic_index = faiss.IndexFlatIP(dim)
        self.topic_index.add(self.topic_embeddings.astype('float32'))
    
    def _embed_topics(self):
        """Embed all topics with the distilled model."""
        topic_texts = [
            f"{row['topic_name']}. Keywords: {row.get('keywords', '')}"
            for _, row in self.topics_df.iterrows()
        ]
        
        embeddings = self.model.encode(topic_texts)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.topic_embeddings = (embeddings / norms).astype('float32')
        
        # Cache
        embeddings_path = self.config.topic_embeddings_path
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, self.topic_embeddings)
        logger.info(f"Cached topic embeddings to {embeddings_path}")
    
    def classify(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a single record.
        
        Args:
            record: Dataset metadata with at least 'title' field.
            
        Returns:
            Classification result with topic, subfield, field, domain.
        """
        return self.classify_batch([record])[0]
    
    def classify_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify a batch of records.
        
        Args:
            records: List of dataset metadata dictionaries.
            
        Returns:
            List of classification results.
        """
        if not self._initialized:
            self.initialize()
        
        if not records:
            return []
        
        # Extract text and IDs
        texts = [prepare_record_text(r) for r in records]
        dataset_ids = [get_dataset_id(r) for r in records]
        
        # Embed
        embeddings = self.model.encode(texts)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = (embeddings / norms).astype('float32')
        
        # Search
        scores, indices = self.topic_index.search(embeddings, 1)
        
        # Build results
        results = []
        min_score = self.config.min_score
        
        for i in range(len(records)):
            result = {'dataset_id': dataset_ids[i]}
            score = float(scores[i, 0])
            
            if score >= min_score:
                row = self.topics_df.iloc[indices[i, 0]]
                result['topic'] = {
                    'id': int(row['topic_id']),
                    'name': str(row['topic_name']),
                    'score': round(score, 4),
                }
                result['subfield'] = {
                    'id': int(row['subfield_id']),
                    'name': str(row['subfield_name'])
                }
                result['field'] = {
                    'id': int(row['field_id']),
                    'name': str(row['field_name'])
                }
                result['domain'] = {
                    'id': int(row['domain_id']),
                    'name': str(row['domain_name'])
                }
            else:
                result['topic'] = None
                result['subfield'] = None
                result['field'] = None
                result['domain'] = None
            
            results.append(result)
        
        return results
    
    def classify_file(
        self,
        input_path: Path,
        output_path: Path,
        show_progress: bool = True
    ) -> int:
        """
        Classify all records in an NDJSON file.
        
        Args:
            input_path: Path to input NDJSON file.
            output_path: Path to output NDJSON file.
            show_progress: Whether to show progress bar.
            
        Returns:
            Number of records processed.
        """
        import json
        from tqdm import tqdm
        
        if not self._initialized:
            self.initialize()
        
        # Count lines for progress bar
        if show_progress:
            with open(input_path) as f:
                total = sum(1 for _ in f)
        else:
            total = None
        
        batch = []
        processed = 0
        
        with open(input_path) as fin, open(output_path, 'w') as fout:
            iterator = tqdm(fin, total=total, desc="Classifying") if show_progress else fin
            
            for line in iterator:
                try:
                    record = json.loads(line)
                    batch.append(record)
                except json.JSONDecodeError:
                    continue
                
                if len(batch) >= self.config.batch_size:
                    results = self.classify_batch(batch)
                    for result in results:
                        fout.write(json.dumps(result) + '\n')
                    processed += len(batch)
                    batch = []
            
            # Final batch
            if batch:
                results = self.classify_batch(batch)
                for result in results:
                    fout.write(json.dumps(result) + '\n')
                processed += len(batch)
        
        return processed

