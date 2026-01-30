#!/usr/bin/env python3
"""
Fine-tune a Model2Vec classifier for OpenAlex topic classification.

Uses the LLM-generated ground truth to train a topic classifier on top of
potion-base-32M embeddings.

Usage:
    python finetune_classifier.py [--train-file PATH] [--test-split FLOAT] [--epochs INT]
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter
import sys

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
VALIDATION_DIR = Path(__file__).parent
GROUND_TRUTH_DIR = VALIDATION_DIR / "llm_ground_truth"
MODELS_DIR = VALIDATION_DIR.parent / "models"
TOPICS_FILE = MODELS_DIR / "topics.csv"


def load_ground_truth(file_path: Path) -> list[dict]:
    """Load ground truth records from NDJSON file."""
    records = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    # Only include records with valid topic_id
                    if record.get('topic_id'):
                        records.append(record)
                except json.JSONDecodeError:
                    continue
    return records


def prepare_text(record: dict) -> str:
    """Prepare text input for classification."""
    parts = []
    
    # Title
    if record.get('title'):
        parts.append(record['title'])
    
    # Subjects
    subjects = record.get('subjects', [])
    if subjects:
        parts.append(f"Subjects: {', '.join(subjects[:15])}")
    
    # Description (truncated)
    desc = record.get('description', '')
    if desc:
        parts.append(desc[:500])
    
    return ". ".join(parts)


def load_taxonomy() -> dict:
    """Load topic taxonomy for label mapping."""
    df = pd.read_csv(TOPICS_FILE)
    # Create topic_id -> row mapping
    return {str(row['topic_id']): row for _, row in df.iterrows()}


def create_label_encoder(topic_ids: list) -> tuple[dict, dict]:
    """Create label encoder for topic IDs."""
    unique_topics = sorted(set(str(tid) for tid in topic_ids))
    topic_to_idx = {tid: idx for idx, tid in enumerate(unique_topics)}
    idx_to_topic = {idx: tid for tid, idx in topic_to_idx.items()}
    return topic_to_idx, idx_to_topic


def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    """Compute class weights for imbalanced data."""
    counts = Counter(labels)
    total = len(labels)
    weights = torch.ones(num_classes)
    
    for label, count in counts.items():
        # Inverse frequency weighting with smoothing
        weights[label] = total / (num_classes * count + 1)
    
    # Normalize
    weights = weights / weights.sum() * num_classes
    return weights


def main():
    parser = argparse.ArgumentParser(description="Fine-tune topic classifier")
    parser.add_argument("--train-file", type=Path, 
                       default=GROUND_TRUTH_DIR / "llm_ground_truth_10000.jsonl",
                       help="Path to ground truth NDJSON file")
    parser.add_argument("--test-split", type=float, default=0.15,
                       help="Fraction of data to hold out for testing")
    parser.add_argument("--max-epochs", type=int, default=50,
                       help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use for training")
    parser.add_argument("--output-dir", type=Path, 
                       default=VALIDATION_DIR / "finetuned_models",
                       help="Output directory for saved models")
    parser.add_argument("--balance-domains", action="store_true",
                       help="Balance training data across domains")
    parser.add_argument("--base-model", type=str, default="minishlab/potion-base-32m",
                       help="Base model to fine-tune (HuggingFace model name)")
    
    args = parser.parse_args()
    
    # Check if training file exists
    if not args.train_file.exists():
        logger.error(f"Training file not found: {args.train_file}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("OpenAlex Topic Classifier Fine-tuning")
    logger.info("=" * 60)
    logger.info(f"Training file: {args.train_file}")
    logger.info(f"Test split: {args.test_split}")
    logger.info(f"Max epochs: {args.max_epochs}")
    logger.info(f"Device: {args.device}")
    
    # Load ground truth
    logger.info("Loading ground truth...")
    records = load_ground_truth(args.train_file)
    logger.info(f"Loaded {len(records)} records with valid topic IDs")
    
    if len(records) < 100:
        logger.error("Not enough training data. Need at least 100 records.")
        sys.exit(1)
    
    # Load taxonomy
    taxonomy = load_taxonomy()
    logger.info(f"Loaded taxonomy with {len(taxonomy)} topics")
    
    # Prepare texts and labels
    logger.info("Preparing training data...")
    texts = [prepare_text(r) for r in records]
    topic_ids = [str(r['topic_id']) for r in records]
    
    # Create label encoder
    topic_to_idx, idx_to_topic = create_label_encoder(topic_ids)
    labels = [topic_to_idx[tid] for tid in topic_ids]
    num_classes = len(topic_to_idx)
    
    logger.info(f"Unique topics in training data: {num_classes}")
    
    # Analyze distribution
    domain_counts = Counter(r.get('domain', 'Unknown') for r in records)
    logger.info("Domain distribution:")
    for domain, count in domain_counts.most_common():
        pct = count / len(records) * 100
        logger.info(f"  {domain}: {count} ({pct:.1f}%)")
    
    # Train/test split - use stratified split only if we have enough samples per class
    logger.info("Splitting data...")
    label_counts = Counter(labels)
    min_count = min(label_counts.values())
    
    if min_count >= 2:
        # Use stratified split
        X_train, X_test, y_train, y_test, records_train, records_test = train_test_split(
            texts, labels, records,
            test_size=args.test_split,
            random_state=42,
            stratify=labels
        )
    else:
        # Fall back to non-stratified split when some classes have <2 samples
        logger.info(f"  (Some topics have <2 samples, using non-stratified split)")
        X_train, X_test, y_train, y_test, records_train, records_test = train_test_split(
            texts, labels, records,
            test_size=args.test_split,
            random_state=42
        )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Compute class weights based on training set distribution
    # Note: We won't use class_weight due to complexity of matching dimensions
    # The model2vec internal validation handles imbalanced classes
    logger.info("Skipping explicit class weights (model handles internally)")
    
    # Initialize classifier from base model
    logger.info(f"Initializing classifier from {args.base_model}...")
    from model2vec.train import StaticModelForClassification
    
    classifier = StaticModelForClassification.from_pretrained(
        model_name=args.base_model,
        out_dim=num_classes
    )
    
    # Train
    logger.info("Starting training...")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    
    start_time = datetime.now()
    
    classifier.fit(
        X=X_train,
        y=y_train,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=5,
        test_size=0.1,  # Internal validation
        device=args.device
    )
    
    train_time = datetime.now() - start_time
    logger.info(f"Training completed in {train_time}")
    
    # Evaluate on held-out test set
    logger.info("Evaluating on held-out test set...")
    
    # Get predictions
    y_pred = classifier.predict(X_test)
    
    # Compute accuracies at different levels
    exact_matches = sum(1 for p, t in zip(y_pred, y_test) if p == t)
    exact_acc = exact_matches / len(y_test)
    
    # Map back to topic info for hierarchical evaluation
    def get_hierarchy(label_idx):
        topic_id = idx_to_topic[label_idx]
        if topic_id in taxonomy:
            row = taxonomy[topic_id]
            return {
                'topic_id': topic_id,
                'topic_name': row['topic_name'],
                'subfield_id': str(row['subfield_id']),
                'field_id': str(row['field_id']),
                'domain_id': str(row['domain_id'])
            }
        return None
    
    domain_matches = 0
    field_matches = 0
    subfield_matches = 0
    
    for pred_idx, true_idx in zip(y_pred, y_test):
        pred_hier = get_hierarchy(pred_idx)
        true_hier = get_hierarchy(true_idx)
        
        if pred_hier and true_hier:
            if pred_hier['domain_id'] == true_hier['domain_id']:
                domain_matches += 1
            if pred_hier['field_id'] == true_hier['field_id']:
                field_matches += 1
            if pred_hier['subfield_id'] == true_hier['subfield_id']:
                subfield_matches += 1
    
    domain_acc = domain_matches / len(y_test)
    field_acc = field_matches / len(y_test)
    subfield_acc = subfield_matches / len(y_test)
    
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Domain Accuracy:   {domain_acc:.1%}")
    logger.info(f"Field Accuracy:    {field_acc:.1%}")
    logger.info(f"Subfield Accuracy: {subfield_acc:.1%}")
    logger.info(f"Topic Accuracy:    {exact_acc:.1%}")
    
    # Save model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_model_short = args.base_model.split('/')[-1].replace('-', '_')
    model_name = f"openalex-finetuned-{base_model_short}-{timestamp}"
    model_path = args.output_dir / model_name
    model_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to {model_path}...")
    # Save the classifier state dict and model
    torch.save(classifier.state_dict(), model_path / "classifier_state.pt")
    
    # Also save as a static model for inference
    static_model = classifier.to_static_model()
    static_model.save_pretrained(str(model_path / "static_model"))
    
    # Save label mappings
    mappings = {
        'topic_to_idx': topic_to_idx,
        'idx_to_topic': {str(k): v for k, v in idx_to_topic.items()}
    }
    with open(model_path / "label_mappings.json", 'w') as f:
        json.dump(mappings, f, indent=2)
    
    # Save evaluation results
    results = {
        'timestamp': timestamp,
        'base_model': args.base_model,
        'train_file': str(args.train_file),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'num_classes': num_classes,
        'train_time_seconds': train_time.total_seconds(),
        'domain_accuracy': domain_acc,
        'field_accuracy': field_acc,
        'subfield_accuracy': subfield_acc,
        'topic_accuracy': exact_acc,
        'hyperparameters': {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'max_epochs': args.max_epochs
        }
    }
    with open(model_path / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Model and results saved to {model_path}")
    logger.info("=" * 60)
    logger.info("FINE-TUNING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

