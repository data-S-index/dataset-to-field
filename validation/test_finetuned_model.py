#!/usr/bin/env python3
"""
Test the fine-tuned classifier against the original 500-record validation set.
Compares performance with base embedding models.
"""

import json
import logging
from pathlib import Path
from collections import Counter

import pandas as pd
from model2vec import StaticModel

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
FINETUNED_DIR = VALIDATION_DIR / "finetuned_models"


def load_ground_truth(file_path: Path) -> list[dict]:
    """Load ground truth records."""
    records = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    if record.get('topic_id'):
                        records.append(record)
                except json.JSONDecodeError:
                    continue
    return records


def prepare_text(record: dict) -> str:
    """Prepare text for classification."""
    parts = []
    if record.get('title'):
        parts.append(record['title'])
    subjects = record.get('subjects', [])
    if subjects:
        parts.append(f"Subjects: {', '.join(subjects[:15])}")
    desc = record.get('description', '')
    if desc:
        parts.append(desc[:500])
    return ". ".join(parts)


def load_taxonomy() -> tuple[pd.DataFrame, dict]:
    """Load taxonomy and create lookup."""
    df = pd.read_csv(TOPICS_FILE)
    lookup = {str(row['topic_id']): row for _, row in df.iterrows()}
    return df, lookup


def find_latest_finetuned_model() -> Path:
    """Find the most recently trained model."""
    models = sorted(FINETUNED_DIR.glob("openalex-topic-classifier-*"))
    if not models:
        raise FileNotFoundError("No fine-tuned models found")
    return models[-1] / "static_model"


def classify_with_embeddings(model: StaticModel, texts: list[str], 
                            taxonomy_df: pd.DataFrame) -> list[dict]:
    """Classify texts using embedding similarity."""
    import numpy as np
    import faiss
    
    # Get topic embeddings
    topic_texts = taxonomy_df['topic_name'].tolist()
    topic_embeddings = model.encode(topic_texts)
    topic_embeddings = topic_embeddings / np.linalg.norm(topic_embeddings, axis=1, keepdims=True)
    
    # Build FAISS index
    index = faiss.IndexFlatIP(topic_embeddings.shape[1])
    index.add(topic_embeddings.astype('float32'))
    
    # Classify texts
    text_embeddings = model.encode(texts)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    scores, indices = index.search(text_embeddings.astype('float32'), 1)
    
    results = []
    for i, (idx, score) in enumerate(zip(indices[:, 0], scores[:, 0])):
        row = taxonomy_df.iloc[idx]
        results.append({
            'topic_id': str(row['topic_id']),
            'topic_name': row['topic_name'],
            'subfield_id': str(row['subfield_id']),
            'subfield_name': row['subfield_name'],
            'field_id': str(row['field_id']),
            'field_name': row['field_name'],
            'domain_id': str(row['domain_id']),
            'domain_name': row['domain_name'],
            'score': float(score)
        })
    
    return results


def evaluate(ground_truth: list[dict], predictions: list[dict], 
             taxonomy_lookup: dict, model_name: str) -> dict:
    """Evaluate predictions against ground truth."""
    domain_correct = 0
    field_correct = 0
    subfield_correct = 0
    topic_correct = 0
    
    for gt, pred in zip(ground_truth, predictions):
        gt_topic_id = str(gt.get('topic_id', ''))
        pred_topic_id = str(pred.get('topic_id', ''))
        
        # Get hierarchies
        gt_row = taxonomy_lookup.get(gt_topic_id)
        pred_row = taxonomy_lookup.get(pred_topic_id)
        
        if gt_row is not None and pred_row is not None:
            if str(gt_row['domain_id']) == str(pred_row['domain_id']):
                domain_correct += 1
            if str(gt_row['field_id']) == str(pred_row['field_id']):
                field_correct += 1
            if str(gt_row['subfield_id']) == str(pred_row['subfield_id']):
                subfield_correct += 1
            if gt_topic_id == pred_topic_id:
                topic_correct += 1
    
    n = len(ground_truth)
    results = {
        'model': model_name,
        'samples': n,
        'domain_accuracy': domain_correct / n,
        'field_accuracy': field_correct / n,
        'subfield_accuracy': subfield_correct / n,
        'topic_accuracy': topic_correct / n
    }
    
    return results


def main():
    logger.info("=" * 60)
    logger.info("Testing Fine-tuned Model vs Base Models")
    logger.info("=" * 60)
    
    # Load ground truth (500-record set)
    gt_file = GROUND_TRUTH_DIR / "llm_ground_truth_500.jsonl"
    if not gt_file.exists():
        gt_file = GROUND_TRUTH_DIR / "llm_ground_truth_500.ndjson"
    
    logger.info(f"Loading ground truth from {gt_file.name}...")
    ground_truth = load_ground_truth(gt_file)
    logger.info(f"Loaded {len(ground_truth)} records")
    
    # Load taxonomy
    taxonomy_df, taxonomy_lookup = load_taxonomy()
    
    # Prepare texts
    texts = [prepare_text(r) for r in ground_truth]
    
    # Test models
    models_to_test = [
        ("minishlab/potion-base-32m", "potion-base-32M (base)"),
    ]
    
    # Add fine-tuned model
    try:
        finetuned_path = find_latest_finetuned_model()
        models_to_test.append((str(finetuned_path), "Fine-tuned"))
        logger.info(f"Found fine-tuned model: {finetuned_path.parent.name}")
    except FileNotFoundError:
        logger.warning("No fine-tuned model found, skipping")
    
    all_results = []
    
    for model_path, model_name in models_to_test:
        logger.info(f"\nTesting: {model_name}")
        logger.info("-" * 40)
        
        # Load model
        model = StaticModel.from_pretrained(model_path)
        
        # Classify
        predictions = classify_with_embeddings(model, texts, taxonomy_df)
        
        # Evaluate
        results = evaluate(ground_truth, predictions, taxonomy_lookup, model_name)
        all_results.append(results)
        
        logger.info(f"  Domain:   {results['domain_accuracy']:.1%}")
        logger.info(f"  Field:    {results['field_accuracy']:.1%}")
        logger.info(f"  Subfield: {results['subfield_accuracy']:.1%}")
        logger.info(f"  Topic:    {results['topic_accuracy']:.1%}")
    
    # Print comparison table
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    
    print("\n| Model | Domain | Field | Subfield | Topic |")
    print("|-------|--------|-------|----------|-------|")
    for r in all_results:
        print(f"| {r['model']:<25} | {r['domain_accuracy']:.1%} | {r['field_accuracy']:.1%} | {r['subfield_accuracy']:.1%} | {r['topic_accuracy']:.1%} |")
    
    # Calculate improvement
    if len(all_results) >= 2:
        base = all_results[0]
        tuned = all_results[-1]
        print("\n| Improvement |", end="")
        print(f" +{(tuned['domain_accuracy'] - base['domain_accuracy'])*100:.1f}% |", end="")
        print(f" +{(tuned['field_accuracy'] - base['field_accuracy'])*100:.1f}% |", end="")
        print(f" +{(tuned['subfield_accuracy'] - base['subfield_accuracy'])*100:.1f}% |", end="")
        print(f" +{(tuned['topic_accuracy'] - base['topic_accuracy'])*100:.1f}% |")


if __name__ == "__main__":
    main()

