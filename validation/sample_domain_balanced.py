#!/usr/bin/env python3
"""
Create domain-balanced samples for fine-tuning.

Uses the embedding classifier to pre-classify records, then samples
evenly from each domain to create a balanced training set.

Usage:
    python sample_domain_balanced.py 100000 --output balanced_100k.jsonl
"""

import argparse
import json
import logging
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/home/joneill/s_index_fast/slim-records/datacite-slim-records")
TOPICS_FILE = Path("/home/joneill/s_index_fast/openalex-topic-classifier/models/topics.csv")
OUTPUT_DIR = Path("/home/joneill/s_index_fast/openalex-topic-classifier/validation/balanced_samples")


def load_taxonomy() -> pd.DataFrame:
    """Load OpenAlex taxonomy."""
    return pd.read_csv(TOPICS_FILE)


def load_records_sample(n_records: int) -> list[dict]:
    """Load a large sample of records for pre-classification."""
    logger.info(f"Loading {n_records * 5} records for pre-classification...")
    
    all_files = sorted(DATA_DIR.glob("*.ndjson"))
    records_per_file = max(1, n_records * 5 // len(all_files))
    
    all_records = []
    
    for file_path in all_files:
        try:
            with open(file_path) as f:
                lines = f.readlines()
                if len(lines) > records_per_file:
                    sample_lines = random.sample(lines, records_per_file)
                else:
                    sample_lines = lines
                
                for line in sample_lines:
                    try:
                        record = json.loads(line)
                        # Filter English
                        lang = record.get('language', 'en')
                        if lang and lang.lower() not in ['en', 'eng', 'english', '']:
                            continue
                        if not record.get('titles'):
                            continue
                        all_records.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
    
    logger.info(f"Loaded {len(all_records)} English records")
    return all_records


def prepare_text(record: dict) -> str:
    """Prepare text for classification."""
    title = ""
    if record.get('titles'):
        for t in record['titles']:
            if isinstance(t, dict):
                title = t.get('title', '')
            else:
                title = str(t)
            if title:
                break
    
    subjects = []
    if record.get('subjects'):
        for s in record['subjects'][:15]:
            if isinstance(s, dict):
                subjects.append(s.get('subject', ''))
            else:
                subjects.append(str(s))
    
    description = ""
    if record.get('descriptions'):
        for d in record['descriptions']:
            if isinstance(d, dict):
                description = d.get('description', '')[:500]
            else:
                description = str(d)[:500]
            if description:
                break
    
    parts = [title]
    if subjects:
        parts.append(f"Subjects: {', '.join(subjects)}")
    if description:
        parts.append(description)
    
    return ". ".join(parts)


def classify_with_embeddings(records: list[dict], taxonomy_df: pd.DataFrame) -> list[str]:
    """Pre-classify records using embedding model to get predicted domains."""
    from model2vec import StaticModel
    import faiss
    
    logger.info("Loading embedding model...")
    model = StaticModel.from_pretrained("minishlab/potion-base-32m")
    
    # Build topic index
    logger.info("Building topic index...")
    topic_texts = taxonomy_df['topic_name'].tolist()
    topic_embeddings = model.encode(topic_texts)
    topic_embeddings = topic_embeddings / np.linalg.norm(topic_embeddings, axis=1, keepdims=True)
    
    index = faiss.IndexFlatIP(topic_embeddings.shape[1])
    index.add(topic_embeddings.astype('float32'))
    
    # Classify in batches
    logger.info("Classifying records...")
    domains = []
    batch_size = 1000
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        texts = [prepare_text(r) for r in batch]
        
        embeddings = model.encode(texts)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        _, indices = index.search(embeddings.astype('float32'), 1)
        
        for idx in indices[:, 0]:
            row = taxonomy_df.iloc[idx]
            domains.append(str(row['domain_name']))
        
        if (i + batch_size) % 10000 == 0:
            logger.info(f"  Classified {i + batch_size}/{len(records)}")
    
    return domains


def balance_by_domain(records: list[dict], domains: list[str], 
                      n_per_domain: int) -> list[dict]:
    """Sample equally from each domain."""
    domain_records = defaultdict(list)
    
    for record, domain in zip(records, domains):
        domain_records[domain].append(record)
    
    logger.info("Domain distribution before balancing:")
    for domain, recs in sorted(domain_records.items()):
        logger.info(f"  {domain}: {len(recs)}")
    
    balanced = []
    for domain, recs in domain_records.items():
        if len(recs) >= n_per_domain:
            balanced.extend(random.sample(recs, n_per_domain))
        else:
            # Oversample if not enough
            balanced.extend(recs)
            if len(recs) < n_per_domain:
                logger.warning(f"  {domain} only has {len(recs)} records (need {n_per_domain})")
    
    random.shuffle(balanced)
    return balanced


def main():
    parser = argparse.ArgumentParser(description="Create domain-balanced samples")
    parser.add_argument("n_records", type=int, help="Total records to sample")
    parser.add_argument("--output", type=str, default=None, help="Output file name")
    parser.add_argument("--n-domains", type=int, default=4, help="Number of domains")
    
    args = parser.parse_args()
    
    n_per_domain = args.n_records // args.n_domains
    
    logger.info("=" * 60)
    logger.info("Domain-Balanced Sampling")
    logger.info("=" * 60)
    logger.info(f"Target: {args.n_records} records ({n_per_domain} per domain)")
    
    # Load taxonomy
    taxonomy_df = load_taxonomy()
    
    # Load extra records for balancing
    records = load_records_sample(args.n_records)
    
    # Pre-classify
    logger.info("Pre-classifying with embedding model...")
    domains = classify_with_embeddings(records, taxonomy_df)
    
    # Balance
    logger.info("Balancing by domain...")
    balanced = balance_by_domain(records, domains, n_per_domain)
    
    logger.info(f"Balanced sample size: {len(balanced)}")
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_file = OUTPUT_DIR / args.output
    else:
        output_file = OUTPUT_DIR / f"balanced_sample_{args.n_records}.jsonl"
    
    with open(output_file, 'w') as f:
        for record in balanced:
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"Saved to {output_file}")
    
    # Verify balance
    logger.info("\nVerifying balance:")
    domain_counts = Counter(domains[records.index(r)] if r in records else "Unknown" 
                           for r in balanced)
    # Re-classify balanced set
    balanced_domains = classify_with_embeddings(balanced[:1000], taxonomy_df)
    logger.info("Sample of 1000 records from balanced set:")
    for domain, count in Counter(balanced_domains).most_common():
        logger.info(f"  {domain}: {count} ({count/10:.1f}%)")


if __name__ == "__main__":
    main()

