#!/usr/bin/env python3
"""
Classify all DataCite slim records using the fine-tuned OpenAlex topic classifier.
Simple single-process version - more reliable than multiprocessing.

Usage:
    python scripts/classify_all_simple.py
"""

import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
INPUT_DIR = Path("/home/joneill/s_index_fast/slim-records/datacite-slim-records")
OUTPUT_DIR = Path("/home/joneill/s_index_fast/classified_output")


def main():
    from openalex_classifier import TopicClassifier
    
    logger.info("=" * 60)
    logger.info("OpenAlex Topic Classification - Simple Version")
    logger.info("=" * 60)
    
    # Initialize classifier once
    logger.info("Initializing classifier...")
    classifier = TopicClassifier()
    classifier.initialize()
    logger.info("Classifier ready!")
    
    # Get all input files
    input_files = sorted(INPUT_DIR.glob("*.ndjson"))
    logger.info(f"Input files: {len(input_files)}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    total_records = 0
    total_classified = 0
    
    for i, input_path in enumerate(input_files):
        output_path = OUTPUT_DIR / input_path.name.replace("-slim.ndjson", "-slim-topic.ndjson")
        
        # Skip if already processed
        if output_path.exists() and output_path.stat().st_size > 0:
            continue
        
        # Load records
        records = []
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        if not records:
            output_path.touch()
            continue
        
        # Classify
        results = classifier.classify_batch(records)
        
        # Write output
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        total_records += len(records)
        total_classified += sum(1 for r in results if r.get('topic'))
        
        # Progress every 50 files
        if (i + 1) % 50 == 0 or i == len(input_files) - 1:
            elapsed = time.time() - start_time
            rate = total_records / elapsed if elapsed > 0 else 0
            pct = (i + 1) / len(input_files) * 100
            eta = (len(input_files) - i - 1) * (elapsed / (i + 1)) / 60 if i > 0 else 0
            logger.info(
                f"Progress: {i+1}/{len(input_files)} ({pct:.1f}%) | "
                f"Records: {total_records:,} | Rate: {rate:.0f}/s | ETA: {eta:.1f}min"
            )
    
    elapsed = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("CLASSIFICATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {len(input_files)}")
    logger.info(f"Total records: {total_records:,}")
    logger.info(f"Classified: {total_classified:,} ({total_classified/total_records*100:.1f}%)" if total_records > 0 else "No records")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info(f"Rate: {total_records/elapsed:.0f} records/second" if elapsed > 0 else "N/A")


if __name__ == "__main__":
    main()

