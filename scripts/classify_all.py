#!/usr/bin/env python3
"""
Classify all DataCite slim records using the fine-tuned OpenAlex topic classifier.

Usage:
    python scripts/classify_all.py [--workers N]
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
INPUT_DIR = Path("/home/joneill/s_index_fast/slim-records/datacite-slim-records")
OUTPUT_DIR = Path("/home/joneill/s_index_fast/classified_output")


def classify_file(input_path: Path) -> tuple[str, int, int, float]:
    """Classify a single input file."""
    from openalex_classifier import TopicClassifier
    
    output_path = OUTPUT_DIR / input_path.name.replace("-slim.ndjson", "-slim-topic.ndjson")
    
    # Initialize classifier for this process
    classifier = TopicClassifier()
    classifier.initialize()
    
    start_time = time.time()
    
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
        # Empty file
        output_path.touch()
        return input_path.name, 0, 0, 0.0
    
    # Classify in batches
    results = classifier.classify_batch(records)
    
    # Write output
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    elapsed = time.time() - start_time
    classified = sum(1 for r in results if r.get('topic'))
    
    return input_path.name, len(records), classified, elapsed


def main():
    parser = argparse.ArgumentParser(description="Classify all DataCite records")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                       help="Number of parallel workers")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("OpenAlex Topic Classification - Full Dataset")
    logger.info("=" * 60)
    logger.info(f"Input: {INPUT_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Workers: {args.workers}")
    
    # Get all input files
    input_files = sorted(INPUT_DIR.glob("*.ndjson"))
    logger.info(f"Input files: {len(input_files)}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    total_records = 0
    total_classified = 0
    completed_files = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(classify_file, f): f for f in input_files}
        
        for future in as_completed(futures):
            try:
                filename, records, classified, elapsed = future.result()
                total_records += records
                total_classified += classified
                completed_files += 1
                
                if completed_files % 50 == 0 or completed_files == len(input_files):
                    elapsed_total = time.time() - start_time
                    rate = total_records / elapsed_total if elapsed_total > 0 else 0
                    pct = completed_files / len(input_files) * 100
                    logger.info(
                        f"Progress: {completed_files}/{len(input_files)} ({pct:.1f}%) | "
                        f"Records: {total_records:,} | Rate: {rate:.0f}/s"
                    )
            except Exception as e:
                logger.error(f"Error processing {futures[future]}: {e}")
    
    elapsed_total = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("CLASSIFICATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {completed_files}")
    logger.info(f"Total records: {total_records:,}")
    logger.info(f"Classified: {total_classified:,} ({total_classified/total_records*100:.1f}%)")
    logger.info(f"Time: {elapsed_total/60:.1f} minutes")
    logger.info(f"Rate: {total_records/elapsed_total:.0f} records/second")


if __name__ == "__main__":
    main()

