#!/usr/bin/env python3
"""
Parallel classification of DataCite records using OpenAlex Topic Classifier.

Based on the original fast classify.py pattern.

Usage:
    python scripts/classify_parallel.py --workers 16
"""

import os
# Set environment variables before imports
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import argparse
import logging
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
INPUT_DIR = Path("/home/joneill/s_index_fast/slim-records/datacite-slim-records")
OUTPUT_DIR = Path("/home/joneill/s_index_fast/classified_output")
BASE_DIR = Path("/home/joneill/s_index_fast/openalex-topic-classifier")


def process_file_worker(args):
    """Worker function for parallel processing - each worker initializes its own classifier."""
    input_path, output_path = args
    
    # Import here to avoid issues with multiprocessing
    sys.path.insert(0, str(BASE_DIR / "src"))
    from openalex_classifier import TopicClassifier
    
    classifier = TopicClassifier()
    classifier.initialize()
    
    count = classifier.classify_file(
        Path(input_path),
        Path(output_path),
        show_progress=False
    )
    
    return Path(input_path).name, count


def main():
    parser = argparse.ArgumentParser(description="Parallel OpenAlex Topic Classification")
    parser.add_argument('--workers', type=int, default=16, help='Number of parallel workers')
    parser.add_argument('--skip-existing', action='store_true', default=True, help='Skip already processed files')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("OpenAlex Topic Classification - PARALLEL")
    logger.info("=" * 60)
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Input: {INPUT_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all input files
    input_files = sorted(INPUT_DIR.glob("*.ndjson"))
    logger.info(f"Total input files: {len(input_files)}")
    
    # Prepare work items (skip existing if requested)
    work_items = []
    for f in input_files:
        output_path = OUTPUT_DIR / f.name.replace("-slim.ndjson", "-slim-topic.ndjson")
        if args.skip_existing and output_path.exists() and output_path.stat().st_size > 0:
            continue
        work_items.append((str(f), str(output_path)))
    
    logger.info(f"Files to process: {len(work_items)}")
    
    if not work_items:
        logger.info("All files already processed!")
        return
    
    start_time = time.time()
    total_records = 0
    completed = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file_worker, item): item for item in work_items}
        
        for future in as_completed(futures):
            try:
                filename, count = future.result()
                total_records += count
                completed += 1
                
                elapsed = time.time() - start_time
                rate = total_records / elapsed if elapsed > 0 else 0
                pct = completed / len(work_items) * 100
                eta = (len(work_items) - completed) * (elapsed / completed) / 60 if completed > 0 else 0
                
                if completed % 10 == 0 or completed == len(work_items):
                    logger.info(
                        f"Progress: {completed}/{len(work_items)} ({pct:.1f}%) | "
                        f"Records: {total_records:,} | Rate: {rate:,.0f}/s | ETA: {eta:.1f}min"
                    )
            except Exception as e:
                logger.error(f"Error: {e}")
    
    elapsed = time.time() - start_time
    rate = total_records / elapsed if elapsed > 0 else 0
    
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {completed}")
    logger.info(f"Total records: {total_records:,}")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info(f"Throughput: {rate:,.0f} records/second")


if __name__ == "__main__":
    main()

