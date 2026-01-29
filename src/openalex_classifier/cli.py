"""Command-line interface for OpenAlex Topic Classifier."""

import argparse
import logging
import sys
import time
from pathlib import Path

from .classifier import TopicClassifier
from .config import Config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Classify scientific datasets with OpenAlex topics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify a file
  openalex-classify input.ndjson output.ndjson
  
  # With custom threshold
  openalex-classify input.ndjson output.ndjson --min-score 0.5
  
  # Quiet mode (no progress bar)
  openalex-classify input.ndjson output.ndjson --quiet
"""
    )
    
    parser.add_argument(
        "input",
        type=Path,
        help="Input NDJSON file with dataset records"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output NDJSON file for classification results"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.40,
        help="Minimum confidence score threshold (default: 0.40)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for processing (default: 256)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bar"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure and run
    config = Config(
        min_score=args.min_score,
        batch_size=args.batch_size
    )
    
    classifier = TopicClassifier(config=config)
    
    print(f"Classifying: {args.input}")
    start = time.time()
    
    count = classifier.classify_file(
        args.input,
        args.output,
        show_progress=not args.quiet
    )
    
    elapsed = time.time() - start
    rate = count / elapsed if elapsed > 0 else 0
    
    print(f"\nCompleted: {count:,} records in {elapsed:.1f}s ({rate:,.0f} rec/s)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

