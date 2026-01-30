#!/usr/bin/env python3
"""
Ultra-parallel LLM ground truth generation using asyncio.

Uses aiohttp for true async I/O with hundreds of concurrent requests.
Designed for generating 100K-500K classifications efficiently.

Usage:
    python generate_llm_ground_truth_async.py 500000 --workers 500 --domain-balanced
"""

import argparse
import asyncio
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Optional
import os

import aiohttp
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-6185ef64c68d473d984963356ab0378e")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

DATA_DIR = Path("/home/joneill/s_index_fast/slim-records/datacite-slim-records")
TOPICS_FILE = Path("/home/joneill/s_index_fast/openalex-topic-classifier/models/topics.csv")
OUTPUT_DIR = Path("/home/joneill/s_index_fast/openalex-topic-classifier/validation/llm_ground_truth")

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 3000  # Adjust based on DeepSeek tier
MAX_CONCURRENT = 500  # Max concurrent requests


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, rate_per_minute: int):
        self.rate = rate_per_minute / 60.0  # Tokens per second
        self.tokens = rate_per_minute
        self.max_tokens = rate_per_minute
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


def load_taxonomy() -> tuple[pd.DataFrame, dict, dict]:
    """Load OpenAlex taxonomy."""
    df = pd.read_csv(TOPICS_FILE)
    
    # Group by domain/field/subfield
    domain_fields = df.groupby('domain_name')['field_name'].unique().to_dict()
    field_subfields = df.groupby('field_name')['subfield_name'].unique().to_dict()
    subfield_topics = df.groupby('subfield_name').apply(
        lambda x: list(zip(x['topic_id'], x['topic_name']))
    ).to_dict()
    
    return df, {
        'domain_fields': domain_fields,
        'field_subfields': field_subfields,
        'subfield_topics': subfield_topics
    }, {str(row['topic_id']): row for _, row in df.iterrows()}


def sample_records_balanced(n_records: int, target_per_domain: Optional[dict] = None) -> list[dict]:
    """Sample records, optionally balanced by predicted domain."""
    logger.info(f"Sampling {n_records} records from DataCite...")
    
    # Load all available files
    all_files = sorted(DATA_DIR.glob("*.ndjson"))
    logger.info(f"Found {len(all_files)} DataCite files")
    
    # For large samples, use more files but fewer records per file
    # For small samples, use fewer files
    if n_records > 100000:
        files_to_use = all_files
    elif n_records > 10000:
        files_to_use = random.sample(all_files, min(200, len(all_files)))
    else:
        files_to_use = random.sample(all_files, min(50, len(all_files)))
    
    # Oversample 2x for filtering
    records_per_file = max(10, (n_records * 2) // len(files_to_use))
    
    all_records = []
    
    for file_path in files_to_use:
        if len(all_records) >= n_records * 2:
            break
            
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
                        # Filter non-English (but allow empty/missing language as English default)
                        lang = record.get('language', '')
                        if lang and lang.lower() not in ['en', 'eng', 'english', '']:
                            continue
                        
                        # Get title
                        title = record.get('title', '')
                        if not title and record.get('titles'):
                            for t in record['titles']:
                                if isinstance(t, dict):
                                    title = t.get('title', '')
                                else:
                                    title = str(t)
                                if title:
                                    break
                        
                        # Skip records with bad/missing titles
                        if not title or len(title) < 10:
                            continue
                        if title.startswith('(:') or title.startswith('CCDC'):
                            continue  # Skip placeholders and CCDC crystal records (too many)
                        
                        all_records.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            continue  # Silent continue for speed
    
    logger.info(f"Loaded {len(all_records)} records")
    
    # Shuffle and take requested amount
    random.shuffle(all_records)
    return all_records[:n_records]


def create_classification_prompt(record: dict, taxonomy_data: dict) -> tuple[str, str]:
    """Create prompts for 2-stage classification."""
    # Extract text - handle both slim format (title/subjects/description) 
    # and full format (titles/subjects/descriptions)
    title = ""
    if record.get('title'):
        title = record['title']
    elif record.get('titles'):
        for t in record['titles']:
            if isinstance(t, dict):
                title = t.get('title', '')
            else:
                title = str(t)
            if title:
                break
    
    subjects = []
    if record.get('subjects'):
        for s in record['subjects'][:20]:
            if isinstance(s, dict):
                subjects.append(s.get('subject', ''))
            else:
                subjects.append(str(s))
    
    description = ""
    if record.get('description'):
        description = record['description'][:1000]
    elif record.get('descriptions'):
        for d in record['descriptions']:
            if isinstance(d, dict):
                description = d.get('description', '')[:1000]
            else:
                description = str(d)[:1000]
            if description:
                break
    
    # Stage 1: Domain/Field/Subfield
    domains = list(taxonomy_data['domain_fields'].keys())
    
    stage1_prompt = f"""Classify this scientific record into the OpenAlex taxonomy.

Title: {title}
Subjects: {', '.join(subjects[:15]) if subjects else 'None'}
Description: {description[:500] if description else 'None'}

Available Domains: {', '.join(domains)}

Return JSON with domain, field, and subfield:
{{"domain": "...", "field": "...", "subfield": "..."}}"""

    return stage1_prompt, json.dumps({
        'id': record.get('id', ''),
        'title': title,
        'subjects': subjects[:15],
        'description': description[:500]
    })


async def classify_record_async(
    session: aiohttp.ClientSession,
    record: dict,
    taxonomy_data: dict,
    topic_lookup: dict,
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
    retry_count: int = 3
) -> Optional[dict]:
    """Classify a single record asynchronously using single-stage approach."""
    
    async with semaphore:
        await rate_limiter.acquire()
        
        stage1_prompt, record_info = create_classification_prompt(record, taxonomy_data)
        record_data = json.loads(record_info)
        
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Get all domains for context
        domains = list(taxonomy_data['domain_fields'].keys())
        
        # Single-stage prompt asking for domain, field, subfield, and topic
        prompt = f"""Classify this scientific record into the OpenAlex taxonomy.

Title: {record_data['title']}
Subjects: {', '.join(record_data['subjects'][:15]) if record_data['subjects'] else 'None'}
Description: {record_data['description'][:300] if record_data['description'] else 'None'}

OpenAlex has 4 domains: {', '.join(domains)}
Each domain has fields, subfields, and specific topics.

Return your classification as JSON with these fields:
- domain: one of the 4 domains above
- field: the appropriate field within that domain
- subfield: the specific subfield
- topic_name: the most specific topic name that matches this record

{{"domain": "...", "field": "...", "subfield": "...", "topic_name": "..."}}"""
        
        for attempt in range(retry_count):
            try:
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You are a scientific classification expert familiar with OpenAlex taxonomy. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 300
                }
                
                async with session.post(DEEPSEEK_URL, json=payload, headers=headers) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** attempt + 1)
                        continue
                    
                    if resp.status != 200:
                        await asyncio.sleep(1)
                        continue
                    
                    data = await resp.json()
                    content = data['choices'][0]['message']['content']
                    
                    # Parse JSON
                    content = content.strip()
                    if content.startswith('```'):
                        content = content.split('```')[1]
                        if content.startswith('json'):
                            content = content[4:]
                    
                    result = json.loads(content)
                
                # Find the topic by name (fuzzy match)
                topic_name_lower = result.get('topic_name', '').lower()
                best_match = None
                best_score = 0
                
                for topic_id, row in topic_lookup.items():
                    name = str(row['topic_name']).lower()
                    # Simple containment matching
                    if topic_name_lower == name:
                        best_match = (topic_id, row)
                        break
                    elif topic_name_lower in name or name in topic_name_lower:
                        score = len(set(topic_name_lower.split()) & set(name.split()))
                        if score > best_score:
                            best_score = score
                            best_match = (topic_id, row)
                
                if best_match is None:
                    # Fallback: just use any topic from the subfield
                    subfield = result.get('subfield', '')
                    for tid, tname in taxonomy_data['subfield_topics'].get(subfield, [])[:1]:
                        if str(tid) in topic_lookup:
                            best_match = (str(tid), topic_lookup[str(tid)])
                            break
                
                if best_match is None:
                    return None
                
                topic_id, topic_row = best_match
                
                # Get record ID
                record_id = (record.get('id') or 
                            record.get('doi') or 
                            (record.get('identifiers', [{}])[0].get('identifier') if record.get('identifiers') else '') or
                            '')
                
                return {
                    'record_id': record_id,
                    'title': record_data['title'],
                    'subjects': record_data['subjects'],
                    'description': record_data['description'],
                    'domain': str(topic_row['domain_name']),
                    'field': str(topic_row['field_name']),
                    'subfield': str(topic_row['subfield_name']),
                    'topic_name': str(topic_row['topic_name']),
                    'topic_id': topic_id
                }
                
            except Exception as e:
                if attempt < retry_count - 1:
                    await asyncio.sleep(1)
                    continue
                return None
        
        return None


async def process_batch(
    records: list[dict],
    taxonomy_data: dict,
    topic_lookup: dict,
    output_file: Path,
    max_concurrent: int = 500,
    rate_per_minute: int = 3000
):
    """Process all records with async concurrency."""
    
    rate_limiter = RateLimiter(rate_per_minute)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    completed = 0
    failed = 0
    start_time = time.time()
    
    # Use connection pooling
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Process in chunks for progress reporting
        chunk_size = 1000
        
        with open(output_file, 'a') as f:
            for chunk_start in range(0, len(records), chunk_size):
                chunk = records[chunk_start:chunk_start + chunk_size]
                
                tasks = [
                    classify_record_async(
                        session, record, taxonomy_data, topic_lookup,
                        rate_limiter, semaphore
                    )
                    for record in chunk
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, dict) and result:
                        f.write(json.dumps(result) + '\n')
                        completed += 1
                    else:
                        failed += 1
                
                f.flush()
                
                # Progress
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = len(records) - chunk_start - len(chunk)
                eta = remaining / rate / 60 if rate > 0 else 0
                
                logger.info(
                    f"Progress: {chunk_start + len(chunk)}/{len(records)} "
                    f"({(chunk_start + len(chunk))/len(records)*100:.1f}%) | "
                    f"Rate: {rate:.1f} rec/s | "
                    f"ETA: {eta:.1f} min | "
                    f"Failed: {failed}"
                )
    
    return completed, failed


def main():
    parser = argparse.ArgumentParser(description="Ultra-parallel LLM ground truth generation")
    parser.add_argument("n_records", type=int, help="Number of records to classify")
    parser.add_argument("--workers", type=int, default=500, help="Max concurrent requests")
    parser.add_argument("--rate-limit", type=int, default=3000, help="Max requests per minute")
    parser.add_argument("--output", type=str, default=None, help="Output file name")
    parser.add_argument("--domain-balanced", action="store_true", help="Balance across domains")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Ultra-Parallel LLM Ground Truth Generation")
    logger.info("=" * 60)
    logger.info(f"Target records: {args.n_records:,}")
    logger.info(f"Max concurrent: {args.workers}")
    logger.info(f"Rate limit: {args.rate_limit}/min")
    
    # Load taxonomy
    logger.info("Loading taxonomy...")
    taxonomy_df, taxonomy_data, topic_lookup = load_taxonomy()
    logger.info(f"Loaded {len(taxonomy_df)} topics")
    
    # Sample records
    records = sample_records_balanced(args.n_records)
    logger.info(f"Sampled {len(records)} records")
    
    # Output file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_file = OUTPUT_DIR / args.output
    else:
        output_file = OUTPUT_DIR / f"llm_ground_truth_{args.n_records}.jsonl"
    
    logger.info(f"Output: {output_file}")
    
    # Clear output file
    output_file.write_text("")
    
    # Run async processing
    logger.info("Starting async classification...")
    start_time = time.time()
    
    completed, failed = asyncio.run(
        process_batch(
            records, taxonomy_data, topic_lookup, output_file,
            max_concurrent=args.workers,
            rate_per_minute=args.rate_limit
        )
    )
    
    elapsed = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Completed: {completed:,}")
    logger.info(f"Failed: {failed:,}")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info(f"Rate: {completed/elapsed:.1f} records/second")
    logger.info(f"Output: {output_file}")


if __name__ == "__main__":
    main()

