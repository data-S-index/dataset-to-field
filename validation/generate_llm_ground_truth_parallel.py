#!/usr/bin/env python3
"""
Parallel LLM ground truth generation using DeepSeek API.
Uses concurrent workers for faster processing.

Usage:
    python generate_llm_ground_truth_parallel.py 10000 --workers 15
"""

import json
import csv
import random
import time
import os
import argparse
import threading
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from queue import Queue

# Configuration
DEEPSEEK_API_KEY = "sk-6185ef64c68d473d984963356ab0378e"
DATA_DIR = Path("/home/joneill/s_index_fast/slim-records/datacite-slim-records")
TOPICS_FILE = Path("/home/joneill/s_index_fast/openalex-topic-classifier/models/topics.csv")
OUTPUT_DIR = Path("/home/joneill/s_index_fast/openalex-topic-classifier/validation/llm_ground_truth")

# Thread-local client
thread_local = threading.local()

def get_client():
    """Get thread-local API client."""
    if not hasattr(thread_local, 'client'):
        thread_local.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
    return thread_local.client


def is_english(text: str) -> bool:
    """Check if text is predominantly English."""
    if not text:
        return False
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    total_letters = sum(1 for c in text if c.isalpha())
    if total_letters == 0:
        return False
    return ascii_letters / total_letters > 0.8


def load_taxonomy():
    """Load OpenAlex taxonomy with full hierarchy."""
    
    domains = {}
    topics_by_subfield = defaultdict(list)
    all_topics = {}
    subfield_to_field = {}
    field_to_domain = {}
    
    valid_domains = set()
    valid_fields = set()
    valid_subfields = set()
    valid_topics = {}
    
    with open(TOPICS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain_id = row['domain_id']
            domain_name = row['domain_name']
            field_id = row['field_id']
            field_name = row['field_name']
            subfield_id = row['subfield_id']
            subfield_name = row['subfield_name']
            topic_id = row['topic_id']
            topic_name = row['topic_name']
            keywords = row.get('keywords', '')
            
            if domain_id not in domains:
                domains[domain_id] = {'name': domain_name, 'fields': {}}
            if field_id not in domains[domain_id]['fields']:
                domains[domain_id]['fields'][field_id] = {'name': field_name, 'subfields': set()}
            domains[domain_id]['fields'][field_id]['subfields'].add(subfield_name)
            
            subfield_to_field[subfield_name] = field_name
            field_to_domain[field_name] = domain_name
            
            topics_by_subfield[subfield_name].append({
                'topic_id': topic_id,
                'topic_name': topic_name,
                'keywords': keywords
            })
            
            all_topics[topic_id] = {
                'topic_id': topic_id,
                'topic_name': topic_name,
                'subfield_id': subfield_id,
                'subfield_name': subfield_name,
                'field_id': field_id,
                'field_name': field_name,
                'domain_id': domain_id,
                'domain_name': domain_name,
                'keywords': keywords
            }
            
            valid_domains.add(domain_name)
            valid_fields.add(field_name)
            valid_subfields.add(subfield_name)
            valid_topics[topic_name.lower().strip()] = topic_id
    
    return {
        'domains': domains,
        'topics_by_subfield': dict(topics_by_subfield),
        'all_topics': all_topics,
        'subfield_to_field': subfield_to_field,
        'field_to_domain': field_to_domain,
        'valid_domains': valid_domains,
        'valid_fields': valid_fields,
        'valid_subfields': valid_subfields,
        'valid_topics': valid_topics
    }


def get_fields_by_domain(taxonomy):
    """Get fields organized by domain."""
    result = defaultdict(list)
    for d in taxonomy['domains'].values():
        for f in d['fields'].values():
            result[d['name']].append(f['name'])
    return dict(result)


def get_subfields_by_field(taxonomy):
    """Get subfields organized by field."""
    result = defaultdict(set)
    for d in taxonomy['domains'].values():
        for f in d['fields'].values():
            for sf in f['subfields']:
                result[f['name']].add(sf)
    return {k: sorted(v) for k, v in result.items()}


def sample_records(n=500, english_only=True):
    """Randomly sample n records from DataCite files."""
    all_files = list(DATA_DIR.glob("*.ndjson"))
    records = []
    
    random.shuffle(all_files)
    for f in all_files:
        if len(records) >= n * 4:
            break
        try:
            with open(f, 'r') as fp:
                for line in fp:
                    try:
                        rec = json.loads(line.strip())
                        title = rec.get('title', '')
                        subjects = rec.get('subjects', [])
                        
                        # Filter: must have title and subjects
                        if not title or len(title) < 20 or not subjects or len(subjects) < 2:
                            continue
                        
                        # Filter: English only
                        if english_only and not is_english(title):
                            continue
                        
                        records.append(rec)
                    except:
                        continue
        except:
            continue
    
    random.shuffle(records)
    return records[:n]


def classify_stage1(record, taxonomy, fields_by_domain, subfields_by_field):
    """Stage 1: Classify to domain/field/subfield."""
    
    client = get_client()
    title = record.get('title', '')
    subjects = record.get('subjects', [])
    description = record.get('description', '')
    
    text = f"Title: {title}\n"
    if subjects:
        text += f"Subjects: {', '.join(subjects[:15])}\n"
    if description:
        text += f"Description: {description[:400]}\n"
    
    hierarchy = "## OpenAlex Taxonomy Hierarchy\n\n"
    for domain, fields in sorted(fields_by_domain.items()):
        hierarchy += f"**{domain}**:\n"
        for field in sorted(fields):
            sfs = subfields_by_field.get(field, [])
            hierarchy += f"  - {field}: {', '.join(sfs[:10])}"
            if len(sfs) > 10:
                hierarchy += f" (+{len(sfs)-10} more)"
            hierarchy += "\n"
    
    prompt = f"""Classify this research record into the OpenAlex academic taxonomy.

{text}

{hierarchy}

Respond with ONLY valid JSON (no markdown):
{{"domain": "one of: Life Sciences, Social Sciences, Physical Sciences, Health Sciences", "field": "exact field name from above", "subfield": "exact subfield name from above", "confidence": 0.0-1.0}}
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert academic classifier. Respond with ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```"):
            lines = result_text.split('\n')
            result_text = '\n'.join(lines[1:-1] if lines[-1].startswith('```') else lines[1:])
        if result_text.startswith("json"):
            result_text = result_text[4:].strip()
        
        return json.loads(result_text)
        
    except Exception as e:
        return {'error': str(e)}


def classify_stage2(record, stage1_result, taxonomy):
    """Stage 2: Given subfield, pick specific topic."""
    
    client = get_client()
    subfield = stage1_result.get('subfield', '')
    topics = taxonomy['topics_by_subfield'].get(subfield, [])
    
    if not topics:
        for sf_name, sf_topics in taxonomy['topics_by_subfield'].items():
            if subfield.lower() in sf_name.lower() or sf_name.lower() in subfield.lower():
                topics = sf_topics
                subfield = sf_name
                break
    
    if not topics:
        return {'error': f'No topics found for subfield: {subfield}'}
    
    title = record.get('title', '')
    subjects = record.get('subjects', [])
    description = record.get('description', '')
    
    text = f"Title: {title}\n"
    if subjects:
        text += f"Subjects: {', '.join(subjects[:12])}\n"
    if description:
        text += f"Description: {description[:300]}\n"
    
    topics_text = f"## Available Topics in '{subfield}' ({len(topics)} total)\n\n"
    for t in topics[:50]:
        topics_text += f"- **{t['topic_name']}** (ID: {t['topic_id']})"
        if t['keywords']:
            topics_text += f": {t['keywords'][:100]}"
        topics_text += "\n"
    
    if len(topics) > 50:
        topics_text += f"\n... and {len(topics)-50} more topics\n"
    
    prompt = f"""Given this research record and subfield "{subfield}", pick the MOST specific matching topic.

{text}

{topics_text}

Respond with ONLY valid JSON (no markdown):
{{"topic_name": "exact topic name from above", "topic_id": "numeric ID from above", "reasoning": "1 sentence why"}}
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert academic classifier. Pick the single best matching topic. Respond with ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```"):
            lines = result_text.split('\n')
            result_text = '\n'.join(lines[1:-1] if lines[-1].startswith('```') else lines[1:])
        if result_text.startswith("json"):
            result_text = result_text[4:].strip()
        
        return json.loads(result_text)
        
    except Exception as e:
        return {'error': str(e)}


def classify_record(args):
    """Full 2-stage classification for a single record."""
    
    record, taxonomy, fields_by_domain, subfields_by_field, record_idx, total = args
    
    title = record.get('title', '')
    subjects = record.get('subjects', [])
    record_id = record.get('identifiers', [{}])[0].get('identifier', f'record_{record_idx}')
    
    # Stage 1
    stage1 = classify_stage1(record, taxonomy, fields_by_domain, subfields_by_field)
    
    if 'error' in stage1:
        return {
            'record_id': record_id,
            'title': title,
            'subjects': subjects[:10],
            'error': stage1['error']
        }
    
    # Small delay between stages
    time.sleep(0.1)
    
    # Stage 2
    stage2 = classify_stage2(record, stage1, taxonomy)
    
    result = {
        'record_id': record_id,
        'title': title,
        'subjects': subjects[:10],
        'domain': stage1.get('domain', ''),
        'field': stage1.get('field', ''),
        'subfield': stage1.get('subfield', ''),
        'topic_name': stage2.get('topic_name', ''),
        'topic_id': stage2.get('topic_id', ''),
        'confidence': stage1.get('confidence', 0),
        'reasoning': stage2.get('reasoning', '')
    }
    
    if 'error' in stage2:
        result['topic_error'] = stage2['error']
    
    return result


# Global counters for progress
progress_lock = threading.Lock()
progress_count = 0
start_time = None


def process_with_progress(args, output_file, total):
    """Process record and update progress."""
    global progress_count, start_time
    
    result = classify_record(args)
    
    with progress_lock:
        progress_count += 1
        
        # Write result
        with open(output_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        # Progress update every 10 records
        if progress_count % 10 == 0:
            elapsed = time.time() - start_time
            rate = progress_count / elapsed
            eta = (total - progress_count) / rate if rate > 0 else 0
            print(f"Progress: {progress_count}/{total} ({100*progress_count/total:.1f}%) | "
                  f"Rate: {rate:.1f} rec/s | ETA: {eta/60:.1f} min")
    
    return result


def main():
    global progress_count, start_time
    
    parser = argparse.ArgumentParser(description='Parallel LLM ground truth generation')
    parser.add_argument('n_records', type=int, nargs='?', default=10000, help='Number of records')
    parser.add_argument('--workers', type=int, default=15, help='Number of parallel workers')
    parser.add_argument('--english-only', action='store_true', default=True, help='Filter English only')
    args = parser.parse_args()
    
    n_records = args.n_records
    n_workers = args.workers
    
    print(f"Loading OpenAlex taxonomy...")
    taxonomy = load_taxonomy()
    fields_by_domain = get_fields_by_domain(taxonomy)
    subfields_by_field = get_subfields_by_field(taxonomy)
    print(f"  {len(taxonomy['valid_domains'])} domains, {len(taxonomy['all_topics'])} topics")
    
    print(f"\nSampling {n_records} English records from DataCite...")
    records = sample_records(n_records, english_only=True)
    print(f"  Sampled {len(records)} records")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"llm_ground_truth_{len(records)}.jsonl"
    
    # Clear if exists
    if output_file.exists():
        output_file.unlink()
    
    print(f"\nClassifying with {n_workers} parallel workers...")
    print(f"Output: {output_file}")
    print(f"Estimated time: {len(records) / (n_workers * 0.4):.0f} seconds ({len(records) / (n_workers * 0.4) / 60:.1f} min)\n")
    
    # Prepare args for all records
    task_args = [
        (rec, taxonomy, fields_by_domain, subfields_by_field, i, len(records))
        for i, rec in enumerate(records)
    ]
    
    progress_count = 0
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(process_with_progress, args, output_file, len(records))
            for args in task_args
        ]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error: {e}")
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(results)} records in {elapsed:.1f}s ({len(results)/elapsed:.1f} rec/s)")
    print(f"Output: {output_file}")
    
    # Domain distribution
    domain_counts = {}
    errors = 0
    for r in results:
        if 'error' in r:
            errors += 1
        else:
            d = r.get('domain', 'unknown')
            domain_counts[d] = domain_counts.get(d, 0) + 1
    
    print(f"\nDomain distribution:")
    for d, c in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {d}: {c} ({100*c/len(results):.1f}%)")
    print(f"  Errors: {errors}")
    
    # Validation
    print(f"\n{'='*60}")
    print("Validating against taxonomy...")
    
    valid_counts = {'domain': 0, 'field': 0, 'subfield': 0, 'topic': 0}
    n_valid = len(results) - errors
    
    for r in results:
        if 'error' in r:
            continue
        if r.get('domain') in taxonomy['valid_domains']:
            valid_counts['domain'] += 1
        if r.get('field') in taxonomy['valid_fields']:
            valid_counts['field'] += 1
        if r.get('subfield') in taxonomy['valid_subfields']:
            valid_counts['subfield'] += 1
        topic_id = str(r.get('topic_id', ''))
        if topic_id in taxonomy['all_topics']:
            valid_counts['topic'] += 1
    
    for level, count in valid_counts.items():
        print(f"  Valid {level}: {count}/{n_valid} ({100*count/n_valid:.1f}%)")


if __name__ == "__main__":
    main()

