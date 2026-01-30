#!/usr/bin/env python3
"""
Generate ground truth classifications using DeepSeek API.
Sends DataCite records to LLM with OpenAlex taxonomy for classification.
Two-stage approach: domain/field/subfield → specific topic
"""

import json
import csv
import random
import time
import os
from pathlib import Path
from openai import OpenAI
from collections import defaultdict

# Configuration
DEEPSEEK_API_KEY = "sk-6185ef64c68d473d984963356ab0378e"
DATA_DIR = Path("/home/joneill/s_index_fast/slim-records/datacite-slim-records")
TOPICS_FILE = Path("/home/joneill/s_index_fast/openalex-topic-classifier/models/topics.csv")
OUTPUT_DIR = Path("/home/joneill/s_index_fast/openalex-topic-classifier/validation/llm_ground_truth")

# Initialize DeepSeek client (OpenAI-compatible API)
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

def load_taxonomy():
    """Load OpenAlex taxonomy with full hierarchy."""
    
    # Hierarchical structures
    domains = {}  # domain_id -> {name, fields: {field_id -> {name, subfields}}}
    topics_by_subfield = defaultdict(list)  # subfield_name -> [{topic_id, topic_name, keywords}]
    all_topics = {}  # topic_id -> full info
    subfield_to_field = {}  # subfield_name -> field_name
    field_to_domain = {}  # field_name -> domain_name
    
    # For validation
    valid_domains = set()
    valid_fields = set()
    valid_subfields = set()
    valid_topics = {}  # topic_name.lower() -> topic_id
    
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
            
            # Build hierarchy
            if domain_id not in domains:
                domains[domain_id] = {'name': domain_name, 'fields': {}}
            if field_id not in domains[domain_id]['fields']:
                domains[domain_id]['fields'][field_id] = {'name': field_name, 'subfields': set()}
            domains[domain_id]['fields'][field_id]['subfields'].add(subfield_name)
            
            # Track mappings
            subfield_to_field[subfield_name] = field_name
            field_to_domain[field_name] = domain_name
            
            # Topics by subfield for stage 2
            topics_by_subfield[subfield_name].append({
                'topic_id': topic_id,
                'topic_name': topic_name,
                'keywords': keywords
            })
            
            # Full topic info
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
            
            # Validation sets
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

def sample_records(n=500):
    """Randomly sample n records from DataCite files."""
    all_files = list(DATA_DIR.glob("*.ndjson"))
    records = []
    
    random.shuffle(all_files)
    for f in all_files:
        if len(records) >= n * 3:
            break
        try:
            with open(f, 'r') as fp:
                for line in fp:
                    try:
                        rec = json.loads(line.strip())
                        title = rec.get('title', '')
                        subjects = rec.get('subjects', [])
                        if title and len(title) > 20 and subjects and len(subjects) >= 2:
                            records.append(rec)
                    except:
                        continue
        except:
            continue
    
    random.shuffle(records)
    return records[:n]

def classify_stage1(record, taxonomy):
    """Stage 1: Classify to domain/field/subfield."""
    
    title = record.get('title', '')
    subjects = record.get('subjects', [])
    description = record.get('description', '')
    
    # Get hierarchy for prompt
    fields_by_domain = get_fields_by_domain(taxonomy)
    subfields_by_field = get_subfields_by_field(taxonomy)
    
    text = f"Title: {title}\n"
    if subjects:
        text += f"Subjects: {', '.join(subjects[:15])}\n"
    if description:
        text += f"Description: {description[:400]}\n"
    
    # Build domain/field/subfield hierarchy for prompt
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
    
    subfield = stage1_result.get('subfield', '')
    topics = taxonomy['topics_by_subfield'].get(subfield, [])
    
    if not topics:
        # Try fuzzy match on subfield
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
    
    # List topics with keywords
    topics_text = f"## Available Topics in '{subfield}' ({len(topics)} total)\n\n"
    for t in topics[:50]:  # Limit to 50 to fit in context
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

def classify_record(record, taxonomy, record_idx, total):
    """Full 2-stage classification."""
    
    title = record.get('title', '')
    subjects = record.get('subjects', [])
    record_id = record.get('identifiers', [{}])[0].get('identifier', f'record_{record_idx}')
    
    # Stage 1: domain/field/subfield
    stage1 = classify_stage1(record, taxonomy)
    time.sleep(0.3)  # Rate limit
    
    if 'error' in stage1:
        print(f"[{record_idx+1}/{total}] STAGE1 ERROR: {stage1['error']}")
        return {
            'record_id': record_id,
            'title': title,
            'subjects': subjects[:10],
            'error': stage1['error']
        }
    
    # Stage 2: specific topic
    stage2 = classify_stage2(record, stage1, taxonomy)
    time.sleep(0.3)  # Rate limit
    
    # Combine results
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
    
    print(f"[{record_idx+1}/{total}] {result['domain'][:15]:15} | {result['field'][:25]:25} | {result.get('topic_name', 'N/A')[:30]} - {title[:40]}...")
    
    return result

def validate_results(results, taxonomy):
    """Validate all outputs against actual taxonomy."""
    
    validation = {
        'total': len(results),
        'errors': 0,
        'valid_domain': 0,
        'valid_field': 0,
        'valid_subfield': 0,
        'valid_topic': 0,
        'invalid_domains': [],
        'invalid_fields': [],
        'invalid_subfields': [],
        'invalid_topics': []
    }
    
    for r in results:
        if 'error' in r:
            validation['errors'] += 1
            continue
        
        # Check domain
        if r.get('domain') in taxonomy['valid_domains']:
            validation['valid_domain'] += 1
        else:
            validation['invalid_domains'].append(r.get('domain'))
        
        # Check field
        if r.get('field') in taxonomy['valid_fields']:
            validation['valid_field'] += 1
        else:
            validation['invalid_fields'].append(r.get('field'))
        
        # Check subfield
        if r.get('subfield') in taxonomy['valid_subfields']:
            validation['valid_subfield'] += 1
        else:
            validation['invalid_subfields'].append(r.get('subfield'))
        
        # Check topic (by ID or name)
        topic_id = str(r.get('topic_id', ''))
        topic_name = r.get('topic_name', '').lower().strip()
        
        if topic_id in taxonomy['all_topics']:
            validation['valid_topic'] += 1
        elif topic_name in taxonomy['valid_topics']:
            validation['valid_topic'] += 1
        else:
            validation['invalid_topics'].append(r.get('topic_name'))
    
    return validation

def main(n_records=500):
    """Main function."""
    
    print(f"Loading OpenAlex taxonomy...")
    taxonomy = load_taxonomy()
    print(f"  {len(taxonomy['valid_domains'])} domains")
    print(f"  {len(taxonomy['valid_fields'])} fields")
    print(f"  {len(taxonomy['valid_subfields'])} subfields")
    print(f"  {len(taxonomy['all_topics'])} topics")
    
    print(f"\nSampling {n_records} records from DataCite...")
    records = sample_records(n_records)
    print(f"  Sampled {len(records)} records")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"llm_ground_truth_{len(records)}.jsonl"
    
    # Clear if exists
    if output_file.exists():
        output_file.unlink()
    
    print(f"\nClassifying with 2-stage DeepSeek approach...")
    print(f"Output: {output_file}\n")
    print(f"{'IDX':>6} | {'DOMAIN':15} | {'FIELD':25} | {'TOPIC':30} | TITLE")
    print("-" * 100)
    
    results = []
    for i, rec in enumerate(records):
        result = classify_record(rec, taxonomy, i, len(records))
        results.append(result)
        
        # Save incrementally
        with open(output_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
    
    # Summary
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(results)} records classified")
    print(f"Output: {output_file}")
    
    # Domain distribution
    domain_counts = {}
    for r in results:
        if 'error' not in r:
            d = r.get('domain', 'unknown')
            domain_counts[d] = domain_counts.get(d, 0) + 1
    
    print(f"\nDomain distribution:")
    for d, c in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = 100*c/len(results) if results else 0
        print(f"  {d}: {c} ({pct:.1f}%)")
    
    # Validation
    print(f"\n{'='*60}")
    print("VALIDATION: Checking for hallucinations...")
    validation = validate_results(results, taxonomy)
    
    n_success = validation['total'] - validation['errors']
    if n_success > 0:
        print(f"  Total: {validation['total']}, Errors: {validation['errors']}")
        print(f"  Valid domains:   {validation['valid_domain']}/{n_success} ({100*validation['valid_domain']/n_success:.1f}%)")
        print(f"  Valid fields:    {validation['valid_field']}/{n_success} ({100*validation['valid_field']/n_success:.1f}%)")
        print(f"  Valid subfields: {validation['valid_subfield']}/{n_success} ({100*validation['valid_subfield']/n_success:.1f}%)")
        print(f"  Valid topics:    {validation['valid_topic']}/{n_success} ({100*validation['valid_topic']/n_success:.1f}%)")
    
    if validation['invalid_topics']:
        print(f"\n  Sample invalid topics:")
        for t in list(set(validation['invalid_topics']))[:10]:
            print(f"    - '{t}'")
    
    # Save validation
    val_file = OUTPUT_DIR / f"validation_{len(records)}.json"
    with open(val_file, 'w') as f:
        json.dump(validation, f, indent=2)
    print(f"\nValidation saved: {val_file}")

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    main(n_records=n)
