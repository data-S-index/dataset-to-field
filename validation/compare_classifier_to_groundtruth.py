#!/usr/bin/env python3
"""
Compare our embedding classifier against LLM ground truth.

Loads the DeepSeek-generated ground truth and runs our classifier
on the same records, then computes accuracy at each level.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openalex_classifier import TopicClassifier


def is_english(text: str) -> bool:
    """Check if text is predominantly English (ASCII-based)."""
    if not text:
        return False
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    total_letters = sum(1 for c in text if c.isalpha())
    if total_letters == 0:
        return False
    return ascii_letters / total_letters > 0.7


def load_ground_truth(filepath: Path, filter_english: bool = True):
    """Load LLM ground truth, optionally filtering non-English."""
    records = []
    skipped_non_english = 0
    skipped_errors = 0
    
    with open(filepath) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except:
                continue
            
            if 'error' in rec or 'topic_error' in rec:
                skipped_errors += 1
                continue
            
            if filter_english:
                title = rec.get('title', '')
                if not is_english(title):
                    skipped_non_english += 1
                    continue
            
            # Normalize topic_id to string
            rec['topic_id'] = str(rec.get('topic_id', ''))
            records.append(rec)
    
    print(f"Loaded {len(records)} records from ground truth")
    print(f"  Skipped {skipped_non_english} non-English")
    print(f"  Skipped {skipped_errors} with errors")
    
    return records


def run_classifier(ground_truth_records):
    """Run our classifier on the ground truth records."""
    
    print("\nInitializing classifier...")
    classifier = TopicClassifier()
    classifier.initialize()
    
    input_records = []
    for gt in ground_truth_records:
        rec = {
            'title': gt.get('title', ''),
            'subjects': gt.get('subjects', []),
        }
        if gt.get('record_id'):
            rec['identifiers'] = [{'identifier': gt['record_id'], 'identifier_type': 'doi'}]
        input_records.append(rec)
    
    print(f"Classifying {len(input_records)} records...")
    results = classifier.classify_batch(input_records)
    
    return results


def compare_results(ground_truth, classifier_results):
    """Compare ground truth to classifier predictions at each level."""
    
    metrics = {
        'total': len(ground_truth),
        'classified': 0,
        'domain_match': 0,
        'field_match': 0,
        'subfield_match': 0,
        'topic_exact': 0,
    }
    
    # Score distribution
    match_scores = []
    mismatch_scores = []
    
    domain_confusion = defaultdict(lambda: defaultdict(int))
    comparisons = []
    
    for gt, pred in zip(ground_truth, classifier_results):
        topic = pred.get('topic', {})
        if not topic:
            continue
        
        metrics['classified'] += 1
        
        # Extract predictions (with IDs)
        pred_topic_id = str(topic.get('id', ''))
        pred_topic_name = topic.get('name', '')
        pred_score = topic.get('score', 0)
        pred_subfield = pred.get('subfield', {})
        pred_subfield_id = str(pred_subfield.get('id', ''))
        pred_subfield_name = pred_subfield.get('name', '')
        pred_field = pred.get('field', {})
        pred_field_id = str(pred_field.get('id', ''))
        pred_field_name = pred_field.get('name', '')
        pred_domain = pred.get('domain', {})
        pred_domain_id = str(pred_domain.get('id', ''))
        pred_domain_name = pred_domain.get('name', '')
        
        # Ground truth
        gt_domain = gt.get('domain', '')
        gt_field = gt.get('field', '')
        gt_subfield = gt.get('subfield', '')
        gt_topic_id = str(gt.get('topic_id', ''))
        gt_topic_name = gt.get('topic_name', '')
        
        # Compare
        domain_match = gt_domain == pred_domain_name
        field_match = gt_field == pred_field_name
        subfield_match = gt_subfield == pred_subfield_name
        topic_match = gt_topic_id == pred_topic_id
        
        if domain_match:
            metrics['domain_match'] += 1
        if field_match:
            metrics['field_match'] += 1
        if subfield_match:
            metrics['subfield_match'] += 1
        if topic_match:
            metrics['topic_exact'] += 1
            match_scores.append(pred_score)
        else:
            mismatch_scores.append(pred_score)
        
        domain_confusion[gt_domain][pred_domain_name] += 1
        
        comparisons.append({
            'record_id': gt.get('record_id'),
            'title': gt.get('title', '')[:80],
            # Ground truth with IDs
            'gt_domain': gt_domain,
            'gt_field': gt_field,
            'gt_subfield': gt_subfield,
            'gt_topic_id': gt_topic_id,
            'gt_topic_name': gt_topic_name,
            # Predictions with IDs
            'pred_domain': pred_domain_name,
            'pred_domain_id': pred_domain_id,
            'pred_field': pred_field_name,
            'pred_field_id': pred_field_id,
            'pred_subfield': pred_subfield_name,
            'pred_subfield_id': pred_subfield_id,
            'pred_topic_id': pred_topic_id,
            'pred_topic_name': pred_topic_name,
            'score': pred_score,
            # Match flags
            'domain_match': domain_match,
            'field_match': field_match,
            'subfield_match': subfield_match,
            'topic_match': topic_match,
        })
    
    # Score statistics
    score_stats = {
        'match_avg': sum(match_scores) / len(match_scores) if match_scores else 0,
        'match_min': min(match_scores) if match_scores else 0,
        'mismatch_avg': sum(mismatch_scores) / len(mismatch_scores) if mismatch_scores else 0,
        'mismatch_min': min(mismatch_scores) if mismatch_scores else 0,
    }
    
    return metrics, domain_confusion, comparisons, score_stats


def print_results(metrics, domain_confusion, comparisons, score_stats):
    """Print comparison results with IDs."""
    
    total = metrics['total']
    n = metrics['classified']
    
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS: Embedding Classifier vs LLM Ground Truth")
    print("=" * 80)
    
    print(f"\nTotal records: {total}")
    print(f"Classified: {n} ({100*n/total:.1f}%)")
    
    print(f"\n{'Level':<20} {'Matches':<15} {'Accuracy':<15}")
    print("-" * 50)
    
    if n > 0:
        levels = [
            ('Domain', metrics['domain_match']),
            ('Field', metrics['field_match']),
            ('Subfield', metrics['subfield_match']),
            ('Topic (exact ID)', metrics['topic_exact']),
        ]
        
        for name, count in levels:
            print(f"{name:<20} {count}/{n:<12} {100*count/n:.1f}%")
    
    # Score analysis
    print(f"\n{'-'*50}")
    print("Score Analysis:")
    print(f"  Exact topic matches:    avg={score_stats['match_avg']:.3f}, min={score_stats['match_min']:.3f}")
    print(f"  Topic mismatches:       avg={score_stats['mismatch_avg']:.3f}, min={score_stats['mismatch_min']:.3f}")
    
    # Domain confusion
    print(f"\n{'-'*50}")
    print("Domain Confusion Matrix:")
    domains = sorted(set(domain_confusion.keys()) | 
                    set(d for conf in domain_confusion.values() for d in conf.keys()))
    
    print(f"\n{'GT \\ Pred':<20}", end="")
    for d in domains:
        print(f"{d[:12]:<14}", end="")
    print()
    
    for gt_d in domains:
        print(f"{gt_d:<20}", end="")
        for pred_d in domains:
            count = domain_confusion[gt_d][pred_d]
            print(f"{count:<14}", end="")
        print()
    
    # Side-by-side comparisons with IDs
    print(f"\n{'='*80}")
    print("SIDE-BY-SIDE COMPARISONS (with Topic IDs)")
    print("="*80)
    
    # Show exact matches
    matches = [c for c in comparisons if c['topic_match']][:5]
    print(f"\n✅ EXACT TOPIC MATCHES ({len([c for c in comparisons if c['topic_match']])} total):")
    for m in matches:
        print(f"\n  📄 {m['title'][:70]}...")
        print(f"     LLM:  ID={m['gt_topic_id']:>5} \"{m['gt_topic_name']}\"")
        print(f"     Emb:  ID={m['pred_topic_id']:>5} \"{m['pred_topic_name']}\" (score: {m['score']:.3f})")
    
    # Show field matches but topic mismatches
    field_matches = [c for c in comparisons if c['field_match'] and not c['topic_match']][:5]
    print(f"\n🔶 SAME FIELD, DIFFERENT TOPIC ({len([c for c in comparisons if c['field_match'] and not c['topic_match']])} total):")
    for m in field_matches:
        print(f"\n  📄 {m['title'][:70]}...")
        print(f"     LLM:  ID={m['gt_topic_id']:>5} \"{m['gt_topic_name']}\"")
        print(f"     Emb:  ID={m['pred_topic_id']:>5} \"{m['pred_topic_name']}\" (score: {m['score']:.3f})")
        print(f"     Field: {m['gt_field']}")
    
    # Show domain mismatches
    domain_mismatches = [c for c in comparisons if not c['domain_match']][:5]
    print(f"\n❌ DOMAIN MISMATCHES ({len([c for c in comparisons if not c['domain_match']])} total):")
    for m in domain_mismatches:
        print(f"\n  📄 {m['title'][:70]}...")
        print(f"     LLM:  {m['gt_domain']} → {m['gt_field']} → ID={m['gt_topic_id']} \"{m['gt_topic_name'][:40]}\"")
        print(f"     Emb:  {m['pred_domain']} → {m['pred_field']} → ID={m['pred_topic_id']} \"{m['pred_topic_name'][:40]}\" (score: {m['score']:.3f})")
    
    return {
        'total': total,
        'classified': n,
        'domain_accuracy': metrics['domain_match'] / n if n else 0,
        'field_accuracy': metrics['field_match'] / n if n else 0,
        'subfield_accuracy': metrics['subfield_match'] / n if n else 0,
        'topic_accuracy': metrics['topic_exact'] / n if n else 0,
        'score_stats': score_stats,
    }


def main():
    """Main comparison function."""
    
    # Try validation set first, fall back to raw files
    gt_file = Path(__file__).parent / "llm_ground_truth" / "validation_set_500.jsonl"
    if not gt_file.exists():
        gt_file = Path(__file__).parent / "llm_ground_truth" / "llm_ground_truth_500.jsonl"
    
    if not gt_file.exists():
        print(f"Ground truth file not found: {gt_file}")
        return
    
    # Load ground truth
    ground_truth = load_ground_truth(gt_file, filter_english=True)
    
    if len(ground_truth) < 10:
        print("Not enough records for validation")
        return
    
    # Run classifier
    predictions = run_classifier(ground_truth)
    
    # Compare
    metrics, domain_conf, comparisons, score_stats = compare_results(ground_truth, predictions)
    
    # Print results
    summary = print_results(metrics, domain_conf, comparisons, score_stats)
    
    # Save detailed results
    output_file = Path(__file__).parent / "llm_ground_truth" / "comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': summary,
            'comparisons': comparisons
        }, f, indent=2)
    print(f"\n\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
