"""Text preprocessing utilities for topic classification."""

import re
from typing import Any, Dict, List, Optional

# Compiled regex for efficiency
_WHITESPACE_RE = re.compile(r'\s+')


def sanitize_text(text: Optional[str]) -> str:
    """
    Clean and normalize text for embedding.
    
    - Handles None/empty values
    - Normalizes whitespace
    - Strips leading/trailing whitespace
    """
    if not text:
        return ""
    
    if not isinstance(text, str):
        text = str(text)
    
    # Normalize whitespace
    text = _WHITESPACE_RE.sub(' ', text)
    
    return text.strip()


def prepare_record_text(record: Dict[str, Any]) -> str:
    """
    Extract and prepare text from a dataset record for classification.
    
    Combines title, description/abstract, and subjects/keywords.
    
    Args:
        record: Dataset metadata dictionary
        
    Returns:
        Combined text string for embedding
    """
    parts = []
    
    # Title (required)
    title = record.get('title') or record.get('titles')
    if title:
        if isinstance(title, list):
            title = title[0] if title else ''
            if isinstance(title, dict):
                title = title.get('title', '')
        parts.append(sanitize_text(title))
    
    # Description/Abstract - KEY for accurate classification
    description = (
        record.get('description') or 
        record.get('abstract') or 
        record.get('abstract_inverted_index') or  # OpenAlex format
        ''
    )
    if description:
        # Handle inverted index format from OpenAlex
        if isinstance(description, dict):
            # Reconstruct text from inverted index
            try:
                words = sorted(description.items(), key=lambda x: min(x[1]) if x[1] else 0)
                description = ' '.join(w[0] for w in words)
            except:
                description = ''
        elif isinstance(description, list):
            description = ' '.join(str(d) for d in description)
        
        # Truncate to 1000 chars for better context
        desc_text = sanitize_text(str(description))[:1000]
        if desc_text:
            parts.append(desc_text)
    
    # Subjects/keywords
    subjects = record.get('subjects') or record.get('keywords') or []
    if subjects:
        if isinstance(subjects, str):
            subjects = [subjects]
        
        # Extract subject text (handle various formats)
        subj_texts = []
        for s in subjects[:25]:  # Increased to 25 for more context
            if isinstance(s, dict):
                subj_text = (
                    s.get('subject') or 
                    s.get('value') or 
                    s.get('name') or 
                    s.get('display_name', '')
                )
            else:
                subj_text = str(s)
            if subj_text:
                subj_texts.append(sanitize_text(subj_text))
        
        if subj_texts:
            parts.append(' '.join(subj_texts))
    
    return ' '.join(parts) if parts else '[no metadata]'


def get_dataset_id(record: Dict[str, Any]) -> str:
    """Extract dataset identifier from record."""
    # Try simple ID fields first
    for field in ['id', 'doi', 'identifier', 'dataset_id']:
        if field in record and record[field]:
            val = record[field]
            if isinstance(val, list):
                val = val[0]
            if isinstance(val, dict):
                val = val.get('identifier') or val.get('value', '')
            return str(val)
    
    # Handle DataCite identifiers array format
    identifiers = record.get('identifiers', [])
    if identifiers:
        for ident in identifiers:
            if isinstance(ident, dict):
                # Prefer DOI
                if ident.get('identifier_type', '').lower() == 'doi':
                    return str(ident.get('identifier', ''))
        # Fall back to first identifier
        if isinstance(identifiers[0], dict):
            return str(identifiers[0].get('identifier', 'unknown'))
        return str(identifiers[0])
    
    return 'unknown'

