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
    
    Combines title and subjects/keywords into a single string.
    
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
    
    # Subjects/keywords
    subjects = record.get('subjects') or record.get('keywords') or []
    if subjects:
        if isinstance(subjects, str):
            subjects = [subjects]
        
        # Extract subject text (handle various formats)
        subj_texts = []
        for s in subjects[:10]:  # Limit to 10 subjects
            if isinstance(s, dict):
                subj_text = s.get('subject') or s.get('value') or s.get('name', '')
            else:
                subj_text = str(s)
            if subj_text:
                subj_texts.append(sanitize_text(subj_text))
        
        if subj_texts:
            parts.append(' '.join(subj_texts))
    
    return ' '.join(parts) if parts else '[no metadata]'


def get_dataset_id(record: Dict[str, Any]) -> str:
    """Extract dataset identifier from record."""
    # Try common ID fields
    for field in ['id', 'doi', 'identifier', 'dataset_id']:
        if field in record and record[field]:
            val = record[field]
            if isinstance(val, list):
                val = val[0]
            if isinstance(val, dict):
                val = val.get('identifier') or val.get('value', '')
            return str(val)
    
    return 'unknown'

