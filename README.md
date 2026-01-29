# OpenAlex Topic Classifier

**Lightweight CPU-based topic classification for scientific datasets using OpenAlex taxonomy.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This classifier assigns [OpenAlex topics](https://docs.openalex.org/api-entities/topics) to scientific dataset metadata using semantic embedding similarity. It maps datasets to the 4,516 topics in the OpenAlex taxonomy, along with their hierarchical subfield, field, and domain classifications.

**Key features:**
- 🚀 **~3,000 records/second** on CPU (no GPU required)
- 📊 **4,516 OpenAlex topics** with full hierarchy
- 🎯 **94.6% classification rate** above 0.50 confidence threshold
- 💻 **Single dependency install** — works on any machine

## Quick Start

```bash
# Install directly from GitHub
pip install git+https://github.com/jimnoneill/openalex-topic-classifier.git

# Or clone and install locally
git clone https://github.com/jimnoneill/openalex-topic-classifier.git
cd openalex-topic-classifier
pip install -e .
```

```python
# Classify a single record
from openalex_classifier import TopicClassifier

classifier = TopicClassifier()
classifier.initialize()

result = classifier.classify({
    "title": "Climate change impact on marine ecosystems",
    "subjects": ["Climate", "Marine Biology", "Ecology"]
})

print(result)
# {
#   'topic': {'id': 1234, 'name': 'Marine Ecology and Climate Change', 'score': 0.72},
#   'subfield': {'id': 23, 'name': 'Ecology'},
#   'field': {'id': 5, 'name': 'Environmental Science'},
#   'domain': {'id': 2, 'name': 'Life Sciences'}
# }
```

## Installation

```bash
# Option 1: Install directly from GitHub (recommended)
pip install git+https://github.com/jimnoneill/openalex-topic-classifier.git

# Option 2: Clone and install locally
git clone https://github.com/jimnoneill/openalex-topic-classifier.git
cd openalex-topic-classifier
pip install -e .

# Models are downloaded automatically on first run
```

### Requirements
- Python 3.10+
- ~500MB disk space for models
- No GPU required

## Usage

### Python API

```python
from openalex_classifier import TopicClassifier

# Initialize (loads model and topic embeddings)
classifier = TopicClassifier()
classifier.initialize()

# Single record
result = classifier.classify(record)

# Batch processing (faster)
results = classifier.classify_batch(records)
```

### Command Line

```bash
# Classify NDJSON file
python -m openalex_classifier.cli input.ndjson output.ndjson

# With progress bar
python -m openalex_classifier.cli input.ndjson output.ndjson --progress
```

### Input Format

Records should have at minimum a `title` field. Additional fields improve classification:

```json
{
  "id": "10.5281/zenodo.123456",
  "title": "Dataset of marine temperature measurements",
  "subjects": ["Oceanography", "Climate Science"],
  "description": "Temperature readings from Pacific Ocean buoys..."
}
```

### Output Format

```json
{
  "dataset_id": "10.5281/zenodo.123456",
  "topic": {
    "id": 1234,
    "name": "Ocean Temperature and Climate Variability",
    "score": 0.68
  },
  "subfield": {"id": 23, "name": "Oceanography"},
  "field": {"id": 5, "name": "Earth and Planetary Sciences"},
  "domain": {"id": 2, "name": "Physical Sciences"}
}
```

## Performance

| Metric | Value |
|--------|-------|
| Throughput | ~3,000 records/sec |
| Mean confidence score | 0.62 |
| Records above 0.50 threshold | 94.6% |
| Classification rate | 99.8% |
| 50M records processing time | ~4.6 hours |

Tested on 32-core AMD Threadripper with real DataCite metadata. Performance scales linearly with CPU cores.

## Method

1. **Text Extraction**: Concatenate title + subjects/keywords from metadata
2. **Semantic Embedding**: Distilled BGE-small model (Model2Vec compression)
3. **Topic Matching**: Cosine similarity against 4,516 pre-embedded topics
4. **Hierarchical Output**: Return topic → subfield → field → domain

The distilled model achieves 91% of the full transformer's quality at 10x the speed.

## OpenAlex Topic Hierarchy

```
Domain (4)
└── Field (26)
    └── Subfield (254)
        └── Topic (4,516)
```

Example:
- **Domain**: Physical Sciences
- **Field**: Computer Science  
- **Subfield**: Artificial Intelligence
- **Topic**: Natural Language Processing

## Citation

If you use this classifier, please cite:

```bibtex
@software{openalex_topic_classifier,
  author = {O'Neill, James and Patel, Bhavesh},
  title = {OpenAlex Topic Classifier},
  year = {2026},
  url = {https://github.com/jimnoneill/openalex-topic-classifier}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [OpenAlex](https://openalex.org/) for the topic taxonomy
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) base embedding model
- [Model2Vec](https://github.com/MinishLab/model2vec) for model distillation

