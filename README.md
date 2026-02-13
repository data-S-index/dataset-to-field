# Dataset Research Field Classifier

**Fine-tuned CPU-based classification of scientific datasets into research fields using OpenAlex topics taxonomy.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/jimnoneill/dataset-to-field)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗-Dataset-blue)](https://huggingface.co/datasets/jimnoneill/dataset-to-field-training-10k)

## Overview

This classifier assigns a research field to scientific datasets based on their metadata using a fine-tuned embedding model. It maps datasets to the 4,516 topics in the [OpenAlex taxonomy](https://docs.openalex.org/api-entities/topics), along with their hierarchical subfield, field, and domain classifications. This was developed as part of our NIH S-index Challenge Phase 2 proposal. We refer to the [S-index Hub](https://github.com/data-S-index/hub) for more information about our S-index and the Challenge.

**Key features:**
- 🚀 **~48,000 records/second** on CPU with parallel processing (no GPU required)
- 📊 **4,516 OpenAlex topics** with full hierarchy
- 🎯 **92.6% domain accuracy, 62.6% topic accuracy** (fine-tuned model)
- 💻 **Single dependency install** — works on any machine

## Quick Start

```bash
# Clone the repository (requires Git LFS for model files)
git lfs install
git clone https://github.com/data-S-index/dataset-to-field.git
cd dataset-to-field
git lfs pull  # Download model files (~140 MB)

# Install
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

This repository uses **Git LFS** to store model files. Make sure Git LFS is installed before cloning.

```bash
# 1. Install Git LFS (if not already installed)
git lfs install

# 2. Clone the repository
git clone https://github.com/data-S-index/dataset-to-field.git
cd dataset-to-field

# 3. Pull model files (~140 MB)
git lfs pull

# 4. Install the package
pip install -e .
```

### Requirements
- Python 3.10+
- Git LFS (for model files)
- ~200MB disk space for models
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
| Throughput | **48,304 records/sec** (32 workers) |
| Total Records Classified | 48.7 million |
| Processing Time | ~15 minutes |
| Compressed Output | 394 MB |

Tested on 32-core AMD Threadripper with real DataCite metadata. Performance scales linearly with CPU cores.

### Pre-classified Output

The complete classified DataCite dataset (46.2M records) is available for download:
- [**classified_output.zip**](https://drive.google.com/file/d/1QJ3AYBzKOnCRTMCd0V-8HLkjGRko99Q5/view?usp=sharing) (394 MB)

## Validation Results

The fine-tuned model was validated against a held-out test set of 1,525 records with ground truth OpenAlex classifications:

| Level | Accuracy | Description |
|-------|----------|-------------|
| **Domain** | **92.6%** | 4 domains: Physical Sciences, Life Sciences, Social Sciences, Health Sciences |
| **Field** | **85.8%** | ~26 fields: Chemistry, Medicine, Computer Science, etc. |
| **Subfield** | **73.6%** | ~250 subfields: more specific research areas |
| **Topic** | **62.6%** | 4,516 topics: granular research topics |

### Model Comparison

| Model | Domain | Field | Subfield | Topic |
|-------|--------|-------|----------|-------|
| Base (potion-32m) | 77.2% | 60.5% | 27.9% | 16.2% |
| **Fine-tuned** | **92.6%** | **85.8%** | **73.6%** | **62.6%** |

The fine-tuned model achieves +15% improvement on domain accuracy and +46% improvement on exact topic matching compared to the base embedding model.

### Training Data

The model was fine-tuned on 10,500 scientific records with ground truth topic classifications aligned with the OpenAlex taxonomy.

- **Dataset**: [jimnoneill/dataset-to-field-training-10k](https://huggingface.co/datasets/jimnoneill/dataset-to-field-training-10k)
- **Model**: [jimnoneill/dataset-to-field](https://huggingface.co/jimnoneill/dataset-to-field)

## Method

1. **Text Extraction**: Concatenate title + subjects/keywords + description from metadata
2. **Semantic Embedding**: Fine-tuned Model2Vec static embeddings
3. **Topic Matching**: Cosine similarity against 4,516 pre-embedded topics
4. **Hierarchical Output**: Return topic → subfield → field → domain

## Topic Hierarchy

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
@software{dataset-to-field,
  author = {O'Neill, James, Patel, Bhavesh},
  title = {Dataset Research Field Classifier},
  year = {2026},
  url = {https://github.com/data-S-index/dataset-to-field}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [OpenAlex](https://openalex.org/) for the topic taxonomy
- [minishlab/potion-base-32m](https://huggingface.co/minishlab/potion-base-32m) base embedding model
- [Model2Vec](https://github.com/MinishLab/model2vec) for model distillation and training
