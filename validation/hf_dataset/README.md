---
language: en
license: apache-2.0
task_categories:
  - text-classification
task_ids:
  - topic-classification
tags:
  - openalex
  - scientific-classification
  - topic-classification
  - s-index
  - datacite
size_categories:
  - 10K<n<100K
---

# OpenAlex Topic Classification Dataset

A dataset of scientific records classified according to the [OpenAlex taxonomy](https://docs.openalex.org/api-entities/topics) for training topic classifiers. Created for the **S-Index Challenge**.

## Dataset Description

This dataset contains 10,500 scientific records (titles, subjects, descriptions) with ground truth topic classifications aligned with the OpenAlex taxonomy hierarchy:

- **Domain** (4 categories): Physical Sciences, Life Sciences, Social Sciences, Health Sciences
- **Field** (~26 categories): Chemistry, Medicine, Computer Science, etc.
- **Subfield** (~250 categories): More specific research areas
- **Topic** (4,516 categories): Granular research topics with numeric IDs

## Dataset Structure

Each record contains:

| Field | Type | Description |
|-------|------|-------------|
| `record_id` | string | DOI or unique identifier |
| `title` | string | Title of the scientific record |
| `subjects` | list[string] | Subject keywords |
| `description` | string | Abstract or description (truncated) |
| `domain` | string | OpenAlex domain classification |
| `field` | string | OpenAlex field classification |
| `subfield` | string | OpenAlex subfield classification |
| `topic_name` | string | OpenAlex topic name |
| `topic_id` | string | OpenAlex topic ID (numeric) |

## Domain Distribution

| Domain | Count | Percentage |
|--------|-------|------------|
| Physical Sciences | 5,680 | 54.1% |
| Life Sciences | 2,882 | 27.4% |
| Social Sciences | 1,116 | 10.6% |
| Health Sciences | 822 | 7.8% |

## Topic Coverage

- **Unique Topics**: 1,471 / 4,516 (32.6%)
- **Topics with 1 example**: 694
- **Topics with 3+ examples**: 514

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("jimnoneill/openalex-topic-classification-10k")

# Access training data
for example in dataset["train"]:
    print(f"Title: {example['title']}")
    print(f"Topic: {example['topic_name']} (ID: {example['topic_id']})")
    print(f"Domain: {example['domain']}")
    break
```

## Training a Classifier

```python
from model2vec import StaticModel
from model2vec.train import StaticModelForClassification

# Load dataset
texts = [f"{r['title']}. {', '.join(r['subjects'][:10])}" for r in dataset["train"]]
labels = [r['topic_id'] for r in dataset["train"]]

# Initialize and train
classifier = StaticModelForClassification.from_pretrained(
    model_name="minishlab/potion-base-32m",
    out_dim=len(set(labels))
)
classifier.fit(texts, labels, max_epochs=30)
```

## Source

Records sampled from DataCite with ground truth classifications derived from the OpenAlex taxonomy. English-language records only.

## Related Resources

- [OpenAlex API](https://docs.openalex.org/)
- [OpenAlex Topics Documentation](https://docs.openalex.org/api-entities/topics)
- [Fine-tuned Model](https://huggingface.co/jimnoneill/openalex-topic-classifier)
- [S-Index Project](https://github.com/jamesoneill12/openalex-topic-classifier)

## Citation

```bibtex
@misc{oneill2026openalex-dataset,
  author = {O'Neill, James},
  title = {OpenAlex Topic Classification Dataset},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/jimnoneill/openalex-topic-classification-10k}
}
```

## License

Apache 2.0

