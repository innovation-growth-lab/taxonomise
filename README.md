# taxonomise

[![PyPI version](https://badge.fury.io/py/taxonomise.svg)](https://pypi.org/project/taxonomise/)
[![Python versions](https://img.shields.io/pypi/pyversions/taxonomise)](https://pypi.org/project/taxonomise/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Semantic taxonomy classification for document corpora.

`taxonomise` uses embedding-based similarity matching to classify documents against hierarchical taxonomies. It provides a Python API and CLI tool for applying custom taxonomies to any text corpus.

## When to Use taxonomise

**Good fit:**
- Classifying research papers, articles, or documents against topic taxonomies
- Working with hierarchical category systems (e.g., academic fields, product categories)
- Need confidence scores to filter uncertain classifications
- Batch processing of document collections

**Not ideal for:**
- Real-time classification (model loading adds latency)
- Very short texts (tweets, titles alone) - sentence matching needs content
- Taxonomies with 1000+ labels (memory scales with label count)

## Features

- **Multi-level matching**: Global (document), sentence, and keyword-level similarity
- **Confidence scoring**: Quantile-based global and local confidence bins
- **Optional zero-shot validation**: NLI model validation for higher precision
- **Multiple input formats**: CSV, Parquet, JSON, JSONL
- **Flexible taxonomies**: Flat or hierarchical, CSV or JSON tree format
- **Extensible keyword extraction**: RAKE, YAKE, KeyBERT, DBpedia support

## Installation

```bash
# Core installation
pip install taxonomise

# With keyword extraction support
pip install taxonomise[keywords]

# With zero-shot validation support
pip install taxonomise[zeroshot]

# All optional features
pip install taxonomise[all]
```

Or with uv:

```bash
uv add taxonomise
```

## Quick Start

### CLI Usage

```bash
# Basic classification
taxonomise classify \
    --corpus documents.csv \
    --taxonomy categories.json \
    --output results.parquet

# With custom column mapping
taxonomise classify \
    --corpus projects.csv \
    --taxonomy taxonomy.csv \
    --output results.json \
    --format json \
    --id-column project_id \
    --text-columns "title,abstract"

# Disable keyword matching (faster)
taxonomise classify \
    --corpus data.jsonl \
    --taxonomy tax.json \
    --output out.csv \
    --disable-keywords

# Enable zero-shot validation
taxonomise classify \
    --corpus corpus.csv \
    --taxonomy taxonomy.json \
    --output results.parquet \
    --enable-zeroshot
```

### Python API

```python
from taxonomise import ClassificationPipeline, PipelineConfig
from taxonomise.data import load_corpus, load_taxonomy

# Load data
corpus = load_corpus("documents.csv")
taxonomy = load_taxonomy("taxonomy.json")

# Configure pipeline
config = PipelineConfig(
    enable_sentence_matching=True,
    enable_keyword_matching=False,
    embedding_model="all-MiniLM-L6-v2",
)

# Run classification
pipeline = ClassificationPipeline(config)
results = pipeline.classify(corpus, taxonomy)

# Filter high-confidence results
high_confidence = results.filter_by_confidence("high")

# Export
for r in high_confidence:
    print(f"{r.document_id} -> {r.taxonomy_label} ({r.confidence_bin})")
```

## Example Dataset

The `examples/` directory contains a sample dataset for testing and experimentation:

- **`astrophysics_corpus.csv`**: 100 fictional astrophysics research abstracts
- **`astrophysics_taxonomy.json`**: 3-level hierarchical taxonomy with 181 topics
  - 5 top-level categories: Stellar Physics, Cosmology, Planetary Science, Galactic Astrophysics, High-Energy Astrophysics
  - 25 level-2 subcategories
  - 150 level-3 specific topics

Try it out:

```bash
taxonomise classify \
    -c examples/astrophysics_corpus.csv \
    -t examples/astrophysics_taxonomy.json \
    -o results.csv \
    -f csv
```

Or with Python:

```python
from taxonomise import ClassificationPipeline
from taxonomise.data import load_corpus, load_taxonomy

corpus = load_corpus("examples/astrophysics_corpus.csv")
taxonomy = load_taxonomy("examples/astrophysics_taxonomy.json")

pipeline = ClassificationPipeline()
results = pipeline.classify(corpus, taxonomy)

for r in results.filter_by_confidence("high"):
    print(f"{r.document_id} -> {r.taxonomy_label}")
```

## Input Formats

### Taxonomy

Each taxonomy entity **must have an ID**.

**CSV format:**
```csv
id,label,parent_id,description
sci,Science,,Top-level category
phys,Physics,sci,Study of matter
quantum,Quantum Physics,phys,Subatomic phenomena
```

**JSON tree format:**
```json
{
  "id": "sci",
  "label": "Science",
  "children": [
    {
      "id": "phys",
      "label": "Physics",
      "children": [
        {"id": "quantum", "label": "Quantum Physics"}
      ]
    }
  ]
}
```

### Corpus

**CSV/Parquet:** Specify columns with `--id-column` and `--text-columns`

**JSON/JSONL:**
```json
{"id": "doc1", "text": "Document content..."}
```

## Output Format

Results include:

| Field | Description |
|-------|-------------|
| `document_id` | Document identifier |
| `taxonomy_label_id` | Taxonomy label ID |
| `taxonomy_label` | Full taxonomy path |
| `relevance_score` | Max weighted sentence score |
| `confidence_bin` | Combined confidence (high/medium/low) |
| `global_bin` | Global quantile bin |
| `local_bin` | Per-document bin |
| `similarity_score_global` | Document-level similarity |
| `similarity_score_sent` | Best sentence similarity |
| `similarity_score_key` | Best keyword similarity |

With `--enable-zeroshot`, additional fields:
- `zeroshot_score`, `zeroshot_bin`
- `max_confidence`, `zeroshot_favouring_confidence`, `sentence_favouring_confidence`

## Configuration

Configuration can be provided via:
1. CLI flags
2. YAML config file (`--config pipeline.yaml`)
3. Python `PipelineConfig` object

Example config file:

```yaml
# Pipeline stages
enable_global_matching: true
enable_sentence_matching: true
enable_keyword_matching: true
enable_zeroshot: false

# Models
embedding_model: "all-MiniLM-L6-v2"
zeroshot_model: "tasksource/ModernBERT-large-nli"

# Processing
batch_size: 1000
top_n_matches: 10
n_jobs: 8

# Score weights
sentence_weight: 0.5
global_weight: 0.3

# Confidence binning
global_q2_threshold: 0.5
global_q3_threshold: 0.75
```

## How It Works

1. **Embedding generation**: Documents and taxonomy labels are embedded using SentenceTransformers
2. **Multi-level matching**:
   - **Global**: Full document vs taxonomy labels
   - **Sentence**: Each sentence vs taxonomy labels
   - **Keyword**: Consensus keywords vs taxonomy labels
3. **Score combination**: Weighted sum of all matching levels
4. **Confidence binning**: Global quantiles + per-document quantiles and dropoff detection
5. **Optional zero-shot**: NLI model validates classifications

## Development

```bash
# Clone and install
git clone https://github.com/igl/taxonomise.git
cd taxonomise
uv sync --all-extras

# Run tests
uv run pytest

# Run CLI
uv run taxonomise --help
```

## Accuracy

Tested against AI 'expert'-labelled samples (300 projects), we applied the [CWTS Research Topics](https://www.leidenmadtrics.nl/articles/an-open-approach-for-classifying-research-publications) taxonomy to research project abstracts from Gateway to Research, obtaining the following classification results:

| Method            | Precision | Recall | F1    |
|------------------|-----------|--------|-------|
| Zero-shot favoured     | 0.765     | 0.691  | 0.726 |
| Zero-shot        | 0.784     | 0.661  | 0.717 |
| Maximum          | 0.645     | 0.747  | 0.692 |

Note: These metrics show the best performing configuration for each approach on a limited sample in a single context. Your results may vary depending on the corpus and taxonomies used.

## Limitations

- **First run downloads models**: ~90MB for default embedding model
- **Memory usage**: ~2GB RAM per 1,000 documents; GPU helps with large datasets (>10,000 documents)
- **spaCy model required**: Auto-downloads `en_core_web_sm` (~12MB) for sentence splitting
- **English-focused**: Default models optimized for English text
- **Text length**: Works best with detailed descriptions; very short texts may get fewer matches

## Troubleshooting

**Q: "No module named 'en_core_web_sm'"**
A: Run `python -m spacy download en_core_web_sm`

**Q: Classification is slow**
A: Try `--disable-keywords` and `--disable-sentence` for faster (but less accurate) results

**Q: Low confidence scores across all documents**
A: Check that taxonomy labels are descriptive phrases, not just IDs or codes

## License

MIT
