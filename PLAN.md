# Plan: Extract `taxonomise` Package from dsit-taxonomies

## Overview

Create a standalone Python package `taxonomise` that extracts the core taxonomy classification methodology from the dsit-taxonomies Kedro project. The package will be framework-agnostic, supporting any corpus and custom taxonomies.

---

## Package Structure

```
taxonomise/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── src/taxonomise/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                    # Click CLI with rich progress bars
│   ├── config.py                 # PipelineConfig dataclass
│   ├── logging.py                # Logging configuration
│   ├── pipeline.py               # ClassificationPipeline orchestrator
│   ├── embeddings.py             # EmbeddingProvider (SentenceTransformer)
│   ├── similarity.py             # Cosine similarity computation
│   ├── matching.py               # Global, sentence, keyword matching (consolidated)
│   ├── confidence.py             # Binning (global, local, dropoff) + zero-shot
│   ├── keywords.py               # Extractors + consensus aggregation
│   ├── data.py                   # Taxonomy + Corpus models, loaders, preprocessor
│   └── io.py                     # Multi-format readers/writers
│
└── tests/
    ├── fixtures/
    └── unit/, integration/
```

---

## Core Interfaces

### PipelineConfig (config.py)
```python
@dataclass
class PipelineConfig:
    # Matching toggles
    enable_global_matching: bool = True
    enable_sentence_matching: bool = True
    enable_keyword_matching: bool = True
    enable_zeroshot: bool = False

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    zeroshot_model: str = "tasksource/ModernBERT-large-nli"
    zeroshot_hypothesis_template: str = "This research project is about {}."

    # Processing
    batch_size: int = 1000
    top_n_matches: int = 10
    n_jobs: int = 8

    # Score weights (keyword_weight = 1 - sentence_weight - global_weight)
    sentence_weight: float = 0.5
    global_weight: float = 0.3

    # Global binning thresholds (quantiles)
    global_q2_threshold: float = 0.5
    global_q3_threshold: float = 0.75

    # Local binning thresholds (quantiles)
    local_q2_threshold: float = 0.5
    local_q3_threshold: float = 0.75

    # Pruning
    prune_sentence_threshold: float = 0.5    # Quantile threshold
    prune_global_threshold: float = 0.5
    prune_keyword_threshold: float = 0.5
    use_quantile_pruning: bool = True

    # Score normalization
    normalise_by_matches: bool = False  # Weight by num_matching_sentences / num_sentences

    # Keywords
    keyword_extractors: list[str] = field(default_factory=lambda: ["rake", "yake", "keybert", "dbpedia"])
    keyword_min_agreement: int = 2  # Requires > (min_agreement - 1) extractors, i.e., >= min_agreement
```

### ClassificationPipeline (pipeline.py)
```python
class ClassificationPipeline:
    def __init__(self, config: PipelineConfig | None = None): ...
    def classify(self, corpus: Corpus, taxonomy: Taxonomy) -> ClassificationResults: ...
```

---

## CLI Design

Single command: `taxonomise classify`

```bash
taxonomise classify \
    --corpus documents.csv \
    --taxonomy categories.json \
    --output results.parquet \
    --format parquet \
    --id-column doc_id \
    --text-columns "title,abstract" \
    --enable-zeroshot \
    --config pipeline.yaml
```

Key flags:
- `--corpus`, `--taxonomy`, `--output` (required)
- `--format` (parquet|csv|json|jsonl)
- `--id-column`, `--text-columns` (column mapping for corpus)
- `--enable-global/--disable-global` (and similar for sentences, keywords, zeroshot)
- `--embedding-model`, `--zeroshot-model`
- `--config` (YAML config file)
- `--verbose` / `--debug` / `--quiet` (logging verbosity)

---

## Standardized Formats

### Taxonomy Input

Each taxonomy entity **must have an ID**. If not provided, an error is raised.

**CSV format:**
```csv
id,label,parent_id,description
sci,Science,,Top-level category
phys,Physics,sci,Study of matter
qphys,Quantum Physics,phys,Subatomic phenomena
```

Required columns: `id`, `label`
Optional columns: `parent_id` (for hierarchy), `description`

**JSON tree format:**
```json
{
  "id": "sci",
  "label": "Science",
  "description": "...",
  "children": [
    {"id": "phys", "label": "Physics", "children": [...]}
  ]
}
```

Required fields: `id`, `label`
Optional fields: `description`, `children`

### Corpus Input

**CSV/Parquet:** User specifies `--id-column` and `--text-columns`

**JSON/JSONL:**
```json
{"id": "doc1", "text": "Document content..."}
```

### Output Format

All formats contain:
- `document_id`, `taxonomy_label_id`, `taxonomy_label`

**Core scores:**
- `relevance_score` - Max sentence score (used for binning)
- `similarity_score_global` - Document-level similarity
- `similarity_score_sent` - Best sentence similarity
- `similarity_score_key` - Best keyword similarity

**Confidence bins:**
- `global_bin` - Based on global quantile thresholds (high/medium/low)
- `local_bin` - Min of quantile and dropoff methods per document (high/medium/low)
- `confidence_bin` - Min of global_bin and local_bin (high/medium/low)

**Metadata:**
- `num_matching_sentences` - Sentences matching this label
- `num_sentences` - Total sentences in document

**Zero-shot (when enabled):**
- `zeroshot_score` - NLI classification confidence
- `zeroshot_bin` - 5-level scale (very high/high/medium/low/very low)
- `sentence_bin` - Renamed from confidence_bin when zero-shot enabled
- `max_confidence` - Highest of sentence_bin and zeroshot_bin
- `zeroshot_favouring_confidence` - Favors zeroshot when disagreement > 1 level
- `sentence_favouring_confidence` - Favors sentence when disagreement > 1 level

### Local Binning Algorithm

For each document, local bins are computed as the minimum of two methods:

**Quantile-based:**
- Score > q3_local quantile → "high"
- Score > q2_local quantile → "medium"
- Otherwise → "low"

**Dropoff-based (relative gap detection):**
1. Sort scores descending
2. Compute relative gaps: `(score[i-1] - score[i]) / score[i-1]`
3. Find positions of two largest gaps (i*, j*)
4. Scores at positions 0 to i* → "high"
5. Scores at positions i*+1 to j* → "medium"
6. Scores at positions j*+1 onwards → "low"

**Edge cases:**
- 1 label: Always "high"
- 2 labels: First is "high", second is "medium"

---

## Logging & Progress

### Logging (core package)

The package uses Python's standard `logging` module for structured logging:

```python
# src/taxonomise/logging.py
import logging

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the taxonomise prefix."""
    return logging.getLogger(f"taxonomise.{name}")

def configure_logging(level: int = logging.INFO, format: str = None):
    """Configure logging for the package."""
    ...
```

Log levels by component:
- **INFO**: Pipeline stage transitions, summary statistics
- **DEBUG**: Batch processing details, intermediate results
- **WARNING**: Missing optional dependencies, fallback behavior

Example log output:
```
INFO  taxonomise.pipeline: Loading corpus from documents.csv (1523 documents)
INFO  taxonomise.pipeline: Loading taxonomy from categories.json (156 labels)
INFO  taxonomise.embeddings: Embedding 1523 documents with all-MiniLM-L6-v2
DEBUG taxonomise.embeddings: Batch 1/2 complete (1000 documents)
INFO  taxonomise.matching: Running global matching...
INFO  taxonomise.confidence: Assigning confidence bins...
INFO  taxonomise.pipeline: Classification complete. Results: 4521 matches
```

### Progress Bars (CLI)

The CLI uses `rich` for progress bars and status display:

```python
# In cli.py
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console

# Progress display during classification
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("{task.percentage:>3.0f}%"),
) as progress:
    task = progress.add_task("Embedding documents...", total=len(corpus))
    # ... update progress as batches complete
```

CLI verbosity:
- Default: Progress bars only
- `--verbose`: Progress bars + INFO logs
- `--debug`: Progress bars + DEBUG logs
- `--quiet`: No output except errors

---

## Dependencies

**Core (required):**
- numpy, sentence-transformers, spacy, click, pyyaml, rich, joblib

**Optional extras:**
- `[keywords]`: rake-nltk, yake, keybert, requests (for DBpedia)
- `[zeroshot]`: transformers, torch, accelerate
- `[all]`: All optional features

---

## Key Source Files to Extract From

| Component | Source File |
|-----------|-------------|
| Similarity computation | `dsit-taxonomies/src/dsit_taxonomy/pipelines/project_similarity_matching/utils.py` |
| Document preprocessing | `dsit-taxonomies/src/dsit_taxonomy/pipelines/project_similarity_matching/nodes.py` |
| Confidence binning | `dsit-taxonomies/src/dsit_taxonomy/pipelines/project_similarity_refinement/utils.py` |
| Zero-shot validation | `dsit-taxonomies/src/dsit_taxonomy/pipelines/project_similarity_refinement/nodes.py` |
| Keyword extraction | `dsit-taxonomies/src/dsit_taxonomy/pipelines/keyword_processing_gtr/nodes.py` |
| Taxonomy normalization | `dsit-taxonomies/src/dsit_taxonomy/pipelines/data_processing_taxonomies/nodes.py` |
| Embeddings hook | `dsit-taxonomies/src/dsit_taxonomy/hooks.py` |

---

## Implementation Phases

### Phase 1: Foundation
1. Initialize uv project with pyproject.toml
2. Create package structure
3. Implement `logging.py` - Logger configuration
4. Implement `config.py` - PipelineConfig dataclass with YAML loading
5. Implement `data.py` - Taxonomy/Corpus models, loaders (CSV/JSON/Parquet), sentence splitter

### Phase 2: Core Matching
1. Implement `embeddings.py` - SentenceTransformer wrapper with pre-normalization
2. Implement `similarity.py` - Batch cosine similarity (vectorized numpy)
3. Implement `matching.py`:
   - `Matcher` protocol
   - `GlobalMatcher`, `SentenceMatcher`, `KeywordMatcher` classes
   - `combine_scores()` function for weighted combination
   - User can enable/disable each matcher via config

### Phase 3: Keywords
1. Implement `keywords.py`:
   - `KeywordExtractor` protocol
   - `RakeExtractor`, `YakeExtractor`, `KeyBertExtractor`, `DBPediaExtractor`
   - `aggregate_keywords()` - Multi-extractor consensus (n > 1 threshold)
   - KeyBERT-only mode when other extractors unavailable

### Phase 4: Confidence Scoring
1. Implement `confidence.py`:
   - `prune_matches()` - Quantile-based pruning
   - `assign_global_bins()` - Global quantile binning
   - `assign_local_bins()` - Quantile + dropoff binning with edge cases
   - `combine_bins()` - Take minimum of global and local
   - `enhance_with_zeroshot()` - NLI validation with 5-level scale
   - `refine_confidence_bins()` - Combine sentence and zeroshot bins

### Phase 5: Pipeline & CLI
1. Implement `pipeline.py`:
   - `ClassificationPipeline` orchestrator
   - Memory management (torch.cuda.empty_cache() after chunks)
   - Progress callbacks for CLI integration
2. Implement `io.py` - Multi-format readers/writers
3. Implement `cli.py` - Click interface with rich progress bars

### Phase 6: Testing & Documentation
1. Unit tests for each module
2. Integration tests for CLI
3. Property-based tests (score invariants)
4. Regression tests against reference implementation
5. README with examples

---

## Verification Plan

1. **Unit tests**: Each module has corresponding tests
   - Score invariants: values in [0, 1] range
   - Binning monotonicity: higher scores never get lower bins
   - Edge cases: 1-2 label documents

2. **Integration test**: End-to-end CLI test with sample data

3. **Regression tests**: Compare outputs with reference implementation
   - Store expected outputs from dsit-taxonomies for test fixtures
   - Verify score distributions match within tolerance

4. **Performance benchmarks**: Track embedding/matching throughput

5. **Memory profiling**: Ensure large corpus handling doesn't OOM

6. **Manual verification**:
   - Create sample corpus (5-10 documents)
   - Create sample taxonomy (flat and hierarchical)
   - Run: `taxonomise classify --corpus sample.csv --taxonomy tax.json --output out.parquet`
   - Verify output contains all expected columns and confidence bins

---

## Critical Algorithm Details from Reference

### Score Flow (from reference implementation)

1. **Per-sentence weighted score** (`matching/nodes.py:360-365`):
   ```python
   sentence_score = (
       sentence_weight * similarity_score
       + global_weight * similarity_score_global.fillna(0)
       + (1 - sentence_weight - global_weight) * similarity_score_key.fillna(0)
   )
   ```

2. **Aggregation to relevance_score** (`refinement/nodes.py:64`):
   ```python
   "sentence_score": ["max", "count"]  # Takes MAX per document-label
   ```

3. **Global binning** (`refinement/nodes.py:92-96`):
   ```python
   q2 = relevance_score.quantile(global_q2_threshold)
   q3 = relevance_score.quantile(global_q3_threshold)
   global_bin = "high" if x > q3 else ("medium" if x > q2 else "low")
   ```

4. **Local binning** (`refinement/utils.py:120-150`):
   - Compute both quantile bins AND dropoff bins
   - Take minimum of both for final local_bin

5. **Final confidence** (`refinement/utils.py:61-65`):
   ```python
   confidence_bin = min(global_bin, local_bin)  # by bin order
   ```

6. **Zero-shot bins** (`refinement/nodes.py:280-289`):
   ```python
   "very high" if x >= 0.9
   "high" if x >= 0.7
   "medium" if x >= 0.5
   "low" if x >= 0.25
   "very low" otherwise
   ```

7. **Keyword agreement** (`keyword_processing_gtr/nodes.py:346`):
   ```python
   output_df = output_df[output_df["num_annotators"] > 1]  # strictly > 1, so >= 2
   ```
