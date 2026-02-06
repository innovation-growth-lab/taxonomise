# taxonomise

A Python package for semantic taxonomy classification of document corpora.

## Project Structure

- `src/taxonomise/` - Main package source
- `tests/` - Test suite
- `dsit-taxonomies/` - Reference implementation (read-only, do not modify)

## Development

```bash
uv sync                    # Install dependencies
uv run pytest              # Run tests
uv run taxonomise --help   # Run CLI
```

## Key Design Decisions

1. **Protocol-based interfaces** - Use `typing.Protocol` for Matcher, KeywordExtractor, LLMProvider
2. **NumPy arrays internally** - Use numpy for batch operations, plain Python for data structures
3. **Optional dependencies** - Core package lightweight, extras for keywords/zeroshot/llm
4. **Configuration hierarchy** - Defaults < config file < CLI flags
5. **Structured logging** - Use Python logging module with `taxonomise.*` loggers
6. **Taxonomy IDs required** - All taxonomy entities must have an explicit ID field

## Core Algorithm

1. **Embedding generation** - SentenceTransformer encodes documents and taxonomy labels
2. **Multi-level matching**
   - Global: Full document embedding vs taxonomy label similarity
   - Sentence: Per-sentence similarity, max taken per label
   - Keyword: Consensus keywords (n>1 extractors) similarity
3. **Score combination** - Per-sentence weighted sum: `sentence_weight*sent + global_weight*global + (1-both)*keyword`
4. **Aggregation** - Max sentence score per document-label pair becomes `relevance_score`
5. **Confidence binning**
   - Global: Quantile-based bins across all matches
   - Local: Min of quantile-based AND dropoff-based (relative gap detection) per document
   - Final: Min of global and local bins (high/medium/low)
6. **Optional zero-shot** - NLI model validates with 5-level scale (very high/high/medium/low/very low)

## Key Configuration Parameters

- `sentence_weight`, `global_weight` - Score combination weights (keyword = 1 - both)
- `global_q2_threshold`, `global_q3_threshold` - Global bin quantile cutoffs
- `local_q2_threshold`, `local_q3_threshold` - Per-document bin quantile cutoffs
- `normalise_by_matches` - Weight relevance by matching sentence ratio
- `global_embedding_threshold` - Pruning quantile for low-scoring matches
- `zeroshot_hypothesis_template` - NLI prompt template

## Reference Code

The `dsit-taxonomies/` directory contains the original Kedro implementation.
Key files for reference:
- `pipelines/project_similarity_matching/utils.py` - `search_batch()` function
- `pipelines/project_similarity_refinement/utils.py` - `assign_local_bins()` function
- `pipelines/keyword_processing_gtr/nodes.py` - Keyword extraction

Do NOT modify files in `dsit-taxonomies/`.

## Testing

Run specific test files:
```bash
uv run pytest tests/unit/test_similarity.py -v
```

## CLI Usage

```bash
taxonomise classify \
    --corpus data.csv \
    --taxonomy taxonomy.json \
    --output results.parquet
```
