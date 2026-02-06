# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-02-06

### Added

- Initial release
- Multi-level matching: global (document), sentence, and keyword-level similarity
- Confidence scoring with global and local quantile-based bins
- Dropoff-based confidence detection for per-document scoring
- CLI tool (`taxonomise classify`) with comprehensive options
- Python API with `ClassificationPipeline` and `PipelineConfig`
- Support for CSV, Parquet, JSON, JSONL input/output formats
- Hierarchical taxonomy support (JSON tree or flat CSV with parent IDs)
- Optional zero-shot NLI validation for higher precision
- Extensible keyword extraction: RAKE, YAKE, KeyBERT, DBpedia
- Example astrophysics dataset (100 documents, 181 taxonomy labels)
- YAML configuration file support
