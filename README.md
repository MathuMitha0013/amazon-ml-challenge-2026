# Amazon ML Challenge 2026 — Business Entity Resolution

This repository contains our team's end-to-end solution for the **Amazon ML Challenge 2026 Business Entity Resolution (Entity Matching)** competition.

## Project Structure Overview
- `code/business_entity_resolution/`: Core source code, modules, pipeline entry points, and documentation.
  - `src/`: Modular Python codebase (data engine, text/address preprocessing, 4-route blocking, candidate ranking, 28 pairwise features, LightGBM model, evaluation, and pipelines).
  - `experiments/`: Reproducible blocking benchmarks, baseline training, and 4-route model training scripts.
  - `tests/`: Automated unit and integration test suite (`pytest`).
  - `README.md`: In-depth architecture specification, benchmark tables, and execution guide.
- `requirements.txt`: Python dependencies required for the project.
- `Documentation_template.md`: Formal competition solution documentation.
- `output/`: Directory designated for submission outputs (`matching_results.tsv` and `candidate_pairs.tsv`).
- `utils/`: Submission validation scripts (`validate_submission.py`).

## Key Performance Highlights (10k S1 Training Slice Benchmark)
- **Multi-Route Candidate Recall:** **96.80%** (33,640 of 34,752 ground-truth links captured across 1.71M candidates).
- **Entity-Level Macro $F_{0.5}$:** **0.9726** (Macro Precision: 0.9872, Macro Recall: 0.9428).
- **Singleton Accuracy:** **100.00%** at calibrated threshold $T=0.70$.
- **Exact Match Set Rate:** **80.62%** exact ground-truth set recovery.

For full architectural details, data flow diagrams, feature descriptions, and pipeline usage, please refer to [code/business_entity_resolution/README.md](code/business_entity_resolution/README.md).
