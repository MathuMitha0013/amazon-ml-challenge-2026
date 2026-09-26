# Business Entity Resolution — Amazon ML Challenge 2026

## 1. Project Purpose & Problem Statement
This repository houses the end-to-end, production-grade solution for the **Amazon ML Challenge 2026 Business Entity Resolution (Entity Matching)** task.

Given three heterogeneous data sources:
- **Source1 ($S1$):** Deduplicated master business records (~1.73M test / ~2.2M train).
- **Source2 ($S2$):** Highly noisy business records (~4.89M test / ~5.03M train).
- **Source3 ($S3$):** Complementary noisy business records (~5.08M test / ~5.29M train).

The objective is to map each test Source1 record (`S1-...`) to its corresponding records in Source2 (`S2-...`) and Source3 (`S3-...`). The solution addresses real-world noise including typos, legal suffix variations, abbreviations, component reordering, missing address fields, and transliteration across multi-country distributions.

The evaluation metric is **entity-level Macro F0.5** (which heavily weights precision over recall, heavily penalizing false positive associations and misclassified singletons) along with a required **compact, high-recall candidate set** (`candidate_pairs.tsv`).

> **Note:** This repository currently establishes the complete architectural skeleton, modular packaging, interfaces, and validation contracts. Core algorithms, models, and blocking indexes will be implemented incrementally across subsequent modules.

---

## 2. End-to-End System Architecture & Data Flow

```
+-------------------------------------------------------------------------------+
|                               RAW DATASETS                                    |
|   dataset/train/ (source1, source2, source3, ground_truth)                   |
|   dataset/test/  (source1, source2, source3)                                 |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|  1. DATA ENGINE (src/data/)                                                  |
|     - Chunked TSV ingestion & Parquet columnar conversion                     |
|     - Out-of-core profiling & ground truth distribution analysis              |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|  2. PREPROCESSING (src/preprocessing/)                                        |
|     - Multi-representation preservation (raw, normalized, compact, tokens)   |
|     - Business name canonicalisation & legal suffix normalization             |
|     - Address tokenization, numeric extraction & locality tagging            |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|  3. MULTI-STAGE BLOCKING & CANDIDATE GENERATION (src/blocking/)               |
|     - Route 1: Exact Normalized Name Lookup                                   |
|     - Route 2: Compact / Phonetic Key Indexing                                |
|     - Route 3: Rare-Token Inverted Index                                      |
|     - Route 4: Address Locality + Numeric Token Blocking                      |
|     - Route 5: Character N-Gram / MinHash Signatures                          |
|     - Union Aggregation -> High-Recall Candidate Pool                         |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|  4. CANDIDATE RANKING & BUDGETING (src/ranking/)                              |
|     - Vectorized cheap similarity pre-filtering (RapidFuzz / Polars)          |
|     - Adaptive candidate budget truncation per S1 -> output/candidate_pairs.tsv|
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|  5. PAIRWISE FEATURE ENGINEERING (src/features/)                              |
|     - Name similarity metrics (Levenshtein, Token Sort/Set, Jaccard)         |
|     - Address similarity & numeric/locality overlap metrics                   |
|     - Contextual features: Country match, source pair type, route hit counts  |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|  6. ML MATCHING & INFERENCE (src/models/)                                     |
|     - LightGBM Pairwise Classifier / Ranker with hard negative mining         |
|     - Entity-level Macro F0.5 Threshold Tuning (src/evaluation/)              |
|     - Precision-calibrated singleton decision gating                          |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|  7. SUBMISSION ARTIFACTS & VALIDATION (output/ & utils/)                      |
|     - output/matching_results.tsv (S1 -> matched S2/S3 IDs)                  |
|     - output/candidate_pairs.tsv  (S1 -> compact candidate S2/S3 IDs)        |
|     - utils/validate_submission.py validation check                           |
+-------------------------------------------------------------------------------+
```

---

## 3. Directory & Module Structure

```
code/business_entity_resolution/
├── README.md                          # Comprehensive project documentation
├── requirements.txt                   # Dependency specification
└── src/
    ├── __init__.py                    # Root package initializer
    ├── data/                          # Out-of-core ingestion, Parquet conversion, profiling
    │   ├── __init__.py
    │   ├── convert_parquet.py         # Converts raw TSVs to indexed Parquet
    │   ├── load_data.py               # Chunked / streaming dataset loader
    │   └── profile_data.py            # Summary statistics & schema verification
    ├── preprocessing/                 # Text & address normalization
    │   ├── __init__.py
    │   ├── normalize_address.py       # Address tokenization & numeric extraction
    │   └── normalize_name.py          # Legal suffix stripping & name normalization
    ├── blocking/                      # Multi-route blocking & union candidate generation
    │   ├── __init__.py
    │   ├── address_blocking.py        # Locality & numeric address key blocking
    │   ├── candidate_generation.py    # Multi-route union & candidate builder
    │   ├── exact_blocking.py          # Exact normalized name hash lookup
    │   ├── ngram_blocking.py          # Character n-gram index blocking
    │   └── token_blocking.py          # Rare name token inverted index blocking
    ├── ranking/                       # Candidate compression & scoring
    │   ├── __init__.py
    │   └── candidate_ranker.py        # Fast pre-filtering & adaptive budget allocator
    ├── features/                      # Pairwise feature extraction
    │   ├── __init__.py
    │   └── pair_features.py           # RapidFuzz string, address & metadata feature extractor
    ├── models/                        # Classifier training & inference
    │   ├── __init__.py
    │   ├── predict.py                 # Batch inference & probability generation
    │   └── train_model.py             # LightGBM model training with validation
    ├── evaluation/                    # Metrics, error analysis & threshold tuning
    │   ├── __init__.py
    │   ├── candidate_recall.py        # Measures blocking recall & reduction ratio
    │   ├── evaluate.py                # Computes entity-level Macro F0.5
    │   ├── ground_truth_analysis.py   # Analyzes match counts, singletons & distributions
    │   └── threshold_tuning.py        # Grid search optimizer for F0.5 cutoff
    └── pipeline/                      # End-to-end orchestration
        ├── __init__.py
        ├── inference_pipeline.py      # End-to-end test execution -> submission TSVs
        └── train_pipeline.py          # End-to-end training & threshold calibration
```

---

## 4. Technology Stack & Design Principles

- **Data Engines:** `DuckDB` and `Polars` for out-of-core chunked processing and disk-backed Parquet storage. Eliminates RAM bottlenecks on ~26M+ records.
- **String Matching:** Rust-backed `RapidFuzz` for vectorized, ultra-fast Levenshtein and token similarity computations.
- **Machine Learning:** `LightGBM` (<8B parameters, Apache-2.0 licensed) for fast, memory-efficient pairwise classification and ranking.
- **Validation:** Zero-leakage entity-disjoint splitting and strict submission format compliance.
- **Fair Play:** Uses **only** provided challenge datasets; strictly no external APIs, search engines, geocoders, or proprietary lookups.

---

## 5. Execution Workflow (Planned)

### 5.1 Environment Setup
```bash
pip install -r requirements.txt
```

### 5.2 Training & Validation Pipeline
```bash
python -m src.pipeline.train_pipeline \
    --train-dir dataset/train \
    --output-model-dir models_saved/
```

### 5.3 Inference & Submission Generation
```bash
python -m src.pipeline.inference_pipeline \
    --test-dir dataset/test \
    --model-path models_saved/lgbm_matcher.joblib \
    --output-dir output/
```

### 5.4 Submission Validation
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 6. Submission Outputs

The final submission generates:
1. `output/matching_results.tsv`:
   - Columns: `source1_entity_id`, `matched_entity_ids` (comma-separated $S2$/$S3$ IDs or empty for singletons).
   - Every test $S1$ entity appears exactly once.
2. `output/candidate_pairs.tsv`:
   - Columns: `source1_entity_id`, `candidate_entity_ids` (compact, high-recall candidate pool containing all final predicted matches).
