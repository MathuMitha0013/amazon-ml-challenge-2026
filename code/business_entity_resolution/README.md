# Business Entity Resolution — Amazon ML Challenge 2026

## 1. Project Purpose & Problem Statement
This repository houses the end-to-end, production-grade solution for the **Amazon ML Challenge 2026 Business Entity Resolution (Entity Matching)** task.

Given three heterogeneous data sources:
- **Source1 ($S1$):** Deduplicated master business records (~1.73M test / ~2.2M train).
- **Source2 ($S2$):** Highly noisy business records (~4.89M test / ~5.03M train).
- **Source3 ($S3$):** Complementary noisy business records (~5.08M test / ~5.29M train).

The objective is to map each test Source1 record (`S1-...`) to its corresponding records in Source2 (`S2-...`) and Source3 (`S3-...`). The solution addresses real-world noise including typos, legal suffix variations, abbreviations, component reordering, missing address fields, and transliteration across multi-country distributions.

The evaluation metric is **entity-level Macro F0.5** (which heavily weights precision over recall, heavily penalizing false positive associations and misclassified singletons) along with a required **compact, high-recall candidate set** (`candidate_pairs.tsv`).

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
|     - Route 1: Exact Normalized Name Hash Lookup                              |
|     - Route 2: Rare Name Token Inverted Index                                 |
|     - Route 3: Address Locality & Numeric Token Inverted Index                |
|     - Route 4: Character 3-Gram Typo / Spelling Index                         |
|     - Multi-Route UNION -> 96.80% Ground-Truth Candidate Recall               |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|  4. CANDIDATE RANKING & BUDGETING (src/ranking/)                              |
|     - Schema standardization & attribute hydration (Polars lazy join)         |
|     - Lightweight composite score pre-filtering (RapidFuzz / Polars)          |
|     - Adaptive candidate budget truncation per S1 -> output/candidate_pairs.tsv|
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|  5. PAIRWISE FEATURE ENGINEERING (src/features/)                              |
|     - 28 Vectorized string, token, address, and metadata features             |
|     - RapidFuzz C-extensions: Levenshtein, partial ratio, token sort/set      |
|     - Address number overlap, locality token matching, length differences     |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|  6. ML MATCHING & INFERENCE (src/models/)                                     |
|     - LightGBM Pairwise Binary Classifier with hard-negative mining           |
|     - Entity-disjoint 80/20 train/val split (zero entity leakage)             |
|     - Entity-level Macro F0.5 Threshold Tuning (src/evaluation/) -> T=0.70    |
|     - Precision-calibrated singleton decision gating (100% singleton accuracy)|
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
├── experiments/                       # Reproducible benchmark & training experiments
│   ├── benchmark_person2_blocking.py  # 4-Route candidate recall benchmark
│   ├── train_baseline_model.py        # Baseline 2-route LightGBM training
│   └── train_4route_model.py          # Full 4-route LightGBM training & tuning
├── tests/                             # Pytest suite
│   └── test_candidate_ranking_ml.py   # Unit & end-to-end integration tests
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
    ├── features/                      # Pairwise feature engineering
    │   ├── __init__.py
    │   └── pair_features.py           # 28 RapidFuzz string, address & metadata features
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

## 4. Benchmark & Validation Results

Evaluated on the 10,000 Source 1 training entity benchmark:

### 4.1 Blocking & Candidate Recall
| Blocking Route | Pairs Generated | Captured Links | Recall |
| :--- | :---: | :---: | :---: |
| Exact Normalized Name | 11,646 | 9,273 / 34,752 | 26.68% |
| Rare Name Token | 524,062 | 22,030 / 34,752 | 63.39% |
| Address Token & Numbers | 975,967 | 31,179 / 34,752 | 89.72% |
| Character 3-Gram | 314,970 | 12,308 / 34,752 | 35.42% |
| **Combined 4-Route UNION** | **1,717,549** | **33,640 / 34,752** | $\mathbf{96.80\%}$ |

### 4.2 LightGBM Model Evaluation (Entity-Disjoint Validation)
| Metric | 2-Route Baseline | 4-Route Multi-Blocking | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Entity-Level Macro F0.5** | 0.8398 | $\mathbf{0.9726}$ | $\mathbf{+0.1328}$ |
| **Macro Precision** | 0.9101 | **0.9872** | **+0.0771** |
| **Macro Recall** | 0.7287 | **0.9428** | **+0.2141** |
| **Singleton Accuracy** | 97.83% | **100.00%** | **+2.17%** |
| **Exact Match Set Rate** | 48.03% | **80.62%** | **+32.59%** |
| **Optimal Threshold** | 0.55 | **0.70** | +0.15 |

---

## 5. Technology Stack & Design Principles

- **Data Engines:** `DuckDB` and `Polars` for out-of-core chunked processing and disk-backed Parquet storage. Eliminates RAM bottlenecks on ~26M+ records.
- **String Matching:** Rust-backed `RapidFuzz` for vectorized, ultra-fast Levenshtein and token similarity computations.
- **Machine Learning:** `LightGBM` (<8B parameters, Apache-2.0 licensed) for fast, memory-efficient pairwise classification and ranking.
- **Validation:** Zero-leakage entity-disjoint splitting and strict submission format compliance.
- **Fair Play:** Uses **only** provided challenge datasets; strictly no external APIs, search engines, geocoders, or proprietary lookups.

---

## 6. Execution & Testing

### 6.1 Run Test Suite
```bash
python -m pytest tests/test_candidate_ranking_ml.py -v
```

### 6.2 Run Multi-Route Blocking Benchmark
```bash
python experiments/benchmark_person2_blocking.py
```

### 6.3 Train LightGBM & Optimize Threshold
```bash
python experiments/train_4route_model.py
```
