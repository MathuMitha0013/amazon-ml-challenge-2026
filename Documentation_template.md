# Amazon ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** [To be updated]  
**Team Members:** [To be updated]  
**Submission Date:** [To be updated]  

---

## 1. Problem Statement
The objective is large-scale Business Entity Resolution (Entity Matching) across three heterogeneous data sources:
- **Source1 ($S1$):** Deduplicated reference/master business records.
- **Source2 ($S2$):** Noisy business records with variations, abbreviations, typos, and formatting differences.
- **Source3 ($S3$):** Noisy business records with complementary variations.

The goal is to map each test Source1 record to its corresponding matching entity IDs in Source2 and Source3. Evaluation is driven by entity-level Macro F0.5 (favoring precision over recall) alongside a compact, high-recall candidate pair submission (`candidate_pairs.tsv`).

---

## 2. Dataset
- **Data Scale:** ~26M+ records across train and test sets (~2+ GB uncompressed TSV files).
- **Attributes:** `entity_id`, `business_name`, `business_address`, `country`.
- **Training Sets:** `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`.
- **Test Sets:** `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`.
- **Key Characteristics:** Multi-country open-set distribution (US, India, France, etc.), heavy skew in match counts (including true singletons with 0 matches), legal suffix variations, component reordering, and noisy address structures.

---

## 3. Data Preprocessing
- **Multi-Representation Preservation:** Raw records are preserved while generating normalized and tokenized variants to prevent irreversible information loss.
- **Business Name Processing:** Lowercasing, punctuation standardisation, legal suffix canonicalisation (e.g., Ltd/Corp/Inc), sorted token signatures, character n-gram extraction.
- **Address Processing:** Component extraction, numeric token isolation (postal/pin codes, building numbers), locality normalisation, abbreviation expansion.

---

## 4. Candidate Generation (Blocking)
To eliminate the infeasible $O(N \times M)$ trillion-pair comparison space, candidate generation employs a multi-route union blocking framework:
1. **Exact Normalized Name Key:** Direct lookup on standardized name representation.
2. **Compact & Phonetic Key:** Whitespace/punctuation-stripped signatures and phonetic representations.
3. **Rare Name Token Inverted Index:** Blocking on highly discriminative, low-frequency name tokens.
4. **Locality + Numeric Address Key:** Conjunction of geographic locality / postal prefix with street numbers.
5. **Character N-Gram / MinHash Blocking:** Robust indexing against OCR/spelling corruption.
- *Union aggregation ensures candidate recall is maximised while keeping candidate sizes tractable.*

---

## 5. Candidate Ranking & Budget Allocation
- **Cheap Heuristic Scoring:** Fast vectorised similarity filters (token Jaccard, character overlap, route hit count) rank initial candidate pools.
- **Adaptive Candidate Budget:** Dynamic candidate capping per Source1 record derived from empirical validation distributions to ensure compact `candidate_pairs.tsv` without sacrificing ground truth recall.

---

## 6. Feature Engineering
Pairwise similarity feature vector generation for candidate pairs $(S1, S2/S3)$:
- **Name Similarities:** RapidFuzz token sort ratio, token set ratio, partial ratio, Levenshtein distance, exact match indicator, length disparity.
- **Address Similarities:** Token Jaccard, numeric component exact match, postal code matching, substring containment.
- **Contextual & Structural:** Country agreement indicator, source indicator ($S1\text{-}S2$ vs $S1\text{-}S3$), number of active blocking routes.

---

## 7. Matching Model
- **Model Choice:** Gradient Boosted Decision Trees (LightGBM) trained on pair similarity vectors with hard-negative mining.
- **Compliance:** Under 8B parameters, Apache 2.0 / MIT license compliant, lightweight CPU/GPU inference footprint.
- **Loss Function:** Binary classification / Pairwise ranking objective tailored for precision-weighted separation.

---

## 8. Threshold Selection
- **Metric-Driven Calibration:** Direct grid search and optimization on validation folds targeting Macro F0.5.
- **Singleton Handling:** High-precision cutoff thresholds preventing false-positive association on master entities with zero true matches.

---

## 9. Evaluation
- **Candidate Stage Metrics:** Candidate Recall (Target $>98\%$), Mean / Median / P95 / P99 Candidates per $S1$, Candidate Reduction Ratio.
- **Final Matching Metrics:** Precision, Recall, Entity-level Macro F0.5.
- **Validation Split Strategy:** Entity-disjoint validation splits to guarantee zero data leakage between training and evaluation folds.

---

## 10. Scalability
- **Out-of-Core Processing:** DuckDB and Polars engines for out-of-core chunked query execution and zero-copy transformations.
- **Disk-Backed Storage:** Fast columnar Parquet format with optimized compression for minimal memory pressure on large test sets.

---

## 11. Computational Efficiency
- **Parallel Execution:** Multi-threaded vectorised tokenization and distance calculations using Rust-accelerated libraries (Polars, RapidFuzz).
- **Memory Footprint:** Controlled RAM ceiling through streaming / batch-wise feature generation and inference.

---

## 12. Reproducibility
- Deterministic seeding across all data splitting, negative sampling, and model training routines.
- Modular, parameterised pipeline driven by reproducible configuration scripts and standard Python virtual environments.

---

## 13. Limitations
- Potential degradation on extreme edge cases involving transliterated non-Latin scripts with no overlapping tokens.
- Highly generic business names sharing common city locations requiring subtle address distinctions.

---

## 14. Future Improvements
- Embedding-based dense retrieval via lightweight, quantized transformer encoders for cross-language / semantic alignment.
- Graph-based connected component clustering for multi-source consensus resolution.
