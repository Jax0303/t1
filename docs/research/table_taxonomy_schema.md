# Detailed Table Taxonomy Schema (Advanced)

This schema expands the previous 3-level classification into a diverse set of sub-types to enable granular causal analysis of failure modes in Table Structure Recognition (TSR) and RAG.

## 1. Macro: Global Topology (거시적 위상)
Determines the overall grid structure and presence of major visual/logical boundaries.

| Category | Sub-type | Definition | Causal Impact |
| :--- | :--- | :--- | :--- |
| **Grid-Native** | **Pure-Matrix** | No spanning, clean N x M grid. | High performance expected for all models. |
| | **Standard-Spanning** | Occasional spanning (e.g., header only). | Common baseline for modern TSR models. |
| **Non-Grid** | **Heavily-Nested** | Spanning cells within spanning cells. | RCA (Global Grid) should outperform local baselines. |
| | **Frameless-Sparse** | No visible lines, rely on alignment. | OCR/Spatial alignment crucial; RCA robustness test. |
| **Complex** | **Multi-Chunked** | Table split by text segments or sub-titles. | Routing/Sectioning failure point. |
| | **Inconsistent-Width** | Columns shift width significantly across rows. | Grid-line prediction jittering. |

## 2. Structural: Header Logic (구조적 헤더)
Maps how table semantics are organized via labels.

| Category | Sub-type | Definition | Causal Impact |
| :--- | :--- | :--- | :--- |
| **Flat** | **Single-Row** | Header is exactly 1 row deep. | RAG: Simple row-level retrieval works. |
| **Hierarchical** | **Multi-Deep** | 2+ rows of nested headers. | RAG: Needs "Header-Path" indexing (Tree). |
| | **Cross-Tab** | Header logic on both Top and Left (Stub). | Row-Col intersection accuracy is paramount. |
| **Recursive** | **Section-Header** | Intermediate headers appearing mid-table. | Segmentation error; causes row-count mismatch. |
| **Legacy** | **Implicit-NoHeader** | No clear header area (e.g., pure list). | Retrieval strategy: fallback to embedding-only. |

## 3. Micro: Data Cell Nature (미시적 셀 특성)
Analysis of the content and layout within individual cells.

| Category | Sub-type | Definition | Causal Impact |
| :--- | :--- | :--- | :--- |
| **Primitive** | **Numerical-Short** | Single numbers or short tokens. | Cleanest for LLM/RAG intake. |
| **Dense** | **Multi-line-Text** | Cells with paragraphs/wrapped text. | Causes Row-Shift errors due to high bbox H. |
| | **Symbolic-Mixed** | Mix of text, numbers, markers (e.g., *, ^). | Parsing/Normalizing quality bottleneck. |
| **Irregular** | **Empty-Dominant** | High sparsity (%) of cells. | Edge-case for adjacency relation metrics. |
| | **Footnote-Linked** | Cells referencing notes outside the table. | Information loss in standard RAG pipeline. |
| | **Merged-Value** | Data (non-header) cells span multiple rows. | Most common failure for "row-first" parsers. |

## Automation Logic (Implementation Strategy)
The `TableClassifier` module will use the following heuristics:
- **Span Density**: `total_span_area / total_grid_area`
- **Header Depth**: Count of top rows with specific span/label traits.
- **Header Intersection**: Check for both Top-Header and Left-Stub presence.
- **Cell Entropy**: Variance in cell content length and aspect ratios.
- **Sparsity Index**: Ratio of empty logical slots to total slots.
