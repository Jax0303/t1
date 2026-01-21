# TSR Causal Analysis Report (Advanced Visualization)

## 1. Global Performance Profile
> [!NOTE]
> Radar charts provide a holistic view of model strengths across different table complexities.

![Radar Performance Profile](radar_performance_profile.png)

## 2. Error Mode Causal Analysis
> [!IMPORTANT]
> This heatmap correlates table types with specific failure modes to identify structural bottlenecks.

![Error Causal Heatmap](error_causal_heatmap.png)

## 3. Stratified Metrics Table
### Level: Macro Type
| macro_type               |   Proposed_RCA_TEDS |   GraphTSR_Baseline_TEDS |   LORE_Baseline_TEDS |
|:-------------------------|--------------------:|-------------------------:|---------------------:|
| Macro: Complex-Irregular |            0.365619 |                 0.655508 |             0.651795 |
| Macro: Pure-Matrix       |            0.350742 |                 0.488173 |             0.584336 |
| Macro: Standard-Spanning |            0.671131 |                 0.353588 |             0.328125 |

### Level: Structural Type
| structural_type           |   Proposed_RCA_TEDS |   GraphTSR_Baseline_TEDS |   LORE_Baseline_TEDS |
|:--------------------------|--------------------:|-------------------------:|---------------------:|
| Struct: Single-Row-Header |            0.385012 |                 0.499815 |             0.568834 |

### Level: Micro Type
| micro_type             |   Proposed_RCA_TEDS |   GraphTSR_Baseline_TEDS |   LORE_Baseline_TEDS |
|:-----------------------|--------------------:|-------------------------:|---------------------:|
| Micro: Numerical-Short |            0.385012 |                 0.499815 |             0.568834 |

## 4. TEDS Distribution
![TEDS Distribution](teds_distribution.png)
