"""
Visualization Script for Attribution Results (Detailed)
Supports Systematic Error Analysis based on user taxonomy.
"""
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_visualizations(results_path="results/production_attribution_final.json"):
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    # 1. Expand Data for Detailed Metrics
    rows = []
    for item in data:
        tid = item['table_id']
        for s_name, s_data in item['scenarios'].items():
            metrics = s_data['metrics']
            rows.append({
                'table_id': tid,
                'scenario': s_name,
                'f1': metrics.get('f1', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'horz_f1': metrics.get('horizontal_f1', 0),
                'vert_f1': metrics.get('vertical_f1', 0),
                'ocr_tokens': s_data.get('ocr_tokens', 0)
            })
    
    df = pd.DataFrame(rows)
    
    # Set aesthetics
    sns.set_theme(style="whitegrid")
    
    # --- Figure 1: Comprehensive Metrics Comparison ---
    # Melt for side-by-side bar chart
    metric_df = df.melt(id_vars=['scenario', 'table_id'], 
                        value_vars=['f1', 'precision', 'recall'], 
                        var_name='Metric', value_name='Score')
    
    avg_metrics = metric_df.groupby(['scenario', 'Metric'])['Score'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=avg_metrics, 
        x='scenario', 
        y='Score', 
        hue='Metric', 
        palette='viridis'
    )
    plt.title('Detailed Performance Metrics by Scenario', fontsize=14, pad=15)
    plt.ylim(0, 1.1)
    
    # Add values on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/detailed_metrics_comparison.png', dpi=300)
    print("Saved results/detailed_metrics_comparison.png")

    # --- Figure 2: Structural Breakdown (Horizontal vs Vertical) ---
    struct_df = df.melt(id_vars=['scenario', 'table_id'], 
                        value_vars=['horz_f1', 'vert_f1'], 
                        var_name='Relation Type', value_name='F1 Score')
    
    avg_struct = struct_df.groupby(['scenario', 'Relation Type'])['F1 Score'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(
        data=avg_struct, 
        x='scenario', 
        y='F1 Score', 
        hue='Relation Type',
        palette='magma'
    )
    plt.title('Structural Relation Performance (Row vs Col)', fontsize=14, pad=15)
    plt.ylim(0, 1.1)
    
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.2f', padding=3)

    plt.legend(title='Relation Type')
    plt.tight_layout()
    plt.savefig('results/structural_breakdown.png', dpi=300)
    print("Saved results/structural_breakdown.png")

    # --- Figure 3: Attribution Impact Heatmap (Enhanced) ---
    # S1: Real, S2: Perfect OCR, S3: Real+GT Structure(simulated), S4: Perfect+GT(simulated)
    # Note: S3/S4 usually 1.0 if GT used. S3 depends on OCR.
    
    # Attribution Logic
    # OCR Error Contribution = F1(S2) - F1(S1)
    # TSR Error Contribution = F1(S4) - F1(S2) (assuming S4 is max potential of TSR)
    # Actually:
    # Gap = 1.0 - S1
    # OCR Part = S2 - S1
    # TSR Part = S4 - S2 (if S4=1.0)
    # Residual = 1.0 - S4
    
    def get_f1(s_name):
        subset = df[df['scenario'] == s_name]
        return subset['f1'].mean() if not subset.empty else 0.0

    s1 = get_f1('S1_RealReal')
    s2 = get_f1('S2_ChunkReal') # Perfect OCR + Model TSR
    # s3 = get_f1('S3_RealGT')
    # s4 = get_f1('S4_ChunkGT') # Perfect OCR + GT Struct (should be 1.0)
    
    ocr_impact = max(0, s2 - s1)
    # TSR impact is the gap remaining even with perfect OCR
    tsr_impact = max(0, 1.0 - s2) 
    
    impact_data = pd.DataFrame([{
        'OCR Quality': ocr_impact,
        'TSR Model': tsr_impact,
        'Residual': 0.0 # Assuming target 1.0
    }])
    
    plt.figure(figsize=(8, 3))
    sns.heatmap(impact_data, annot=True, cmap='Reds', fmt=".3f", 
                cbar_kws={'label': 'F1 Score Loss contribution'},
                annot_kws={"size": 14, "weight": "bold"})
    plt.title('Systematic Error Attribution Analysis', fontsize=14, pad=15)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('results/systematic_attribution_heatmap.png', dpi=300)
    print("Saved results/systematic_attribution_heatmap.png")
    
    # --- Report Generation Text ---
    # Determine Dominant Error Stage
    dominant_stage = "Unknown"
    if ocr_impact > tsr_impact and ocr_impact > 0.1:
        dominant_stage = "OCR Detection/Recognition"
    elif tsr_impact > ocr_impact and tsr_impact > 0.1:
        dominant_stage = "TSR Grid Detection"
        
    print(f"\n[Systematic Error Report]")
    print(f"Dominant Fault Stage: {dominant_stage}")
    print(f"OCR Impact: {ocr_impact:.3f}")
    print(f"TSR Impact: {tsr_impact:.3f}")

if __name__ == "__main__":
    generate_visualizations()
