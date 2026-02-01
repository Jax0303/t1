
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def generate_heatmap(csv_path: str = 'results/sota_error_db.csv', output_dir: str = 'docs/visualizations/sota'):
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Prepare Data for Error Type Heatmap
    heatmap_data = []
    category_map = {
        'Row Merge/Split': 'TSR',
        'Col Merge/Split': 'TSR',
        'Row/Col Shift': 'TSR',
        'Span Error': 'TSR',
        'Table Boundary Error': 'TSR',
        'Pure TSR Error': 'TSR',
        'OCR Content Error': 'OCR',
        'Combined Error': 'Mixed',
    }

    for _, row in df.iterrows():
        primary = row['primary_type']
        if primary == 'Success': continue
        
        secondaries = str(row['secondary_types']).split(',') if pd.notna(row['secondary_types']) and row['secondary_types'] else []
        broad_cat = category_map.get(primary, 'Unknown')
        
        specific_type = primary
        if primary == 'Pure TSR Error' and secondaries:
             interesting = [s for s in secondaries if s != 'Table Boundary Error']
             if interesting: specific_type = interesting[0]
             else: specific_type = secondaries[0]
        
        heatmap_data.append({
            'Specific Type': specific_type.strip(),
            'Category': broad_cat,
            'Stage': row['stage'],
            'Count': 1
        })

    if not heatmap_data:
        print("No error data found to visualize.")
        return

    plot_df = pd.DataFrame(heatmap_data)
    
    # --- Plot 1: Error Type by Category ---
    pivot_type = pd.pivot_table(plot_df, values='Count', index='Specific Type', columns='Category', aggfunc='sum', fill_value=0)
    for col in ['OCR', 'TSR', 'Mixed']:
        if col not in pivot_type.columns: pivot_type[col] = 0
    pivot_type = pivot_type[['OCR', 'TSR', 'Mixed']] # Order columns

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", font_scale=1.2)
    sns.heatmap(pivot_type, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Error Count'})
    plt.title('Distribution of Error Types across OCR vs TSR', pad=20, fontsize=16, fontweight='bold')
    plt.ylabel('Specific Error Type', fontsize=14)
    plt.xlabel('System Category', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_type_heatmap.png', dpi=300)
    print(f"Saved error type heatmap to {output_dir}/error_type_heatmap.png")

    # --- Plot 2: Stage-wise Error Distribution ---
    pivot_stage = pd.pivot_table(plot_df, values='Count', index='Specific Type', columns='Stage', aggfunc='sum', fill_value=0)
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_stage, annot=True, fmt='d', cmap='PuBu', cbar_kws={'label': 'Error Count'})
    plt.title('Error Types by Processing Stage', pad=20, fontsize=16, fontweight='bold')
    plt.ylabel('Specific Error Type', fontsize=14)
    plt.xlabel('Processing Stage', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_stage_heatmap.png', dpi=300)
    print(f"Saved error stage heatmap to {output_dir}/error_stage_heatmap.png")

if __name__ == "__main__":
    generate_heatmap()
