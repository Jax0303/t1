"""
Error Analysis by Table Taxonomy
Classifies tables by type, header structure, and cell types, then generates heatmaps for error patterns.
Uses REAL error cases collected by collect_error_cases.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_error_data():
    """Load error cases and taxonomy counts."""
    data_path = Path("outputs/error_cases.json")
    
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run collect_error_cases.py first.")
        return None, None
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['error_cases'])} error cases from {data['total_samples']} samples")
    
    df_errors = pd.DataFrame(data['error_cases'])
    counts = data.get('taxonomy_counts', {})
    
    return df_errors, counts

def calculate_error_rates(df_errors, taxonomy_counts):
    """
    Calculate error rates for each error type across taxonomies.
    Rate = Count(ErrorType in Category) / Total(Category)
    """
    error_types = ['Row-Shift', 'Col-Misalignment', 'Cell-Merge-Error', 
                   'Row-Count-Mismatch', 'Col-Count-Mismatch', 'Content-Loss']
    
    # helper to flatten the one-hot encoding of failure modes
    # We want a DataFrame where each row is a failure case, and columns are error types (1/0)
    # Actually, we aggregate by category first.
    
    results = []
    
    # 1. By Macro Type
    macro_counts = taxonomy_counts.get('macro_type', {})
    for cat, total in macro_counts.items():
        if total == 0: continue
        subset = df_errors[df_errors['table_type'] == cat]
        
        row = {'Category Type': 'Macro Type', 'Category Value': cat, 'Total Samples': total}
        for et in error_types:
            count = sum(1 for modes in subset['failure_modes'] if et in modes)
            row[et] = count / total
        results.append(row)

    # 2. By Header Structure
    header_counts = taxonomy_counts.get('header_structure', {})
    for cat, total in header_counts.items():
        if total == 0: continue
        subset = df_errors[df_errors['header_structure'] == cat]
        
        row = {'Category Type': 'Header Structure', 'Category Value': cat, 'Total Samples': total}
        for et in error_types:
            count = sum(1 for modes in subset['failure_modes'] if et in modes)
            row[et] = count / total
        results.append(row)

    # 3. By Cell Type
    cell_counts = taxonomy_counts.get('cell_characteristics', {})
    for cat, total in cell_counts.items():
        if total == 0: continue
        subset = df_errors[df_errors['cell_characteristics'] == cat]
        
        row = {'Category Type': 'Cell Type', 'Category Value': cat, 'Total Samples': total}
        for et in error_types:
            count = sum(1 for modes in subset['failure_modes'] if et in modes)
            row[et] = count / total
        results.append(row)

    return pd.DataFrame(results)

def generate_heatmaps(rates_df, output_dir):
    """Generate heatmaps for error patterns."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    error_types = ['Row-Shift', 'Col-Misalignment', 'Cell-Merge-Error', 
                   'Row-Count-Mismatch', 'Col-Count-Mismatch', 'Content-Loss']
    
    # 1. Macro Type
    df_macro = rates_df[rates_df['Category Type'] == 'Macro Type'].set_index('Category Value')[error_types]
    if not df_macro.empty:
        plt.figure(figsize=(12, 6))
        sns.heatmap(df_macro, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Error Rate'})
        plt.title('Error Rate by Table Macro Type', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'heatmap_macro_type.png', dpi=300)
        plt.close()

    # 2. Header Structure
    df_header = rates_df[rates_df['Category Type'] == 'Header Structure'].set_index('Category Value')[error_types]
    if not df_header.empty:
        plt.figure(figsize=(12, 6))
        sns.heatmap(df_header, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Error Rate'})
        plt.title('Error Rate by Header Structure', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'heatmap_header_structure.png', dpi=300)
        plt.close()

    # 3. Cell Type
    df_cell = rates_df[rates_df['Category Type'] == 'Cell Type'].set_index('Category Value')[error_types]
    if not df_cell.empty:
        plt.figure(figsize=(12, 6))
        sns.heatmap(df_cell, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Error Rate'})
        plt.title('Error Rate by Cell Characteristics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'heatmap_cell_type.png', dpi=300)
        plt.close()
    
    print(f"✓ Heatmaps saved to {output_dir}")

def generate_summary_json(rates_df, output_file='outputs/error_taxonomy_summary.json'):
    """Generate summary JSON."""
    summary = {
        'overall': {}, # Pending overall calculation
        'by_macro_type': {},
        'by_header_structure': {},
        'by_cell_type': {}
    }
    
    error_types = ['Row-Shift', 'Col-Misalignment', 'Cell-Merge-Error', 
                   'Row-Count-Mismatch', 'Col-Count-Mismatch', 'Content-Loss']
    
    for _, row in rates_df.iterrows():
        cat_type = row['Category Type']
        cat_val = row['Category Value']
        
        dist = {et: row[et] for et in error_types}
        
        if cat_type == 'Macro Type':
            summary['by_macro_type'][cat_val] = dist
        elif cat_type == 'Header Structure':
            summary['by_header_structure'][cat_val] = dist
        elif cat_type == 'Cell Type':
            summary['by_cell_type'][cat_val] = dist
            
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary JSON saved to {output_file}")

def main():
    print("\n" + "="*70)
    print("TSR ERROR ANALYSIS (REAL DATA)")
    print("="*70)
    
    df_errors, counts = load_error_data()
    if df_errors is None: return
    
    print("[1/3] Calculating error rates...")
    rates_df = calculate_error_rates(df_errors, counts)
    
    print("[2/3] Generating heatmaps...")
    generate_heatmaps(rates_df, 'outputs/error_heatmaps')
    
    print("[3/3] Saving summary...")
    generate_summary_json(rates_df)
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
