import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_intuitive_viz(sota_csv: str = 'results/sota_error_db.csv', fallback_csv: str = 'results/real_error_db.csv', output_dir: str = 'docs/visualizations/intuitive'):
    print("Generating intuitive visualizations...")
    try:
        sota_df = pd.read_csv(sota_csv)
        fallback_df = pd.read_csv(fallback_csv)
    except FileNotFoundError:
        print("Error: Input CSVs not found.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Global Style: Clean, Modern, Professional
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'normal',
        'axes.titleweight': 'bold',
        'axes.titlesize': 20,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'figure.facecolor': 'white'
    })

    # 1. "The Performance Leap" (Mock vs SOTA Comparison)
    plt.figure(figsize=(10, 7))
    comparison_data = [
        {'Version': 'Fallback (Mock)', 'Avg F1': fallback_df['struct_f1'].mean()},
        {'Version': 'SOTA (Current)', 'Avg F1': sota_df['struct_f1'].mean()}
    ]
    comp_df = pd.DataFrame(comparison_data)
    
    ax = sns.barplot(x='Version', y='Avg F1', data=comp_df, palette=['#bdc3c7', '#3498db'])
    # Add Large Data Labels
    for i, v in enumerate(comp_df['Avg F1']):
        ax.text(i, v + 0.02, f'{v:.2f}', color='black', ha='center', va='bottom', fontsize=18, fontweight='bold')
    
    plt.title('Performance Quantum Leap: Mock vs SOTA', pad=30)
    plt.ylim(0, 1.0)
    plt.ylabel('Extraction Accuracy (F1)')
    plt.xlabel('')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_performance_leap.png', dpi=300)

    # 2. "Stage-wise Survival Rate" (Waterfall/Funnel Logic)
    # How many pass OCR vs How many pass TSR
    # OCR Success: CER < 0.05 (Arbitrary high bar)
    # TSR Success: F1 > 0.8
    total = len(sota_df)
    ocr_pass = (sota_df['cer'] < 0.05).sum()
    tsr_pass = (sota_df['struct_f1'] > 0.8).sum()
    
    funnel_data = {
        'Stage': ['Input Data', 'OCR Recognition', 'TSR Structure'],
        'Success Rate': [100.0, (ocr_pass/total)*100, (tsr_pass/total)*100]
    }
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Success Rate', y='Stage', data=pd.DataFrame(funnel_data), palette="GnBu_d")
    for i, v in enumerate(funnel_data['Success Rate']):
        ax.text(v - 8, i, f'{v:.0f}%', color='white', va='center', fontweight='bold', fontsize=14)
        
    plt.title('The Bottleneck: Stage-wise Success Rate', pad=30)
    plt.xlabel('Successful Extraction %')
    plt.xlim(0, 110)
    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_bottleneck_analysis.png', dpi=300)

    # 3. "Error Attribution: OCR vs TSR" (The most intuitive pie)
    # Categorize into OCR-heavy, TSR-heavy, or Mixed
    # OCR: cer > 0.1
    # TSR: cer < 0.1 and F1 < 0.9
    categories = []
    for _, row in sota_df.iterrows():
        if row['cer'] > 0.1: categories.append('OCR Artifacts')
        elif row['struct_f1'] < 0.9: categories.append('TSR Structure Errors')
        else: categories.append('Perfect Extraction')
        
    attr_counts = pd.Series(categories).value_counts()
    
    plt.figure(figsize=(10, 10))
    colors = ['#e74c3c', '#34495e', '#2ecc71']
    plt.pie(attr_counts, labels=attr_counts.index, autopct='%1.1f%%', 
            startangle=140, colors=colors, explode=[0.05]*len(attr_counts),
            textprops={'fontsize': 14, 'fontweight': 'bold'})
    plt.title('Root Cause Attribution Map', pad=40)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_root_cause_map.png', dpi=300)

    # 4. "TSR Specifics: Top 3 Pain Points"
    tsr_errors = sota_df[sota_df['primary_type'] != 'Success']['primary_type'].value_counts().head(3)
    
    plt.figure(figsize=(11, 7))
    ax = sns.barplot(x=tsr_errors.values, y=tsr_errors.index, palette="Reds_r")
    plt.title('Immediate Priorities: Key Failure Modes', pad=30)
    plt.xlabel('Incident Frequency')
    plt.ylabel('')
    for i, v in enumerate(tsr_errors.values):
        ax.text(v + 0.1, i, f'{v} cases', va='center', fontweight='bold')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_actionable_priorities.png', dpi=300)

    print(f"Intuitive visualizations saved to {output_dir}")

if __name__ == "__main__":
    generate_intuitive_viz()
