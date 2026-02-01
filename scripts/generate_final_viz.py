import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_production_viz(csv_path: str = 'results/sota_error_db.csv', output_dir: str = 'docs/visualizations/final'):
    print(f"Loading SOTA data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Premium Styling Configuration
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11
    })
    
    # Palette 1: Deep Corporate (Blues/Greys)
    # Palette 2: Vibrant Accent (For errors/highlights)
    COLORS_GRADIENT = ["#1a2a6c", "#2a4858", "#3a6a8c", "#4a8abc", "#5aacec"]
    ERROR_PALETTE = ["#e74c3c", "#f39c12", "#d35400", "#c0392b"]

    # --- 1. F1 Score Distribution (Enhanced Histplot) ---
    plt.figure(figsize=(10, 6))
    sns.set_style("white")
    ax = sns.histplot(df['struct_f1'], kde=True, color='#2c3e50', bins=8, alpha=0.7)
    plt.axvline(df['struct_f1'].mean(), color='#e67e22', linestyle='--', linewidth=2, label=f'Average Performance: {df["struct_f1"].mean():.2f}')
    
    plt.title('Table Structure Recognition Performance (F1)', pad=20)
    plt.xlabel('Structure F1 Score')
    plt.ylabel('Table Count')
    plt.legend(frameon=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/f1_distribution.png', dpi=300)
    
    # --- 2. Error Type Severity (Clean Boxplot) ---
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    ax = sns.boxplot(x='primary_type', y='severity', data=df, palette="Blues_d", width=0.6)
    sns.stripplot(x='primary_type', y='severity', data=df, color="#1a2a6c", alpha=0.4, size=6)
    
    plt.title('Error Impact Analysis (Severity by Type)', pad=20)
    plt.xlabel('Failure Mode')
    plt.ylabel('Impact Score (0 to 1)')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_severity_analysis.png', dpi=300)

    # --- 3. Incident Rate (Modern Horizontal Bar Chart) ---
    plt.figure(figsize=(11, 7))
    primary_counts = df['primary_type'].value_counts()
    
    sns.set_style("white")
    ax = sns.barplot(x=primary_counts.values, y=primary_counts.index, palette="mako")
    
    # Add labels on bars
    for i, v in enumerate(primary_counts.values):
        ax.text(v + 0.1, i, str(v), color='black', va='center', fontweight='bold')

    plt.title('Top Failure Modes in SOTA Pipeline', pad=20)
    plt.xlabel('Number of Occurrences')
    plt.ylabel('')
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/incident_rate.png', dpi=300)

    # --- 4. Reliability Summary (Modern Donut Chart) ---
    plt.figure(figsize=(8, 8))
    success_count = (df['struct_f1'] > 0.9).sum() # Loosen for SOTA realism
    failure_count = len(df) - success_count
    
    data = [success_count, failure_count]
    labels = [f'Near Perfect ({success_count})', f'Structural Issues ({failure_count})']
    colors = ['#27ae60', '#ecf0f1'] # Clean green and light grey
    
    plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, 
            pctdistance=0.85, explode=(0.05, 0), wedgeprops={'edgecolor': 'white', 'linewidth': 3})
    
    # Draw Circle for Donut
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title('System Reliability Overview', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/reliability_summary.png', dpi=300)

    print(f"Refined visualizations saved to {output_dir}")

if __name__ == "__main__":
    generate_production_viz()
