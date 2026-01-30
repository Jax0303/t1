import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set Korean font and style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_results(report_path):
    with open(report_path, 'r') as f:
        return json.load(f)

def plot_teds_comparison(results, output_dir):
    """TEDS-S 비교 바 차트"""
    scenarios = list(results.keys())
    teds_scores = [results[sc]['summary']['mean_teds_s'] for sc in scenarios]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    bars = ax.bar(range(len(scenarios)), teds_scores, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('TEDS-S Score', fontsize=14, fontweight='bold')
    ax.set_title('Table Structure Recognition Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([sc.replace('Exp ', '').replace(': ', '\n') for sc in scenarios], 
                        rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, teds_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/teds_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/teds_comparison.png")

def plot_cer_comparison(results, output_dir):
    """CER 비교 (전체 및 카테고리별)"""
    scenarios = list(results.keys())
    
    metrics = {
        'Overall CER': [results[sc]['summary']['mean_cer'] for sc in scenarios],
        'Numeric CER': [results[sc]['summary'].get('cer_numeric', 0) for sc in scenarios],
        'Alpha CER': [results[sc]['summary'].get('cer_alpha', 0) for sc in scenarios]
    }
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#e74c3c', '#f39c12', '#3498db']
    
    for i, (metric, values) in enumerate(metrics.items()):
        offset = width * (i - 1)
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i], 
                      edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0.001:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Character Error Rate', fontsize=14, fontweight='bold')
    ax.set_title('OCR Error Analysis: Overall and Categorical CER', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([sc.replace('Exp ', '').replace(': ', '\n') for sc in scenarios],
                        rotation=15, ha='right', fontsize=10)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/cer_comparison.png")

def plot_error_distribution_heatmap(results, output_dir):
    """오류 유형 분포 히트맵"""
    scenarios = list(results.keys())
    error_types = set()
    for sc in scenarios:
        error_types.update(results[sc]['summary']['error_distribution'].keys())
    error_types = sorted(list(error_types))
    
    data = []
    for sc in scenarios:
        row = []
        for et in error_types:
            count = results[sc]['summary']['error_distribution'].get(et, 0)
            row.append(count)
        data.append(row)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=error_types, 
                yticklabels=[sc.replace('Exp ', '').replace(': ', '\n') for sc in scenarios],
                cbar_kws={'label': 'Count'}, linewidths=1, linecolor='black')
    
    ax.set_title('Error Type Distribution Heatmap', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Error Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Experiment Scenario', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_distribution_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/error_distribution_heatmap.png")

def plot_adjacency_f1(results, output_dir):
    """Adjacency F1 비교"""
    scenarios = list(results.keys())
    adj_f1_scores = [results[sc]['summary'].get('mean_adj_f1', 0) for sc in scenarios]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#27ae60', '#16a085', '#c0392b', '#8e44ad']
    bars = ax.bar(range(len(scenarios)), adj_f1_scores, color=colors, 
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Adjacency F1 Score', fontsize=14, fontweight='bold')
    ax.set_title('Cell Adjacency Relation Recognition Performance', fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([sc.replace('Exp ', '').replace(': ', '\n') for sc in scenarios],
                        rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, adj_f1_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/adjacency_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/adjacency_f1_comparison.png")

def main():
    report_path = '/root/t1-10/outputs/refined_comparison_report.json'
    output_dir = '/root/t1-10/outputs/visualizations'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Loading experiment results...")
    results = load_results(report_path)
    
    print("\nGenerating visualizations...")
    plot_teds_comparison(results, output_dir)
    plot_cer_comparison(results, output_dir)
    plot_error_distribution_heatmap(results, output_dir)
    plot_adjacency_f1(results, output_dir)
    
    print(f"\n{'='*60}")
    print("✓ All visualizations generated successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
