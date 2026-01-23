"""
Rigorous Error Analysis with Improvement Ceiling Calculation
Analyzes actual error cases following Andrew Ng's error analysis principles.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

def load_error_cases():
    """Load collected error cases."""
    with open('outputs/error_cases.json', 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"LOADED ERROR CASES")
    print(f"{'='*80}")
    print(f"Dataset: {data['dataset']}")
    print(f"Split: {data['split']}")
    print(f"Total samples: {data['total_samples']}")
    print(f"Error cases: {data['error_count']} ({data['error_rate']*100:.2f}%)")
    print(f"Avg TEDS (all): {data['avg_teds_all']:.4f}")
    print(f"Avg TEDS (errors): {data['avg_teds_errors']:.4f}")
    print(f"{'='*80}\n")
    
    return pd.DataFrame(data['error_cases'])

def analyze_error_distribution(df):
    """Analyze error distribution by taxonomy."""
    
    print(f"\n{'='*80}")
    print(f"ERROR DISTRIBUTION ANALYSIS")
    print(f"{'='*80}\n")
    
    total_errors = len(df)
    
    # 1. By Table Type
    print("1. ERROR DISTRIBUTION BY TABLE TYPE")
    print("-" * 80)
    table_type_dist = df['table_type'].value_counts()
    for table_type, count in table_type_dist.items():
        pct = count / total_errors * 100
        avg_teds = df[df['table_type'] == table_type]['teds'].mean()
        print(f"   {table_type:<30} {count:>5} ({pct:>5.2f}%)  Avg TEDS: {avg_teds:.4f}")
    
    # 2. By Header Structure
    print(f"\n2. ERROR DISTRIBUTION BY HEADER STRUCTURE")
    print("-" * 80)
    header_dist = df['header_structure'].value_counts()
    for header, count in header_dist.items():
        pct = count / total_errors * 100
        avg_teds = df[df['header_structure'] == header]['teds'].mean()
        print(f"   {header:<30} {count:>5} ({pct:>5.2f}%)  Avg TEDS: {avg_teds:.4f}")
    
    # 3. By Cell Characteristics
    print(f"\n3. ERROR DISTRIBUTION BY CELL CHARACTERISTICS")
    print("-" * 80)
    cell_dist = df['cell_characteristics'].value_counts()
    for cell_char, count in cell_dist.items():
        pct = count / total_errors * 100
        avg_teds = df[df['cell_characteristics'] == cell_char]['teds'].mean()
        print(f"   {cell_char:<30} {count:>5} ({pct:>5.2f}%)  Avg TEDS: {avg_teds:.4f}")
    
    # 4. By Failure Mode (can have multiple per case)
    print(f"\n4. FAILURE MODE DISTRIBUTION")
    print("-" * 80)
    all_failure_modes = []
    for modes in df['failure_modes']:
        all_failure_modes.extend(modes)
    
    failure_mode_counts = Counter(all_failure_modes)
    for mode, count in failure_mode_counts.most_common():
        pct = count / total_errors * 100
        print(f"   {mode:<30} {count:>5} ({pct:>5.2f}%)")
    
    print(f"{'='*80}\n")
    
    return {
        'table_type': table_type_dist,
        'header_structure': header_dist,
        'cell_characteristics': cell_dist,
        'failure_modes': failure_mode_counts
    }

def calculate_improvement_ceiling(df):
    """
    Calculate maximum potential improvement for each error category.
    Following Andrew Ng's principle: prioritize categories with highest ceiling.
    """
    
    print(f"\n{'='*80}")
    print(f"IMPROVEMENT CEILING CALCULATION")
    print(f"{'='*80}\n")
    
    total_samples = 3000  # From error_cases.json
    current_avg_teds = df['teds'].mean()
    
    ceilings = []
    
    # Calculate ceiling for each table type
    print("CEILING BY TABLE TYPE:")
    print("-" * 80)
    for table_type in df['table_type'].unique():
        subset = df[df['table_type'] == table_type]
        count = len(subset)
        pct_of_total = count / len(df) * 100
        
        # If we fix ALL errors in this category (bring TEDS to 1.0)
        current_contribution = subset['teds'].sum()
        perfect_contribution = count * 1.0
        improvement = perfect_contribution - current_contribution
        
        # Impact on overall TEDS
        overall_impact = improvement / total_samples
        
        ceilings.append({
            'category': 'Table Type',
            'value': table_type,
            'error_count': count,
            'pct_of_errors': pct_of_total,
            'current_avg_teds': subset['teds'].mean(),
            'max_improvement': overall_impact,
            'priority_score': pct_of_total * overall_impact
        })
        
        print(f"   {table_type:<30}")
        print(f"      Errors: {count:>5} ({pct_of_total:>5.2f}% of all errors)")
        print(f"      Current Avg TEDS: {subset['teds'].mean():.4f}")
        print(f"      Max TEDS Improvement: +{overall_impact:.4f}")
        print(f"      Priority Score: {pct_of_total * overall_impact:.2f}")
        print()
    
    # Calculate ceiling for each header structure
    print("\nCEILING BY HEADER STRUCTURE:")
    print("-" * 80)
    for header in df['header_structure'].unique():
        subset = df[df['header_structure'] == header]
        count = len(subset)
        pct_of_total = count / len(df) * 100
        
        current_contribution = subset['teds'].sum()
        perfect_contribution = count * 1.0
        improvement = perfect_contribution - current_contribution
        overall_impact = improvement / total_samples
        
        ceilings.append({
            'category': 'Header Structure',
            'value': header,
            'error_count': count,
            'pct_of_errors': pct_of_total,
            'current_avg_teds': subset['teds'].mean(),
            'max_improvement': overall_impact,
            'priority_score': pct_of_total * overall_impact
        })
        
        print(f"   {header:<30}")
        print(f"      Errors: {count:>5} ({pct_of_total:>5.2f}% of all errors)")
        print(f"      Current Avg TEDS: {subset['teds'].mean():.4f}")
        print(f"      Max TEDS Improvement: +{overall_impact:.4f}")
        print(f"      Priority Score: {pct_of_total * overall_impact:.2f}")
        print()
    
    # Calculate ceiling for each failure mode
    print("\nCEILING BY FAILURE MODE:")
    print("-" * 80)
    
    for mode in ['Row-Shift', 'Col-Misalignment', 'Cell-Merge-Error', 
                 'Row-Count-Mismatch', 'Col-Count-Mismatch', 'Content-Loss']:
        # Find all errors with this failure mode
        subset = df[df['failure_modes'].apply(lambda x: mode in x)]
        if len(subset) == 0:
            continue
        
        count = len(subset)
        pct_of_total = count / len(df) * 100
        
        current_contribution = subset['teds'].sum()
        perfect_contribution = count * 1.0
        improvement = perfect_contribution - current_contribution
        overall_impact = improvement / total_samples
        
        ceilings.append({
            'category': 'Failure Mode',
            'value': mode,
            'error_count': count,
            'pct_of_errors': pct_of_total,
            'current_avg_teds': subset['teds'].mean(),
            'max_improvement': overall_impact,
            'priority_score': pct_of_total * overall_impact
        })
        
        print(f"   {mode:<30}")
        print(f"      Errors: {count:>5} ({pct_of_total:>5.2f}% of all errors)")
        print(f"      Current Avg TEDS: {subset['teds'].mean():.4f}")
        print(f"      Max TEDS Improvement: +{overall_impact:.4f}")
        print(f"      Priority Score: {pct_of_total * overall_impact:.2f}")
        print()
    
    print(f"{'='*80}\n")
    
    return pd.DataFrame(ceilings)

def generate_priority_ranking(ceiling_df):
    """Generate priority ranking based on improvement ceiling."""
    
    print(f"\n{'='*80}")
    print(f"PRIORITY RANKING (by Priority Score)")
    print(f"{'='*80}\n")
    
    # Sort by priority score
    ranked = ceiling_df.sort_values('priority_score', ascending=False)
    
    print(f"{'Rank':<6} {'Category':<20} {'Value':<30} {'Errors':<8} {'Max Δ TEDS':<12} {'Priority':<10}")
    print("-" * 100)
    
    for i, row in enumerate(ranked.head(15).itertuples(), 1):
        print(f"{i:<6} {row.category:<20} {row.value:<30} {row.error_count:<8} "
              f"+{row.max_improvement:<11.4f} {row.priority_score:<10.2f}")
    
    print(f"{'='*80}\n")
    
    # Save to JSON
    ranked.to_json('outputs/priority_ranking.json', orient='records', indent=2)
    print(f"✓ Saved priority ranking to: outputs/priority_ranking.json\n")
    
    return ranked

def generate_visualizations(df, ceiling_df, output_dir='outputs/rigorous_analysis'):
    """Generate comprehensive visualizations."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Generating visualizations...")
    
    # 1. Pareto Chart: Cumulative Error Contribution
    fig, ax = plt.subplots(figsize=(14, 7))
    
    table_type_dist = df['table_type'].value_counts()
    cumulative_pct = (table_type_dist.cumsum() / len(df) * 100)
    
    ax.bar(range(len(table_type_dist)), table_type_dist.values, color='steelblue', alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(range(len(cumulative_pct)), cumulative_pct.values, color='red', marker='o', linewidth=2)
    ax2.axhline(y=80, color='orange', linestyle='--', label='80% Threshold')
    
    ax.set_xlabel('Table Type', fontsize=12)
    ax.set_ylabel('Error Count', fontsize=12)
    ax2.set_ylabel('Cumulative %', fontsize=12)
    ax.set_xticks(range(len(table_type_dist)))
    ax.set_xticklabels(table_type_dist.index, rotation=45, ha='right')
    ax.set_title('Pareto Chart: Error Contribution by Table Type', fontsize=14, fontweight='bold')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_chart.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'pareto_chart.png'}")
    plt.close()
    
    # 2. Priority Matrix: Frequency × Impact
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for category in ceiling_df['category'].unique():
        subset = ceiling_df[ceiling_df['category'] == category]
        ax.scatter(subset['pct_of_errors'], subset['max_improvement'], 
                  s=subset['priority_score']*20, alpha=0.6, label=category)
        
        for _, row in subset.iterrows():
            ax.annotate(row['value'][:15], (row['pct_of_errors'], row['max_improvement']),
                       fontsize=8, alpha=0.7)
    
    ax.set_xlabel('% of Total Errors', fontsize=12)
    ax.set_ylabel('Max TEDS Improvement', fontsize=12)
    ax.set_title('Priority Matrix: Error Frequency × Impact', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'priority_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'priority_matrix.png'}")
    plt.close()
    
    # 3. Heatmap: Failure Mode × Table Type
    failure_mode_by_type = defaultdict(lambda: defaultdict(int))
    
    for _, row in df.iterrows():
        table_type = row['table_type']
        for mode in row['failure_modes']:
            failure_mode_by_type[mode][table_type] += 1
    
    heatmap_data = pd.DataFrame(failure_mode_by_type).fillna(0).T
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Error Count'})
    ax.set_title('Failure Mode Distribution by Table Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Table Type', fontsize=12)
    ax.set_ylabel('Failure Mode', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_failure_mode_by_type.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'heatmap_failure_mode_by_type.png'}")
    plt.close()
    
    print(f"\n✓ All visualizations saved to: {output_dir}/\n")

def main():
    print(f"\n{'='*80}")
    print(f"RIGOROUS ERROR ANALYSIS")
    print(f"Following Andrew Ng's Error Analysis Principles")
    print(f"{'='*80}")
    
    # Load error cases
    df = load_error_cases()
    
    # Analyze distribution
    distributions = analyze_error_distribution(df)
    
    # Calculate improvement ceiling
    ceiling_df = calculate_improvement_ceiling(df)
    
    # Generate priority ranking
    ranked = generate_priority_ranking(ceiling_df)
    
    # Generate visualizations
    generate_visualizations(df, ceiling_df)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"KEY FINDINGS")
    print(f"{'='*80}\n")
    
    top_priority = ranked.iloc[0]
    print(f"🎯 TOP PRIORITY:")
    print(f"   Category: {top_priority['category']}")
    print(f"   Value: {top_priority['value']}")
    print(f"   Affects: {top_priority['error_count']} errors ({top_priority['pct_of_errors']:.2f}%)")
    print(f"   Max TEDS Improvement: +{top_priority['max_improvement']:.4f}")
    print(f"   Current Avg TEDS: {top_priority['current_avg_teds']:.4f}")
    
    print(f"\n📊 RECOMMENDED ACTION:")
    print(f"   Focus on fixing '{top_priority['value']}' errors first.")
    print(f"   This could improve overall TEDS by up to {top_priority['max_improvement']:.4f} points.")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
