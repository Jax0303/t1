import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set Korean font and style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# Create output directory
output_dir = Path('outputs/verified_analysis')
output_dir.mkdir(exist_ok=True)

# Load verified experimental data
with open('outputs/comparison_report.json') as f:
    comparison_data = json.load(f)

with open('outputs/error_analysis_report.json') as f:
    error_analysis = json.load(f)

# Extract metrics from comparison data
experiments = {
    'GNN-CSP\n(Perfect OCR)': {
        'TEDS-S': comparison_data['GNN-CSP (Perfect OCR)']['summary']['mean_teds_s'],
        'CER': comparison_data['GNN-CSP (Perfect OCR)']['summary']['mean_cer'],
        'samples': len(comparison_data['GNN-CSP (Perfect OCR)']['details'])
    },
    'GNN-CSP\n(Noisy OCR)': {
        'TEDS-S': comparison_data['GNN-CSP (Noisy OCR)']['summary']['mean_teds_s'],
        'CER': comparison_data['GNN-CSP (Noisy OCR)']['summary']['mean_cer'],
        'samples': len(comparison_data['GNN-CSP (Noisy OCR)']['details'])
    },
    'Baseline\n(No CSP, Noisy)': {
        'TEDS-S': comparison_data['Baseline (No CSP, Noisy OCR)']['summary']['mean_teds_s'],
        'CER': comparison_data['Baseline (No CSP, Noisy OCR)']['summary']['mean_cer'],
        'samples': len(comparison_data['Baseline (No CSP, Noisy OCR)']['details'])
    }
}

print("=== 실험 데이터 요약 ===")
for exp_name, metrics in experiments.items():
    print(f"\n{exp_name.replace(chr(10), ' ')}:")
    print(f"  TEDS-S: {metrics['TEDS-S']:.4f}")
    print(f"  CER: {metrics['CER']:.4f}")
    print(f"  Samples: {metrics['samples']}")

# 1. TEDS-S Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
exp_names = list(experiments.keys())
teds_values = [experiments[name]['TEDS-S'] for name in exp_names]
colors = ['#2ecc71', '#e74c3c', '#3498db']

bars = ax.bar(exp_names, teds_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars, teds_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('TEDS-S Score', fontsize=14, fontweight='bold')
ax.set_xlabel('Experiment Scenario', fontsize=14, fontweight='bold')
ax.set_title('Table Structure Recognition Performance (TEDS-S)', fontsize=16, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.9)')
ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (>0.7)')
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Poor (<0.5)')
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'verified_teds_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / 'verified_teds_comparison.png'}")
plt.close()

# 2. CER Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
cer_values = [experiments[name]['CER'] for name in exp_names]

bars = ax.bar(exp_names, cer_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

for bar, val in zip(bars, cer_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Character Error Rate (CER)', fontsize=14, fontweight='bold')
ax.set_xlabel('Experiment Scenario', fontsize=14, fontweight='bold')
ax.set_title('OCR Quality Comparison (Lower is Better)', fontsize=16, fontweight='bold')
ax.set_ylim(0, max(cer_values) * 1.3)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'verified_cer_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'verified_cer_comparison.png'}")
plt.close()

# 3. TEDS-S vs CER Scatter Plot
fig, ax = plt.subplots(figsize=(10, 8))

for i, (name, metrics) in enumerate(experiments.items()):
    ax.scatter(metrics['CER'], metrics['TEDS-S'], 
              s=500, color=colors[i], alpha=0.7, 
              edgecolors='black', linewidth=2,
              label=name.replace('\n', ' '))
    
    # Add annotations
    ax.annotate(name.replace('\n', ' '), 
                xy=(metrics['CER'], metrics['TEDS-S']),
                xytext=(15, 15), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc=colors[i], alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax.set_xlabel('Character Error Rate (CER) →', fontsize=14, fontweight='bold')
ax.set_ylabel('TEDS-S Score →', fontsize=14, fontweight='bold')
ax.set_title('OCR Quality vs TSR Performance', fontsize=16, fontweight='bold')
ax.legend(loc='lower left', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'verified_scatter_teds_cer.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'verified_scatter_teds_cer.png'}")
plt.close()

# 4. Error Distribution Pie Chart
error_dist = error_analysis['summary']['error_distribution']
fig, ax = plt.subplots(figsize=(10, 8))

colors_pie = ['#e74c3c', '#2ecc71', '#f39c12']
explode = (0.05, 0.05, 0.05)

wedges, texts, autotexts = ax.pie(error_dist.values(), 
                                    labels=error_dist.keys(),
                                    autopct='%1.1f%%',
                                    explode=explode,
                                    colors=colors_pie,
                                    startangle=90,
                                    textprops={'fontsize': 12, 'fontweight': 'bold'})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')

ax.set_title(f'Error Type Distribution (Total: {sum(error_dist.values())} samples)',
             fontsize=16, fontweight='bold', pad=20)

# Add legend with counts
legend_labels = [f'{k}: {v} samples' for k, v in error_dist.items()]
ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'verified_error_distribution.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'verified_error_distribution.png'}")
plt.close()

# 5. Comparative Performance Table
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
headers = ['Experiment', 'TEDS-S ↑', 'CER ↓', 'Samples', 'Quality']

for name, metrics in experiments.items():
    quality = 'Poor' if metrics['TEDS-S'] < 0.5 else 'Good' if metrics['TEDS-S'] < 0.9 else 'Excellent'
    table_data.append([
        name.replace('\n', ' '),
        f"{metrics['TEDS-S']:.4f}",
        f"{metrics['CER']:.4f}",
        str(metrics['samples']),
        quality
    ])

table = ax.table(cellText=table_data, colLabels=headers,
                cellLoc='center', loc='center',
                colWidths=[0.35, 0.15, 0.15, 0.15, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white', fontsize=12)

# Style rows
row_colors = ['#ecf0f1', '#d5dbdb']
for i, row in enumerate(table_data):
    for j in range(len(headers)):
        cell = table[(i+1, j)]
        cell.set_facecolor(row_colors[i % 2])
        
        # Highlight TEDS-S values
        if j == 1:  # TEDS-S column
            teds_val = float(row[1])
            if teds_val < 0.5:
                cell.set_facecolor('#e74c3c')
                cell.set_text_props(color='white', weight='bold')
            elif teds_val >= 0.9:
                cell.set_facecolor('#2ecc71')
                cell.set_text_props(color='white', weight='bold')

plt.title('Experimental Results Summary', fontsize=16, fontweight='bold', pad=20)
plt.savefig(output_dir / 'verified_results_table.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'verified_results_table.png'}")
plt.close()

# Generate text summary
tsr_error_count = error_dist.get('TSR Error', 0)
success_count = error_dist.get('Success', 0)
minor_noise_count = error_dist.get('Minor Structural Noise', 0)
total_count = sum(error_dist.values())

tsr_error_pct = tsr_error_count / total_count * 100 if total_count > 0 else 0
success_pct = success_count / total_count * 100 if total_count > 0 else 0
minor_noise_pct = minor_noise_count / total_count * 100 if total_count > 0 else 0

# Extract experiment keys without newlines for f-string
exp1_key = 'GNN-CSP\n(Perfect OCR)'
exp2_key = 'GNN-CSP\n(Noisy OCR)'
exp3_key = 'Baseline\n(No CSP, Noisy)'

summary_text = f"""
=== 검증된 실험 결과 요약 ===

실험 데이터셋: SciTSR (400 samples)
분석 날짜: 2026-02-01

【실험 1: GNN-CSP (Perfect OCR)】
- TEDS-S: {experiments[exp1_key]['TEDS-S']:.4f}
- CER: {experiments[exp1_key]['CER']:.4f}
- 샘플 수: {experiments[exp1_key]['samples']}
- 평가: OCR이 완벽해도 GNN-CSP는 30% 성능만 달성 (심각한 TSR 문제)

【실험 2: GNN-CSP (Noisy OCR)】
- TEDS-S: {experiments[exp2_key]['TEDS-S']:.4f}
- CER: {experiments[exp2_key]['CER']:.4f}
- 샘플 수: {experiments[exp2_key]['samples']}
- 평가: OCR 노이즈 추가 시 성능 약간 더 하락 (27.7%)

【실험 3: Baseline (No CSP, Noisy OCR)】
- TEDS-S: {experiments[exp3_key]['TEDS-S']:.4f}
- CER: {experiments[exp3_key]['CER']:.4f}
- 샘플 수: {experiments[exp3_key]['samples']}
- 평가: CSP 없는 단순 베이스라인이 오히려 89.5%로 훨씬 우수!

【핵심 발견】
1. ⚠️ GNN-CSP 모델이 단순 베이스라인보다 59% 낮은 성능
2. ⚠️ Perfect OCR 환경에서도 GNN-CSP는 30%만 달성
3. ✅ CSP 없는 베이스라인이 가장 안정적 (89.5% TEDS-S)
4. OCR 노이즈 영향은 상대적으로 미미 (1.4% CER)

【오류 분류 통계】
- TSR Error: {tsr_error_count} cases ({tsr_error_pct:.1f}%)
- Success: {success_count} cases ({success_pct:.1f}%)
- Minor Noise: {minor_noise_count} cases ({minor_noise_pct:.1f}%)

【결론】
GNN-CSP 모델은 현재 상태로는 실용적이지 않음.
단순 공간 기반 정렬이 훨씬 더 안정적이고 효과적.
"""

with open(output_dir / 'verified_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(f"\n✓ Saved: {output_dir / 'verified_summary.txt'}")
print(summary_text)

print("\n" + "="*50)
print("모든 검증된 분석 결과가 생성되었습니다!")
print(f"출력 디렉토리: {output_dir}")
print("="*50)
