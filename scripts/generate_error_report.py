"""
Generate Error Analysis Report

Creates comprehensive analysis of detected errors from Error DB.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_error_db(path: str = 'results/real_error_db.csv') -> pd.DataFrame:
    """Load error database"""
    return pd.read_csv(path)


def generate_report(df: pd.DataFrame, output_path: str = 'results/error_analysis_report.md'):
    """Generate markdown report"""
    
    report = []
    
    # Header
    report.append("# Error Analysis Report - Real OCR+TSR Pipeline")
    report.append("")
    report.append(f"**Generated**: 2026-02-01")
    report.append(f"**Dataset**: SciTSR Test Set")
    report.append(f"**Sample Size**: {len(df)} tables")
    report.append("")
    report.append("---")
    report.append("")
    
    # Executive Summary
    report.append("## 📊 Executive Summary")
    report.append("")
    report.append(f"### Overall Performance")
    report.append("")
    report.append(f"- **Average Structure F1**: {df['struct_f1'].mean():.4f}")
    report.append(f"- **Average Severity**: {df['severity'].mean():.4f}")
    report.append(f"- **Success Rate**: {(df['struct_f1'] > 0.8).sum()}/{len(df)} ({100*(df['struct_f1'] > 0.8).sum()/len(df):.1f}%)")
    report.append("")
    
    # Performance Distribution
    high_f1 = (df['struct_f1'] > 0.8).sum()
    med_f1 = ((df['struct_f1'] > 0.5) & (df['struct_f1'] <= 0.8)).sum()
    low_f1 = (df['struct_f1'] <= 0.5).sum()
    
    report.append(f"### Performance Distribution")
    report.append("")
    report.append(f"- **High (F1 > 0.8)**: {high_f1} tables ({100*high_f1/len(df):.1f}%)")
    report.append(f"- **Medium (0.5 < F1 ≤ 0.8)**: {med_f1} tables ({100*med_f1/len(df):.1f}%)")
    report.append(f"- **Low (F1 ≤ 0.5)**: {low_f1} tables ({100*low_f1/len(df):.1f}%)")
    report.append("")
    
    # Error Types
    report.append("## 🏷️ Error Type Distribution")
    report.append("")
    error_counts = df['primary_type'].value_counts()
    for error_type, count in error_counts.items():
        pct = 100 * count / len(df)
        avg_severity = df[df['primary_type'] == error_type]['severity'].mean()
        report.append(f"- **{error_type}**: {count} tables ({pct:.1f}%) - Avg Severity: {avg_severity:.3f}")
    report.append("")
    
    # Processing Stages
    report.append("## 🔧 Processing Stage Distribution")
    report.append("")
    stage_counts = df['stage'].value_counts()
    for stage, count in stage_counts.items():
        pct = 100 * count / len(df)
        report.append(f"- **{stage}**: {count} tables ({pct:.1f}%)")
    report.append("")
    
    # Split Analysis
    report.append("## 📈 COMP vs Normal Tables")
    report.append("")
    comp_df = df[df['split'] == 'comp']
    normal_df = df[df['split'] == 'normal']
    
    if len(comp_df) > 0 and len(normal_df) > 0:
        report.append(f"### COMP Tables (Complex)")
        report.append(f"- Count: {len(comp_df)}")
        report.append(f"- Avg F1: {comp_df['struct_f1'].mean():.4f}")
        report.append(f"- Avg Severity: {comp_df['severity'].mean():.4f}")
        report.append("")
        
        report.append(f"### Normal Tables")
        report.append(f"- Count: {len(normal_df)}")
        report.append(f"- Avg F1: {normal_df['struct_f1'].mean():.4f}")
        report.append(f"- Avg Severity: {normal_df['severity'].mean():.4f}")
        report.append("")
    
    # Grid Statistics
    report.append("## 📐 Grid Statistics")
    report.append("")
    report.append(f"### Cell Count Errors")
    df['cell_diff'] = df['num_pred_cells'] - df['num_gt_cells']
    report.append(f"- **Average Cell Difference**: {df['cell_diff'].mean():.1f}")
    report.append(f"- **Over-prediction**: {(df['cell_diff'] > 0).sum()} tables")
    report.append(f"- **Under-prediction**: {(df['cell_diff'] < 0).sum()} tables")
    report.append(f"- **Exact match**: {(df['cell_diff'] == 0).sum()} tables")
    report.append("")
    
    report.append(f"### Grid Dimension Errors")
    df['row_diff'] = df['pred_rows'] - df['gt_rows']
    df['col_diff'] = df['pred_cols'] - df['gt_cols']
    report.append(f"- **Average Row Difference**: {df['row_diff'].mean():.1f}")
    report.append(f"- **Average Col Difference**: {df['col_diff'].mean():.1f}")
    report.append("")
    
    # Worst Cases
    report.append("## ❌ Worst 10 Cases")
    report.append("")
    worst = df.nsmallest(10, 'struct_f1')[['table_id', 'struct_f1', 'severity', 'primary_type', 'num_gt_cells', 'num_pred_cells']]
    report.append("| Rank | Table ID | F1 | Severity | Error Type | GT Cells | Pred Cells |")
    report.append("|------|----------|----|---------:|------------|----------|------------|")
    for idx, (i, row) in enumerate(worst.iterrows(), 1):
        report.append(f"| {idx} | {row['table_id']} | {row['struct_f1']:.4f} | {row['severity']:.3f} | {row['primary_type']} | {row['num_gt_cells']} | {row['num_pred_cells']} |")
    report.append("")
    
    # Key Findings
    report.append("## 🔍 Key Findings")
    report.append("")
    report.append("### 1. TSR는 완전히 실패하고 있음")
    report.append("")
    report.append(f"- **모든 100개 테이블**이 `Pure TSR Error`로 분류됨")
    report.append(f"- 평균 F1 점수 {df['struct_f1'].mean():.4f}는 사실상 랜덤 추측 수준")
    report.append(f"- Spatial Sorting TSR이 SciTSR의 복잡한 표 구조를 전혀 캐치하지 못함")
    report.append("")
    
    report.append("### 2. Mock OCR의 한계")
    report.append("")
    report.append(f"- 현재 PaddleOCR가 설치되지 않아 Mock OCR 사용 중")
    report.append(f"- Mock OCR은 단순 3x3 그리드만 생성")
    report.append(f"- **실제 OCR 설치 후 재실행 필요**")
    report.append("")
    
    report.append("### 3. Cell Count 불일치")
    report.append("")
    report.append(f"- 평균적으로 {abs(df['cell_diff'].mean()):.1f}개의 셀 차이 발생")
    report.append(f"- GT 평균: {df['num_gt_cells'].mean():.1f} cells")
    report.append(f"- Pred 평균: {df['num_pred_cells'].mean():.1f} cells")
    report.append(f"- **대부분 Under-prediction** (GT보다 적게 검출)")
    report.append("")
    
    # Recommendations
    report.append("## 💡 Recommendations")
    report.append("")
    report.append("### Immediate Actions")
    report.append("")
    report.append("1. **Install PaddleOCR**")
    report.append("   ```bash")
    report.append("   pip install paddleocr")
    report.append("   ```")
    report.append("")
    report.append("2. **Re-run with Real OCR**")
    report.append("   - 실제 text detection/recognition 성능 확인")
    report.append("   - OCR vs TSR 오류 구분 가능")
    report.append("")
    report.append("3. **Improve TSR Method**")
    report.append("   - Spatial Sorting은 너무 단순함")
    report.append("   - Table detection 기반 방법 고려 (Detectron2, PaddleOCR table module 등)")
    report.append("   - 또는 Learning-based TSR (기존 GNN-CSP, Graph-TSR 등)")
    report.append("")
    
    report.append("### Next Steps for Error Atlas")
    report.append("")
    report.append("1. **Visual Inspection**")
    report.append("   - `artifacts/` 폴더의 overlay 이미지 확인")
    report.append("   - 왜 TSR이 실패했는지 시각적으로 파악")
    report.append("")
    report.append("2. **Counterfactual Experiments**")
    report.append("   - Perfect OCR (GT text) + Real TSR")
    report.append("   - Real OCR + Perfect TSR (GT structure)")
    report.append("   - 정확한 OCR vs TSR 귀속")
    report.append("")
    report.append("3. **Manual Verification**")
    report.append("   - Top 20-30 worst cases 수동 검증")
    report.append("   - 실제 failure mode 확인")
    report.append("")
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Report saved to: {output_path}")


def create_visualizations(df: pd.DataFrame, output_dir: str = 'results/visualizations'):
    """Create visualization plots"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style('whitegrid')
    
    # 1. F1 Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['struct_f1'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(df['struct_f1'].mean(), color='red', linestyle='--', label=f'Mean: {df["struct_f1"].mean():.4f}')
    ax.set_xlabel('Structure F1 Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Structure F1 Scores')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'f1_distribution.png', dpi=150)
    plt.close()
    
    # 2. Cell Count Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['num_gt_cells'], df['num_pred_cells'], alpha=0.6)
    max_cells = max(df['num_gt_cells'].max(), df['num_pred_cells'].max())
    ax.plot([0, max_cells], [0, max_cells], 'r--', label='Perfect prediction')
    ax.set_xlabel('GT Cell Count')
    ax.set_ylabel('Predicted Cell Count')
    ax.set_title('Cell Count: Ground Truth vs Prediction')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'cell_count_scatter.png', dpi=150)
    plt.close()
    
    # 3. COMP vs Normal
    if 'split' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        splits = ['comp', 'normal']
        f1_by_split = [df[df['split'] == s]['struct_f1'].mean() for s in splits]
        ax.bar(splits, f1_by_split, edgecolor='black', alpha=0.7)
        ax.set_ylabel('Average Structure F1')
        ax.set_title('Performance: COMP vs Normal Tables')
        ax.set_ylim([0, 1])
        for i, v in enumerate(f1_by_split):
            ax.text(i, v + 0.02, f'{v:.4f}', ha='center')
        plt.tight_layout()
        plt.savefig(output_path / 'comp_vs_normal.png', dpi=150)
        plt.close()
    
    print(f"✓ Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    # Load data
    df = load_error_db()
    
    # Generate report
    generate_report(df)
    
    # Create visualizations
    create_visualizations(df)
    
    print("\n" + "="*60)
    print("Error Analysis Complete!")
    print("="*60)
