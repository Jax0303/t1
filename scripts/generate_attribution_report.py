"""
Generate Attribution Analysis Report

Creates comprehensive report from counterfactual attribution experiments.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_attribution_results(path: str = 'results/attribution_results.json') -> list:
    """Load attribution experiment results"""
    with open(path, 'r') as f:
        return json.load(f)


def generate_attribution_report(
    results: list,
    output_path: str = 'results/attribution_report.md'
):
    """Generate markdown attribution report"""
    
    report = []
    
    # Header
    report.append("# OCR vs TSR Attribution Analysis")
    report.append("")
    report.append("## 🎯 Experiment Design")
    report.append("")
    report.append("### Counterfactual Scenarios")
    report.append("")
    report.append("| Scenario | OCR | TSR | Purpose |")
    report.append("|----------|-----|-----|---------|")
    report.append("| **S1** | GT (Perfect) | Real | Isolate TSR errors |")
    report.append("| **S2** | Real | GT (Perfect) | Isolate OCR errors |")
    report.append("| **S3** | GT (Perfect) | GT (Perfect) | Upper bound |")
    report.append("| **S4** | Real | Real | Current baseline |")
    report.append("")
    
    report.append("### Attribution Metrics")
    report.append("")
    report.append("```")
    report.append("OCR_impact = F1(S1) - F1(S4)")
    report.append("  → How much does perfect OCR improve performance?")
    report.append("")
    report.append("TSR_impact = F1(S2) - F1(S4)")
    report.append("  → How much does perfect TSR improve performance?")
    report.append("")
    report.append("OCR_attribution = OCR_impact / (OCR_impact + TSR_impact)")
    report.append("TSR_attribution = TSR_impact / (OCR_impact + TSR_impact)")
    report.append("```")
    report.append("")
    report.append("---")
    report.append("")
    
    # Results
    valid_results = [r for r in results if 'error' not in r]
    
    report.append("## 📊 Experimental Results")
    report.append("")
    report.append(f"**Sample Size**: {len(valid_results)} tables")
    report.append("")
    
    # Average F1 scores
    avg_f1_s1 = np.mean([r['attribution']['f1_s1_gt_ocr_real_tsr'] for r in valid_results])
    avg_f1_s2 = np.mean([r['attribution']['f1_s2_real_ocr_gt_tsr'] for r in valid_results])
    avg_f1_s3 = np.mean([r['attribution']['f1_s3_gt_ocr_gt_tsr'] for r in valid_results])
    avg_f1_s4 = np.mean([r['attribution']['f1_s4_real_ocr_real_tsr'] for r in valid_results])
    
    report.append("### Average F1 Scores by Scenario")
    report.append("")
    report.append("| Scenario | Avg F1 | Description |")
    report.append("|----------|-------:|-------------|")
    report.append(f"| S1 (GT OCR + Real TSR) | {avg_f1_s1:.4f} | **TSR 단독 성능** |")
    report.append(f"| S2 (Real OCR + GT TSR) | {avg_f1_s2:.4f} | **OCR 단독 성능** |")
    report.append(f"| S3 (GT OCR + GT TSR) | {avg_f1_s3:.4f} | **이론적 상한** |")
    report.append(f"| S4 (Real OCR + Real TSR) | {avg_f1_s4:.4f} | **현재 베이스라인** |")
    report.append("")
    
    # Impact analysis
    avg_ocr_impact = np.mean([r['attribution']['ocr_impact'] for r in valid_results])
    avg_tsr_impact = np.mean([r['attribution']['tsr_impact'] for r in valid_results])
    
    report.append("### Impact Analysis")
    report.append("")
    report.append(f"- **OCR Impact**: {avg_ocr_impact:+.4f}")
    report.append(f"  - Perfect OCR로 바꾸면 F1이 평균 {avg_ocr_impact:+.4f} 개선")
    report.append(f"- **TSR Impact**: {avg_tsr_impact:+.4f}")
    report.append(f"  - Perfect TSR로 바꾸면 F1이 평균 {avg_tsr_impact:+.4f} 개선")
    report.append("")
    
    # Attribution
    avg_ocr_attr = np.mean([r['attribution']['ocr_attribution'] for r in valid_results])
    avg_tsr_attr = np.mean([r['attribution']['tsr_attribution'] for r in valid_results])
    
    report.append("### Attribution (원인 귀속)")
    report.append("")
    report.append(f"- **OCR Attribution**: {avg_ocr_attr:.1%}")
    report.append(f"- **TSR Attribution**: {avg_tsr_attr:.1%}")
    report.append("")
    
    # Primary cause
    ocr_primary = sum(1 for r in valid_results if r['attribution']['primary_cause'] == 'OCR')
    tsr_primary = sum(1 for r in valid_results if r['attribution']['primary_cause'] == 'TSR')
    
    report.append("### Primary Cause Distribution")
    report.append("")
    report.append(f"- **OCR-dominant errors**: {ocr_primary} tables ({100*ocr_primary/len(valid_results):.1f}%)")
    report.append(f"- **TSR-dominant errors**: {tsr_primary} tables ({100*tsr_primary/len(valid_results):.1f}%)")
    report.append("")
    
    # Key findings
    report.append("---")
    report.append("")
    report.append("## 🔍 Key Findings")
    report.append("")
    
    if avg_tsr_attr > 0.9:
        report.append("### 🚨 Critical: TSR가 거의 모든 오류의 원인")
        report.append("")
        report.append(f"- TSR Attribution: **{avg_tsr_attr:.1%}**")
        report.append(f"- Perfect TSR를 사용하면 F1이 {avg_f1_s4:.4f} → {avg_f1_s2:.4f} (약 **{avg_tsr_impact:+.4f}** 개선)")
        report.append(f"- 현재 Spatial Sorting TSR이 완전히 실패하고 있음")
        report.append("")
        report.append("**원인**:")
        report.append("- Spatial Sorting은 복잡한 표 구조(merged cells, nested headers) 처리 불가")
        report.append("- 단순 좌표 기반 정렬만으로는 부족")
        report.append("")
    elif avg_ocr_attr > 0.9:
        report.append("### 🚨 Critical: OCR이 거의 모든 오류의 원인")
        report.append("")
        report.append(f"- OCR Attribution: **{avg_ocr_attr:.1%}**")
        report.append(f"- Perfect OCR를 사용하면 F1이 {avg_f1_s4:.4f} → {avg_f1_s1:.4f} (약 **{avg_ocr_impact:+.4f}** 개선)")
        report.append("")
    else:
        report.append("### ⚠️  OCR과 TSR 모두 문제")
        report.append("")
        report.append(f"- OCR Attribution: {avg_ocr_attr:.1%}")
        report.append(f"- TSR Attribution: {avg_tsr_attr:.1%}")
        report.append(f"- 둘 다 개선이 필요함")
        report.append("")
    
    # Evidence
    report.append("## 📐 정량적 증거")
    report.append("")
    report.append("### Scenario 비교")
    report.append("")
    report.append("```")
    report.append(f"S3 (Perfect OCR + Perfect TSR):    F1 = {avg_f1_s3:.4f}  ← Upper bound")
    report.append(f"S2 (Real OCR + Perfect TSR):       F1 = {avg_f1_s2:.4f}  ← OCR만 틀림")
    report.append(f"S1 (Perfect OCR + Real TSR):       F1 = {avg_f1_s1:.4f}  ← TSR만 틀림")
    report.append(f"S4 (Real OCR + Real TSR):          F1 = {avg_f1_s4:.4f}  ← 현재")
    report.append("")
    report.append(f"Gap(S3 - S4) = {avg_f1_s3 - avg_f1_s4:.4f}  ← Total improvement potential")
    report.append(f"Gap(S2 - S4) = {avg_f1_s2 - avg_f1_s4:.4f}  ← TSR improvement potential")
    report.append(f"Gap(S1 - S4) = {avg_f1_s1 - avg_f1_s4:.4f}  ← OCR improvement potential")
    report.append("```")
    report.append("")
    
    # Top cases
    report.append("## 📋 Top 10 Cases by TSR Impact")
    report.append("")
    
    # Sort by TSR impact
    sorted_results = sorted(valid_results, key=lambda x: x['attribution']['tsr_impact'], reverse=True)[:10]
    
    report.append("| Rank | Table ID | S4 F1 | S2 F1 | TSR Impact | TSR Attr |")
    report.append("|------|----------|------:|------:|-----------:|---------:|")
    
    for i, r in enumerate(sorted_results, 1):
        table_id = r['table_id']
        f1_s4 = r['attribution']['f1_s4_real_ocr_real_tsr']
        f1_s2 = r['attribution']['f1_s2_real_ocr_gt_tsr']
        tsr_impact = r['attribution']['tsr_impact']
        tsr_attr = r['attribution']['tsr_attribution']
        
        report.append(f"| {i} | {table_id} | {f1_s4:.4f} | {f1_s2:.4f} | {tsr_impact:+.4f} | {tsr_attr:.1%} |")
    
    report.append("")
    
    # Recommendations
    report.append("## 💡 Recommendations")
    report.append("")
    
    if avg_tsr_attr > 0.7:
        report.append("### Priority 1: TSR 개선 **[HIGH PRIORITY]**")
        report.append("")
        report.append("TSR이 {:.0f}%의 오류를 차지하므로, TSR 개선이 최우선입니다.".format(avg_tsr_attr * 100))
        report.append("")
        report.append("**추천 방법**:")
        report.append("")
        report.append("1. **Learning-based TSR 사용**")
        report.append("   - Graph-TSR (이미 프로젝트에 있음: `src/models/graph_tsr.py`)")
        report.append("   - Table Transformer")
        report.append("   - PaddleOCR Table Module")
        report.append("")
        report.append("2. **Rule-based TSR 개선**")
        report.append("   - Line detection 추가")
        report.append("   - Morphological operations")
        report.append("   - Hough transform for table lines")
        report.append("")
    
    if avg_ocr_attr > 0.3:
        report.append("### Priority 2: OCR 개선")
        report.append("")
        report.append("OCR도 {:.0f}%의 오류를 차지합니다.".format(avg_ocr_attr * 100))
        report.append("")
        report.append("**추천 방법**:")
        report.append("")
        report.append("1. **PaddleOCR 실제 설치 및 사용**")
        report.append("   ```bash")
        report.append("   pip install paddleocr")
        report.append("   ```")
        report.append("")
        report.append("2. **OCR 후처리**")
        report.append("   - Spell checking")
        report.append("   - Context-based correction")
        report.append("")
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Attribution report saved to: {output_path}")


def create_attribution_visualizations(
    results: list,
    output_dir: str = 'results/attribution_viz'
):
    """Create attribution visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    valid_results = [r for r in results if 'error' not in r]
    
    # Extract data
    ocr_attrs = [r['attribution']['ocr_attribution'] for r in valid_results]
    tsr_attrs = [r['attribution']['tsr_attribution'] for r in valid_results]
    ocr_impacts = [r['attribution']['ocr_impact'] for r in valid_results]
    tsr_impacts = [r['attribution']['tsr_impact'] for r in valid_results]
    
    # 1. Attribution pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    avg_ocr = np.mean(ocr_attrs)
    avg_tsr = np.mean(tsr_attrs)
    
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1 if avg_ocr > avg_tsr else 0, 0.1 if avg_tsr > avg_ocr else 0)
    
    ax.pie([avg_ocr, avg_tsr], 
           labels=['OCR', 'TSR'],
           autopct='%1.1f%%',
           colors=colors,
           explode=explode,
           startangle=90)
    ax.set_title('Average Error Attribution\n(OCR vs TSR)', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'attribution_pie.png', dpi=150)
    plt.close()
    
    # 2. Impact comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(valid_results[:20]))  # Top 20
    width = 0.35
    
    ax.bar(x - width/2, ocr_impacts[:20], width, label='OCR Impact', alpha=0.8, color='#ff9999')
    ax.bar(x + width/2, tsr_impacts[:20], width, label='TSR Impact', alpha=0.8, color='#66b3ff')
    
    ax.set_xlabel('Table Index')
    ax.set_ylabel('F1 Impact')
    ax.set_title('OCR vs TSR Impact per Table')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'impact_comparison.png', dpi=150)
    plt.close()
    
    # 3. Scenario waterfall
    avg_f1_s4 = np.mean([r['attribution']['f1_s4_real_ocr_real_tsr'] for r in valid_results])
    avg_f1_s1 = np.mean([r['attribution']['f1_s1_gt_ocr_real_tsr'] for r in valid_results])
    avg_f1_s2 = np.mean([r['attribution']['f1_s2_real_ocr_gt_tsr'] for r in valid_results])
    avg_f1_s3 = np.mean([r['attribution']['f1_s3_gt_ocr_gt_tsr'] for r in valid_results])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = ['S4\n(Real+Real)', 'S1\n(GT OCR)', 'S2\n(GT TSR)', 'S3\n(Perfect)']
    f1_scores = [avg_f1_s4, avg_f1_s1, avg_f1_s2, avg_f1_s3]
    
    bars = ax.bar(scenarios, f1_scores, color=['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1'], alpha=0.8)
    
    ax.set_ylabel('Average F1 Score', fontsize=12)
    ax.set_title('Performance Across Scenarios', fontsize=14, weight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}',
               ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'scenario_waterfall.png', dpi=150)
    plt.close()
    
    print(f"✓ Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    # Load results
    results = load_attribution_results()
    
    # Generate report
    generate_attribution_report(results)
    
    # Create visualizations
    create_attribution_visualizations(results)
    
    print("\n" + "="*60)
    print("Attribution Analysis Complete!")
    print("="*60)
