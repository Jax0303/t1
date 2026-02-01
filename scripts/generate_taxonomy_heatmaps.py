
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def generate_taxonomy_plots():
    with open("results/taxonomy_analysis/taxonomy_summary.json", 'r') as f:
        results = json.load(f)
    
    output_dir = Path("results/taxonomy_analysis/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = []
    for res in results:
        data.append({
            "Taxonomy": res["Type"].split(": ")[1],
            "OCR Error Contribution": res["OCR_Loss"],
            "TSR Error Contribution": res["TSR_Loss"],
            "Full System F1": res["Full_F1"],
            "Cell_Loss": res["Cell_Loss"],
            "Shift_Count": res["Shift_Count"]
        })
    
    df = pd.DataFrame(data)
    
    # Plot 1: OCR vs TSR Error Contribution Stacked
    plt.figure(figsize=(12, 6))
    df_melt = df.melt(id_vars="Taxonomy", value_vars=["OCR Error Contribution", "TSR Error Contribution"], 
                      var_name="Error Type", value_name="F1 Loss")
    
    sns.set_theme(style="whitegrid")
    sns.barplot(data=df_melt, x="Taxonomy", y="F1 Loss", hue="Error Type", palette="muted")
    plt.title("Error Attribution by Table Complexity Type", fontsize=16, fontweight='bold')
    plt.xticks(rotation=15)
    plt.ylabel("F1 Score reduction (Loss)")
    plt.tight_layout()
    plt.savefig(output_dir / "error_attribution_by_type.png", dpi=300)
    
    # Plot 2: Heatmap of F1 scores across types and scenarios
    heatmap_rows = []
    for res in results:
        heatmap_rows.append({
            "Taxonomy": res["Type"].split(": ")[1],
            "Pipeline (A)": res["Raw_Metrics"]["A"]["f1"],
            "Pure TSR (B)": res["Raw_Metrics"]["B"]["f1"],
            "Pure OCR (C)": res["Raw_Metrics"]["C"]["f1"]
        })
    
    df_h = pd.DataFrame(heatmap_rows).set_index("Taxonomy")
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_h, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'F1 Score'})
    plt.title("F1 Performance Heatmap by Scenario & Complexity", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "performance_heatmap.png", dpi=300)

    # Plot 3: Specific Error Metrics (Cell Loss vs Structural Shift)
    plt.figure(figsize=(12, 6))
    df_error = df.copy()
    # Normalize for comparison
    df_error["Cell Loss (Count)"] = df["Cell_Loss"]
    df_error["Structural Shift (Error Edges)"] = df["Shift_Count"]
    
    df_e_melt = df_error.melt(id_vars="Taxonomy", value_vars=["Cell Loss (Count)", "Structural Shift (Error Edges)"],
                              var_name="Error Symptom", value_name="Frequency")
    
    sns.barplot(data=df_e_melt, x="Taxonomy", y="Frequency", hue="Error Symptom", palette="rocket")
    plt.title("Detailed Error Symptoms: Cell Loss & Structural Shift", fontsize=16, fontweight='bold')
    plt.xticks(rotation=15)
    plt.ylabel("Frequency / Count")
    plt.tight_layout()
    plt.savefig(output_dir / "error_symptoms_by_type.png", dpi=300)

    # Plot 4: Error Type Occurrence Heatmap
    # We estimate error types from the relation FP/FN and metric differences
    type_heatmap_data = []
    for res in results:
        m = res["Raw_Metrics"]["A"]
        # Heuristic mapping for heatmap visualization
        type_heatmap_data.append({
            "Taxonomy": res["Type"].split(": ")[1],
            "Row/Col Shift": res["Shift_Count"],
            "Cell Missing": res["Cell_Loss"],
            "Merge/Split": m.get('FP', 0) // 2, # Approximation
            "Context Drift": m.get('FN', 0) - res["Cell_Loss"]
        })
    
    df_t = pd.DataFrame(type_heatmap_data).set_index("Taxonomy")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_t, annot=True, fmt="d", cmap="Reds", cbar_kws={'label': 'Occurrence Count'})
    plt.title("Error Type Distribution Heatmap by Taxonomy", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "error_type_distribution_heatmap.png", dpi=300)

    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    generate_taxonomy_plots()
