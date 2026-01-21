import torch
import torch.utils.data
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.metrics import TEDS, calculate_adjacency_relation_metrics, calculate_physical_overlap_metrics
from src.utils.table_converter import table_data_to_html
from scripts.universal_dataset import UniversalTableDataset
from src.models.baselines.wrapper import RCAWrapper, GraphTSRWrapper, LOREWrapper
from src.evaluation.table_classifier import TableClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TSR_Evaluator")

class TSREvaluator:
    def __init__(self, dataset, models: dict):
        self.dataset = dataset
        self.models = models
        self.teds = TEDS(structure_only=True)
        self.classifier = TableClassifier()
        self.results = []
        self.error_log = []

    def _categorize_error(self, pred_html, gt_html, score):
        """Analyze differences between pred and gt to categorize errors."""
        if score > 0.95:
            return "Success"
        
        # Heuristic rules for error categorization
        if not pred_html or "<td>" not in pred_html:
            return "Empty Detection"
        
        # Count rows/cols in HTML (naive count)
        p_row_count = pred_html.count("<tr>")
        g_row_count = gt_html.count("<tr>")
        
        if abs(p_row_count - g_row_count) > 2:
            return "Row/Col Alignment Error"
        
        if "rowspan" in gt_html and "rowspan" not in pred_html:
            return "Merge Cell Error (Missing)"
        
        if "rowspan" in pred_html and "rowspan" not in gt_html:
            return "Merge Cell Error (Extra)"

        return "Structural/Noise Error"

    def run_evaluation(self, num_samples=100):
        indices = torch.randperm(len(self.dataset))[:num_samples]
        
        for idx in tqdm(indices, desc="Evaluating samples"):
            try:
                sample = self.dataset[idx]
                img_tensor = sample["image"].unsqueeze(0) # Batch dim
                
                # Ground Truth
                gt_cells = sample.get("cells", [])
                gt_html = table_data_to_html(gt_cells) if gt_cells else sample.get("html", "")
                
                # Classify GT table
                tags = self.classifier.classify({"cells": gt_cells})
                
                sample_results = {
                    "index": int(idx),
                    "macro_type": tags["macro"],
                    "structural_type": tags["structural"],
                    "micro_type": tags["micro"]
                }
                
                for name, model in self.models.items():
                    # Prediction
                    pred = model.predict(img_tensor)
                    pred_cells = pred["cells"]
                    pred_html = table_data_to_html(pred_cells)
                    
                    # Metrics
                    teds_score = self.teds.evaluate(pred_html, gt_html)
                    sample_results[f"{name}_TEDS"] = teds_score
                    
                    # Error Analysis
                    err_type = self._categorize_error(pred_html, gt_html, teds_score)
                    self.error_log.append({
                        "sample": int(idx), 
                        "model": name, 
                        "error_type": err_type,
                        "macro_type": tags["macro"]
                    })
                
                self.results.append(sample_results)
            except Exception as e:
                logger.error(f"Error evaluating sample {idx}: {e}")
                continue

    def plot_results(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df_results = pd.DataFrame(self.results)
        df_errors = pd.DataFrame(self.error_log)
        
        # 1. TEDS Distribution (KDE)
        plt.figure(figsize=(10, 6))
        for model in self.models.keys():
            sns.kdeplot(df_results[f"{model}_TEDS"], label=model, fill=True)
        plt.title("TEDS-Struct Score Distribution")
        plt.legend()
        plt.savefig(output_dir / "teds_distribution.png")
        plt.close()

        # 2. Radar Chart (Spider Chart) for Multi-dimensional Comparison
        self._plot_radar_summary(df_results, output_dir)

        # 3. Heatmap: Taxonomy Type vs Error Type
        self._plot_error_heatmap(df_errors, output_dir)

    def _plot_radar_summary(self, df, output_dir):
        """Generates a Radar Chart for model performance across all taxonomy types."""
        from math import pi
        
        # Collect all unique types from all levels
        types = list(df["macro_type"].unique()) + list(df["structural_type"].unique()) + list(df["micro_type"].unique())
        types = [t for t in types if t != "Unknown"]
        
        if not types: return

        # Calculate mean scores for each type
        radar_df = pd.DataFrame(index=types)
        for model in self.models.keys():
            scores = []
            for t in types:
                col = "macro_type" if "Macro" in t else ("structural_type" if "Struct" in t else "micro_type")
                score = df[df[col] == t][f"{model}_TEDS"].mean()
                scores.append(score if not np.isnan(score) else 0)
            radar_df[model] = scores

        # Plotting
        N = len(types)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        
        for model in self.models.keys():
            values = radar_df[model].tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
            ax.fill(angles, values, alpha=0.1)
            
        plt.xticks(angles[:-1], types, color='grey', size=10)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
        plt.ylim(0, 1)
        plt.title("Model Performance Profile (Radar Chart)", size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.savefig(output_dir / "radar_performance_profile.png")
        plt.close()

    def _plot_error_heatmap(self, df_errors, output_dir):
        """Generates a heatmap of Error Type vs Macro Table Type."""
        ct = pd.crosstab(df_errors["macro_type"], df_errors["error_type"], normalize='index')
        plt.figure(figsize=(12, 8))
        sns.heatmap(ct, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Error Mode Distribution by Table Type (Heatmap)")
        plt.tight_layout()
        plt.savefig(output_dir / "error_causal_heatmap.png")
        plt.close()

    def save_report(self, output_path):
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        
        # Summary
        numeric_df = df.select_dtypes(include=[np.number])
        summary = numeric_df.mean().to_dict()
        logger.info(f"Summary Results:\n{summary}")
        
        with open(output_path.replace(".csv", "_summary.md"), "w") as f:
            f.write("# TSR Causal Analysis Report (Advanced Visualization)\n\n")
            f.write("## 1. Global Performance Profile\n")
            f.write("> [!NOTE]\n")
            f.write("> Radar charts provide a holistic view of model strengths across different table complexities.\n\n")
            f.write("![Radar Performance Profile](radar_performance_profile.png)\n\n")
            
            f.write("## 2. Error Mode Causal Analysis\n")
            f.write("> [!IMPORTANT]\n")
            f.write("> This heatmap correlates table types with specific failure modes to identify structural bottlenecks.\n\n")
            f.write("![Error Causal Heatmap](error_causal_heatmap.png)\n\n")

            f.write("## 3. Stratified Metrics Table\n")
            for level in ["macro_type", "structural_type", "micro_type"]:
                f.write(f"### Level: {level.replace('_', ' ').title()}\n")
                # Handle potential empty data
                try:
                    strat_summary = df.groupby(level)[df.columns[df.columns.str.contains("TEDS")]].mean()
                    f.write(strat_summary.to_markdown())
                except:
                    f.write("No data for this level.")
                f.write("\n\n")
            
            f.write("## 4. TEDS Distribution\n")
            f.write("![TEDS Distribution](teds_distribution.png)\n")

if __name__ == "__main__":
    from src.hiertable_rag.core.rca_model import RCAModel
    from src.hiertable_rag.core.grid_restorer import IntersectionRestorer
    
    base_dir = "/home/user/t1-4/data/external"
    # Using real SciTSR data
    dataset = UniversalTableDataset(base_dir, "scitsr", split="val")
    
    rca_model = RCAModel()
    restorer = IntersectionRestorer()
    
    models = {
        "Proposed_RCA": RCAWrapper(rca_model, restorer),
        "GraphTSR_Baseline": GraphTSRWrapper(),
        "LORE_Baseline": LOREWrapper()
    }
    
    evaluator = TSREvaluator(dataset, models)
    # Run a meaningful small batch on real data
    evaluator.run_evaluation(num_samples=20)
    evaluator.plot_results("outputs/plots")
    evaluator.save_report("outputs/tsr_comparison_report.csv")
