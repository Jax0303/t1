import pandas as pd
from pathlib import Path

csv_path = "outputs/tsr_comparison_report.csv"
if not Path(csv_path).exists():
    print("Report not found.")
else:
    # Actually the errors are in a separate log or we need to re-run
    # For now, let's assume we want to analyze the existing outputs
    print("Analyzing error distributions...")
    # Since error_log isn't saved to CSV, let's look at the saved summary.md or re-calculate
    # I will run a tiny evaluation batch to get the real counts
    pass

from src.evaluation.tsr_evaluator import TSREvaluator
from scripts.universal_dataset import UniversalTableDataset
from src.models.baselines.wrapper import RCAWrapper, GraphTSRWrapper, LOREWrapper
from src.hiertable_rag.core.rca_model import RCAModel
from src.hiertable_rag.core.grid_restorer import IntersectionRestorer

base_dir = "data/external"
dataset = UniversalTableDataset(base_dir, "scitsr", split="val")
models = {
    "GraphTSR_Baseline": GraphTSRWrapper(),
    "LORE_Baseline": LOREWrapper()
}
evaluator = TSREvaluator(dataset, models)
evaluator.run_evaluation(num_samples=50)

df_errors = pd.DataFrame(evaluator.error_log)
print("\n=== Global Error Frequency (Baselines) ===")
print(df_errors["error_type"].value_counts())

print("\n=== Error Frequency by Table Type (Macro) ===")
print(df_errors.groupby("macro_type")["error_type"].value_counts().unstack().fillna(0))
