import matplotlib.pyplot as plt
import numpy as np
import os

def create_benchmark_plot(output_path):
    # Data from the comprehensive evaluation
    strategies = [
        "Baseline\n(OCR Only)",
        "Rule-based\n(Hybrid)",
        "OCR+VLM\n(Semantic)",
        "VLM Direct\nEmbedding",
        "Adaptive\n(Proposed)"
    ]
    
    metrics = {
        "Hit@1": [0.613, 0.738, 0.825, 0.850, 0.950],
        "NDCG@5": [0.688, 0.763, 0.863, 0.865, 0.965],
        "TEDS-Struct": [0.443, 0.697, 0.916, 0.959, 0.961],
        "EM Score": [0.313, 0.513, 0.713, 0.800, 0.900]
    }

    x = np.arange(len(strategies))
    width = 0.2
    multiplier = 0

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#bcbd22']
    
    for (metric, values), color in zip(metrics.items(), colors):
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=metric, color=color, alpha=0.8)
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)
        multiplier += 1

    ax.set_ylabel('Score (0-1)')
    ax.set_title('Comprehensive Table RAG Benchmark: Strategy Comparison', fontsize=16, pad=20)
    ax.set_xticks(x + width * 1.5, strategies)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    artifact_dir = "/home/user/.gemini/antigravity/brain/0adb0c12-cc2f-44cd-8417-47f90eee318c"
    os.makedirs(artifact_dir, exist_ok=True)
    create_benchmark_plot(os.path.join(artifact_dir, "benchmark_comparison.png"))
