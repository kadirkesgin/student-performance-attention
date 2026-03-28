import matplotlib.pyplot as plt
import numpy as np
import os

def create_academic_plot():
    datasets = ["UCI Dropout", "Mendeley AI", "Kaggle 2024"]
    
    # 5-Fold CV Mean Accuracies
    rf = [0.760, 0.514, 0.914]
    catboost = [0.762, 0.495, 0.784]
    gb = [0.765, 0.468, 0.450]
    proposed = [0.755, 0.479, 0.602]
    ablation = [0.740, 0.450, 0.380]

    x = np.arange(len(datasets))
    width = 0.15

    plt.style.use('seaborn-v0_8-muted')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.bar(x - 2*width, rf, width, label='Random Forest', alpha=0.8)
    ax.bar(x - width, catboost, width, label='CatBoost (SOTA)', alpha=0.8)
    ax.bar(x, gb, width, label='Grad-Boost', alpha=0.8)
    ax.bar(x + width, proposed, width, label='Proposed (Hybrid-Attn)', color='teal', edgecolor='black', linewidth=1.2)
    ax.bar(x + 2*width, ablation, width, label='Ablation (No-Attn)', hatch='//', alpha=0.6)

    ax.set_ylabel('Accuracy (5-Fold CV Mean)')
    ax.set_title('Academic Performance Comparison: Proposed vs SOTA and Baselines', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1.1)

    # Annotate the attention-boost in Kaggle
    ax.annotate(f'Attention Boost: +22%', 
                xy=(2 + width, 0.602), xytext=(2 + 1.5*width, 0.7),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                fontsize=10, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/final_academic_comparison.png", dpi=300, bbox_inches='tight')
    print("Final academic plot saved to outputs/final_academic_comparison.png")

if __name__ == "__main__":
    create_academic_plot()
