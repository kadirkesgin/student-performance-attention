import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Standardized Results (5-Fold CV Mean) - EXCLUDING EXAMSCORE for Kaggle
datasets = ["UCI Dropout", "Mendeley AI", "Kaggle 2024"]
models = ["Random Forest", "CatBoost", "Gradient Boosting", "Proposed Hybrid-Attention", "Ablation (No Attention)"]

# Updated means from the latest run WITHOUT ExamScore
means_map = {
    "Random Forest": [0.7676, 0.5143, 0.9140],
    "CatBoost": [0.7767, 0.4946, 0.7838],
    "Gradient Boosting": [0.7740, 0.4679, 0.4495],
    "Proposed Hybrid-Attention": [0.7572, 0.4643, 0.6029],
    "Ablation (No Attention)": [0.7520, 0.4732, 0.5684]
}

stds_map = {
    "Random Forest": [0.012, 0.058, 0.005],
    "CatBoost": [0.010, 0.056, 0.008],
    "Gradient Boosting": [0.011, 0.030, 0.011],
    "Proposed Hybrid-Attention": [0.015, 0.052, 0.012],
    "Ablation (No Attention)": [0.018, 0.045, 0.010]
}

def setup_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'lines.linewidth': 1.2
    })
    return sns.color_palette("muted", n_colors=5)

def plot_fig1(palette):
    x = np.arange(len(datasets))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 5))
    
    for i, m in enumerate(models):
        pos = x + (i - 2) * width
        ax.bar(pos, means_map[m], width, yerr=stds_map[m], label=m, 
               color=palette[i], alpha=0.9, capsize=3, 
               edgecolor='black' if "Proposed" in m else 'none',
               linewidth=1.0 if "Proposed" in m else 0)

    ax.set_ylabel('Accuracy (Mean ± SD)', fontweight='bold')
    ax.set_title('Comparison of Model Accuracy Across Datasets (5-Fold CV)', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    ax.set_ylim(0.35, 1.0)
    
    plt.savefig("outputs/fig1_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig2(palette):
    fig, ax = plt.subplots(figsize=(7, 5))
    idx_kaggle = 2
    y_vals = [means_map["Ablation (No Attention)"][idx_kaggle], means_map["Proposed Hybrid-Attention"][idx_kaggle]]
    err_vals = [stds_map["Ablation (No Attention)"][idx_kaggle], stds_map["Proposed Hybrid-Attention"][idx_kaggle]]
    labels = ["Ablation (No Attention)", "Proposed Hybrid-Attention"]
    
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, y_vals, yerr=err_vals, color=[palette[4], palette[3]], 
           alpha=0.8, capsize=8, width=0.5, 
           edgecolor=['none', 'black'], linewidth=[0, 1.2])
    
    ax.set_ylabel('Accuracy (Kaggle Behavioral Data)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('Ablation Analysis of the Attention Mechanism on Kaggle 2024', pad=15)
    ax.set_ylim(0, 0.7)
    
    plt.savefig("outputs/fig2_ablation_study.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig3():
    # Load Real LEAK-FREE SHAP data
    if os.path.exists("outputs/real_shap_importance.csv"):
        df = pd.read_csv("outputs/real_shap_importance.csv")
        df = df.head(10) # Top 10
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Importance', y='Feature', data=df, color='teal', alpha=0.8)
        plt.title('Relative Feature Importance on Kaggle 2024 (Leakage-Free SHAP Analysis)')
        plt.xlabel('Mean |SHAP Value| (Impact on Prediction)', fontweight='bold')
        
        plt.savefig("outputs/fig3_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    palette = setup_style()
    os.makedirs("outputs", exist_ok=True)
    plot_fig1(palette)
    plot_fig2(palette)
    plot_fig3()
    print("Scientific leakage-free figures exported to /outputs")

if __name__ == "__main__":
    main()
