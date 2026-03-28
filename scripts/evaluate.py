import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from models.proposed_model import ProposedModel
import os
import matplotlib.pyplot as plt
from scipy import stats

def train_proposed_model(X_train, y_train, X_test, y_test, num_classes, ablation=False):
    input_dim = X_train.shape[1]
    model = ProposedModel(input_dim, num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    X_train_t = torch.FloatTensor(X_train.values)
    y_train_t = torch.LongTensor(y_train.values)
    X_test_t = torch.FloatTensor(X_test.values)
    
    model.train()
    for epoch in range(150):
        optimizer.zero_grad()
        outputs = model(X_train_t, ablation_no_attn=ablation)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t, ablation_no_attn=ablation)
        _, predicted = torch.max(test_outputs, 1)
        y_pred = predicted.numpy()
        probs = nn.Softmax(dim=1)(test_outputs).numpy()
    
    return y_pred, probs

def evaluate_fold(X, y, train_idx, val_idx, num_classes):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    results = {}
    
    # Base Models
    models = {
        "RF": RandomForestClassifier(n_estimators=100, random_state=42),
        "GB": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
        "CatBoost": CatBoostClassifier(iterations=200, verbose=0, random_state=42)
    }
    
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_val)
        results[f"{name}_Acc"] = accuracy_score(y_val, pred)
        results[f"{name}_F1"] = f1_score(y_val, pred, average='macro', zero_division=0)
        
    # Proposed
    prop_pred, _ = train_proposed_model(X_train, y_train, X_val, y_val, num_classes, ablation=False)
    results["Proposed_Acc"] = accuracy_score(y_val, prop_pred)
    results["Proposed_F1"] = f1_score(y_val, prop_pred, average='macro', zero_division=0)
    
    # Ablation: No-Attention
    abl_pred, _ = train_proposed_model(X_train, y_train, X_val, y_val, num_classes, ablation=True)
    results["Ablation_Acc"] = accuracy_score(y_val, abl_pred)
    results["Ablation_F1"] = f1_score(y_val, abl_pred, average='macro', zero_division=0)
    
    return results

def run_rigorous_experiment(dataset_path, name):
    print(f"\n--- Rigorous Experiment: {name} ---")
    df = pd.read_csv(dataset_path)
    
    leaky_cols = []
    if 'Target' in df.columns: target = 'Target'
    elif 'Grade_Enc' in df.columns:
        target = 'Grade_Enc'
        leaky_cols = ['Grade', 'Categories', 'Category_Enc', 'Final_Exam_Marks', 'Total']
    elif 'FinalGrade' in df.columns:
        target = 'FinalGrade'
        leaky_cols = ['ExamScore', 'Exam_Score'] # Explicitly removing leakage
    else: return

    X = df.drop(columns=[target, 'Student Id'] + leaky_cols, errors='ignore')
    X_norm = (X - X.mean()) / (X.std() + 1e-8)
    y = df[target]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    num_classes = len(y.unique())
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_norm, y)):
        res = evaluate_fold(X_norm, y, train_idx, val_idx, num_classes)
        fold_results.append(res)
        
    df_res = pd.DataFrame(fold_results)
    
    # Output Table for Paper
    metrics = ["Acc", "F1"]
    models = ["RF", "GB", "MLP", "CatBoost", "Proposed", "Ablation"]
    
    print(f"Aggregated Results for {name}:")
    for m in models:
        acc_mean = df_res[f"{m}_Acc"].mean()
        f1_mean = df_res[f"{m}_F1"].mean()
        print(f"  {m}: Acc={acc_mean:.4f}, Macro-F1={f1_mean:.4f}")
    
    # Significance test (Proposed vs GB - our main target comparison)
    _, p_val = stats.ttest_rel(df_res["Proposed_Acc"], df_res["GB_Acc"])
    print(f"  Significance (Prop vs GB): p-value = {p_val:.4f}")
    
    return df_res

def main():
    datasets = [
        ("data/processed/uci_dropout_cleaned.csv", "UCI Dropout"),
        ("data/processed/mendeley_ai_cleaned.csv", "Mendeley AI"),
        ("data/processed/kaggle_2024_cleaned.csv", "Kaggle 2024")
    ]
    
    all_metrics = {}
    for path, name in datasets:
        if os.path.exists(path):
            all_metrics[name] = run_rigorous_experiment(path, name)

if __name__ == "__main__":
    main()
