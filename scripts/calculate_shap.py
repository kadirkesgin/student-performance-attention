import pandas as pd
import torch
import shap
import numpy as np
import os
from models.proposed_model import ProposedModel

def get_real_shap():
    # Load Kaggle 2024 Data
    df = pd.read_csv("data/processed/kaggle_2024_cleaned.csv")
    target = 'FinalGrade'
    leaky_cols = ['ExamScore', 'Exam_Score']
    X = df.drop(columns=[target, 'Student Id'] + leaky_cols, errors='ignore')
    X_norm = (X - X.mean()) / (X.std() + 1e-8)
    y = df[target]
    
    # Quick Train for Proposed Model (to get importance)
    input_dim = X.shape[1]
    num_classes = len(y.unique())
    model = ProposedModel(input_dim, num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    X_t = torch.FloatTensor(X_norm.values)
    y_t = torch.LongTensor(y.values)
    
    print("Training model for SHAP calculation...")
    for epoch in range(50): # Enough for feature importance ranking
        model.train()
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
    
    model.eval()
    # SHAP DeepExplainer (using a subset for speed/stability)
    background = X_t[:100]
    test_samples = X_t[100:300]
    
    print("Calculating SHAP values...")
    # Use KernelExplainer for stability on custom modules if DeepExplainer fails
    def model_predict(x):
        with torch.no_grad():
            t = torch.FloatTensor(x)
            return model(t).numpy()
            
    explainer = shap.KernelExplainer(model_predict, background.numpy())
    shap_vals = explainer.shap_values(test_samples.numpy())
    
    # Aggregate importance (mean absolute shap per feature across classes)
    if isinstance(shap_vals, list):
        # Multi-class: sum mean abs across classes
        importance = np.mean([np.abs(sv).mean(0) for sv in shap_vals], axis=0)
    else:
        importance = np.abs(shap_vals).mean(0)
        
    feature_names = X.columns.tolist()
    print(f"Importance shape: {importance.shape}")
    print(f"Feature names length: {len(feature_names)}")
    
    # If multi-class (features, classes), average across classes
    if len(importance.shape) > 1:
        importance = np.mean(importance, axis=1)
    
    # Ensure they match
    if len(importance) > len(feature_names):
        importance = importance[:len(feature_names)]
    elif len(importance) < len(feature_names):
        feature_names = feature_names[:len(importance)]

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance.flatten()})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    importance_df.to_csv("outputs/real_shap_importance.csv", index=False)
    print("Real SHAP importance saved successfully.")

if __name__ == "__main__":
    get_real_shap()
