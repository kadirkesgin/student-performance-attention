from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def train_baselines(X_train, y_train, X_test, y_test, dataset_name):
    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    # 2. Gradient Boosting (Replacement for XGBoost due to libomp issues)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    
    print(f"Results for {dataset_name}:")
    print(f"  Random Forest Accuracy: {rf_acc:.4f}")
    print(f"  Gradient Boosting Accuracy: {gb_acc:.4f}")
    
    return rf, gb
