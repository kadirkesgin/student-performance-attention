import pandas as pd
import os

def analyze_datasets():
    datasets = [
        ("data/processed/uci_dropout_cleaned.csv", "UCI Dropout"),
        ("data/processed/mendeley_ai_cleaned.csv", "Mendeley AI"),
        ("data/processed/kaggle_2024_cleaned.csv", "Kaggle 2024"),
        ("data/synthetic/uci_synthetic.csv", "UCI Synthetic")
    ]
    
    for path, name in datasets:
        if os.path.exists(path):
            df = pd.read_csv(path)
            target = ""
            if 'Target' in df.columns: target = 'Target'
            elif 'Grade_Enc' in df.columns: target = 'Grade_Enc'
            elif 'FinalGrade' in df.columns: target = 'FinalGrade'
            
            print(f"\n--- Analysis: {name} ---")
            print(f"Total Samples: {len(df)}")
            print(f"Features: {len(df.columns) - 2}")
            print("Class Distribution:")
            print(df[target].value_counts(normalize=True).sort_index())
            
analyze_datasets()
