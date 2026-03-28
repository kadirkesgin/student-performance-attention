import pandas as pd
import numpy as np
import os

def generate_synthetic_uci(n_samples=5000):
    input_path = "data/processed/uci_dropout_cleaned.csv"
    output_path = "data/synthetic/uci_synthetic.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: Processed data {input_path} not found. Run preprocess.py first.")
        return

    df = pd.read_csv(input_path)
    
    # Simple statistical simulation (mimicking distributions)
    synthetic_data = {}
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            mean = df[col].mean()
            std = df[col].std()
            # Generate normal distribution and clip to original range
            synthetic_data[col] = np.random.normal(mean, std, n_samples)
            synthetic_data[col] = np.clip(synthetic_data[col], df[col].min(), df[col].max())
            if df[col].dtype == np.int64:
                synthetic_data[col] = synthetic_data[col].round().astype(int)
        else:
            # For categorical (if any left), pick randomly from original
            synthetic_data[col] = np.random.choice(df[col].dropna().unique(), n_samples)
            
    synthetic_df = pd.DataFrame(synthetic_data)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    synthetic_df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} synthetic records at {output_path}")

if __name__ == "__main__":
    generate_synthetic_uci()
