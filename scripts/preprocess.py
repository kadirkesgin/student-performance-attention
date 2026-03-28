import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_uci():
    input_path = "data/raw/uci_dropout/data.csv"
    output_path = "data/processed/uci_dropout_cleaned.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path, sep=';')
    df.columns = df.columns.str.strip()
    
    le = LabelEncoder()
    df['Target'] = le.fit_transform(df['Target'])
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print(f"UCI Target Label Mapping: {mapping}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed UCI data saved to {output_path}")

def preprocess_mendeley():
    input_path = "data/raw/Student Performance Dataset 2024.csv"
    output_path = "data/processed/mendeley_ai_cleaned.csv"

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    le_grade = LabelEncoder()
    df['Grade_Enc'] = le_grade.fit_transform(df['Grade'])
    
    le_cat = LabelEncoder()
    df['Category_Enc'] = le_cat.fit_transform(df['Categories'])

    print(f"Mendeley Grade Labels: {dict(zip(le_grade.classes_, range(len(le_grade.classes_))))}")
    print(f"Mendeley Category Labels: {dict(zip(le_cat.classes_, range(len(le_cat.classes_))))}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed Mendeley data saved to {output_path}")

def preprocess_kaggle():
    input_path = "data/raw/merged_dataset.csv"
    output_path = "data/processed/kaggle_2024_cleaned.csv"

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    # FinalGrade is already numeric here but let's ensure consistency
    # (If categorical strings were present, we'd use LabelEncoder)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed Kaggle data saved to {output_path}")

if __name__ == "__main__":
    preprocess_uci()
    preprocess_mendeley()
    preprocess_kaggle()
