import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import joblib
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    print("Loading data...")
    for f in os.listdir(args.data):
        if f.endswith('.parquet'):
            df = pd.read_parquet(os.path.join(args.data, f))
            break
    
    # Use sentiment columns
    sent_cols = [c for c in df.columns if 'sentiment' in c]
    print(f"Using columns: {sent_cols}")
    
    X = df[sent_cols].fillna(0)
    y = (df['overall'] >= 4).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.start_run()
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("test_accuracy", acc)
    print(f"Test accuracy: {acc:.4f}")
    
    # Save model to output directory
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    mlflow.end_run()
    print("Done!")

if __name__ == "__main__":
    main()
