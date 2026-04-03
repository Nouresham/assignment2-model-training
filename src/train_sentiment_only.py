import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import mlflow
import joblib
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    
    print("=" * 60)
    print("TRAINING WITH SENTIMENT FEATURES ONLY")
    print("=" * 60)
    
    # Find parquet file
    parquet_file = None
    for file in os.listdir(args.data):
        if file.endswith('.parquet'):
            parquet_file = os.path.join(args.data, file)
            break
    
    if parquet_file is None:
        raise FileNotFoundError("No parquet file found")
    
    print(f"Loading: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    print(f"Shape: {df.shape}")
    
    # Use sentiment columns only
    sentiment_cols = ['sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound']
    available_cols = [c for c in sentiment_cols if c in df.columns]
    
    print(f"Using sentiment columns: {available_cols}")
    
    if len(available_cols) == 0:
        print("ERROR: No sentiment columns found!")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    X = df[available_cols].fillna(0).astype(np.float32)
    y = (df['overall'] >= 4).astype(int)
    
    print(f"X shape: {X.shape}")
    print(f"Positive samples: {y.sum()}/{len(y)}")
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Train
    mlflow.start_run()
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    for name, X_data, y_data in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        acc = accuracy_score(y_data, model.predict(X_data))
        f1 = f1_score(y_data, model.predict(X_data))
        auc = roc_auc_score(y_data, model.predict_proba(X_data)[:, 1])
        mlflow.log_metric(f"{name}_accuracy", acc)
        mlflow.log_metric(f"{name}_f1", f1)
        mlflow.log_metric(f"{name}_auc", auc)
        print(f"{name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # Save model
    os.makedirs(args.output, exist_ok=True)
    joblib.dump(model, os.path.join(args.output, "model.pkl"))
    mlflow.log_artifact(os.path.join(args.output, "model.pkl"))
    mlflow.end_run()
    
    print(f"\n✅ Training complete in {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()
