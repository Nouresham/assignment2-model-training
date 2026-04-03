import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import joblib
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    return parser.parse_args()

def load_data(path):
    for f in os.listdir(path):
        if f.endswith('.parquet'):
            return pd.read_parquet(os.path.join(path, f))
    raise FileNotFoundError(f"No parquet file in {path}")

def main():
    args = parse_args()
    start = time.time()
    
    mlflow.start_run()
    mlflow.log_params(vars(args))
    
    print("Loading data...")
    train_df = load_data(args.train_data)
    val_df = load_data(args.val_data)
    test_df = load_data(args.test_data)
    
    label_col = 'overall'
    id_cols = ['asin', 'reviewerID']
    feature_cols = [c for c in train_df.columns if c not in id_cols + [label_col]]
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = (train_df[label_col] >= 4).astype(int)
    X_val = val_df[feature_cols].fillna(0)
    y_val = (val_df[label_col] >= 4).astype(int)
    X_test = test_df[feature_cols].fillna(0)
    y_test = (test_df[label_col] >= 4).astype(int)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Features: {len(feature_cols)}")
    
    print("Training model...")
    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("\nEvaluation:")
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)
        mlflow.log_metric(f"{name}_accuracy", acc)
        mlflow.log_metric(f"{name}_auc", auc)
        print(f"  {name}: Acc={acc:.4f}, AUC={auc:.4f}")
    
    os.makedirs(args.output, exist_ok=True)
    joblib.dump(model, os.path.join(args.output, "model.pkl"))
    
    mlflow.log_metric("runtime_seconds", time.time() - start)
    mlflow.end_run()
    print(f"\n✅ Done in {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
