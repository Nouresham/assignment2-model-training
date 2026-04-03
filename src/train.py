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
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()

def load_data(data_path):
    """Load parquet data from folder path"""
    print(f"Loading data from: {data_path}")
    
    # Find the parquet file (same as test script saw)
    parquet_file = None
    for file in os.listdir(data_path):
        if file.endswith('.parquet'):
            parquet_file = os.path.join(data_path, file)
            break
    
    if parquet_file is None:
        raise FileNotFoundError(f"No parquet file found in {data_path}")
    
    print(f"Found parquet file: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Identify label column
    label_col = 'overall'
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Columns: {df.columns.tolist()[:10]}")
    
    # Identify feature columns (exclude IDs and label)
    id_cols = ['asin', 'reviewerID']
    feature_cols = [c for c in df.columns if c not in id_cols + [label_col]]
    
    X = df[feature_cols].fillna(0)
    y = (df[label_col] >= 4).astype(int)
    
    print(f"Features: {X.shape[1]}, Positive samples: {y.sum()}/{len(y)}")
    return X, y

def main():
    args = parse_args()
    start_time = time.time()
    
    print("=" * 60)
    print("STARTING TRAINING JOB")
    print("=" * 60)
    
    # Start MLflow run
    mlflow.start_run()
    mlflow.log_params({
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": args.random_state
    })
    
    # Load data
    print("\n[1/4] Loading training data...")
    X_train, y_train = load_data(args.train_data)
    
    print("\n[2/4] Loading validation data...")
    X_val, y_val = load_data(args.val_data)
    
    print("\n[3/4] Loading test data...")
    X_test, y_test = load_data(args.test_data)
    
    print(f"\nDataset shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # Train model
    print("\n[4/4] Training model...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluation results:")
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)
        
        mlflow.log_metric(f"{name}_accuracy", acc)
        mlflow.log_metric(f"{name}_f1", f1)
        mlflow.log_metric(f"{name}_auc", auc)
        
        print(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # Save model
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
    
    runtime = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime)
    mlflow.end_run()
    
    print(f"\n✅ Training complete! Runtime: {runtime:.2f} seconds")
    print(f"✅ Model saved to: {model_path}")

if __name__ == "__main__":
    main()
