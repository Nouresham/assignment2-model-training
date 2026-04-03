import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import joblib
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()

def clean_features(X):
    """Clean feature matrix: replace inf, cap extreme values"""
    # Replace infinity with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with column mean
    X = X.fillna(X.mean())
    
    # Cap extreme values at 3 standard deviations
    for col in X.columns:
        mean = X[col].mean()
        std = X[col].std()
        if std > 0:
            upper = mean + 3 * std
            lower = mean - 3 * std
            X[col] = X[col].clip(lower, upper)
    
    return X

def main():
    args = parse_args()
    start_time = time.time()
    
    print("=" * 60)
    print("STARTING TRAINING JOB")
    print("=" * 60)
    
    mlflow.start_run()
    mlflow.log_params({
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": args.random_state
    })
    
    # Find parquet file
    print("\n[1/5] Loading data...")
    parquet_file = None
    for file in os.listdir(args.data):
        if file.endswith('.parquet'):
            parquet_file = os.path.join(args.data, file)
            break
    
    if parquet_file is None:
        raise FileNotFoundError(f"No parquet file found in {args.data}")
    
    print(f"File: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    print(f"Shape: {df.shape}")
    
    # Identify label
    label_col = 'overall'
    if label_col not in df.columns:
        raise ValueError(f"Label '{label_col}' not found")
    
    # Identify feature columns (exclude label and IDs)
    exclude_cols = [label_col, 'asin', 'reviewerID']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"\n[2/5] Found {len(feature_cols)} feature columns")
    
    # Get features and label
    X = df[feature_cols].copy()
    y = (df[label_col] >= 4).astype(int)
    
    print(f"X shape: {X.shape}")
    print(f"Positive class: {y.sum()}/{len(y)} ({y.sum()/len(y)*100:.1f}%)")
    
    # Clean features
    print("\n[3/5] Cleaning features...")
    X = clean_features(X)
    print(f"Cleaned shape: {X.shape}")
    
    # Split data
    print("\n[4/5] Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=args.random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=args.random_state)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Train model
    print("\n[5/5] Training model...")
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
