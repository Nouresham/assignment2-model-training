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
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    
    print("=" * 60)
    print("STARTING TRAINING JOB")
    print("=" * 60)
    
    mlflow.start_run()
    mlflow.log_params(vars(args))
    
    # Find parquet file
    print("\n[1/4] Loading data...")
    parquet_file = None
    for file in os.listdir(args.data):
        if file.endswith('.parquet'):
            parquet_file = os.path.join(args.data, file)
            break
    
    if parquet_file is None:
        raise FileNotFoundError(f"No parquet file found in {args.data}")
    
    print(f"File: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Print column info
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Simple feature selection - use only sentiment and length features (avoid complex ones)
    feature_cols = []
    for col in df.columns:
        if col in ['overall', 'asin', 'reviewerID']:
            continue
        # Only use simple numeric columns
        if any(term in col for term in ['sentiment', 'length']):
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
    
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
    
    if len(feature_cols) == 0:
        # Fallback: use any numeric column
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c not in ['overall', 'asin', 'reviewerID']]
        print(f"Fallback: using {len(feature_cols)} numeric columns")
    
    X = df[feature_cols].fillna(0).astype(np.float32)
    y = (df['overall'] >= 4).astype(int)
    
    print(f"X shape: {X.shape}")
    print(f"Positive samples: {y.sum()}/{len(y)}")
    
    # Split data
    print("\n[2/4] Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=args.random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=args.random_state)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Train model
    print("\n[3/4] Training model...")
    model = RandomForestClassifier(
        n_estimators=min(args.n_estimators, 50),  # Reduce for speed
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n[4/4] Evaluation results:")
    for name, X_data, y_data in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = model.predict(X_data)
        y_proba = model.predict_proba(X_data)[:, 1]
        
        acc = accuracy_score(y_data, y_pred)
        f1 = f1_score(y_data, y_pred)
        auc = roc_auc_score(y_data, y_proba)
        
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
