import argparse, os, time, mlflow, joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    start = time.time()
    mlflow.start_run()
    
    # Load raw data
    for f in os.listdir(args.data):
        if f.endswith('.parquet'):
            df = pd.read_parquet(os.path.join(args.data, f))
            break
    
    print(f"Loaded {len(df)} rows")
    
    # Create label (positive if rating >= 4)
    y = (df['overall'] >= 4).astype(int)
    print(f"Positive reviews: {y.sum()} / {len(y)} ({y.sum()/len(y)*100:.1f}%)")
    
    # TF-IDF features from reviewText
    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    X_tfidf = tfidf.fit_transform(df['reviewText'].fillna(''))
    
    # Review length feature
    review_length = df['reviewText'].fillna('').str.len().values.reshape(-1, 1)
    
    # Combine features
    X = hstack([review_length, X_tfidf])
    print(f"Feature matrix: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train model
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\nResults:")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test F1 Score: {f1:.4f}")
    print(f"  Test AUC: {auc:.4f}")
    
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_f1", f1)
    mlflow.log_metric("test_auc", auc)
    mlflow.log_metric("training_runtime_seconds", time.time() - start)
    
    # Save model
    os.makedirs(args.output, exist_ok=True)
    joblib.dump(model, os.path.join(args.output, "model.pkl"))
    mlflow.log_artifact(os.path.join(args.output, "model.pkl"))
    
    mlflow.end_run()
    print(f"\n✅ Done in {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
