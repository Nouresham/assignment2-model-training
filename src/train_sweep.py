import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow
import joblib
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    return parser.parse_args()

def main():
    args = parse_args()
    start = time.time()
    
    mlflow.start_run()
    mlflow.log_params({"n_estimators": args.n_estimators, "max_depth": args.max_depth})
    
    # Load data
    for f in os.listdir(args.data):
        if f.endswith('.parquet'):
            df = pd.read_parquet(os.path.join(args.data, f))
            break
    
    # Features
    df['review_length'] = df['reviewText'].fillna('').str.len()
    tfidf = TfidfVectorizer(max_features=50, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['reviewText'].fillna(''))
    X = np.hstack([df[['review_length']].values, tfidf_matrix.toarray()])
    y = (df['overall'] >= 4).astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("test_accuracy", acc)
    
    # Save
    os.makedirs(args.output, exist_ok=True)
    joblib.dump(model, os.path.join(args.output, "model.pkl"))
    
    mlflow.log_metric("runtime", time.time() - start)
    mlflow.end_run()
    print(f"n_estimators={args.n_estimators}, max_depth={args.max_depth}, accuracy={acc:.4f}")

if __name__ == "__main__":
    main()
