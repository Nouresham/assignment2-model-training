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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    start = time.time()
    print("Loading data...")
    
    # Find parquet file
    for f in os.listdir(args.data):
        if f.endswith('.parquet'):
            df = pd.read_parquet(os.path.join(args.data, f))
            break
    
    print(f"Loaded {len(df)} rows")
    
    # Simple features from reviewText
    print("Extracting features...")
    df['review_length'] = df['reviewText'].fillna('').str.len()
    
    # TF-IDF on reviewText (limit to 100 features for speed)
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['reviewText'].fillna(''))
    
    # Create feature matrix
    X = np.hstack([df[['review_length']].values, tfidf_matrix.toarray()])
    y = (df['overall'] >= 4).astype(int)
    
    print(f"Features: {X.shape[1]}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    print("Training model...")
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {acc:.4f}")
    
    # Save
    os.makedirs(args.output, exist_ok=True)
    joblib.dump(model, os.path.join(args.output, "model.pkl"))
    joblib.dump(tfidf, os.path.join(args.output, "tfidf.pkl"))
    
    print(f"✅ Done in {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
