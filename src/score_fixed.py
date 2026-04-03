import json
import joblib
import numpy as np
import os

model = None
tfidf = None

def init():
    global model, tfidf
    print("Initializing model...", flush=True)
    
    # Look for model files
    model_dir = os.environ.get('AZUREML_MODEL_DIR', '.')
    model_path = os.path.join(model_dir, 'model.pkl')
    tfidf_path = os.path.join(model_dir, 'tfidf.pkl')
    
    print(f"Model path: {model_path}", flush=True)
    print(f"TF-IDF path: {tfidf_path}", flush=True)
    
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    
    print("Model loaded successfully!", flush=True)

def run(raw_data):
    try:
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data
        
        text = data.get('reviewText', '') or data.get('text', '')
        
        # Extract features
        length = len(text)
        tfidf_features = tfidf.transform([text]).toarray()
        features = np.hstack([[length], tfidf_features[0]])
        
        prediction = model.predict([features])[0]
        
        return {
            "prediction": int(prediction),
            "sentiment": "Positive" if prediction == 1 else "Negative"
        }
    except Exception as e:
        return {"error": str(e)}
