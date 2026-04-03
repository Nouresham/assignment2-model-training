import json
import joblib
import numpy as np
import os

def init():
    global model, tfidf
    # Try different possible model paths
    model_path = 'model.pkl'
    tfidf_path = 'tfidf.pkl'
    
    # Check if running in Azure ML
    if os.path.exists('/var/azureml-app'):
        model_path = os.path.join('/var/azureml-app', 'model.pkl')
        tfidf_path = os.path.join('/var/azureml-app', 'tfidf.pkl')
    
    # Check AZUREML_MODEL_DIR
    if 'AZUREML_MODEL_DIR' in os.environ:
        model_dir = os.environ['AZUREML_MODEL_DIR']
        model_path = os.path.join(model_dir, 'model.pkl')
        tfidf_path = os.path.join(model_dir, 'tfidf.pkl')
    
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"Loading TF-IDF from: {tfidf_path}")
    tfidf = joblib.load(tfidf_path)
    print("Model loaded successfully!")

def run(raw_data):
    try:
        # Parse input
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data
        
        # Get review text
        text = data.get('reviewText', '')
        if not text:
            text = data.get('text', '')
        
        # Extract features (same as training)
        length = len(text)
        tfidf_features = tfidf.transform([text]).toarray()
        features = np.hstack([[length], tfidf_features[0]])
        
        # Predict
        prediction = model.predict([features])[0]
        
        return {
            "prediction": int(prediction),
            "sentiment": "Positive" if prediction == 1 else "Negative"
        }
    except Exception as e:
        return {"error": str(e)}
