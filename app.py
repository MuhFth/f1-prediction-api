from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os

# --- MOCKING CLASSES FOR LOCAL TESTING ---
class MockScaler:
    """A placeholder for the actual scikit-learn scaler."""
    def transform(self, X):
        print(f"Scaler transform called with shape: {X.shape}")
        return X

class MockModel:
    """A placeholder for the actual joblib-loaded model."""
    def predict_proba(self, X):
        print(f"Model predict_proba called with shape: {X.shape}")
        return np.array([[0.2, 0.8]])

# Initialize the mock objects
try:
    model = MockModel()
    scaler = MockScaler()
    print("✓ Mock Model and Scaler initialized successfully")
except Exception as e:
    print(f"✗ Error initializing models: {e}")
    model = None
    scaler = None

# --- FLASK SETUP ---
app = Flask(__name__)

# CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Untuk production, ganti dengan domain spesifik
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Startup message
print("=" * 50)
print("🚀 Flask Application Starting...")
print(f"Model loaded: {model is not None}")
print(f"Scaler loaded: {scaler is not None}")
print(f"Port: {os.environ.get('PORT', '8000')}")
print("=" * 50)

@app.route("/", methods=["GET"])
def root():
    """Root endpoint - basic info."""
    return jsonify({
        "message": "F1 Winner Predictor API",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for Railway."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })

@app.route("/test", methods=["GET"])
def test():
    """Simple test endpoint."""
    return jsonify({
        "message": "API is working!", 
        "test": "success"
    })

@app.route("/predict", methods=["GET"])
def predict_info():
    """Info about predict endpoint."""
    return jsonify({
        "message": "Use POST method to make predictions",
        "method": "POST",
        "endpoint": "/predict",
        "example": {
            "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 
                        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Receives feature data and returns win probability."""
    
    if model is None or scaler is None:
        return jsonify({
            "error": "Model or scaler not loaded"
        }), 503
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or "features" not in data:
            return jsonify({
                "error": "Missing 'features' field in request body"
            }), 400
        
        # Convert to numpy array
        input_array = np.array(data["features"]).reshape(1, -1)
        
        # Validate feature count
        expected_features = 20
        if input_array.shape[1] != expected_features:
            return jsonify({
                "error": f"Expected {expected_features} features, got {input_array.shape[1]}"
            }), 400
        
        # Scale and predict
        input_scaled = scaler.transform(input_array)
        probability = model.predict_proba(input_scaled)[:, 1].item()
        
        return jsonify({
            "winner_probability": round(probability, 4),
            "status": "success"
        })
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "error": "Internal server error"
    }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)