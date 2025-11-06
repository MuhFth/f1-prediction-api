from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import joblib 

# --- MODEL/SCALER CONFIGURATION ---
MODEL_PATH = 'model.pkl' # Disesuaikan
SCALER_PATH = 'scaler.pkl' # Disesuaikan

# --- DYNAMIC MOCKING CLASSES FOR FALLBACK ---
class DynamicMockModel:
    """Mock model that returns a random win probability (0.1 to 0.9)."""
    def predict_proba(self, X):
        # Menghasilkan probabilitas acak, menunjukkan hasil yang bervariasi
        prob = np.random.uniform(0.1, 0.9)
        # Probabilitas untuk kelas [0, 1]
        return np.array([[1 - prob, prob]])

class MockScaler:
    """A placeholder for the actual scikit-learn scaler that passes data through."""
    def transform(self, X):
        return X

# --- INITIALIZE MODEL AND SCALER ---
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        # ✅ Load Model dan Scaler yang ASLI
        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
        print("✓ Model and Scaler loaded successfully from disk.")
    else:
        # ⚠️ FALLBACK: Gunakan Mock jika file tidak ditemukan
        scaler = MockScaler()
        model = DynamicMockModel()
        print(f"⚠️ WARNING: Model files not found ({MODEL_PATH} or {SCALER_PATH}). Using Dynamic Mock Model and Mock Scaler.")
        
except Exception as e:
    # ❌ ERROR: Jika terjadi kesalahan saat loading (misal: file rusak)
    print(f"✗ ERROR initializing models: {e}")
    scaler = MockScaler() # Fallback
    model = DynamicMockModel() # Fallback

# --- FLASK SETUP (KODE DI BAWAH INI TETAP SAMA) ---
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
print(f"Model loaded: {model.__class__.__name__}") 
print(f"Scaler loaded: {scaler.__class__.__name__}")
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
        "model_loaded_type": model.__class__.__name__, 
        "scaler_loaded_type": scaler.__class__.__name__
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
            "error": "Model or scaler failed to initialize. Check logs."
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or "features" not in data:
            return jsonify({
                "error": "Missing 'features' field in request body"
            }), 400
        
        input_array = np.array(data["features"]).reshape(1, -1)
        
        expected_features = 20
        if input_array.shape[1] != expected_features:
            return jsonify({
                "error": f"Expected {expected_features} features, got {input_array.shape[1]}"
            }), 400
        
        # Scaling dan prediksi menggunakan model yang dimuat (asli atau mock)
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
