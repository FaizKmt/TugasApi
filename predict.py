from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import requests
from datasets import load_dataset

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables
model = None
id_to_label = None
emotion_labels = {}  # Akan diisi otomatis dari dataset

def load_emotion_labels():
    """Load emotion labels dari dataset"""
    global emotion_labels
    try:
        print("Loading emotion labels from dataset...")
        ds = load_dataset("elvanromp/emosi", split='train')
        
        # Mengumpulkan unique labels dan contoh teks
        unique_emotions = {}
        for item in ds:
            label = item['label']
            if label not in unique_emotions:
                unique_emotions[label] = {
                    'text_example': item['text'],
                    'label_text': item['label_text']  # Mengambil label_text dari dataset
                }
        
        emotion_labels = unique_emotions
        print(f"Loaded {len(emotion_labels)} emotion labels")
        return True
    except Exception as e:
        print(f"Error loading emotion labels: {e}")
        return False

def initialize_model():
    global model, id_to_label
    try:
        print("Loading model...")
        with open('emotion_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            id_to_label = model_data['id_to_label']
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "Server is running",
        "available_emotions": list(emotion_labels.keys())
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        print("Received request:", request.form)
        
        text = request.form.get('text')
        if not text:
            print("No text provided")
            return jsonify({
                'status': 'error',
                'error': 'No text provided'
            }), 400

        if not model:
            print("Model not loaded")
            return jsonify({
                'status': 'error',
                'error': 'Model not loaded'
            }), 500

        # Make prediction
        print(f"Making prediction for text: {text}")
        numeric_prediction = model.predict([text])[0]
        emotion = id_to_label[numeric_prediction]
        
        # Get emotion details from loaded labels
        emotion_info = emotion_labels.get(emotion, {})
        label_text = emotion_info.get('label_text', 'Label tidak tersedia')
        example_text = emotion_info.get('text_example', 'Contoh tidak tersedia')

        return jsonify({
            'status': 'success',
            'prediction': emotion,
            'label_text': label_text,
            'example_text': example_text,
            'input_text': text
        })

    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "status": "success",
        "message": "Test endpoint working",
        "available_emotions": emotion_labels
    })

if __name__ == '__main__':
    if initialize_model() and load_emotion_labels():
        print("Starting server on http://localhost:8000")
        print("Available emotions:", list(emotion_labels.keys()))
        print("Available endpoints:")
        print("  - GET  /")
        print("  - GET  /test")
        print("  - POST /api/predict")
        app.run(host='0.0.0.0', port=8000, debug=True)
    else:
        print("Failed to initialize server. Check errors above.")