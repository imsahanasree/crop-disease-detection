from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load the model
try:
    model = tf.keras.models.load_model('model_vgg19.h5')
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

class_dict = {
    'Tomato Bacterial spot': 0,
    'Tomato Early blight': 1,
    'Tomato Late blight': 2,
    'Tomato Leaf Mold': 3,
    'Tomato Septoria leaf spot': 4,
    'Tomato Spider mites Two-spotted spider mite': 5,
    'Tomato Target Spot': 6,
    'Tomato Tomato Yellow Leaf Curl Virus': 7,
    'Tomato Tomato mosaic virus': 8,
    'Tomato healthy': 9
}

def prepare(img_array):
    """Prepare image array for prediction by normalizing and reshaping"""
    img_array = img_array.astype('float32') / 255.0
    return img_array.reshape(-1, 128, 128, 3)

def prediction_cls(prediction):
    for key, clss in class_dict.items():
        if np.argmax(prediction) == clss:
            return key

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/details')
def details():
    disease = request.args.get('disease')
    return render_template('details.html', disease=disease)

@app.route('/calendar')
def calendar():
    return render_template('calendar.html')

@app.route('/growing-tips')
def growing_tips():
    return render_template('growing-tips.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        if not model:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Process the image
        img = Image.open(io.BytesIO(file.read()))
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img = img.resize((128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = prepare(img_array)
        
        # Make prediction
        prediction = model.predict(img_array)
        result = prediction_cls(prediction)
        
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
