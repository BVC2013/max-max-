from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import numpy as np
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

# Custom function to override layer config
def custom_depthwise_conv2d(*args, **kwargs):
    if 'groups' in kwargs:
        del kwargs['groups']  # Remove 'groups' argument
    return DepthwiseConv2D(*args, **kwargs)

# Load the model with the custom layer
model = load_model('keras_model.h5', custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d})

app = Flask(__name__)

# Load labels from labels.txt (adjust path if needed)
with open("labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode the image from base64 and preprocess it
    image_data = data['image'].split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = (image_array / 127.5) - 1  # Normalize
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction = model.predict(image_array, verbose=0)
    predicted_index = np.argmax(prediction)

    # Map specific classes to custom labels
    if predicted_index == 5:
        predicted_class = "bhav"
    elif predicted_index == 6:
        predicted_class = "max"
    elif predicted_index == 7:
        predicted_class = "Maxon"
    else:
        predicted_class = labels[predicted_index]

    return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
