from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import io
import base64
from torchvision import transforms
from model import model
from predict import predict_mask
import numpy as np

app = Flask(__name__)

model.load_state_dict(torch.load("models\\facemask_model_statedict__f.pth", map_location=torch.device('cpu')))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        image = request.files['image']
        predicted_class, probability = predict_mask(image, model)
        print(f'Predicted Class: {predicted_class}')
        print(f'Probability: {probability}')
        
        # Convert the image to base64
        image_base64 = base64.b64encode(image.read()).decode('utf-8')

        # Return the prediction, probability, and image as base64
        return jsonify({'prediction': predicted_class, 'probability': probability, 'image': image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
