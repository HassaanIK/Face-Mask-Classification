from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from io import BytesIO
import base64
from model import model
from predict import predict_mask

app = Flask(__name__)

model.load_state_dict(torch.load("models\\facemask_model_statedict__f.pth", map_location=torch.device('cpu')))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['file']

        class_name, probability, image = predict_mask(file, model)
        print(f'Predicted Class: {class_name}')
        print(f'Probability: {probability}')
        
        # Convert image to base64 format
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Return the prediction, probability, and image as base64
        return render_template('index.html', image=img_str, class_name=class_name, probability=probability)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
