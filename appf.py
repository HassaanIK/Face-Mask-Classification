from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import io
import base64
from torchvision import transforms
from face_mask_detection import FaceMaskDetectionModel
import numpy as np

app = Flask(__name__)

# Load the model
model = torch.load("model\\facemask_detection_model_f.pth", map_location=torch.device('cpu'))
model.eval()

# Define the pre-processing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.9),
    transforms.ToTensor()
])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        image = request.files['image']
        
        # Pre-process the image
        image_tensor = transform(Image.open(io.BytesIO(image.read())).convert("RGB")).unsqueeze(0)
        
        # Set the model to evaluation mode
        model.eval()

        # Make a prediction
        with torch.no_grad():
            output = model(image_tensor)

        # Convert the output to probabilities using softmax
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get the predicted class
        predicted_class = torch.argmax(probabilities).item()

        # Determine the prediction label based on the predicted class
        prediction = "with mask" if predicted_class == 0 else "without mask"

        # Return the prediction along with the uploaded image
        image.seek(0)  # Reset the file pointer
        image_base64 = base64.b64encode(image.read()).decode('utf-8')
        return jsonify({'prediction': prediction, 'image': image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
@app.route('/show_image', methods=['POST'])
def show_image():
    try:
        # Get the image from the request
        image = request.files['image']

        # Convert the image to a NumPy array
        image_array = np.array(Image.open(io.BytesIO(image.read())))

        # Return the base64-encoded image as a JSON response
        image.seek(0)  # Reset the file pointer
        image_base64 = base64.b64encode(image.read()).decode('utf-8')
        return jsonify({'image': image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True)
