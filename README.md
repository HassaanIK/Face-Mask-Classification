# Face Mask Classification

### Overview
This project focuses on building a deep learning model for detecting whether a person is wearing a face mask or not in an image. The model is trained on a dataset containing images of people with and without masks. The goal is to predict the class of an input image as either "With Mask" or "Without Mask" and display the prediction along with the uploaded image.

### Steps
- Data Collection: The data used for this project was taken from [Kaggle](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset).
- Data Preprocessing:Images are resized to 224x224 pixels and normalized to values between 0 and 1. Data augmentation techniques are applied to it.
- Model Architecture:
  - The model architecture consists of a convolutional neural network `CNN` for feature extraction followed by a fully connected `FC` layer for classification.
  - The CNN part includes three convolutional layers with `batch normalization`, `ReLU` activation, `max pooling`, and `dropout layers` for regularization.
  - The `FC` layer consists of two linear layers with batch normalization, ReLU activation, and dropout.
- Model Training:
  - The model is trained using the `Adam` optimizer with a `learning rate` of 0.001 and a `batch size` of 32.
  - The loss function used is the `cross-entropy loss`.
  - The model is trained for 10 `epochs`.
- Model Evaluation:
  - The model is evaluated on a separate test set to assess its performance.
  - Evaluation metrics accuracy is used to measure the model's performance, 98.89 of which is achieved on test set.
- Flask Web App:
  - A Flask web application is created to provide a user interface for entering an image and getting the prediction masked or not masked.
 
### Usage 
1. Install the required dependencies using `pip install -r requirements.txt`.
2. Download the pre-trained model weights and place them in the `models/` directory.
3. Run the Flask web application using `python app.py`.
4. Access the application in your web browser at `http://localhost:5000`.

### Web App
![Screenshot (29)](https://github.com/HassaanIK/Face-Mask-Classification/assets/139614780/d7321e80-9200-40ae-bca8-e489cdadd3de)
