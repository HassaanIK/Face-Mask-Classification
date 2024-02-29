import torch
from torchvision import transforms
from PIL import Image


def predict_mask(image_path, model):

    # Define the pre-processing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")

    # Pre-process the image
    image_tensor = transform(image).unsqueeze(0)

    # Set the model to evaluation mode
    model.eval()

    # Make a prediction
    with torch.no_grad():
        output = model(image_tensor)

    # Convert the output to probabilities using softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the predicted class
    predicted_class = torch.argmax(probabilities).item()

    # Get the probability for the predicted class
    predicted_probability = probabilities[predicted_class].item()

    # Define class labels
    class_labels = ['with mask', 'without mask']

    return class_labels[predicted_class], predicted_probability
