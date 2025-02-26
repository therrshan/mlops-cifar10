import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from models.simple_cnn import SimpleCNN
from models.logistic_regression import LogisticRegression
import src.config as cfg


# Load model function
def load_model(model_path, model_class, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(**params).to(device)
    model.load_state_dict(torch.load(model_path))  # Load saved model weights
    model.eval()  # Set the model to evaluation mode
    return model


# Preprocessing function for images
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resizing image to 32x32 for CIFAR-10
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 normalization
    ])

    image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
    return transform(image).unsqueeze(0)  # Add batch dimension


# Run inference on a single image
def run_inference(model, image_path):
    input_tensor = preprocess_image(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)

    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(input_tensor)  # Pass the input through the model
        _, predicted_class = torch.max(output, 1)  # Get the predicted class label

    return predicted_class.item()


# Example usage
if __name__ == "__main__":
    # Accept model name as input
    model_name = input("Enter the model name (simple_cnn or logistic_regression): ").strip().lower()

    if model_name not in ['simple_cnn', 'logistic_regression']:
        print("Invalid model name. Please enter 'simple_cnn' or 'logistic_regression'.")
        exit(1)

    # Set up model configuration based on input
    model_class = SimpleCNN if model_name == 'simple_cnn' else LogisticRegression
    model_dir = os.path.join(cfg.CHECKPOINT_DIR, model_name)
    model_path = os.path.join(model_dir, 'final_model.pth')

    # Load the model
    model = load_model(model_path, model_class, cfg.models_config[model_name]['params'])

    # Path to the image to be classified
    image_path = os.path.join(cfg.REPO_ROOT, "images.jpg")  # Change this to the actual image path

    # Run inference on the image
    prediction = run_inference(model, image_path)

    # Output the prediction
    print(f"Predicted class: {prediction}")
