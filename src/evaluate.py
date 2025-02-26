import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from src.dataset import *
import numpy as np
from models.simple_cnn import SimpleCNN
from models.logistic_regression import LogisticRegression
import src.config as cfg

def load_model(model_path, model_class,params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(**params).to(device)
    model.load_state_dict(torch.load(model_path))  # Load saved model weights
    model.eval()  # Set the model to evaluation mode
    return model


# Define a transformation pipeline for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 normalization
])



# Run inference on a single batch
def run_inference(model, data_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in data_loader:
            # Send data to the correct device (CPU/GPU)
            inputs, labels = inputs.cuda(), labels.cuda()  # Use `.to(device)` if running on GPU

            # Forward pass
            outputs = model(inputs)  # Pass inputs through the model
            _, preds = torch.max(outputs, 1)  # Get predicted class

            all_preds.extend(preds.cpu().numpy())  # Move to CPU and collect predictions
            all_labels.extend(labels.cpu().numpy())  # Collect true labels

    return np.array(all_preds), np.array(all_labels)


# Example usage
if __name__ == "__main__":

    for model_name, config in cfg.models_config.items():
        model_dir = os.path.join(cfg.CHECKPOINT_DIR, model_name)
        model_path = os.path.join(model_dir, 'final_model.pth')
        model = load_model(model_path, config['model'], config['params'])

        # Load test data
        test_dataset = CIFAR10Dataset(train=False)
        test_loader = DataLoader(test_dataset, shuffle=False)

        # Run inference
        predictions, labels = run_inference(model, test_loader)

        # Print results
        print(f"Predictions: {predictions[:10]}")
        print(f"True Labels: {labels[:10]}")
