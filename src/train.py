# src/train.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.config import *
from src.dataset import CIFAR10Dataset  # Your custom dataset loader
from src.logger import CustomLogger  # Logger for training details

# Initialize Logger
logger = CustomLogger(log_dir="../logs").get_logger()

# Directory for saving models and checkpoints
checkpoint_dir = "../checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

def train_model(model, params, train_loader, val_loader, model_name, epochs=10, checkpoint_interval=5):
    """Train the model and save checkpoints."""
    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model(**params).to(device)  # Initialize model with params
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    current_checkpoint_dir = os.path.join(checkpoint_dir, model_name)
    os.makedirs(current_checkpoint_dir, exist_ok=True)

    best_val_accuracy = 0.0

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Training loop
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Log loss and save checkpoint
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

        # Validation loop
        val_accuracy = validate_model(model, val_loader, device)
        logger.info(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_accuracy:.4f}")

        # Save checkpoint if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path = os.path.join(current_checkpoint_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")

        # Save model at the checkpoint interval
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(current_checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(current_checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")


def validate_model(model, val_loader, device):
    """Validation loop to calculate accuracy."""
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def get_data_loaders(batch_size=64):
    """Get CIFAR-10 dataset loaders."""
    train_dataset = CIFAR10Dataset(train=True)
    val_dataset = CIFAR10Dataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    # Load data
    train_loader, val_loader = get_data_loaders()

    # Train models as per the configuration
    for model_name, config in models_config.items():
        logger.info(f"Training model: {model_name}")
        model_class = config["model"]
        params = config["params"]
        train_model(model_class, params, train_loader, val_loader, model_name, epochs=10)
