import os
from src.models.simple_cnn import SimpleCNN
from src.models.mlp import MLP
from src.models.resnet import ResNet
from src.models.logistic_regression import LogisticRegression

# Find the root directory of the project (parent of 'src')
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define directories
DATA_DIR = os.path.join(REPO_ROOT, "data", "cifar-10")
LOG_DIR = os.path.join(REPO_ROOT, "logs")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")

# CIFAR-10 dataset download URL and batch paths
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_DOWNLOAD_DIR = os.path.join(DATA_DIR, "cifar-10-batches-py")

# src/models_config.py


# Dictionary of models with editable parameters
models_config = {
"logistic_regression": {
        "model": LogisticRegression,
        "params": {
            "num_classes": 10,
            "input_dim": 32 * 32 * 3  # CIFAR-10 image size flattened
        }
    },
    "simple_cnn": {
        "model": SimpleCNN,
        "params": {
            "num_classes": 10
        }
    }
}


