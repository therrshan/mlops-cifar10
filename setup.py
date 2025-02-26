import os

# Define the directory structure relative to the current base directory
directories = [
    "data/cifar-10",
    "notebooks",
    "src/models",
    "logs",
    "results"
]

# Define files to be created
files_to_create = [
    "README.md",
    "requirements.txt",
    "train.sh",
    "src/__init__.py",
    "src/config.py",
    "src/dataset.py",
    "src/train.py",
    "src/evaluate.py",
    "src/infer.py",
    "src/utils.py",
    "src/models/__init__.py",
    "src/models/logistic_regression.py",
    "src/models/mlp.py",
    "src/models/simple_cnn.py",
    "src/models/deeper_cnn.py",
    "src/models/vgg.py",
    "src/models/resnet.py",
    "src/models/wideresnet.py",
    "src/models/efficientnet.py",
    "notebooks/1_data_exploration.ipynb",
    "notebooks/2_baseline_models.ipynb",
    "notebooks/3_simple_cnn.ipynb",
    "notebooks/4_advanced_cnns.ipynb",
    "notebooks/5_model_comparison.ipynb",
]

# Function to create directories
def create_directories(directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

# Function to create files
def create_files(files):
    for file in files:
        directory = os.path.dirname(file)
        if directory:  # Only create a directory if it's not empty
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(file):
            with open(file, "w") as f:
                f.write("")  # Create an empty file
            print(f"Created file: {file}")

# Create directories
create_directories(directories)

# Create files
create_files(files_to_create)

print("\n✅ Project setup complete!")
