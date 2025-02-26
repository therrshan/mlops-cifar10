import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import src.config as cfg


class CIFAR10Parser:
    def __init__(self, normalize=False):
        self.repo_root = cfg.REPO_ROOT  # Dynamically get the repo root
        self.data_dir = cfg.DATA_DIR  # Use repo root for data_dir
        self.download_dir = cfg.CIFAR10_DOWNLOAD_DIR
        self.normalize = normalize

        if not os.path.exists(self.download_dir):
            raise FileNotFoundError("CIFAR-10 dataset not found. Please download and extract it first.")

    def _load_data_batch(self, batch_file):
        """
        Load a batch of CIFAR-10 data from the specified batch file.
        """
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            images = batch[b'data']
            labels = batch[b'labels']

        # Reshape images to (N, 3, 32, 32) format
        images = np.array(images).reshape(-1, 3, 32, 32).astype(np.float32)
        labels = np.array(labels)

        # Normalize images if needed
        if self.normalize:
            images = images / 255.0  # Normalize to [0, 1]

        return images, labels

    def _load_all_batches(self):
        """
        Load all CIFAR-10 training batches and combine them.
        """
        data_batches = []
        labels_batches = []

        for i in range(1, 6):
            batch_file = os.path.join(self.download_dir, f'data_batch_{i}')
            data, labels = self._load_data_batch(batch_file)
            data_batches.append(data)
            labels_batches.append(labels)

        data_batches = np.concatenate(data_batches, axis=0)
        labels_batches = np.concatenate(labels_batches, axis=0)

        return data_batches, labels_batches

    def get_train_data(self):
        """
        Get the training data (images and labels).
        """
        return self._load_all_batches()

    def get_test_data(self):
        """
        Get the test data (images and labels).
        """
        test_batch_file = os.path.join(self.download_dir, 'test_batch')
        return self._load_data_batch(test_batch_file)


class CIFAR10Dataset(Dataset):
    def __init__(self, train=True, normalize=False, transform=None):
        """
        Args:
            data_dir (str): Path to the dataset directory.
            train (bool): If True, loads training data; otherwise, validation data.
            normalize (bool): Whether to normalize the data or not.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.train = train
        self.normalize = normalize
        self.transform = transform

        # Initialize CIFAR10Parser to load data
        self.parser = CIFAR10Parser(normalize=self.normalize)

        # Load data
        if self.train:
            self.data, self.labels = self.parser.get_train_data()
        else:
            self.data, self.labels = self.parser.get_test_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        # Apply any transformation (e.g., normalization, augmentation)
        if self.transform:
            image = self.transform(image)

        # Convert image to PyTorch tensor
        image = torch.tensor(image, dtype=torch.float32)

        return image, label


def get_data_loaders(batch_size=64, data_dir=cfg.CIFAR10_DOWNLOAD_DIR, normalize=False):
    """Returns DataLoader for CIFAR-10 dataset with applied transformations."""
    # Define the transformations (convert to tensor, normalize, etc.)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to ImageNet values
    ])

    # Create training and validation datasets
    train_dataset = CIFAR10Dataset(data_dir=data_dir, train=True, normalize=normalize, transform=transform)
    val_dataset = CIFAR10Dataset(data_dir=data_dir, train=False, normalize=normalize, transform=transform)

    # Create DataLoader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
