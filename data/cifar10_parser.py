import os
import tarfile
import urllib.request


class CIFAR10Downloader:
    def __init__(self, data_dir="data/cifar-10"):
        self.data_dir = data_dir
        self.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        self.download_path = os.path.join(self.data_dir, "cifar-10-python.tar.gz")
        self.extract_dir = os.path.join(self.data_dir, "cifar-10-batches-py")

        if not os.path.exists(self.extract_dir):
            self._download_and_extract()

    def _download_and_extract(self):
        """
        Download and extract CIFAR-10 dataset if not already present.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Download the dataset if not already present
        print("Downloading CIFAR-10 dataset...")
        if not os.path.exists(self.download_path):
            urllib.request.urlretrieve(self.url, self.download_path)

        # Extract if not already extracted
        if not os.path.exists(self.extract_dir):
            with tarfile.open(self.download_path, 'r:gz') as tar:
                tar.extractall(path=self.data_dir)
            print("CIFAR-10 dataset extracted.")
        else:
            print("CIFAR-10 dataset already exists.")
