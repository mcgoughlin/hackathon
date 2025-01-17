import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from random import random
import torchvision

class BrainMRI_Dataset(Dataset):
    def __init__(self, path, is_train=True, device=None):
        """
        Dataset loader for labelled brain MRI slices (NumPy format with 3 channels).
        
        Parameters:
        - path: str, path to the base data directory.
        - is_train: bool, whether the dataset is for training or validation.
        - device: torch.device, device to load tensors on (CPU or GPU).
        """
        self.is_train = is_train
        self.device = device

        # Determine subdirectory (train or val) based on is_train
        name = "train" if is_train else "val"
        self.data_dir = os.path.join(path, "slices", name)
        assert os.path.exists(self.data_dir), f"Directory {self.data_dir} does not exist!"

        # CSV for labels
        csv_path = os.path.join(self.data_dir, "labels.csv")
        assert os.path.exists(csv_path), f"CSV file {csv_path} not found!"
        self.data_df = pd.read_csv(csv_path)

        # Image directory
        self.image_dir = os.path.join(self.data_dir, "images")
        assert os.path.exists(self.image_dir), f"Image directory {self.image_dir} does not exist!"

        print(f"Loaded {len(self.data_df)} samples from {name}.")

    def __len__(self):
        return len(self.data_df)

    def _add_noise(self, tensor, p=0.3, noise_strength=0.3):
        if random() > p:
            return tensor
        noise = torch.randn_like(tensor, device=self.device) * noise_strength
        return tensor + noise

    def _rotate(self, tensor, p=0.5):
        if random() > p:
            return tensor
        rot_extent = torch.randint(1, 4, (1,)).item()
        return torch.rot90(tensor, rot_extent, dims=[-2, -1])

    def _flip(self, tensor, p=0.5):
        if random() > p:
            return tensor
        flip_dim = torch.randint(1, 3, (1,)).item()
        return torch.flip(tensor, dims=[-flip_dim])

    def _contrast(self, tensor, p=0.3):
        if random() > p:
            return tensor
        factor = random() * 0.2 + 0.9
        mean = tensor.mean()
        min_val, max_val = tensor.min().item(), tensor.max().item()
        tensor = (tensor - mean) * factor + mean
        return tensor.clip(min_val, max_val)

    def _blur(self, tensor, p=0.3):
        if random() > p:
            return tensor
        kernel_size = 3
        blur_transform = torchvision.transforms.GaussianBlur(kernel_size=kernel_size)
        return blur_transform(tensor)

    def __getitem__(self, idx):
        # Retrieve file path and label
        img_name = self.data_df.iloc[idx]["Filename"]
        label = self.data_df.iloc[idx]["Label"]

        # Load NumPy image
        img_path = os.path.join(self.image_dir, img_name)
        image = np.load(img_path).astype(np.float32)
        image = torch.tensor(image, device=self.device)

        # Apply augmentations if it's training data
        if self.is_train:
            transforms = [self._add_noise, self._rotate, self._flip, self._contrast, self._blur]
            np.random.shuffle(transforms)
            for transform in transforms:
                image = transform(image)

        return image, torch.tensor(label, device=self.device)


def get_data_loaders(data_dir, batch_size=16, num_workers=4, device=None):
    """
    Prepare and return DataLoaders for train and val datasets.

    Parameters:
    - data_dir: str, path to the base directory containing "slices".
    - batch_size: int, batch size for DataLoader.
    - num_workers: int, number of workers for DataLoader.
    - device: torch.device, device to load tensors on.

    Returns:
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    """
    # Create datasets
    train_dataset = BrainMRI_Dataset(path=data_dir, is_train=True, device=device)
    val_dataset = BrainMRI_Dataset(path=data_dir, is_train=False, device=device)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


if __name__ == "__main__":
    data_dir = "absolute/path/to/data"
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_data_loaders(data_dir, batch_size=batch_size, device=device)

    for images, labels in train_loader:
        print(f"Train batch - Images: {images.shape}, Labels: {labels}")
        break

    for images, labels in val_loader:
        print(f"Validation batch - Images: {images.shape}, Labels: {labels}")
        break
