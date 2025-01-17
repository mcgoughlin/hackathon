import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from random import random
import torchvision


class BrainMRI_Dataset(Dataset):
    def __init__(self, slices_dir_path, metadata_file_name, is_train=True, device=None):
        """
        Dataset loader for labelled brain MRI slices using a metadata CSV.

        Parameters:
        - slices_dir_path: str, path to the directory containing slices and metadata.
        - metadata_file_name: str, name of the metadata CSV file.
        - is_train: bool, whether the dataset is for training or validation.
        - device: torch.device, device to load tensors on (CPU or GPU).
        """
        self.is_train = is_train
        self.device = device

        # Resolve full path to metadata
        metadata_path = os.path.join(slices_dir_path, metadata_file_name)
        assert os.path.exists(metadata_path), f"Metadata file {metadata_path} does not exist!"
        self.data_df = pd.read_csv(metadata_path)


        # Filter rows based on TRAIN column
        self.data_df = self.data_df[self.data_df["TRAIN"] == (1 if is_train else 0)]

        # Ensure there are samples
        assert len(self.data_df) > 0, f"No data found for {'train' if is_train else 'val'} split!"

        # Set subdirectory based on train/val
        sub_dir = "train" if is_train else "val"
        self.data_dir = os.path.join(slices_dir_path, sub_dir)

        print(f"Loaded {len(self.data_df)} samples for {'train' if is_train else 'val'}.")

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
        rot_extent = np.random.randint(1, 4)  # Randomly select 90, 180, or 270 degrees
        return torch.rot90(tensor, rot_extent, dims=[0, 1])  # Rotate height and width


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
        row = self.data_df.iloc[idx]
        img_name = row["FP"]
        label = row["LABEL"]

        # Resolve full path to image
        img_path = os.path.join(self.data_dir, img_name)
        assert os.path.exists(img_path), f"Image file {img_path} does not exist!"

        # Load NumPy image
        image = np.load(img_path).astype(np.float32)
        image = torch.tensor(image, device=self.device)  # Convert to PyTorch tensor
        
        # Apply augmentations if it's training data
        if self.is_train:
            transforms = [self._add_noise, self._rotate, self._flip, self._contrast, self._blur]
            np.random.shuffle(transforms)
            for transform in transforms:
                image = transform(image)

        # swap first and third axes
        image = torch.swapaxes(image,0,2)

        return image, torch.tensor(label, device=self.device)


def get_data_loaders(slices_dir_path, metadata_file_name, batch_size=16, num_workers=4, device=None):
    """
    Prepare and return DataLoaders for train and val datasets.

    Parameters:
    - slices_dir_path: str, path to the directory containing slices and metadata.
    - metadata_file_name: str, name of the metadata CSV file.
    - batch_size: int, batch size for DataLoader.
    - num_workers: int, number of workers for DataLoader.
    - device: torch.device, device to load tensors on.

    Returns:
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    """
    # Create datasets
    train_dataset = BrainMRI_Dataset(slices_dir_path, metadata_file_name, is_train=True, device=device)
    val_dataset = BrainMRI_Dataset(slices_dir_path, metadata_file_name, is_train=False, device=device)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    slices_dir_path = "/absolute/path/to/slices"
    metadata_file_name = "slice_meta.csv"
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_data_loaders(slices_dir_path, metadata_file_name, batch_size=batch_size, device=device)

    for images, labels in train_loader:
        print(f"Train batch - Images: {images.shape}, Labels: {labels}")
        break

    for images, labels in val_loader:
        print(f"Validation batch - Images: {images.shape}, Labels: {labels}")
        break
