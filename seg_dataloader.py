import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from random import random
import torchvision


class Seg_Dataset(Dataset):
    def __init__(self, base_dir, metadata_file, is_train=True, device=None):
        """
        Dataset loader for labeled brain MRI slices, segmentation masks, and labels.

        Parameters:
        - base_dir: str, path to the directory containing train/val folders.
        - metadata_file: str, name of the metadata CSV file.
        - is_train: bool, whether the dataset is for training or validation.
        - device: torch.device, device to load tensors on (CPU or GPU).
        """
        self.device = device
        self.is_train = is_train

        # Load metadata
        metadata_path = os.path.join(base_dir, metadata_file)
        assert os.path.exists(metadata_path), f"Metadata file {metadata_path} not found!"
        self.data_df = pd.read_csv(metadata_path)
        self.data_df = self.data_df[self.data_df["TRAIN"] == (1 if is_train else 0)]
        print(self.data_df['SURV'].mean(),self.data_df['SURV'].std())
        longs = (self.data_df['SURV']>365).sum()
        print('##################')
        print(longs)

        # Ensure there are samples
        assert len(self.data_df) > 0, f"No data found for {'train' if is_train else 'val'} split!"

        # Define paths
        split = "train" if is_train else "val"
        self.img_dir = os.path.join(base_dir, split, "img")
        self.seg_dir = os.path.join(base_dir, split, "mask")

        print(f"Loaded {len(self.data_df)} samples for {'train' if is_train else 'val'}.")

    def __len__(self):
        return len(self.data_df)

    def _add_noise(self, tensor, p=0.3, noise_strength=0.3):
        """Add random noise to the tensor."""
        if random() > p:
            return tensor
        noise = torch.randn_like(tensor, device=self.device) * noise_strength
        return tensor + noise

    def _rotate(self, tensor, p=0.5):
        """Rotate the tensor by 90, 180, or 270 degrees."""
        if random() > p:
            return tensor
        rot_extent = np.random.randint(1, 4)  # Randomly select 90, 180, or 270 degrees
        return torch.rot90(tensor, rot_extent, dims=[0, 1])

    def _flip(self, tensor, p=0.5):
        """Flip the tensor horizontally or vertically."""
        if random() > p:
            return tensor
        flip_dim = torch.randint(1, 3, (1,)).item()
        return torch.flip(tensor, dims=[flip_dim])

    def _contrast(self, tensor, p=0.3):
        """Adjust the contrast of the tensor."""
        if random() > p:
            return tensor
        factor = random() * 0.2 + 0.9
        mean = tensor.mean()
        min_val, max_val = tensor.min().item(), tensor.max().item()
        tensor = (tensor - mean) * factor + mean
        return tensor.clip(min_val, max_val)

    def _blur(self, tensor, p=0.3):
        """Apply Gaussian blur to the tensor."""
        if random() > p:
            return tensor
        kernel_size = 3
        blur_transform = torchvision.transforms.GaussianBlur(kernel_size=kernel_size)
        return blur_transform(tensor)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        file_name = row["FP"]
        label = row["SURV"]

        img_path = os.path.join(self.img_dir, file_name)
        seg_path = os.path.join(self.seg_dir, file_name)

        assert os.path.exists(img_path), f"Image file {img_path} not found!"
        assert os.path.exists(seg_path), f"Segmentation file {seg_path} not found!"

        # Load image and segmentation mask
        img = torch.tensor(np.load(img_path), dtype=torch.float32, device=self.device)
        seg = torch.tensor(np.load(seg_path), dtype=torch.float32, device=self.device)

        # Apply augmentations
        if self.is_train:
            transforms_mri = [self._add_noise, self._rotate, self._flip, self._contrast, self._blur]
            transforms_seg = [self._rotate, self._flip]
            np.random.shuffle(transforms_mri)
            for transform in transforms_mri:
                img = transform(img)
            for transform in transforms_seg:
                seg = transform(seg)

        img = torch.swapaxes(img, 0, 2)
        seg = torch.swapaxes(seg, 0, 2)
        seg = seg[1]
        # print(seg.shape)




        return file_name,img, seg, torch.tensor(label>300, device=self.device)


def get_data_loaders(base_dir, metadata_file, batch_size=16, num_workers=4, device=None):
    """
    Prepare and return DataLoaders for train and validation datasets.

    Parameters:
    - base_dir: str, path to the directory containing train/val folders.
    - metadata_file: str, name of the metadata CSV file.
    - batch_size: int, batch size for DataLoader.
    - num_workers: int, number of workers for DataLoader.
    - device: torch.device, device to load tensors on.

    Returns:
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    """
    train_dataset = Seg_Dataset(base_dir, metadata_file, is_train=True, device=device)
    val_dataset = Seg_Dataset(base_dir, metadata_file, is_train=False, device=device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    base_dir = "/absolute/path/to/slices"
    metadata_file = "slice_meta.csv"
    batch_size = 16
    device = torch.device("cpu")

    train_loader, val_loader = get_data_loaders(base_dir, metadata_file, batch_size=batch_size, device=device)

    for imgs, segs, labels in train_loader:
        print(f"Train batch - Images: {imgs.shape}, Segmentation Maps: {segs.shape}, Labels: {labels.shape}")
        break

    for imgs, segs, labels in val_loader:
        print(f"Validation batch - Images: {imgs.shape}, Segmentation Maps: {segs.shape}, Labels: {labels.shape}")
        break
