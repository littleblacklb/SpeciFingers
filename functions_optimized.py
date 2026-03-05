"""
Optimized data loading for SpeciFingers training.
Replaces functions.py with faster dataset classes that:
- Load pre-packed .npz files (1 file read instead of 49)
- Support full RAM caching
- Use efficient prefetching
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils import data
from pathlib import Path
from tqdm import tqdm

# Re-export original utilities and models
from functions import (
    labels2cat,
    labels2onehot,
    onehot2labels,
    cat2labels,
    conv2D_output_size,
    EncoderCNN,
    ResCNNEncoder,
    AlexCNNEncoder,
    DecoderRNN,
    CRNN_final_prediction,
    CRNN_final_prediction_R,
)


class ViTCNNEncoder(nn.Module):
    """
    ViT-based encoder for video sequences.
    Drop-in replacement for AlexCNNEncoder with same interface.
    Uses pretrained ViT-B/16 as backbone.
    """

    def __init__(self, fc_hidden1=512, fc_hidden2=256, drop_p=0.3, CNN_embed_dim=256):
        super(ViTCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        # Load pretrained ViT-B/16
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # Remove classification head (returns 768-dim features)
        self.vit.heads = nn.Identity()

        # Freeze ViT backbone (same as AlexNet approach)
        for param in self.vit.parameters():
            param.requires_grad = False

        # Custom FC layers (ViT-B outputs 768-dim)
        self.fc1 = nn.Linear(768, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ViT forward (frozen)
            with torch.no_grad():
                x = self.vit(x_3d[:, t, :, :, :])  # (batch, 768)

            # FC layers (trainable)
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # Shape: (batch, time_step, CNN_embed_dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        return cnn_embed_seq


class FasterViTCNNEncoder(nn.Module):
    """
    FasterViT-based encoder for video sequences.
    """

    def __init__(
        self,
        fc_hidden1=512,
        fc_hidden2=256,
        drop_p=0.3,
        CNN_embed_dim=256,
        model_name="faster_vit_0_224",
    ):
        super(FasterViTCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        try:
            from fastervit import create_model
        except ImportError:
            raise ImportError("Please install fastervit: pip install fastervit")

        # Fix for PyTorch 2.6+ weights_only default change
        # Add argparse.Namespace to safe globals for FasterViT pretrained weights
        import argparse

        try:
            torch.serialization.add_safe_globals([argparse.Namespace])
        except AttributeError:
            pass  # Older PyTorch versions don't have this

        # Load pretrained FasterViT
        try:
            self.vit = create_model(model_name, pretrained=True)
            print(f"Loaded pretrained weights for {model_name}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights for {model_name}: {e}")
            self.vit = create_model(model_name, pretrained=False)

        # Determine feature dim and remove head
        # faster_vit_0_224 usually has 512 features
        # We need to find the correct attribute
        num_features = 512
        if hasattr(self.vit, "num_features"):
            num_features = self.vit.num_features
        elif hasattr(self.vit, "head"):
            if isinstance(self.vit.head, nn.Linear):
                num_features = self.vit.head.in_features
            self.vit.head = nn.Identity()

        # Fallback if unsure, do a dry run
        with torch.no_grad():
            try:
                dummy = torch.zeros(1, 3, 224, 224)
                out = self.vit(dummy)
                num_features = out.shape[1]
            except:
                pass

        # Freeze backbone
        for param in self.vit.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(num_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.vit(x_3d[:, t, :, :, :])  # (batch, features)

            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # Shape: (batch, time_step, CNN_embed_dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        return cnn_embed_seq


class EfficientViTCNNEncoder(nn.Module):
    """
    EfficientViT-based encoder for video sequences.
    Uses pretrained EfficientViT from timm as backbone.
    Default model: efficientvit_b1.r224_in1k (~9M params, 224x224 input).
    """

    def __init__(
        self,
        fc_hidden1=512,
        fc_hidden2=256,
        drop_p=0.3,
        CNN_embed_dim=256,
        model_name="efficientvit_b1.r224_in1k",
    ):
        super(EfficientViTCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")

        # Load pretrained EfficientViT with classification head removed
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        print(f"Loaded EfficientViT '{model_name}' with {num_features}-dim features")

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Trainable projection head
        self.fc1 = nn.Linear(num_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.backbone(x_3d[:, t, :, :, :])  # (batch, num_features)

            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # Shape: (batch, time_step, CNN_embed_dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        return cnn_embed_seq


class Dataset_CRNN_Fast(data.Dataset):
    """
    Optimized dataset that loads pre-packed .npz files.
    Supports optional RAM caching with memory limit to prevent OOM.
    """

    def __init__(
        self, data_path, samples, labels, cache_in_ram=True, max_cache_gb=None
    ):
        """
        Args:
            data_path: Path to packed_data directory
            samples: List of .npz file paths (relative to data_path)
            labels: List of integer labels
            cache_in_ram: If True, cache samples in RAM (with memory limit)
            max_cache_gb: Maximum GB to use for cache (None = auto-detect available)
        """
        self.data_path = Path(data_path)
        self.samples = samples
        self.labels = labels
        self.cache_in_ram = cache_in_ram
        self._cache = {} if cache_in_ram else None
        self._cache_full = False
        self._cache_bytes = 0

        # Set max cache size
        if cache_in_ram and max_cache_gb is None:
            # Auto-detect: use 50% of available RAM, max 8GB
            try:
                import psutil

                available_gb = psutil.virtual_memory().available / (1024**3)
                self._max_cache_bytes = int(min(available_gb * 0.5, 8) * 1024**3)
            except ImportError:
                # psutil not available, use conservative 4GB limit
                self._max_cache_bytes = 4 * 1024**3
        elif cache_in_ram:
            self._max_cache_bytes = int(max_cache_gb * 1024**3)
        else:
            self._max_cache_bytes = 0

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, idx):
        """Load a single sample from disk or cache."""
        if self.cache_in_ram and idx in self._cache:
            return self._cache[idx]

        npz_path = self.data_path / self.samples[idx]
        with np.load(npz_path) as data:
            frames = data["frames"]  # Shape: (num_frames, 3, H, W)

        # Convert to tensor
        X = torch.from_numpy(frames)

        # Cache if enabled and not full
        if self.cache_in_ram and not self._cache_full:
            sample_bytes = X.numel() * X.element_size()
            if self._cache_bytes + sample_bytes <= self._max_cache_bytes:
                self._cache[idx] = X
                self._cache_bytes += sample_bytes
            else:
                self._cache_full = True
                print(
                    f"  Cache limit reached ({self._cache_bytes / 1024**3:.1f}GB), "
                    f"cached {len(self._cache)}/{len(self)} samples"
                )

        return X

    def __getitem__(self, idx):
        X = self._load_sample(idx)
        y = torch.LongTensor([self.labels[idx]])
        return X, y

    def get_cache_info(self):
        """Return cache statistics."""
        if self._cache is None:
            return {"enabled": False, "cached": 0, "total": len(self), "bytes": 0}
        return {
            "enabled": True,
            "cached": len(self._cache),
            "total": len(self),
            "bytes": self._cache_bytes,
            "full": self._cache_full,
        }


def create_fast_dataloaders(
    packed_data_path: str,
    user_index: int,
    batch_size: int = 512,
    num_workers: int = 8,
    cache_in_ram: bool = True,
    test_mode: bool = False,
    max_cache_gb: float = None,
):
    """
    Create optimized train/validation dataloaders for a specific user.

    Args:
        packed_data_path: Path to packed_data directory
        user_index: User index to use as validation (leave-one-out)
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        cache_in_ram: Whether to cache dataset in RAM
        test_mode: If True, load only 10 samples for quick verification
        max_cache_gb: Maximum GB to use for RAM cache (None = auto-detect)

    Returns:
        train_loader, valid_loader, label_encoder
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    packed_path = Path(packed_data_path)
    manifest_file = packed_path / "manifest.json"

    with open(manifest_file) as f:
        manifest = json.load(f)

    # Filter out ThumbFront (as in original)
    samples = [s for s in manifest["samples"] if "ThumbFront" not in s["folder"]]

    # Split by user for leave-one-out validation
    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []

    for s in samples:
        if s["user"] == str(user_index):
            test_samples.append(s["file"])
            test_labels.append(s["label"])
        else:
            train_samples.append(s["file"])
            train_labels.append(s["label"])

    # Encode labels
    action_names = ["ThumbSide", "3Middle", "LittleFinger"]
    le = LabelEncoder()
    le.fit(action_names)

    train_labels_encoded = le.transform(train_labels).tolist()
    test_labels_encoded = le.transform(test_labels).tolist()

    # Limit samples in test mode
    if test_mode:
        max_samples = 10
        train_samples = train_samples[:max_samples]
        train_labels_encoded = train_labels_encoded[:max_samples]
        test_samples = test_samples[:max_samples]
        test_labels_encoded = test_labels_encoded[:max_samples]
        print(f"TEST MODE: Limited to {max_samples} samples each")

    # Create datasets
    train_set = Dataset_CRNN_Fast(
        packed_path,
        train_samples,
        train_labels_encoded,
        cache_in_ram=cache_in_ram,
        max_cache_gb=max_cache_gb,
    )
    valid_set = Dataset_CRNN_Fast(
        packed_path,
        test_samples,
        test_labels_encoded,
        cache_in_ram=cache_in_ram,
        max_cache_gb=max_cache_gb,
    )

    # Lazy caching - samples cached on first access during training
    # This avoids OOM from trying to preload entire dataset
    print(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")
    print(f"RAM caching: {'enabled (lazy)' if cache_in_ram else 'disabled'}")

    # DataLoader parameters optimized for speed
    use_cuda = torch.cuda.is_available()
    loader_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "drop_last": False,
    }

    # These params only work with num_workers > 0
    if num_workers > 0:
        loader_params["prefetch_factor"] = 4
        loader_params["persistent_workers"] = True

    if not use_cuda:
        # Reduce params for CPU
        loader_params = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": num_workers,
        }

    train_loader = data.DataLoader(train_set, **loader_params)

    # Validation loader doesn't need shuffling
    valid_params = loader_params.copy()
    valid_params["shuffle"] = False
    valid_loader = data.DataLoader(valid_set, **valid_params)

    return train_loader, valid_loader, le


def get_dataloader_for_arranged_data(
    data_path: str,
    batch_size: int = 512,
    num_workers: int = 8,
    cache_in_ram: bool = True,
):
    """
    Alternative loader for pre-arranged jpg_video_arrange structure.
    Loads from existing JPG files but with caching.
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from PIL import Image
    import torchvision.transforms as transforms

    class Dataset_CRNN_Cached(data.Dataset):
        """Dataset with lazy loading and optional RAM caching."""

        def __init__(
            self, data_path, folders, labels, frames, transform=None, cache=True
        ):
            self.data_path = data_path
            self.folders = folders
            self.labels = labels
            self.frames = frames
            self.transform = transform
            self.cache = cache
            self._cache = {} if cache else None

        def __len__(self):
            return len(self.folders)

        def __getitem__(self, idx):
            if self.cache and idx in self._cache:
                X = self._cache[idx]
            else:
                folder = self.folders[idx]
                images = []
                for i in self.frames:
                    img_path = os.path.join(
                        self.data_path, folder, f"image_{i:05d}.jpg"
                    )
                    img = Image.open(img_path)
                    if self.transform:
                        img = self.transform(img)
                    images.append(img)
                X = torch.stack(images, dim=0)

                if self.cache:
                    self._cache[idx] = X

            y = torch.LongTensor([self.labels[idx]])
            return X, y

    return Dataset_CRNN_Cached
