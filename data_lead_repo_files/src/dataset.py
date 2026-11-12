"""PyTorch data loader for the HC18 fetal head segmentation project.

The loader reads the ultrasound images and the filled head masks listed in
``splits.csv``, resizes both to the same size and, when asked, applies the
same random augmentation to the image and its mask.

Expected folder layout under ``data_dir``::

    data_dir/
        splits.csv              filename, split, pixel_size_mm, true_hc_mm
        raw/training_set/       000_HC.png, 001_HC.png, ...
        masks/                  000_HC.png, 001_HC.png, ... (0 or 255)

Example
-------
>>> from dataset import get_dataloaders
>>> loaders = get_dataloaders("/content/drive/MyDrive/aai-521-hc18",
...                           image_size=256, batch_size=16)
>>> images, masks = next(iter(loaders["train"]))
>>> images.shape, masks.shape
(torch.Size([16, 1, 256, 256]), torch.Size([16, 1, 256, 256]))
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SPLIT_NAMES = ("train", "val", "test")
IMAGE_SUBDIR = Path("raw") / "training_set"
MASK_SUBDIR = Path("masks")
SPLITS_FILENAME = "splits.csv"


def _to_height_width(image_size):
    """Return ``image_size`` as a ``(height, width)`` tuple."""
    if isinstance(image_size, int):
        return image_size, image_size
    height, width = image_size
    return int(height), int(width)


def get_transforms(image_size=256, augment=False):
    """Build the resize and augmentation pipeline.

    Parameters
    ----------
    image_size : int or tuple of int
        Output size as one number (square) or ``(height, width)``.
    augment : bool
        If True, add a random horizontal flip, a small rotation and a
        brightness and contrast change. Use this for the training split only.

    Returns
    -------
    albumentations.Compose
        Callable used as ``transform(image=image, mask=mask)``. Flips and
        rotations are applied to the image and the mask together, while
        brightness and contrast only touch the image. Masks are always
        resized with nearest neighbour interpolation so they stay binary.
    """
    height, width = _to_height_width(image_size)
    steps = [A.Resize(height=height, width=width,
                      interpolation=cv2.INTER_LINEAR)]
    if augment:
        steps += [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.5),
        ]
    return A.Compose(steps)


class HC18Dataset(Dataset):
    """Ultrasound images and binary head masks for one split.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Project data folder that holds ``splits.csv``, ``raw`` and ``masks``.
    split : {"train", "val", "test"}
        Which rows of ``splits.csv`` to load.
    image_size : int or tuple of int, default 256
        Output size as one number (square) or ``(height, width)``.
    augment : bool, default False
        Apply random augmentation. Leave False for validation and test.
    in_channels : {1, 3}, default 1
        Number of image channels returned. Use 3 to repeat the grayscale
        image for encoders that expect RGB input.
    return_meta : bool, default False
        If True, each item also carries a dict with the filename, pixel
        size, true head circumference and the original image size, which is
        what is needed to turn a predicted mask back into millimetres.

    Notes
    -----
    Each item is ``(image, mask)`` or ``(image, mask, meta)``. ``image`` is
    a float32 tensor of shape ``(in_channels, H, W)`` scaled to 0-1 and
    ``mask`` is a float32 tensor of shape ``(1, H, W)`` holding 0 or 1.
    """

    def __init__(self, data_dir, split, image_size=256, augment=False,
                 in_channels=1, return_meta=False):
        if split not in SPLIT_NAMES:
            raise ValueError(
                f"split must be one of {SPLIT_NAMES}, got {split!r}")
        if in_channels not in (1, 3):
            raise ValueError("in_channels must be 1 or 3")

        self.data_dir = Path(data_dir)
        self.split = split
        self.in_channels = in_channels
        self.return_meta = return_meta
        self.transform = get_transforms(image_size, augment)

        splits = pd.read_csv(self.data_dir / SPLITS_FILENAME)
        self.table = splits[splits["split"] == split].reset_index(drop=True)
        if self.table.empty:
            raise ValueError(f"No rows found for split {split!r}")

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        image = self._read_gray(self.data_dir / IMAGE_SUBDIR / row.filename)
        mask = self._read_gray(self.data_dir / MASK_SUBDIR / row.filename)
        original_height, original_width = image.shape
        mask = (mask > 127).astype(np.uint8)

        out = self.transform(image=image, mask=mask)
        image = out["image"].astype(np.float32) / 255.0
        mask = out["mask"].astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)
        if self.in_channels == 3:
            image = image.repeat(3, 1, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        if not self.return_meta:
            return image, mask

        meta = {
            "filename": row.filename,
            "pixel_size_mm": float(row.pixel_size_mm),
            "true_hc_mm": float(row.true_hc_mm),
            "original_height": original_height,
            "original_width": original_width,
        }
        return image, mask, meta

    @staticmethod
    def _read_gray(path):
        """Read an image as a 2D uint8 array and fail loudly if missing."""
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read {path}")
        return image


def get_dataloaders(data_dir, image_size=256, batch_size=16, augment=True,
                    in_channels=1, num_workers=2, return_meta=False):
    """Create one DataLoader per split.

    Augmentation and shuffling are only ever applied to the training split,
    so validation and test scores stay comparable between runs.

    Returns
    -------
    dict
        ``{"train": DataLoader, "val": DataLoader, "test": DataLoader}``
    """
    loaders = {}
    for split in SPLIT_NAMES:
        is_train = split == "train"
        dataset = HC18Dataset(
            data_dir,
            split,
            image_size=image_size,
            augment=augment and is_train,
            in_channels=in_channels,
            return_meta=return_meta,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
