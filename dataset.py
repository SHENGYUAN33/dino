# dataset.py
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random


class TwoCropTransform:
    """Create two augmented views for MoCo"""

    def __init__(self, base_transform):
        self.base = base_transform

    def __call__(self, x):
        return [self.base(x), self.base(x)]


def get_moco_transforms():
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    return TwoCropTransform(augmentation)


class RetinaUnlabeled(Dataset):
    def __init__(self, data_dir, transform):
        self.paths = [os.path.join(data_dir, f)
                      for f in os.listdir(data_dir)
                      if f.lower().endswith((".png", ".jpg", "bmp"))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        im1, im2 = self.transform(img)
        return im1, im2


def get_moco_dataloader(cfg):
    ds = RetinaUnlabeled(cfg.data_dir, get_moco_transforms())
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                      num_workers=cfg.num_workers, drop_last=True)
