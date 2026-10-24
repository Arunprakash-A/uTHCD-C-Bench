"""Dataset and preprocessing for uTHCD-C evaluation.

The test images are read with ``torchvision.datasets.ImageFolder`` from a
user-supplied directory (one sub-directory per character class). A corruption is
applied to each PIL image at its native resolution, after which the standard
evaluation transform resizes, center-crops and normalizes it for the network.
"""
from __future__ import annotations

from torchvision import transforms
from torchvision.datasets import ImageFolder

from .corruptions import apply_corruption

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_eval_transform(size: int = 224):
    """Resize -> center-crop -> tensor -> ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CorruptedImageFolder(ImageFolder):
    """ImageFolder that corrupts each image before applying ``transform``.

    Set the active corruption/severity with :meth:`set_corruption`; ``'clean'``
    leaves the image unchanged.
    """

    def __init__(self, root, transform=None, corruption="clean", severity=3):
        super().__init__(root, transform=None)
        self.eval_transform = transform or build_eval_transform()
        self.corruption = corruption
        self.severity = severity

    def set_corruption(self, corruption, severity=3):
        self.corruption = corruption
        self.severity = severity

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path).convert("RGB")
        img = apply_corruption(img, self.corruption, self.severity)
        return self.eval_transform(img), target
