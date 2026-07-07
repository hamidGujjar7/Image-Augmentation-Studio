"""
augmentation_engine.py
-----------------------------------------------------------------------------
Core image-augmentation logic. Deliberately kept 100% free of any GUI
imports so it can be unit-tested or reused (e.g. in a CLI or notebook)
independently of CustomTkinter.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


@dataclass
class AugmentationParams:
    """Snapshot of every augmentation control exposed in the UI."""

    rotation_enabled: bool = False
    rotation_degrees: float = 15.0
    rotation_probability: float = 0.75

    flip_h_enabled: bool = False
    flip_h_probability: float = 0.5
    flip_v_enabled: bool = False
    flip_v_probability: float = 0.5

    scale_enabled: bool = False
    scale_factor: float = 1.0
    scale_probability: float = 0.5

    brightness_enabled: bool = False
    brightness_factor: float = 1.0
    brightness_probability: float = 0.5

    contrast_enabled: bool = False
    contrast_factor: float = 1.0
    contrast_probability: float = 0.5

    saturation_enabled: bool = False
    saturation_factor: float = 1.0
    saturation_probability: float = 0.5

    hue_enabled: bool = False
    hue_shift: float = 0.0
    hue_probability: float = 0.5

    gaussian_blur_enabled: bool = False
    gaussian_blur_kernel: int = 3
    gaussian_blur_probability: float = 0.5

    motion_blur_enabled: bool = False
    motion_blur_kernel: int = 3
    motion_blur_probability: float = 0.5

    gauss_noise_enabled: bool = False
    gauss_noise_amount: float = 0.02
    gauss_noise_probability: float = 0.5

    grid_dropout_enabled: bool = False
    grid_dropout_ratio: float = 0.3
    grid_dropout_probability: float = 0.5

    coarse_dropout_enabled: bool = False
    coarse_dropout_intensity: float = 0.3
    coarse_dropout_probability: float = 0.5

    random_order_enabled: bool = True
    seed_value: int = 1234
    lock_seed: bool = False

    def any_enabled(self) -> bool:
        return any(
            getattr(self, name)
            for name in self.__dataclass_fields__
            if name.endswith("_enabled")
        )


class AugmentationEngine:
    """Builds Albumentations pipelines from AugmentationParams and runs them."""

    def __init__(self, params: AugmentationParams):
        self.params = params

    def build_pipeline(self, deterministic: bool = True) -> A.Compose:
        p = self.params
        transforms: List[A.BasicTransform] = []

        def spread(value: float, amount: float) -> Tuple[float, float]:
            if deterministic:
                return (value, value)
            return (value - amount, value + amount)

        if p.rotation_enabled:
            lo, hi = spread(p.rotation_degrees, 10)
            lo, hi = max(-180, lo), min(180, hi)
            transforms.append(
                A.Rotate(limit=(lo, hi), p=self._probability(p.rotation_probability), border_mode=cv2.BORDER_CONSTANT)
            )

        if p.flip_h_enabled:
            transforms.append(A.HorizontalFlip(p=self._probability(p.flip_h_probability)))

        if p.flip_v_enabled:
            transforms.append(A.VerticalFlip(p=self._probability(p.flip_v_probability)))

        if p.scale_enabled:
            lo, hi = spread(p.scale_factor, 0.1)
            lo = max(0.1, lo)
            transforms.append(A.Affine(scale=(lo, max(lo, hi)), p=self._probability(p.scale_probability)))

        if p.brightness_enabled or p.contrast_enabled or p.saturation_enabled or p.hue_enabled:
            b = spread(p.brightness_factor, 0.1) if p.brightness_enabled else (1.0, 1.0)
            c = spread(p.contrast_factor, 0.1) if p.contrast_enabled else (1.0, 1.0)
            s = spread(p.saturation_factor, 0.1) if p.saturation_enabled else (1.0, 1.0)
            h = spread(p.hue_shift, 0.05) if p.hue_enabled else (0.0, 0.0)
            transforms.append(
                A.ColorJitter(
                    brightness=b,
                    contrast=c,
                    saturation=s,
                    hue=h,
                    p=self._probability(p.brightness_probability if p.brightness_enabled else p.contrast_probability if p.contrast_enabled else p.saturation_probability if p.saturation_enabled else p.hue_probability),
                )
            )

        if p.gaussian_blur_enabled:
            k = _odd(p.gaussian_blur_kernel)
            lo = k if deterministic else _odd(max(3, k - 4))
            transforms.append(A.GaussianBlur(blur_limit=(min(lo, k), k), p=self._probability(p.gaussian_blur_probability)))

        if p.motion_blur_enabled:
            k = _odd(p.motion_blur_kernel)
            lo = k if deterministic else _odd(max(3, k - 4))
            transforms.append(A.MotionBlur(blur_limit=(min(lo, k), k), p=self._probability(p.motion_blur_probability)))

        if p.gauss_noise_enabled:
            std = max(1.0, p.gauss_noise_amount * 255)
            var_hi = std ** 2
            var_lo = var_hi if deterministic else var_hi * 0.4
            transforms.append(A.GaussNoise(var_limit=(var_lo, var_hi), p=self._probability(p.gauss_noise_probability)))

        if p.grid_dropout_enabled:
            ratio = min(0.9, max(0.1, p.grid_dropout_ratio))
            transforms.append(A.GridDropout(ratio=ratio, p=self._probability(p.grid_dropout_probability)))

        if p.coarse_dropout_enabled:
            intensity = min(1.0, max(0.05, p.coarse_dropout_intensity))
            num_holes = max(1, int(intensity * 12))
            hole_size = int(8 + intensity * 40)
            hole_range = (max(0.05, hole_size / 100.0), max(0.1, hole_size / 100.0))
            transforms.append(
                A.CoarseDropout(
                    num_holes_range=(1, num_holes),
                    hole_height_range=hole_range,
                    hole_width_range=hole_range,
                    fill=0,
                    p=self._probability(p.coarse_dropout_probability),
                )
            )

        if not transforms:
            transforms.append(A.NoOp())

        if p.random_order_enabled and len(transforms) > 1:
            return A.Compose([A.RandomOrder(transforms, p=1.0)])
        return A.Compose(transforms)

    def apply(self, image: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self.params.lock_seed:
            random.seed(self.params.seed_value)
            np.random.seed(self.params.seed_value)
        pipeline = self.build_pipeline(deterministic=deterministic)
        return pipeline(image=image)["image"]

    def generate_variants(self, image: np.ndarray, count: int) -> List[np.ndarray]:
        count = max(1, int(count))
        return [self.apply(image, deterministic=False) for _ in range(count)]

    @staticmethod
    def _probability(value: float) -> float:
        return max(0.0, min(1.0, float(value)))


def _odd(value: int) -> int:
    value = int(round(value))
    value = max(3, value)
    return value if value % 2 == 1 else value + 1


def load_image_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Supported: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    data = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if data is None:
        raise ValueError(f"Could not decode image (corrupt or unsupported): {path}")
    return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)


def save_image_rgb(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise IOError(f"Failed to write image to: {path}")


def find_images_in_folder(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Not a valid folder: {folder}")
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )