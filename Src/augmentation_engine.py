"""
augmentation_engine.py
-----------------------------------------------------------------------------
Core image-augmentation logic. Deliberately kept 100% free of any GUI
imports so it can be unit-tested or reused (e.g. in a CLI or notebook)
independently of CustomTkinter.

Supports:
  - Geometry: Rotate, Flip H/V, Scale, Random Crop, Center Crop, Pad,
              Translation, Perspective, Affine, Shear, Elastic Transform
  - Weather:  Rain, Snow, Fog, Sun Flare, Shadow
  - Blur:     Gaussian, Motion, Median, Glass, Defocus, Zoom
  - Noise:    Gaussian, ISO, Salt & Pepper, Speckle, Poisson
  - Color:    Brightness, Contrast, Saturation, Hue, Gamma, CLAHE,
              Equalization, RGB Shift, Channel Shuffle, Grayscale, Sepia
  - Compression: JPEG Compression, Downscale (Low Resolution)
  - Advanced: Mosaic, MixUp, CutMix, Random Erasing, Coarse Dropout,
              Grid Dropout
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import albumentations as A
import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


@dataclass
class AugmentationParams:
    """Snapshot of every augmentation control exposed in the UI."""

    # ── Geometry ──────────────────────────────────────────────────────────
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

    random_crop_enabled: bool = False
    random_crop_width: int = 256
    random_crop_height: int = 256
    random_crop_probability: float = 0.5

    center_crop_enabled: bool = False
    center_crop_width: int = 256
    center_crop_height: int = 256
    center_crop_probability: float = 0.5

    pad_enabled: bool = False
    pad_pixels: int = 32
    pad_probability: float = 0.5

    translation_enabled: bool = False
    translation_x: float = 0.1
    translation_y: float = 0.1
    translation_probability: float = 0.5

    perspective_enabled: bool = False
    perspective_scale: float = 0.05
    perspective_probability: float = 0.5

    affine_enabled: bool = False
    affine_rotate: float = 15.0
    affine_scale_min: float = 0.9
    affine_scale_max: float = 1.1
    affine_probability: float = 0.5

    shear_enabled: bool = False
    shear_degrees: float = 10.0
    shear_probability: float = 0.5

    elastic_enabled: bool = False
    elastic_alpha: float = 120.0
    elastic_sigma: float = 6.0
    elastic_probability: float = 0.5

    # ── Weather ───────────────────────────────────────────────────────────
    rain_enabled: bool = False
    rain_intensity: float = 0.3
    rain_probability: float = 0.5

    snow_enabled: bool = False
    snow_intensity: float = 0.3
    snow_probability: float = 0.5

    fog_enabled: bool = False
    fog_intensity: float = 0.3
    fog_probability: float = 0.5

    sun_flare_enabled: bool = False
    sun_flare_intensity: float = 0.3
    sun_flare_probability: float = 0.5

    shadow_enabled: bool = False
    shadow_intensity: float = 0.3
    shadow_probability: float = 0.5

    # ── Blur ──────────────────────────────────────────────────────────────
    gaussian_blur_enabled: bool = False
    gaussian_blur_kernel: int = 3
    gaussian_blur_probability: float = 0.5

    motion_blur_enabled: bool = False
    motion_blur_kernel: int = 3
    motion_blur_probability: float = 0.5

    median_blur_enabled: bool = False
    median_blur_kernel: int = 3
    median_blur_probability: float = 0.5

    glass_blur_enabled: bool = False
    glass_blur_sigma: float = 0.7
    glass_blur_probability: float = 0.5

    defocus_blur_enabled: bool = False
    defocus_blur_radius: int = 5
    defocus_blur_probability: float = 0.5

    zoom_blur_enabled: bool = False
    zoom_blur_steps: int = 3
    zoom_blur_probability: float = 0.5

    # ── Noise ─────────────────────────────────────────────────────────────
    gauss_noise_enabled: bool = False
    gauss_noise_amount: float = 0.02
    gauss_noise_probability: float = 0.5

    iso_noise_enabled: bool = False
    iso_noise_intensity: float = 0.3
    iso_noise_probability: float = 0.5

    salt_pepper_enabled: bool = False
    salt_pepper_amount: float = 0.02
    salt_pepper_probability: float = 0.5

    speckle_noise_enabled: bool = False
    speckle_noise_intensity: float = 0.3
    speckle_noise_probability: float = 0.5

    # ── Color ─────────────────────────────────────────────────────────────
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

    gamma_enabled: bool = False
    gamma_value: float = 1.0
    gamma_probability: float = 0.5

    clahe_enabled: bool = False
    clahe_clip_limit: float = 4.0
    clahe_probability: float = 0.5

    equalization_enabled: bool = False
    equalization_probability: float = 0.5

    rgb_shift_enabled: bool = False
    rgb_shift_r: float = 20.0
    rgb_shift_g: float = 20.0
    rgb_shift_b: float = 20.0
    rgb_shift_probability: float = 0.5

    channel_shuffle_enabled: bool = False
    channel_shuffle_probability: float = 0.5

    grayscale_enabled: bool = False
    grayscale_probability: float = 0.5

    sepia_enabled: bool = False
    sepia_probability: float = 0.5

    # ── Compression ───────────────────────────────────────────────────────
    jpeg_compression_enabled: bool = False
    jpeg_quality_lower: int = 40
    jpeg_quality_upper: int = 80
    jpeg_compression_probability: float = 0.5

    downscale_enabled: bool = False
    downscale_min: float = 0.5
    downscale_max: float = 0.8
    downscale_probability: float = 0.5

    # ── Advanced ──────────────────────────────────────────────────────────
    random_erasing_enabled: bool = False
    random_erasing_ratio: float = 0.3
    random_erasing_probability: float = 0.5

    grid_dropout_enabled: bool = False
    grid_dropout_ratio: float = 0.3
    grid_dropout_probability: float = 0.5

    coarse_dropout_enabled: bool = False
    coarse_dropout_intensity: float = 0.3
    coarse_dropout_probability: float = 0.5

    # ── Pipeline ──────────────────────────────────────────────────────────
    random_order_enabled: bool = True
    seed_value: int = 1234
    lock_seed: bool = False

    def any_enabled(self) -> bool:
        return any(
            getattr(self, name)
            for name in self.__dataclass_fields__
            if name.endswith("_enabled")
        )

    def enabled_count(self) -> int:
        return sum(
            1 for name in self.__dataclass_fields__
            if name.endswith("_enabled") and getattr(self, name)
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

        # ── Geometry ──────────────────────────────────────────────────────
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

        if p.random_crop_enabled:
            transforms.append(
                A.RandomCrop(
                    width=max(16, p.random_crop_width),
                    height=max(16, p.random_crop_height),
                    p=self._probability(p.random_crop_probability),
                )
            )

        if p.center_crop_enabled:
            transforms.append(
                A.CenterCrop(
                    width=max(16, p.center_crop_width),
                    height=max(16, p.center_crop_height),
                    p=self._probability(p.center_crop_probability),
                )
            )

        if p.pad_enabled:
            px = max(1, int(p.pad_pixels))
            transforms.append(
                A.PadIfNeeded(
                    min_height=None, min_width=None,
                    pad_height_divisor=None, pad_width_divisor=None,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=self._probability(p.pad_probability),
                )
            )

        if p.translation_enabled:
            tx = max(0.0, min(0.5, p.translation_x))
            ty = max(0.0, min(0.5, p.translation_y))
            transforms.append(
                A.Affine(
                    translate_percent={"x": (-tx, tx), "y": (-ty, ty)},
                    p=self._probability(p.translation_probability),
                )
            )

        if p.perspective_enabled:
            s = max(0.01, min(0.2, p.perspective_scale))
            transforms.append(
                A.Perspective(scale=(s * 0.5, s), p=self._probability(p.perspective_probability))
            )

        if p.affine_enabled:
            transforms.append(
                A.Affine(
                    rotate=(-p.affine_rotate, p.affine_rotate),
                    scale=(p.affine_scale_min, p.affine_scale_max),
                    p=self._probability(p.affine_probability),
                )
            )

        if p.shear_enabled:
            s = max(1.0, abs(p.shear_degrees))
            transforms.append(
                A.Affine(shear=(-s, s), p=self._probability(p.shear_probability))
            )

        if p.elastic_enabled:
            transforms.append(
                A.ElasticTransform(
                    alpha=max(1.0, p.elastic_alpha),
                    sigma=max(1.0, p.elastic_sigma),
                    p=self._probability(p.elastic_probability),
                )
            )

        # ── Weather ───────────────────────────────────────────────────────
        if p.rain_enabled:
            transforms.append(
                A.RandomRain(
                    slant_lower=-10, slant_upper=10,
                    drop_length=10, drop_width=1,
                    brightness_coefficient=max(0.5, 1.0 - p.rain_intensity),
                    p=self._probability(p.rain_probability),
                )
            )

        if p.snow_enabled:
            coeff = max(0.1, min(0.9, p.snow_intensity))
            transforms.append(
                A.RandomSnow(
                    snow_point_lower=max(0.1, coeff - 0.2),
                    snow_point_upper=min(0.9, coeff + 0.2),
                    brightness_coeff=2.0 + coeff,
                    p=self._probability(p.snow_probability),
                )
            )

        if p.fog_enabled:
            lo = max(0.1, min(0.8, p.fog_intensity - 0.1))
            hi = max(lo + 0.05, min(1.0, p.fog_intensity + 0.1))
            transforms.append(
                A.RandomFog(
                    fog_coef_lower=lo,
                    fog_coef_upper=hi,
                    alpha_coef=0.08,
                    p=self._probability(p.fog_probability),
                )
            )

        if p.sun_flare_enabled:
            transforms.append(
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    angle_lower=0.0,
                    src_radius=int(200 + p.sun_flare_intensity * 200),
                    p=self._probability(p.sun_flare_probability),
                )
            )

        if p.shadow_enabled:
            transforms.append(
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_limit=(1, max(1, int(p.shadow_intensity * 5))),
                    shadow_dimension=5,
                    p=self._probability(p.shadow_probability),
                )
            )

        # ── Blur ──────────────────────────────────────────────────────────
        if p.gaussian_blur_enabled:
            k = _odd(p.gaussian_blur_kernel)
            lo = k if deterministic else _odd(max(3, k - 4))
            transforms.append(A.GaussianBlur(blur_limit=(min(lo, k), k), p=self._probability(p.gaussian_blur_probability)))

        if p.motion_blur_enabled:
            k = _odd(p.motion_blur_kernel)
            lo = k if deterministic else _odd(max(3, k - 4))
            transforms.append(A.MotionBlur(blur_limit=(min(lo, k), k), p=self._probability(p.motion_blur_probability)))

        if p.median_blur_enabled:
            k = _odd(p.median_blur_kernel)
            lo = k if deterministic else _odd(max(3, k - 4))
            transforms.append(A.MedianBlur(blur_limit=(min(lo, k), k), p=self._probability(p.median_blur_probability)))

        if p.glass_blur_enabled:
            transforms.append(
                A.GlassBlur(
                    sigma=max(0.1, p.glass_blur_sigma),
                    max_delta=4, iterations=2,
                    p=self._probability(p.glass_blur_probability),
                )
            )

        if p.defocus_blur_enabled:
            r = max(3, int(p.defocus_blur_radius))
            transforms.append(
                A.Defocus(
                    radius=(max(3, r - 2), r),
                    p=self._probability(p.defocus_blur_probability),
                )
            )

        if p.zoom_blur_enabled:
            steps = max(1, int(p.zoom_blur_steps))
            transforms.append(
                A.ZoomBlur(
                    max_factor=1.05 + steps * 0.03,
                    p=self._probability(p.zoom_blur_probability),
                )
            )

        # ── Noise ─────────────────────────────────────────────────────────
        if p.gauss_noise_enabled:
            std = max(1.0, p.gauss_noise_amount * 255)
            var_hi = std ** 2
            var_lo = var_hi if deterministic else var_hi * 0.4
            transforms.append(A.GaussNoise(var_limit=(var_lo, var_hi), p=self._probability(p.gauss_noise_probability)))

        if p.iso_noise_enabled:
            intensity = max(0.01, min(1.0, p.iso_noise_intensity))
            transforms.append(
                A.ISONoise(
                    color_shift=(0.01, intensity * 0.3),
                    intensity=(intensity * 0.3, intensity),
                    p=self._probability(p.iso_noise_probability),
                )
            )

        if p.salt_pepper_enabled:
            amount = max(0.001, min(0.1, p.salt_pepper_amount))
            # Implemented via custom pixel manipulation since albumentations
            # doesn't have a dedicated salt-and-pepper transform.  We use
            # a Compose with a Lambda or a Pixel Dropout as a reasonable
            # approximation.
            transforms.append(
                A.PixelDropout(
                    dropout_prob=amount,
                    per_channel=False,
                    drop_value=0,
                    p=self._probability(p.salt_pepper_probability) / 2,
                )
            )
            transforms.append(
                A.PixelDropout(
                    dropout_prob=amount,
                    per_channel=False,
                    drop_value=255,
                    p=self._probability(p.salt_pepper_probability) / 2,
                )
            )

        if p.speckle_noise_enabled:
            intensity = max(0.01, min(1.0, p.speckle_noise_intensity))
            transforms.append(
                A.MultiplicativeNoise(
                    multiplier=(1.0 - intensity * 0.5, 1.0 + intensity * 0.5),
                    per_channel=True,
                    p=self._probability(p.speckle_noise_probability),
                )
            )

        # ── Color ─────────────────────────────────────────────────────────
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

        if p.gamma_enabled:
            g = max(10, min(200, int(p.gamma_value * 100)))
            lo = g if deterministic else max(10, g - 20)
            transforms.append(
                A.RandomGamma(gamma_limit=(lo, g), p=self._probability(p.gamma_probability))
            )

        if p.clahe_enabled:
            cl = max(1.0, min(16.0, p.clahe_clip_limit))
            transforms.append(
                A.CLAHE(clip_limit=(cl * 0.5, cl), tile_grid_size=(8, 8), p=self._probability(p.clahe_probability))
            )

        if p.equalization_enabled:
            transforms.append(
                A.Equalize(p=self._probability(p.equalization_probability))
            )

        if p.rgb_shift_enabled:
            transforms.append(
                A.RGBShift(
                    r_shift_limit=max(1, int(p.rgb_shift_r)),
                    g_shift_limit=max(1, int(p.rgb_shift_g)),
                    b_shift_limit=max(1, int(p.rgb_shift_b)),
                    p=self._probability(p.rgb_shift_probability),
                )
            )

        if p.channel_shuffle_enabled:
            transforms.append(
                A.ChannelShuffle(p=self._probability(p.channel_shuffle_probability))
            )

        if p.grayscale_enabled:
            transforms.append(
                A.ToGray(p=self._probability(p.grayscale_probability))
            )

        if p.sepia_enabled:
            # Sepia approximated via ToSepia or a custom lambda
            transforms.append(
                A.ToSepia(p=self._probability(p.sepia_probability))
            )

        # ── Compression ───────────────────────────────────────────────────
        if p.jpeg_compression_enabled:
            lo = max(1, min(100, p.jpeg_quality_lower))
            hi = max(lo, min(100, p.jpeg_quality_upper))
            transforms.append(
                A.ImageCompression(
                    quality_lower=lo,
                    quality_upper=hi,
                    compression_type=A.ImageCompression.ImageCompressionType.JPEG,
                    p=self._probability(p.jpeg_compression_probability),
                )
            )

        if p.downscale_enabled:
            lo = max(0.1, min(0.9, p.downscale_min))
            hi = max(lo, min(1.0, p.downscale_max))
            transforms.append(
                A.Downscale(
                    scale_min=lo, scale_max=hi,
                    p=self._probability(p.downscale_probability),
                )
            )

        # ── Advanced / Dropout ────────────────────────────────────────────
        if p.random_erasing_enabled:
            ratio = min(0.8, max(0.05, p.random_erasing_ratio))
            transforms.append(
                A.CoarseDropout(
                    num_holes_range=(1, max(1, int(ratio * 8))),
                    hole_height_range=(ratio * 0.5, ratio),
                    hole_width_range=(ratio * 0.5, ratio),
                    fill=random.randint(0, 255),
                    p=self._probability(p.random_erasing_probability),
                )
            )

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


def save_image_rgb(image: np.ndarray, path: Path, quality: int = 95,
                   png_compression: int = 3, webp_quality: int = 90) -> None:
    """Save an RGB numpy array to disk with format-specific quality settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ext = path.suffix.lower()
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, quality))]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, png_compression))]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, max(1, min(100, webp_quality))]
    if not cv2.imwrite(str(path), bgr, params if params else None):
        raise IOError(f"Failed to write image to: {path}")


def find_images_in_folder(folder: Path, recursive: bool = False) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Not a valid folder: {folder}")
    if recursive:
        return sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )