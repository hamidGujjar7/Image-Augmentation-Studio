"""
image_utilities.py
-----------------------------------------------------------------------------
Image utility functions: rename, resize, format conversion, crop, rotate,
split into tiles, merge tiles, and more.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from Src.augmentation_engine import SUPPORTED_EXTENSIONS, find_images_in_folder


# ═══════════════════════════════════════════════════════════════════════════
#  Rename Files
# ═══════════════════════════════════════════════════════════════════════════

def rename_files(folder: Path, pattern: str = "{index:04d}",
                 prefix: str = "", suffix: str = "",
                 start_index: int = 1) -> List[Tuple[str, str]]:
    """
    Rename all images in a folder.
    Pattern supports: {index}, {original}, {uuid}
    Returns list of (old_name, new_name) tuples.
    """
    images = find_images_in_folder(folder)
    renames = []
    for i, img_path in enumerate(images, start=start_index):
        ext = img_path.suffix
        new_name = pattern.format(
            index=i,
            original=img_path.stem,
            uuid=uuid.uuid4().hex[:8],
        )
        new_name = f"{prefix}{new_name}{suffix}{ext}"
        new_path = folder / new_name
        if new_path != img_path:
            img_path.rename(new_path)
            renames.append((img_path.name, new_name))
    return renames


# ═══════════════════════════════════════════════════════════════════════════
#  Resize All Images
# ═══════════════════════════════════════════════════════════════════════════

def resize_all_images(folder: Path, width: int, height: int,
                      keep_aspect: bool = True,
                      output_folder: Optional[Path] = None,
                      interpolation: int = cv2.INTER_LANCZOS4) -> int:
    """Resize all images in a folder. Returns number of images processed."""
    images = find_images_in_folder(folder)
    out = output_folder or folder
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if keep_aspect:
            h, w = img.shape[:2]
            scale = min(width / w, height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        else:
            resized = cv2.resize(img, (width, height), interpolation=interpolation)
        cv2.imwrite(str(out / img_path.name), resized)
        count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════════
#  Format Conversion
# ═══════════════════════════════════════════════════════════════════════════

def convert_format(folder: Path, source_ext: str, target_ext: str,
                   output_folder: Optional[Path] = None,
                   quality: int = 95) -> int:
    """Convert images from one format to another. Returns count of converted images."""
    if not source_ext.startswith("."):
        source_ext = "." + source_ext
    if not target_ext.startswith("."):
        target_ext = "." + target_ext

    out = output_folder or folder
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in folder.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() != source_ext.lower():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        new_path = out / (img_path.stem + target_ext)
        params = []
        if target_ext.lower() in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif target_ext.lower() == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        elif target_ext.lower() == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]
        cv2.imwrite(str(new_path), img, params if params else None)
        count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════════
#  Convert to Grayscale / RGB
# ═══════════════════════════════════════════════════════════════════════════

def convert_to_grayscale(folder: Path, output_folder: Optional[Path] = None) -> int:
    """Convert all images to grayscale."""
    images = find_images_in_folder(folder)
    out = output_folder or folder
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(out / img_path.name), gray)
        count += 1
    return count


def convert_to_rgb(folder: Path, output_folder: Optional[Path] = None) -> int:
    """Convert all images to RGB (3-channel)."""
    images = find_images_in_folder(folder)
    out = output_folder or folder
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(str(out / img_path.name), img)
        count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════════
#  Rotate / Crop Images
# ═══════════════════════════════════════════════════════════════════════════

def rotate_all_images(folder: Path, angle: float,
                      output_folder: Optional[Path] = None) -> int:
    """Rotate all images by a fixed angle."""
    images = find_images_in_folder(folder)
    out = output_folder or folder
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(str(out / img_path.name), rotated)
        count += 1
    return count


def crop_all_images(folder: Path, x: int, y: int, w: int, h: int,
                    output_folder: Optional[Path] = None) -> int:
    """Crop all images to a fixed region."""
    images = find_images_in_folder(folder)
    out = output_folder or folder
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        cx = max(0, min(x, iw))
        cy = max(0, min(y, ih))
        cw = min(w, iw - cx)
        ch = min(h, ih - cy)
        cropped = img[cy:cy + ch, cx:cx + cw]
        if cropped.size > 0:
            cv2.imwrite(str(out / img_path.name), cropped)
            count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════════
#  Split into Tiles / Merge Tiles
# ═══════════════════════════════════════════════════════════════════════════

def split_into_tiles(image_path: Path, rows: int, cols: int,
                     output_folder: Optional[Path] = None) -> List[Path]:
    """Split a single image into a grid of tiles."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    out = output_folder or image_path.parent / "tiles"
    out.mkdir(parents=True, exist_ok=True)

    tile_h = h // rows
    tile_w = w // cols
    paths = []
    for r in range(rows):
        for c in range(cols):
            y1 = r * tile_h
            x1 = c * tile_w
            y2 = y1 + tile_h if r < rows - 1 else h
            x2 = x1 + tile_w if c < cols - 1 else w
            tile = img[y1:y2, x1:x2]
            tile_path = out / f"{image_path.stem}_r{r}_c{c}{image_path.suffix}"
            cv2.imwrite(str(tile_path), tile)
            paths.append(tile_path)
    return paths


def merge_tiles(tile_folder: Path, rows: int, cols: int,
                output_path: Optional[Path] = None) -> Path:
    """Merge tiles back into a single image. Tiles must follow _rN_cN naming."""
    tiles = {}
    for f in sorted(tile_folder.iterdir()):
        if not f.is_file():
            continue
        match = re.search(r"_r(\d+)_c(\d+)", f.stem)
        if match:
            r, c = int(match.group(1)), int(match.group(2))
            tiles[(r, c)] = cv2.imread(str(f))

    if not tiles:
        raise ValueError(f"No tiles found in {tile_folder}")

    # Build rows
    row_images = []
    for r in range(rows):
        row_tiles = [tiles[(r, c)] for c in range(cols) if (r, c) in tiles]
        if row_tiles:
            row_images.append(np.concatenate(row_tiles, axis=1))
    merged = np.concatenate(row_images, axis=0)

    if output_path is None:
        output_path = tile_folder / "merged.png"
    cv2.imwrite(str(output_path), merged)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
#  EXIF / Metadata
# ═══════════════════════════════════════════════════════════════════════════

def remove_exif(image_path: Path, output_path: Optional[Path] = None) -> None:
    """Remove EXIF data from an image by reading and re-saving."""
    img = cv2.imread(str(image_path))
    if img is None:
        return
    out = output_path or image_path
    cv2.imwrite(str(out), img)


def remove_exif_batch(folder: Path, output_folder: Optional[Path] = None) -> int:
    """Remove EXIF from all images in a folder."""
    images = find_images_in_folder(folder)
    out = output_folder or folder
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in images:
        remove_exif(img_path, out / img_path.name)
        count += 1
    return count
