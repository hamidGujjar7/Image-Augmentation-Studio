"""
batch_processor.py
-----------------------------------------------------------------------------
Batch augmentation processor with selection modes, save modes, naming,
progress tracking, pause/resume/cancel, and final report generation.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import datetime
import os
import random
import shutil
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from Src.augmentation_engine import (
    AugmentationEngine,
    AugmentationParams,
    SUPPORTED_EXTENSIONS,
    find_images_in_folder,
    load_image_rgb,
    save_image_rgb,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════════════════

class SelectionMode(Enum):
    ALL_IMAGES = "all_images"
    SELECTED_IMAGES = "selected_images"
    SELECTED_FOLDER = "selected_folder"
    SELECTED_CLASS = "selected_class"
    IMAGES_WITHOUT_LABELS = "images_without_labels"
    IMAGES_WITH_LABELS = "images_with_labels"
    RANDOM_IMAGES = "random_images"
    RECENTLY_ADDED = "recently_added"
    FILTER_BY_FILENAME = "filter_by_filename"


class SaveMode(Enum):
    OVERWRITE_ORIGINAL = "overwrite_original"
    SAVE_AS_NEW = "save_as_new"
    SAVE_IN_NEW_FOLDER = "save_in_new_folder"
    SAVE_NEXT_TO_ORIGINAL = "save_next_to_original"
    REPLACE_EXISTING = "replace_existing"


class NamingMode(Enum):
    SUFFIX = "suffix"
    PREFIX = "prefix"
    TIMESTAMP = "timestamp"
    UUID = "uuid"
    KEEP_ORIGINAL = "keep_original"


class LabelSaveMode(Enum):
    COPY = "copy"
    TRANSFORM = "transform"
    VERIFY = "verify"
    SKIP = "skip"


# ═══════════════════════════════════════════════════════════════════════════
#  Batch Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BatchConfig:
    """All settings for a batch augmentation run."""
    # Selection
    selection_mode: SelectionMode = SelectionMode.ALL_IMAGES
    selected_images: List[Path] = field(default_factory=list)
    selected_class: str = ""
    filename_filter: str = ""
    random_percentage: float = 100.0  # 10, 25, 50, 100

    # Copies
    copies_per_image: int = 1

    # Save
    save_mode: SaveMode = SaveMode.SAVE_IN_NEW_FOLDER
    output_folder: Optional[Path] = None
    naming_mode: NamingMode = NamingMode.SUFFIX
    custom_suffix: str = "_aug"
    custom_prefix: str = "aug_"

    # Label handling
    label_save_mode: LabelSaveMode = LabelSaveMode.COPY
    labels_dir: Optional[Path] = None

    # File options
    output_format: str = ".png"
    jpeg_quality: int = 95
    png_compression: int = 3
    webp_quality: int = 90
    skip_existing: bool = False
    replace_existing: bool = False

    # Metadata
    keep_exif: bool = False

    # Safety
    max_output_images: int = 10000
    create_backup: bool = False
    backup_folder: Optional[Path] = None

    # Image filters
    min_width: int = 0
    min_height: int = 0
    max_width: int = 99999
    max_height: int = 99999


# ═══════════════════════════════════════════════════════════════════════════
#  Batch Report
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BatchReport:
    """Final report for a completed batch job."""
    original_images: int = 0
    generated_images: int = 0
    skipped_images: int = 0
    failed_images: int = 0
    total_time_seconds: float = 0.0
    output_folder: str = ""
    errors: List[str] = field(default_factory=list)

    @property
    def total_time_formatted(self) -> str:
        m, s = divmod(int(self.total_time_seconds), 60)
        return f"{m}m {s}s"

    @property
    def images_per_second(self) -> float:
        if self.total_time_seconds > 0:
            return self.generated_images / self.total_time_seconds
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Batch Processor
# ═══════════════════════════════════════════════════════════════════════════

class BatchProcessor:
    """Runs batch augmentation with full control over selection, naming, saving."""

    def __init__(self, engine: AugmentationEngine, config: BatchConfig):
        self.engine = engine
        self.config = config
        self._cancelled = False
        self._paused = False
        self._progress_callback: Optional[Callable[[float, str], None]] = None

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set a callback for progress updates: callback(fraction, status_text)."""
        self._progress_callback = callback

    def cancel(self):
        self._cancelled = True

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self, source_folder: Path) -> BatchReport:
        """Execute the batch augmentation. Returns a report."""
        report = BatchReport()
        start_time = time.time()

        # 1) Gather images based on selection mode
        image_paths = self._select_images(source_folder)
        report.original_images = len(image_paths)

        if not image_paths:
            report.total_time_seconds = time.time() - start_time
            return report

        # 2) Apply image filters
        image_paths = self._filter_images(image_paths)

        # 3) Apply random percentage
        if self.config.random_percentage < 100.0:
            count = max(1, int(len(image_paths) * self.config.random_percentage / 100.0))
            image_paths = random.sample(image_paths, count)

        # 4) Safety check
        total_output = len(image_paths) * self.config.copies_per_image
        if total_output > self.config.max_output_images:
            self._report_progress(0, f"WARNING: Output ({total_output}) exceeds max ({self.config.max_output_images})")

        # 5) Setup output directory
        output_dir = self._resolve_output_dir(source_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        report.output_folder = str(output_dir)

        # 6) Create backup if requested
        if self.config.create_backup:
            self._create_backup(source_folder)

        # 7) Process images
        total_ops = len(image_paths) * self.config.copies_per_image
        done = 0

        for img_path in image_paths:
            if self._cancelled:
                break

            while self._paused:
                time.sleep(0.1)
                if self._cancelled:
                    break

            try:
                image_np = load_image_rgb(img_path)
            except Exception as exc:
                report.failed_images += 1
                report.errors.append(f"Failed to load {img_path.name}: {exc}")
                done += self.config.copies_per_image
                self._report_progress(done / total_ops, f"Skipped {img_path.name}")
                continue

            for copy_idx in range(self.config.copies_per_image):
                if self._cancelled:
                    break

                try:
                    augmented = self.engine.apply(image_np, deterministic=False)
                    out_name = self._generate_filename(img_path, copy_idx)
                    out_path = self._resolve_save_path(img_path, out_name, output_dir)

                    if self.config.skip_existing and out_path.exists():
                        report.skipped_images += 1
                        done += 1
                        continue

                    save_image_rgb(
                        augmented, out_path,
                        quality=self.config.jpeg_quality,
                        png_compression=self.config.png_compression,
                        webp_quality=self.config.webp_quality,
                    )

                    # Handle labels
                    self._handle_labels(img_path, out_path)

                    report.generated_images += 1
                except Exception as exc:
                    report.failed_images += 1
                    report.errors.append(f"Failed augmenting {img_path.name} copy {copy_idx + 1}: {exc}")

                done += 1
                elapsed = time.time() - start_time
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total_ops - done) / speed if speed > 0 else 0
                self._report_progress(
                    done / total_ops,
                    f"Image {done}/{total_ops}  |  {speed:.1f} img/s  |  ETA: {int(eta)}s"
                )

        report.total_time_seconds = time.time() - start_time
        self._report_progress(1.0, f"Done! {report.generated_images} images in {report.total_time_formatted}")
        return report

    def _select_images(self, source_folder: Path) -> List[Path]:
        """Select images based on the configured selection mode."""
        cfg = self.config
        mode = cfg.selection_mode

        if mode == SelectionMode.SELECTED_IMAGES:
            return [p for p in cfg.selected_images if p.exists()]

        all_images = find_images_in_folder(source_folder)

        if mode == SelectionMode.ALL_IMAGES:
            return all_images

        if mode == SelectionMode.IMAGES_WITH_LABELS:
            labels_dir = cfg.labels_dir or source_folder
            return [p for p in all_images if (labels_dir / (p.stem + ".txt")).exists()]

        if mode == SelectionMode.IMAGES_WITHOUT_LABELS:
            labels_dir = cfg.labels_dir or source_folder
            return [p for p in all_images if not (labels_dir / (p.stem + ".txt")).exists()]

        if mode == SelectionMode.FILTER_BY_FILENAME:
            pattern = cfg.filename_filter.lower()
            return [p for p in all_images if pattern in p.name.lower()]

        if mode == SelectionMode.RECENTLY_ADDED:
            # Sort by creation time, take most recent 20%
            by_time = sorted(all_images, key=lambda p: p.stat().st_ctime, reverse=True)
            count = max(1, len(by_time) // 5)
            return by_time[:count]

        if mode == SelectionMode.RANDOM_IMAGES:
            count = max(1, int(len(all_images) * cfg.random_percentage / 100))
            return random.sample(all_images, min(count, len(all_images)))

        return all_images

    def _filter_images(self, image_paths: List[Path]) -> List[Path]:
        """Apply dimension filters."""
        cfg = self.config
        if cfg.min_width == 0 and cfg.min_height == 0 and cfg.max_width >= 99999 and cfg.max_height >= 99999:
            return image_paths

        filtered = []
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            if w >= cfg.min_width and h >= cfg.min_height and w <= cfg.max_width and h <= cfg.max_height:
                filtered.append(p)
        return filtered

    def _resolve_output_dir(self, source_folder: Path) -> Path:
        if self.config.output_folder:
            return self.config.output_folder
        if self.config.save_mode == SaveMode.SAVE_IN_NEW_FOLDER:
            return source_folder / "augmented"
        return source_folder

    def _generate_filename(self, original: Path, copy_index: int) -> str:
        """Generate output filename based on naming mode."""
        stem = original.stem
        ext = self.config.output_format or original.suffix
        if not ext.startswith("."):
            ext = "." + ext
        mode = self.config.naming_mode

        if mode == NamingMode.SUFFIX:
            return f"{stem}{self.config.custom_suffix}{copy_index + 1}{ext}"
        elif mode == NamingMode.PREFIX:
            return f"{self.config.custom_prefix}{stem}_{copy_index + 1}{ext}"
        elif mode == NamingMode.TIMESTAMP:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            return f"{stem}_{ts}{ext}"
        elif mode == NamingMode.UUID:
            return f"{stem}_{uuid.uuid4().hex[:8]}{ext}"
        else:  # KEEP_ORIGINAL
            if copy_index == 0:
                return f"{stem}{ext}"
            return f"{stem}_{copy_index + 1}{ext}"

    def _resolve_save_path(self, original: Path, filename: str, output_dir: Path) -> Path:
        """Resolve the full save path based on save mode."""
        mode = self.config.save_mode
        if mode == SaveMode.OVERWRITE_ORIGINAL:
            return original
        elif mode == SaveMode.SAVE_NEXT_TO_ORIGINAL:
            return original.parent / filename
        else:
            return output_dir / filename

    def _handle_labels(self, source_img: Path, dest_img: Path):
        """Copy/transform labels alongside augmented images."""
        if self.config.label_save_mode == LabelSaveMode.SKIP:
            return
        labels_dir = self.config.labels_dir
        if labels_dir is None:
            # Try common label locations
            for candidate in [source_img.parent.parent / "labels", source_img.parent / "labels", source_img.parent]:
                label_file = candidate / (source_img.stem + ".txt")
                if label_file.exists():
                    labels_dir = candidate
                    break
        if labels_dir is None:
            return

        src_label = labels_dir / (source_img.stem + ".txt")
        if not src_label.exists():
            return

        dest_labels_dir = dest_img.parent
        if dest_img.parent.name != "labels":
            dest_labels_dir = dest_img.parent.parent / "labels"
            if not dest_labels_dir.parent.exists():
                dest_labels_dir = dest_img.parent

        dest_labels_dir.mkdir(parents=True, exist_ok=True)
        dest_label = dest_labels_dir / (dest_img.stem + ".txt")
        shutil.copy2(str(src_label), str(dest_label))

    def _create_backup(self, source_folder: Path):
        """Create a backup of the source folder."""
        backup_dir = self.config.backup_folder or source_folder.parent / "backups"
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{source_folder.name}_{ts}"
        shutil.copytree(str(source_folder), str(backup_path))

    def _report_progress(self, fraction: float, text: str):
        if self._progress_callback:
            self._progress_callback(fraction, text)
