"""
dataset_manager.py
-----------------------------------------------------------------------------
Handles dataset import / export for YOLO, COCO, Pascal VOC and classification
formats.  Also provides dataset information, cleaning, balancing, splitting,
merging, and statistics.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from Src.augmentation_engine import SUPPORTED_EXTENSIONS, find_images_in_folder


# ═══════════════════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BBox:
    """Bounding box with class id and coordinates in *absolute* pixels."""
    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return max(0, self.width) * max(0, self.height)


@dataclass
class ImageAnnotation:
    image_path: Path
    width: int
    height: int
    bboxes: List[BBox] = field(default_factory=list)


@dataclass
class DatasetInfo:
    """Summary statistics for a loaded dataset."""
    total_images: int = 0
    total_classes: int = 0
    class_names: List[str] = field(default_factory=list)
    images_per_class: Dict[str, int] = field(default_factory=dict)
    total_bboxes: int = 0
    avg_width: float = 0.0
    avg_height: float = 0.0
    min_width: int = 0
    max_width: int = 0
    min_height: int = 0
    max_height: int = 0
    dataset_size_bytes: int = 0
    empty_labels: int = 0
    missing_labels: int = 0
    bbox_size_distribution: Dict[str, int] = field(default_factory=dict)  # small/medium/large
    aspect_ratios: List[float] = field(default_factory=list)
    class_distribution: Dict[str, int] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
#  YOLO Format
# ═══════════════════════════════════════════════════════════════════════════

def load_yolo_dataset(images_dir: Path, labels_dir: Path,
                      class_names: Optional[List[str]] = None
                      ) -> Tuple[List[ImageAnnotation], List[str]]:
    """Load a YOLO-format dataset (normalized xywh)."""
    images = find_images_in_folder(images_dir)
    annotations: List[ImageAnnotation] = []

    detected_classes: set = set()
    for img_path in images:
        label_path = labels_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        bboxes: List[BBox] = []
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x_min = (cx - bw / 2) * w
                y_min = (cy - bh / 2) * h
                x_max = (cx + bw / 2) * w
                y_max = (cy + bh / 2) * h
                bboxes.append(BBox(cls_id, x_min, y_min, x_max, y_max))
                detected_classes.add(cls_id)
        annotations.append(ImageAnnotation(img_path, w, h, bboxes))

    if class_names is None:
        class_names = [str(i) for i in sorted(detected_classes)]
    return annotations, class_names


def export_yolo_dataset(annotations: List[ImageAnnotation],
                        class_names: List[str],
                        output_dir: Path,
                        copy_images: bool = True) -> None:
    """Export annotations in YOLO format."""
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for ann in annotations:
        if copy_images and ann.image_path.exists():
            shutil.copy2(str(ann.image_path), str(images_out / ann.image_path.name))
        label_file = labels_out / (ann.image_path.stem + ".txt")
        lines = []
        for bb in ann.bboxes:
            cx = ((bb.x_min + bb.x_max) / 2) / ann.width
            cy = ((bb.y_min + bb.y_max) / 2) / ann.height
            bw = bb.width / ann.width
            bh = bb.height / ann.height
            lines.append(f"{bb.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label_file.write_text("\n".join(lines))

    # Write classes.txt
    (output_dir / "classes.txt").write_text("\n".join(class_names))


# ═══════════════════════════════════════════════════════════════════════════
#  COCO Format
# ═══════════════════════════════════════════════════════════════════════════

def load_coco_dataset(json_path: Path, images_dir: Path
                      ) -> Tuple[List[ImageAnnotation], List[str]]:
    """Load a COCO-format dataset from its JSON annotation file."""
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco.get("categories", [])}
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(coco.get("categories", []))}
    class_names = [cat_map[c["id"]] for c in coco.get("categories", [])]

    img_map: Dict[int, dict] = {img["id"]: img for img in coco.get("images", [])}
    ann_by_img: Dict[int, List[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        ann_by_img[ann["image_id"]].append(ann)

    annotations: List[ImageAnnotation] = []
    for img_id, img_info in img_map.items():
        img_path = images_dir / img_info["file_name"]
        w = img_info.get("width", 0)
        h = img_info.get("height", 0)
        bboxes: List[BBox] = []
        for ann in ann_by_img.get(img_id, []):
            if "bbox" not in ann:
                continue
            bx, by, bw, bh = ann["bbox"]  # COCO: x, y, width, height
            cls_idx = cat_id_to_idx.get(ann["category_id"], 0)
            bboxes.append(BBox(cls_idx, bx, by, bx + bw, by + bh))
        annotations.append(ImageAnnotation(img_path, w, h, bboxes))

    return annotations, class_names


def export_coco_dataset(annotations: List[ImageAnnotation],
                        class_names: List[str],
                        output_dir: Path,
                        copy_images: bool = True) -> None:
    """Export annotations in COCO JSON format."""
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    coco: Dict[str, Any] = {"images": [], "annotations": [], "categories": []}
    for i, name in enumerate(class_names):
        coco["categories"].append({"id": i, "name": name})

    ann_id = 1
    for img_id, ann in enumerate(annotations, start=1):
        if copy_images and ann.image_path.exists():
            shutil.copy2(str(ann.image_path), str(images_out / ann.image_path.name))
        coco["images"].append({
            "id": img_id,
            "file_name": ann.image_path.name,
            "width": ann.width,
            "height": ann.height,
        })
        for bb in ann.bboxes:
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": bb.class_id,
                "bbox": [bb.x_min, bb.y_min, bb.width, bb.height],
                "area": bb.area,
                "iscrowd": 0,
            })
            ann_id += 1

    with open(output_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  Pascal VOC Format
# ═══════════════════════════════════════════════════════════════════════════

def load_voc_dataset(annotations_dir: Path, images_dir: Path,
                     ) -> Tuple[List[ImageAnnotation], List[str]]:
    """Load Pascal VOC XML annotations."""
    class_set: set = set()
    annotations: List[ImageAnnotation] = []
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        filename = root.findtext("filename", "")
        img_path = images_dir / filename
        size = root.find("size")
        w = int(size.findtext("width", "0")) if size is not None else 0
        h = int(size.findtext("height", "0")) if size is not None else 0
        bboxes: List[BBox] = []
        for obj in root.findall("object"):
            name = obj.findtext("name", "unknown")
            class_set.add(name)
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            x_min = float(bndbox.findtext("xmin", "0"))
            y_min = float(bndbox.findtext("ymin", "0"))
            x_max = float(bndbox.findtext("xmax", "0"))
            y_max = float(bndbox.findtext("ymax", "0"))
            bboxes.append(BBox(-1, x_min, y_min, x_max, y_max))  # class_id resolved below
        annotations.append(ImageAnnotation(img_path, w, h, bboxes))

    class_names = sorted(class_set)
    name_to_id = {name: i for i, name in enumerate(class_names)}

    # Resolve class IDs (requires re-parsing names)
    for xml_path, ann in zip(sorted(annotations_dir.glob("*.xml")), annotations):
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        for i, obj in enumerate(root.findall("object")):
            name = obj.findtext("name", "unknown")
            if i < len(ann.bboxes):
                ann.bboxes[i].class_id = name_to_id.get(name, 0)

    return annotations, class_names


def export_voc_dataset(annotations: List[ImageAnnotation],
                       class_names: List[str],
                       output_dir: Path,
                       copy_images: bool = True) -> None:
    """Export annotations in Pascal VOC XML format."""
    images_out = output_dir / "JPEGImages"
    ann_out = output_dir / "Annotations"
    images_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    for ann in annotations:
        if copy_images and ann.image_path.exists():
            shutil.copy2(str(ann.image_path), str(images_out / ann.image_path.name))
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = ann.image_path.name
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(ann.width)
        ET.SubElement(size, "height").text = str(ann.height)
        ET.SubElement(size, "depth").text = "3"
        for bb in ann.bboxes:
            obj = ET.SubElement(root, "object")
            name = class_names[bb.class_id] if bb.class_id < len(class_names) else str(bb.class_id)
            ET.SubElement(obj, "name").text = name
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(bb.x_min))
            ET.SubElement(bndbox, "ymin").text = str(int(bb.y_min))
            ET.SubElement(bndbox, "xmax").text = str(int(bb.x_max))
            ET.SubElement(bndbox, "ymax").text = str(int(bb.y_max))
        tree = ET.ElementTree(root)
        tree.write(str(ann_out / (ann.image_path.stem + ".xml")), encoding="unicode")


# ═══════════════════════════════════════════════════════════════════════════
#  Label Converter
# ═══════════════════════════════════════════════════════════════════════════

def convert_labels(annotations: List[ImageAnnotation], class_names: List[str],
                   source_format: str, target_format: str,
                   output_dir: Path) -> None:
    """Convert labels from one format to another."""
    exporters = {
        "yolo": export_yolo_dataset,
        "coco": export_coco_dataset,
        "voc": export_voc_dataset,
    }
    exporter = exporters.get(target_format.lower())
    if exporter is None:
        raise ValueError(f"Unknown target format: {target_format}")
    exporter(annotations, class_names, output_dir, copy_images=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Label Checker
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LabelIssue:
    image_path: Path
    issue_type: str
    details: str


def check_labels(annotations: List[ImageAnnotation],
                 num_classes: int) -> List[LabelIssue]:
    """Check labels for common issues."""
    issues: List[LabelIssue] = []
    for ann in annotations:
        if not ann.bboxes:
            issues.append(LabelIssue(ann.image_path, "empty_label", "No bounding boxes"))
            continue

        seen_boxes = set()
        for bb in ann.bboxes:
            # Invalid class IDs
            if bb.class_id < 0 or bb.class_id >= num_classes:
                issues.append(LabelIssue(ann.image_path, "wrong_class_id",
                                         f"Class ID {bb.class_id} out of range [0, {num_classes})"))

            # Negative values
            if bb.x_min < 0 or bb.y_min < 0:
                issues.append(LabelIssue(ann.image_path, "negative_values",
                                         f"Negative coordinates: ({bb.x_min}, {bb.y_min})"))

            # Boxes outside image
            if bb.x_max > ann.width or bb.y_max > ann.height:
                issues.append(LabelIssue(ann.image_path, "box_outside_image",
                                         f"Box exceeds image bounds ({ann.width}x{ann.height})"))

            # Invalid coordinates (inverted)
            if bb.x_min >= bb.x_max or bb.y_min >= bb.y_max:
                issues.append(LabelIssue(ann.image_path, "invalid_coordinates",
                                         f"Inverted coords: ({bb.x_min},{bb.y_min})->({bb.x_max},{bb.y_max})"))

            # Tiny boxes (< 4px in either dimension)
            if bb.width < 4 or bb.height < 4:
                issues.append(LabelIssue(ann.image_path, "tiny_box",
                                         f"Tiny box: {bb.width:.1f}x{bb.height:.1f}px"))

            # Duplicate boxes
            key = (bb.class_id, round(bb.x_min), round(bb.y_min), round(bb.x_max), round(bb.y_max))
            if key in seen_boxes:
                issues.append(LabelIssue(ann.image_path, "duplicate_box",
                                         f"Duplicate box for class {bb.class_id}"))
            seen_boxes.add(key)

    return issues


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset Cleaning
# ═══════════════════════════════════════════════════════════════════════════

def find_duplicate_images(image_paths: List[Path]) -> List[List[Path]]:
    """Find duplicate images using file hash comparison."""
    hash_map: Dict[str, List[Path]] = defaultdict(list)
    for path in image_paths:
        if not path.exists():
            continue
        file_hash = hashlib.md5(path.read_bytes()).hexdigest()
        hash_map[file_hash].append(path)
    return [group for group in hash_map.values() if len(group) > 1]


def find_blurry_images(image_paths: List[Path], threshold: float = 100.0
                       ) -> List[Tuple[Path, float]]:
    """Find blurry images using Laplacian variance."""
    blurry = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        variance = cv2.Laplacian(img, cv2.CV_64F).var()
        if variance < threshold:
            blurry.append((path, variance))
    return sorted(blurry, key=lambda x: x[1])


def find_corrupted_images(image_paths: List[Path]) -> List[Path]:
    """Find images that cannot be decoded."""
    corrupted = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            corrupted.append(path)
    return corrupted


def find_tiny_images(image_paths: List[Path], min_width: int = 32, min_height: int = 32
                     ) -> List[Path]:
    """Find images smaller than the given minimum dimensions."""
    tiny = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if w < min_width or h < min_height:
            tiny.append(path)
    return tiny


def find_grayscale_images(image_paths: List[Path]) -> List[Path]:
    """Find images that are effectively grayscale."""
    gray_imgs = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None or len(img.shape) < 3:
            continue
        b, g, r = cv2.split(img)
        if np.allclose(b, g, atol=5) and np.allclose(g, r, atol=5):
            gray_imgs.append(path)
    return gray_imgs


def remove_empty_folders(root: Path) -> int:
    """Recursively remove empty folders. Returns number of folders removed."""
    removed = 0
    for dirpath, dirnames, filenames in os.walk(str(root), topdown=False):
        dp = Path(dirpath)
        if dp == root:
            continue
        if not any(dp.iterdir()):
            dp.rmdir()
            removed += 1
    return removed


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset Balancing
# ═══════════════════════════════════════════════════════════════════════════

def get_class_distribution(annotations: List[ImageAnnotation],
                           class_names: List[str]) -> Dict[str, int]:
    """Count images containing each class."""
    dist: Dict[str, int] = {name: 0 for name in class_names}
    for ann in annotations:
        seen_classes: set = set()
        for bb in ann.bboxes:
            if 0 <= bb.class_id < len(class_names):
                seen_classes.add(class_names[bb.class_id])
        for cls in seen_classes:
            dist[cls] += 1
    return dist


def compute_balance_plan(distribution: Dict[str, int],
                         target_count: Optional[int] = None
                         ) -> Dict[str, int]:
    """Compute how many augmented images each class needs to reach the target."""
    if not distribution:
        return {}
    if target_count is None:
        target_count = max(distribution.values())
    return {cls: max(0, target_count - count) for cls, count in distribution.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset Split
# ═══════════════════════════════════════════════════════════════════════════

def split_dataset(image_paths: List[Path],
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.2,
                  test_ratio: float = 0.1,
                  seed: Optional[int] = 42,
                  stratify_labels_dir: Optional[Path] = None
                  ) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split images into train/val/test sets."""
    if seed is not None:
        random.seed(seed)

    paths = list(image_paths)
    random.shuffle(paths)

    total = len(paths)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = paths[:train_end]
    val = paths[train_end:val_end]
    test = paths[val_end:]
    return train, val, test


def export_split(train: List[Path], val: List[Path], test: List[Path],
                 output_dir: Path, labels_dir: Optional[Path] = None) -> None:
    """Copy split images (and labels) into train/val/test folders."""
    for split_name, split_paths in [("train", train), ("valid", val), ("test", test)]:
        img_dir = output_dir / split_name / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for p in split_paths:
            shutil.copy2(str(p), str(img_dir / p.name))
        if labels_dir:
            lbl_dir = output_dir / split_name / "labels"
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for p in split_paths:
                lbl = labels_dir / (p.stem + ".txt")
                if lbl.exists():
                    shutil.copy2(str(lbl), str(lbl_dir / lbl.name))


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset Merge
# ═══════════════════════════════════════════════════════════════════════════

def merge_datasets(datasets: List[Tuple[Path, Path]],
                   output_dir: Path) -> Tuple[int, int]:
    """
    Merge multiple datasets. Each dataset is (images_dir, labels_dir).
    Returns (total_images, total_renamed).
    """
    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    seen_names: set = set()
    total = 0
    renamed = 0

    for images_dir, labels_dir in datasets:
        for img_path in find_images_in_folder(images_dir):
            name = img_path.stem
            ext = img_path.suffix
            final_name = name
            if final_name in seen_names:
                counter = 1
                while f"{name}_{counter}" in seen_names:
                    counter += 1
                final_name = f"{name}_{counter}"
                renamed += 1
            seen_names.add(final_name)
            shutil.copy2(str(img_path), str(out_images / f"{final_name}{ext}"))
            label_path = labels_dir / (name + ".txt")
            if label_path.exists():
                shutil.copy2(str(label_path), str(out_labels / f"{final_name}.txt"))
            total += 1

    return total, renamed


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset Statistics
# ═══════════════════════════════════════════════════════════════════════════

def compute_dataset_info(annotations: List[ImageAnnotation],
                         class_names: List[str]) -> DatasetInfo:
    """Compute comprehensive dataset statistics."""
    info = DatasetInfo()
    info.total_images = len(annotations)
    info.total_classes = len(class_names)
    info.class_names = class_names

    widths, heights, aspects = [], [], []
    total_size = 0
    bbox_count = 0
    small, medium, large = 0, 0, 0
    class_counter: Counter = Counter()

    for ann in annotations:
        widths.append(ann.width)
        heights.append(ann.height)
        if ann.width > 0 and ann.height > 0:
            aspects.append(ann.width / ann.height)
        if ann.image_path.exists():
            total_size += ann.image_path.stat().st_size

        if not ann.bboxes:
            info.empty_labels += 1

        for bb in ann.bboxes:
            bbox_count += 1
            if 0 <= bb.class_id < len(class_names):
                class_counter[class_names[bb.class_id]] += 1
            area = bb.area
            img_area = max(1, ann.width * ann.height)
            ratio = area / img_area
            if ratio < 0.01:
                small += 1
            elif ratio < 0.1:
                medium += 1
            else:
                large += 1

    info.total_bboxes = bbox_count
    info.dataset_size_bytes = total_size
    info.aspect_ratios = aspects

    if widths:
        info.avg_width = sum(widths) / len(widths)
        info.avg_height = sum(heights) / len(heights)
        info.min_width = min(widths)
        info.max_width = max(widths)
        info.min_height = min(heights)
        info.max_height = max(heights)

    info.class_distribution = dict(class_counter)
    info.images_per_class = dict(get_class_distribution(annotations, class_names))
    info.bbox_size_distribution = {"small": small, "medium": medium, "large": large}

    return info


# ═══════════════════════════════════════════════════════════════════════════
#  ZIP Import / Export
# ═══════════════════════════════════════════════════════════════════════════

def import_zip(zip_path: Path, extract_to: Path) -> Path:
    """Extract a ZIP dataset and return the extraction folder."""
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(extract_to))
    return extract_to


def export_zip(source_dir: Path, zip_path: Path) -> None:
    """Zip a dataset directory."""
    with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in source_dir.rglob("*"):
            if file.is_file():
                zf.write(str(file), str(file.relative_to(source_dir)))
