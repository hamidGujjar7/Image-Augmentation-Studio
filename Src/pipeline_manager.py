"""
pipeline_manager.py
-----------------------------------------------------------------------------
Preset & pipeline management. Save/load complete augmentation configurations
with named presets for common use cases (YOLO Training, Classification,
Segmentation, Medical, Warehouse, Traffic, etc.).
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from Src.augmentation_engine import AugmentationParams


# ═══════════════════════════════════════════════════════════════════════════
#  Built-in Presets
# ═══════════════════════════════════════════════════════════════════════════

BUILTIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "YOLO Training": {
        "rotation_enabled": True, "rotation_degrees": 15, "rotation_probability": 0.7,
        "flip_h_enabled": True, "flip_h_probability": 0.5,
        "scale_enabled": True, "scale_factor": 1.0, "scale_probability": 0.5,
        "brightness_enabled": True, "brightness_factor": 1.05, "brightness_probability": 0.5,
        "contrast_enabled": True, "contrast_factor": 1.02, "contrast_probability": 0.4,
        "gauss_noise_enabled": True, "gauss_noise_amount": 0.01, "gauss_noise_probability": 0.3,
        "random_order_enabled": True,
    },
    "Classification": {
        "rotation_enabled": True, "rotation_degrees": 20, "rotation_probability": 0.6,
        "flip_h_enabled": True, "flip_h_probability": 0.5,
        "flip_v_enabled": True, "flip_v_probability": 0.3,
        "brightness_enabled": True, "brightness_factor": 1.1, "brightness_probability": 0.5,
        "contrast_enabled": True, "contrast_factor": 1.05, "contrast_probability": 0.5,
        "saturation_enabled": True, "saturation_factor": 1.1, "saturation_probability": 0.4,
        "gaussian_blur_enabled": True, "gaussian_blur_kernel": 3, "gaussian_blur_probability": 0.3,
        "random_order_enabled": True,
    },
    "Segmentation": {
        "rotation_enabled": True, "rotation_degrees": 15, "rotation_probability": 0.5,
        "flip_h_enabled": True, "flip_h_probability": 0.5,
        "elastic_enabled": True, "elastic_alpha": 80, "elastic_sigma": 5, "elastic_probability": 0.3,
        "brightness_enabled": True, "brightness_factor": 1.05, "brightness_probability": 0.4,
        "gaussian_blur_enabled": True, "gaussian_blur_kernel": 3, "gaussian_blur_probability": 0.2,
        "random_order_enabled": True,
    },
    "OCR": {
        "rotation_enabled": True, "rotation_degrees": 5, "rotation_probability": 0.5,
        "perspective_enabled": True, "perspective_scale": 0.03, "perspective_probability": 0.4,
        "brightness_enabled": True, "brightness_factor": 1.1, "brightness_probability": 0.5,
        "contrast_enabled": True, "contrast_factor": 1.1, "contrast_probability": 0.5,
        "gaussian_blur_enabled": True, "gaussian_blur_kernel": 3, "gaussian_blur_probability": 0.3,
        "gauss_noise_enabled": True, "gauss_noise_amount": 0.005, "gauss_noise_probability": 0.3,
        "jpeg_compression_enabled": True, "jpeg_quality_lower": 50, "jpeg_quality_upper": 85, "jpeg_compression_probability": 0.3,
        "random_order_enabled": True,
    },
    "Medical": {
        "rotation_enabled": True, "rotation_degrees": 10, "rotation_probability": 0.5,
        "flip_h_enabled": True, "flip_h_probability": 0.5,
        "flip_v_enabled": True, "flip_v_probability": 0.5,
        "elastic_enabled": True, "elastic_alpha": 60, "elastic_sigma": 4, "elastic_probability": 0.3,
        "brightness_enabled": True, "brightness_factor": 1.05, "brightness_probability": 0.4,
        "contrast_enabled": True, "contrast_factor": 1.05, "contrast_probability": 0.4,
        "clahe_enabled": True, "clahe_clip_limit": 4.0, "clahe_probability": 0.3,
        "random_order_enabled": True,
    },
    "Warehouse": {
        "rotation_enabled": True, "rotation_degrees": 10, "rotation_probability": 0.5,
        "flip_h_enabled": True, "flip_h_probability": 0.5,
        "brightness_enabled": True, "brightness_factor": 1.15, "brightness_probability": 0.6,
        "contrast_enabled": True, "contrast_factor": 1.1, "contrast_probability": 0.5,
        "fog_enabled": True, "fog_intensity": 0.15, "fog_probability": 0.2,
        "motion_blur_enabled": True, "motion_blur_kernel": 5, "motion_blur_probability": 0.3,
        "gauss_noise_enabled": True, "gauss_noise_amount": 0.01, "gauss_noise_probability": 0.3,
        "random_order_enabled": True,
    },
    "Traffic": {
        "rotation_enabled": True, "rotation_degrees": 8, "rotation_probability": 0.4,
        "flip_h_enabled": True, "flip_h_probability": 0.5,
        "scale_enabled": True, "scale_factor": 0.95, "scale_probability": 0.4,
        "brightness_enabled": True, "brightness_factor": 1.2, "brightness_probability": 0.6,
        "rain_enabled": True, "rain_intensity": 0.3, "rain_probability": 0.2,
        "fog_enabled": True, "fog_intensity": 0.2, "fog_probability": 0.2,
        "sun_flare_enabled": True, "sun_flare_intensity": 0.3, "sun_flare_probability": 0.15,
        "motion_blur_enabled": True, "motion_blur_kernel": 7, "motion_blur_probability": 0.3,
        "random_order_enabled": True,
    },
    "Drone": {
        "rotation_enabled": True, "rotation_degrees": 25, "rotation_probability": 0.6,
        "flip_h_enabled": True, "flip_h_probability": 0.5,
        "flip_v_enabled": True, "flip_v_probability": 0.3,
        "scale_enabled": True, "scale_factor": 0.9, "scale_probability": 0.5,
        "perspective_enabled": True, "perspective_scale": 0.05, "perspective_probability": 0.4,
        "brightness_enabled": True, "brightness_factor": 1.15, "brightness_probability": 0.5,
        "shadow_enabled": True, "shadow_intensity": 0.3, "shadow_probability": 0.3,
        "fog_enabled": True, "fog_intensity": 0.15, "fog_probability": 0.15,
        "random_order_enabled": True,
    },
    "Night": {
        "brightness_enabled": True, "brightness_factor": 0.7, "brightness_probability": 0.7,
        "contrast_enabled": True, "contrast_factor": 1.15, "contrast_probability": 0.5,
        "gauss_noise_enabled": True, "gauss_noise_amount": 0.03, "gauss_noise_probability": 0.6,
        "iso_noise_enabled": True, "iso_noise_intensity": 0.5, "iso_noise_probability": 0.5,
        "gaussian_blur_enabled": True, "gaussian_blur_kernel": 3, "gaussian_blur_probability": 0.3,
        "gamma_enabled": True, "gamma_value": 0.6, "gamma_probability": 0.5,
        "random_order_enabled": True,
    },
    "Low Light": {
        "brightness_enabled": True, "brightness_factor": 0.6, "brightness_probability": 0.8,
        "contrast_enabled": True, "contrast_factor": 1.2, "contrast_probability": 0.6,
        "gauss_noise_enabled": True, "gauss_noise_amount": 0.04, "gauss_noise_probability": 0.7,
        "iso_noise_enabled": True, "iso_noise_intensity": 0.6, "iso_noise_probability": 0.6,
        "clahe_enabled": True, "clahe_clip_limit": 6.0, "clahe_probability": 0.4,
        "gamma_enabled": True, "gamma_value": 0.5, "gamma_probability": 0.6,
        "random_order_enabled": True,
    },
    "Industrial": {
        "rotation_enabled": True, "rotation_degrees": 5, "rotation_probability": 0.4,
        "flip_h_enabled": True, "flip_h_probability": 0.5,
        "brightness_enabled": True, "brightness_factor": 1.15, "brightness_probability": 0.6,
        "contrast_enabled": True, "contrast_factor": 1.1, "contrast_probability": 0.5,
        "gaussian_blur_enabled": True, "gaussian_blur_kernel": 3, "gaussian_blur_probability": 0.3,
        "gauss_noise_enabled": True, "gauss_noise_amount": 0.01, "gauss_noise_probability": 0.3,
        "jpeg_compression_enabled": True, "jpeg_quality_lower": 60, "jpeg_quality_upper": 90, "jpeg_compression_probability": 0.3,
        "random_order_enabled": True,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline Manager
# ═══════════════════════════════════════════════════════════════════════════

class PipelineManager:
    """Save, load, and manage augmentation presets/pipelines."""

    def __init__(self, presets_dir: Optional[Path] = None):
        self.presets_dir = presets_dir or Path.home() / ".augmentation_studio" / "presets"
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def get_builtin_preset_names(self) -> List[str]:
        """Return names of all built-in presets."""
        return list(BUILTIN_PRESETS.keys())

    def get_custom_preset_names(self) -> List[str]:
        """Return names of all user-saved presets."""
        return [p.stem for p in self.presets_dir.glob("*.json")]

    def get_all_preset_names(self) -> List[str]:
        """Return all preset names (built-in + custom)."""
        return self.get_builtin_preset_names() + self.get_custom_preset_names()

    def load_preset(self, name: str) -> Dict[str, Any]:
        """Load a preset by name. Checks built-in first, then custom."""
        if name in BUILTIN_PRESETS:
            return BUILTIN_PRESETS[name].copy()
        custom_path = self.presets_dir / f"{name}.json"
        if custom_path.exists():
            with open(custom_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise ValueError(f"Preset not found: {name}")

    def save_preset(self, name: str, params: AugmentationParams,
                    extra: Optional[Dict[str, Any]] = None) -> Path:
        """Save current params as a named preset."""
        data = {}
        for key, val in params.__dict__.items():
            data[key] = val
        if extra:
            data.update(extra)
        out_path = self.presets_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return out_path

    def delete_preset(self, name: str) -> bool:
        """Delete a custom preset. Cannot delete built-ins."""
        if name in BUILTIN_PRESETS:
            return False
        path = self.presets_dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def apply_preset(self, name: str, params: AugmentationParams) -> AugmentationParams:
        """Load a preset and apply it to the given params object."""
        preset = self.load_preset(name)
        # Reset all _enabled fields to False first
        for field_name in params.__dataclass_fields__:
            if field_name.endswith("_enabled"):
                setattr(params, field_name, False)
        # Apply preset values
        for key, value in preset.items():
            if hasattr(params, key):
                setattr(params, key, value)
        return params

    def export_pipeline(self, params: AugmentationParams, path: Path,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Export a complete pipeline configuration to a JSON file."""
        data = {"augmentation_params": {}}
        for key, val in params.__dict__.items():
            data["augmentation_params"][key] = val
        if metadata:
            data["metadata"] = metadata
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def import_pipeline(self, path: Path) -> Dict[str, Any]:
        """Import a pipeline configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("augmentation_params", data)
