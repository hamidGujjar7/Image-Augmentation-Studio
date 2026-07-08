"""
gui_app.py
-----------------------------------------------------------------------------
Modern dark-mode GUI for the Image Augmentation Tool, built with
CustomTkinter. This module only handles presentation and user interaction;
all image-processing logic lives in augmentation_engine.py and related
modules.

Features:
  - Full augmentation controls (Geometry, Weather, Blur, Noise, Color,
    Compression, Advanced)
  - Dataset management (Import/Export YOLO, COCO, VOC)
  - Batch processing with selection modes, save modes, naming
  - Pipeline presets (11 built-in + custom save/load)
  - Dataset tools (cleaning, balancing, splitting, merging, statistics)
  - Image utilities (rename, resize, convert, crop, tiles)
  - Preview system with before/after comparison
  - Undo/Redo system
  - Progress tracking with ETA
  - Label checker
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import json
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Callable, Optional

import customtkinter as ctk
import numpy as np
from PIL import Image

from Src.augmentation_engine import (
    AugmentationEngine,
    AugmentationParams,
    find_images_in_folder,
    load_image_rgb,
    save_image_rgb,
)
from Src.batch_processor import (
    BatchConfig,
    BatchProcessor,
    NamingMode,
    SaveMode,
    SelectionMode,
    LabelSaveMode,
)
from Src.pipeline_manager import PipelineManager
from Src.dataset_manager import (
    load_yolo_dataset,
    load_coco_dataset,
    load_voc_dataset,
    export_yolo_dataset,
    export_coco_dataset,
    export_voc_dataset,
    check_labels,
    split_dataset,
    export_split,
    merge_datasets,
    compute_dataset_info,
    find_duplicate_images,
    find_blurry_images,
    find_corrupted_images,
    find_tiny_images,
    remove_empty_folders,
    get_class_distribution,
    compute_balance_plan,
    import_zip,
    export_zip,
)
from Src.image_utilities import (
    rename_files,
    resize_all_images,
    convert_format,
    convert_to_grayscale,
    convert_to_rgb,
    split_into_tiles,
    merge_tiles,
    remove_exif_batch,
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

FONT_HEADER = ("Segoe UI", 16, "bold")
FONT_SECTION = ("Segoe UI", 13, "bold")
FONT_LABEL = ("Segoe UI", 12)
FONT_SMALL = ("Segoe UI", 11)

PREVIEW_MAX_SIZE = 480
PAD = 12


# ═══════════════════════════════════════════════════════════════════════════
#  Reusable Widget Rows
# ═══════════════════════════════════════════════════════════════════════════

class SliderRow(ctk.CTkFrame):
    def __init__(self, master, label: str, from_: float, to: float, initial: float, on_change: Callable[[float], None], is_int: bool = False, fmt: str = "{:.2f}", search_tags: Optional[list[str]] = None):
        super().__init__(master, fg_color="transparent")
        self.is_int = is_int
        self.fmt = fmt
        self.on_change = on_change
        self.search_tags = search_tags or [label.lower()]
        self._updating_programmatically = False

        self.label = ctk.CTkLabel(self, text=label, font=FONT_LABEL, anchor="w")
        self.label.grid(row=0, column=0, sticky="w")
        self.value_label = ctk.CTkLabel(self, text=self._format(initial), font=FONT_SMALL, text_color="#8ab4f8")
        self.value_label.grid(row=0, column=1, sticky="e")

        steps = int(to - from_) if is_int else 200
        self.slider = ctk.CTkSlider(self, from_=from_, to=to, number_of_steps=max(steps, 1), command=self._handle_change)
        self.slider.set(initial)
        self.slider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 8))
        self.grid_columnconfigure(0, weight=1)

    def _format(self, value: float) -> str:
        return str(int(round(value))) if self.is_int else self.fmt.format(value)

    def _handle_change(self, value: float):
        value = int(round(value)) if self.is_int else float(value)
        self.value_label.configure(text=self._format(value))
        if not self._updating_programmatically:
            self.on_change(value)

    def set_value(self, value: float):
        self._updating_programmatically = True
        try:
            self.slider.set(value)
            self._handle_change(value)
        finally:
            self._updating_programmatically = False

    def matches(self, query: str) -> bool:
        return not query or any(query in tag for tag in self.search_tags)


class ToggleRow(ctk.CTkFrame):
    def __init__(self, master, label: str, on_toggle: Callable[[bool], None], search_tags: Optional[list[str]] = None):
        super().__init__(master, fg_color="transparent")
        self.search_tags = search_tags or [label.lower()]
        self.on_toggle = on_toggle
        self.var = ctk.BooleanVar(value=False)
        self.checkbox = ctk.CTkCheckBox(self, text=label, variable=self.var, font=FONT_LABEL, command=lambda: on_toggle(self.var.get()))
        self.checkbox.pack(anchor="w")

    def set_value(self, value: bool):
        if self.var.get() != value:
            self.var.set(value)
        self.on_toggle(value)

    def matches(self, query: str) -> bool:
        return not query or any(query in tag for tag in self.search_tags)


# ═══════════════════════════════════════════════════════════════════════════
#  Main Application
# ═══════════════════════════════════════════════════════════════════════════

class ImageAugmentationApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Augmentation Studio")
        self.geometry("1480x920")
        self.minsize(1200, 780)
        self.configure(fg_color="#0b1020")

        self.params = AugmentationParams()
        self.engine = AugmentationEngine(self.params)
        self.pipeline_manager = PipelineManager()
        self.current_image_path: Optional[Path] = None
        self.current_image_np: Optional[np.ndarray] = None
        self.selected_folder: Optional[Path] = None
        self.variants_per_image = 6
        self._batch_running = False
        self._batch_processor: Optional[BatchProcessor] = None

        self._original_ctk_image: Optional[ctk.CTkImage] = None
        self._augmented_ctk_image: Optional[ctk.CTkImage] = None
        self.searchable_rows = []
        self.slider_controls = {}
        self.toggle_controls = {}
        self.section_cards = {}

        # Undo/Redo
        self._undo_stack: list = []
        self._redo_stack: list = []
        self._max_undo = 50

        self._build_layout()

    # ── Layout ────────────────────────────────────────────────────────────

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_preview_area()
        self._build_output_sidebar()

    # ── Sidebar (Left) ───────────────────────────────────────────────────

    def _build_sidebar(self):
        sidebar = ctk.CTkScrollableFrame(self, width=380, corner_radius=0, fg_color="#0b1020")
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(sidebar, fg_color="#15233b", corner_radius=16)
        header.pack(fill="x", pady=(0, PAD))
        ctk.CTkLabel(header, text="🎛  Augmentation Studio", font=FONT_HEADER).pack(anchor="w", padx=PAD, pady=(PAD, 4))
        ctk.CTkLabel(header, text="Complete image augmentation & dataset toolkit", font=FONT_SMALL, text_color="#9bb3cf").pack(anchor="w", padx=PAD, pady=(0, PAD))

        # Search
        self.search_var = ctk.StringVar(value="")
        self.search_entry = ctk.CTkEntry(sidebar, placeholder_text="Search controls…", textvariable=self.search_var)
        self.search_entry.pack(fill="x", pady=(0, PAD))
        self.search_entry.bind("<KeyRelease>", lambda _event: self._apply_search())

        # Section cards
        self._build_quick_actions_card(sidebar)
        self._build_input_card(sidebar)
        self.section_cards["Geometry"] = self._build_section_card(sidebar, "Geometry", "Rotate, flip, scale, crop, pad, perspective, shear, elastic", self._build_geometry_controls)
        self.section_cards["Weather"] = self._build_section_card(sidebar, "Weather", "Rain, snow, fog, sun flare, shadow", self._build_weather_controls)
        self.section_cards["Color"] = self._build_section_card(sidebar, "Color", "Brightness, contrast, saturation, hue, gamma, CLAHE, and more", self._build_color_controls)
        self.section_cards["Blur"] = self._build_section_card(sidebar, "Blur", "Gaussian, motion, median, glass, defocus, zoom blur", self._build_blur_controls)
        self.section_cards["Noise"] = self._build_section_card(sidebar, "Noise", "Gaussian, ISO, salt & pepper, speckle noise", self._build_noise_controls)
        self.section_cards["Compression"] = self._build_section_card(sidebar, "Compression", "JPEG compression, downscale / low resolution", self._build_compression_controls)
        self.section_cards["Advanced"] = self._build_section_card(sidebar, "Advanced", "Random erasing, grid dropout, coarse dropout", self._build_advanced_controls)
        self.section_cards["Batch"] = self._build_section_card(sidebar, "Batch Processing", "Selection, copies, save modes, progress", self._build_batch_controls)
        self._build_dataset_tools_card(sidebar)
        self._build_image_utils_card(sidebar)

    # ── Quick Actions ─────────────────────────────────────────────────────

    def _build_quick_actions_card(self, parent):
        card = self._section_card(parent, "Quick Actions", "Presets, seed, order, undo/redo")

        # Preset buttons row 1
        buttons = ctk.CTkFrame(card, fg_color="transparent")
        buttons.pack(fill="x", pady=(0, 4))
        for label in ["Light", "Balanced", "Heavy"]:
            ctk.CTkButton(buttons, text=label, width=80, command=lambda p=label: self._apply_preset(p)).pack(side="left", padx=(0, 6))
        ctk.CTkButton(buttons, text="Reset", width=80, fg_color="#6b7280", hover_color="#4b5563", command=self._reset_defaults).pack(side="left")

        # Pipeline preset dropdown
        preset_frame = ctk.CTkFrame(card, fg_color="transparent")
        preset_frame.pack(fill="x", pady=(4, 6))
        ctk.CTkLabel(preset_frame, text="Pipeline Preset", font=FONT_LABEL).pack(side="left")
        preset_names = self.pipeline_manager.get_all_preset_names()
        self.preset_var = ctk.StringVar(value="Select preset...")
        self.preset_menu = ctk.CTkOptionMenu(
            preset_frame,
            values=preset_names if preset_names else ["No presets"],
            variable=self.preset_var,
            command=self._on_preset_selected,
            width=180,
        )
        self.preset_menu.pack(side="right")

        # Undo/Redo
        undo_frame = ctk.CTkFrame(card, fg_color="transparent")
        undo_frame.pack(fill="x", pady=(2, 6))
        ctk.CTkButton(undo_frame, text="↩ Undo", width=80, fg_color="#4b5563", hover_color="#374151", command=self._undo).pack(side="left", padx=(0, 6))
        ctk.CTkButton(undo_frame, text="↪ Redo", width=80, fg_color="#4b5563", hover_color="#374151", command=self._redo).pack(side="left", padx=(0, 6))
        self.undo_label = ctk.CTkLabel(undo_frame, text="", font=FONT_SMALL, text_color="#9bb3cf")
        self.undo_label.pack(side="left")

        # Random order toggle
        self.random_order_toggle = ToggleRow(card, "Random Order", self._toggle("random_order_enabled"), search_tags=["random", "order", "shuffle"])
        self.random_order_toggle.pack(fill="x", pady=(4, 6))
        self._register_toggle("random_order_enabled", self.random_order_toggle)

        # Seed
        seed_row = ctk.CTkFrame(card, fg_color="transparent")
        seed_row.pack(fill="x", pady=(2, 6))
        ctk.CTkLabel(seed_row, text="Seed", font=FONT_LABEL).pack(side="left")
        self.seed_var = ctk.StringVar(value=str(self.params.seed_value))
        self.seed_entry = ctk.CTkEntry(seed_row, width=100, textvariable=self.seed_var)
        self.seed_entry.pack(side="left", padx=(8, 0))
        self.lock_seed_var = ctk.BooleanVar(value=self.params.lock_seed)
        self.lock_seed_checkbox = ctk.CTkCheckBox(seed_row, text="Lock", variable=self.lock_seed_var, command=self._on_seed_toggle)
        self.lock_seed_checkbox.pack(side="left", padx=(10, 0))

        # Save/Load config
        action_row = ctk.CTkFrame(card, fg_color="transparent")
        action_row.pack(fill="x", pady=(8, 0))
        ctk.CTkButton(action_row, text="Save Config", command=self.on_save_config).pack(side="left", padx=(0, 6))
        ctk.CTkButton(action_row, text="Load Config", command=self.on_load_config).pack(side="left", padx=(0, 6))
        ctk.CTkButton(action_row, text="Save Preset", fg_color="#2563eb", hover_color="#1d4ed8", command=self._save_custom_preset).pack(side="left")

    # ── Input Card ────────────────────────────────────────────────────────

    def _build_input_card(self, parent):
        card = self._section_card(parent, "Input", "Load images, folders, or datasets")
        ctk.CTkButton(card, text="📄  Select Image", command=self.on_select_image).pack(fill="x", pady=3)
        ctk.CTkButton(card, text="📁  Select Folder (Batch)", command=self.on_select_folder).pack(fill="x", pady=3)

        # Dataset import buttons
        dataset_frame = ctk.CTkFrame(card, fg_color="transparent")
        dataset_frame.pack(fill="x", pady=(4, 2))
        ctk.CTkButton(dataset_frame, text="YOLO", width=60, fg_color="#1e40af", hover_color="#1e3a8a", command=self._import_yolo).pack(side="left", padx=(0, 4))
        ctk.CTkButton(dataset_frame, text="COCO", width=60, fg_color="#1e40af", hover_color="#1e3a8a", command=self._import_coco).pack(side="left", padx=(0, 4))
        ctk.CTkButton(dataset_frame, text="VOC", width=60, fg_color="#1e40af", hover_color="#1e3a8a", command=self._import_voc).pack(side="left", padx=(0, 4))
        ctk.CTkButton(dataset_frame, text="ZIP", width=60, fg_color="#1e40af", hover_color="#1e3a8a", command=self._import_zip).pack(side="left")

        self.input_status_label = ctk.CTkLabel(card, text="No file or folder selected.", font=FONT_SMALL, text_color="#9bb3cf", wraplength=320, justify="left")
        self.input_status_label.pack(anchor="w", pady=(4, 6))

    # ── Geometry Controls ─────────────────────────────────────────────────

    def _build_geometry_controls(self, parent):
        # Rotation
        self._add_aug_control(parent, "Rotation", "rotation_enabled", [
            ("Degrees", -45, 45, "rotation_degrees", False, "{:.1f}", ["rotation", "degrees"]),
            ("Probability", 0, 1, "rotation_probability", False, "{:.2f}", ["rotation", "probability"]),
        ], ["rotation", "geometry"])

        # Flip Horizontal
        self._add_aug_control(parent, "Flip Horizontal", "flip_h_enabled", [
            ("Probability", 0, 1, "flip_h_probability", False, "{:.2f}", ["flip", "horizontal"]),
        ], ["flip", "horizontal", "geometry"])

        # Flip Vertical
        self._add_aug_control(parent, "Flip Vertical", "flip_v_enabled", [
            ("Probability", 0, 1, "flip_v_probability", False, "{:.2f}", ["flip", "vertical"]),
        ], ["flip", "vertical", "geometry"])

        # Scale
        self._add_aug_control(parent, "Scale", "scale_enabled", [
            ("Factor", 0.5, 1.5, "scale_factor", False, "{:.2f}", ["scale", "factor"]),
            ("Probability", 0, 1, "scale_probability", False, "{:.2f}", ["scale", "probability"]),
        ], ["scale", "zoom", "geometry"])

        # Random Crop
        self._add_aug_control(parent, "Random Crop", "random_crop_enabled", [
            ("Width", 16, 1024, "random_crop_width", True, "{}", ["crop", "width"]),
            ("Height", 16, 1024, "random_crop_height", True, "{}", ["crop", "height"]),
            ("Probability", 0, 1, "random_crop_probability", False, "{:.2f}", ["crop", "probability"]),
        ], ["crop", "random", "geometry"])

        # Center Crop
        self._add_aug_control(parent, "Center Crop", "center_crop_enabled", [
            ("Width", 16, 1024, "center_crop_width", True, "{}", ["crop", "center", "width"]),
            ("Height", 16, 1024, "center_crop_height", True, "{}", ["crop", "center", "height"]),
            ("Probability", 0, 1, "center_crop_probability", False, "{:.2f}", ["crop", "center"]),
        ], ["crop", "center", "geometry"])

        # Pad
        self._add_aug_control(parent, "Padding", "pad_enabled", [
            ("Pixels", 1, 128, "pad_pixels", True, "{}", ["pad", "pixels"]),
            ("Probability", 0, 1, "pad_probability", False, "{:.2f}", ["pad", "probability"]),
        ], ["pad", "padding", "geometry"])

        # Translation
        self._add_aug_control(parent, "Translation", "translation_enabled", [
            ("X Shift", 0, 0.5, "translation_x", False, "{:.2f}", ["translation", "x"]),
            ("Y Shift", 0, 0.5, "translation_y", False, "{:.2f}", ["translation", "y"]),
            ("Probability", 0, 1, "translation_probability", False, "{:.2f}", ["translation"]),
        ], ["translation", "shift", "geometry"])

        # Perspective
        self._add_aug_control(parent, "Perspective", "perspective_enabled", [
            ("Scale", 0.01, 0.2, "perspective_scale", False, "{:.3f}", ["perspective", "scale"]),
            ("Probability", 0, 1, "perspective_probability", False, "{:.2f}", ["perspective"]),
        ], ["perspective", "geometry"])

        # Affine
        self._add_aug_control(parent, "Affine", "affine_enabled", [
            ("Rotate", 0, 45, "affine_rotate", False, "{:.1f}", ["affine", "rotate"]),
            ("Scale Min", 0.5, 1.0, "affine_scale_min", False, "{:.2f}", ["affine", "scale"]),
            ("Scale Max", 1.0, 1.5, "affine_scale_max", False, "{:.2f}", ["affine", "scale"]),
            ("Probability", 0, 1, "affine_probability", False, "{:.2f}", ["affine"]),
        ], ["affine", "geometry"])

        # Shear
        self._add_aug_control(parent, "Shear", "shear_enabled", [
            ("Degrees", 1, 30, "shear_degrees", False, "{:.1f}", ["shear", "degrees"]),
            ("Probability", 0, 1, "shear_probability", False, "{:.2f}", ["shear"]),
        ], ["shear", "geometry"])

        # Elastic Transform
        self._add_aug_control(parent, "Elastic Transform", "elastic_enabled", [
            ("Alpha", 1, 300, "elastic_alpha", False, "{:.0f}", ["elastic", "alpha"]),
            ("Sigma", 1, 20, "elastic_sigma", False, "{:.1f}", ["elastic", "sigma"]),
            ("Probability", 0, 1, "elastic_probability", False, "{:.2f}", ["elastic"]),
        ], ["elastic", "transform", "geometry"])

    # ── Weather Controls ──────────────────────────────────────────────────

    def _build_weather_controls(self, parent):
        self._add_aug_control(parent, "Rain", "rain_enabled", [
            ("Intensity", 0.1, 1.0, "rain_intensity", False, "{:.2f}", ["rain", "intensity"]),
            ("Probability", 0, 1, "rain_probability", False, "{:.2f}", ["rain"]),
        ], ["rain", "weather"])

        self._add_aug_control(parent, "Snow", "snow_enabled", [
            ("Intensity", 0.1, 0.9, "snow_intensity", False, "{:.2f}", ["snow", "intensity"]),
            ("Probability", 0, 1, "snow_probability", False, "{:.2f}", ["snow"]),
        ], ["snow", "weather"])

        self._add_aug_control(parent, "Fog", "fog_enabled", [
            ("Intensity", 0.1, 1.0, "fog_intensity", False, "{:.2f}", ["fog", "intensity"]),
            ("Probability", 0, 1, "fog_probability", False, "{:.2f}", ["fog"]),
        ], ["fog", "weather"])

        self._add_aug_control(parent, "Sun Flare", "sun_flare_enabled", [
            ("Intensity", 0.1, 1.0, "sun_flare_intensity", False, "{:.2f}", ["sun", "flare"]),
            ("Probability", 0, 1, "sun_flare_probability", False, "{:.2f}", ["sun", "flare"]),
        ], ["sun", "flare", "weather"])

        self._add_aug_control(parent, "Shadow", "shadow_enabled", [
            ("Intensity", 0.1, 1.0, "shadow_intensity", False, "{:.2f}", ["shadow", "intensity"]),
            ("Probability", 0, 1, "shadow_probability", False, "{:.2f}", ["shadow"]),
        ], ["shadow", "weather"])

    # ── Color Controls ────────────────────────────────────────────────────

    def _build_color_controls(self, parent):
        self._add_aug_control(parent, "Brightness", "brightness_enabled", [
            ("Level", 0.5, 1.5, "brightness_factor", False, "{:.2f}", ["brightness"]),
            ("Probability", 0, 1, "brightness_probability", False, "{:.2f}", ["brightness"]),
        ], ["brightness", "color"])

        self._add_aug_control(parent, "Contrast", "contrast_enabled", [
            ("Level", 0.5, 1.5, "contrast_factor", False, "{:.2f}", ["contrast"]),
            ("Probability", 0, 1, "contrast_probability", False, "{:.2f}", ["contrast"]),
        ], ["contrast", "color"])

        self._add_aug_control(parent, "Saturation", "saturation_enabled", [
            ("Level", 0.5, 1.5, "saturation_factor", False, "{:.2f}", ["saturation"]),
            ("Probability", 0, 1, "saturation_probability", False, "{:.2f}", ["saturation"]),
        ], ["saturation", "color"])

        self._add_aug_control(parent, "Hue Shift", "hue_enabled", [
            ("Shift", -0.5, 0.5, "hue_shift", False, "{:.2f}", ["hue"]),
            ("Probability", 0, 1, "hue_probability", False, "{:.2f}", ["hue"]),
        ], ["hue", "color"])

        self._add_aug_control(parent, "Gamma", "gamma_enabled", [
            ("Value", 0.3, 2.0, "gamma_value", False, "{:.2f}", ["gamma"]),
            ("Probability", 0, 1, "gamma_probability", False, "{:.2f}", ["gamma"]),
        ], ["gamma", "color"])

        self._add_aug_control(parent, "CLAHE", "clahe_enabled", [
            ("Clip Limit", 1.0, 16.0, "clahe_clip_limit", False, "{:.1f}", ["clahe"]),
            ("Probability", 0, 1, "clahe_probability", False, "{:.2f}", ["clahe"]),
        ], ["clahe", "color"])

        self._add_aug_control(parent, "Equalization", "equalization_enabled", [
            ("Probability", 0, 1, "equalization_probability", False, "{:.2f}", ["equalization"]),
        ], ["equalization", "color"])

        self._add_aug_control(parent, "RGB Shift", "rgb_shift_enabled", [
            ("R Shift", 0, 50, "rgb_shift_r", False, "{:.0f}", ["rgb", "shift"]),
            ("G Shift", 0, 50, "rgb_shift_g", False, "{:.0f}", ["rgb", "shift"]),
            ("B Shift", 0, 50, "rgb_shift_b", False, "{:.0f}", ["rgb", "shift"]),
            ("Probability", 0, 1, "rgb_shift_probability", False, "{:.2f}", ["rgb", "shift"]),
        ], ["rgb", "shift", "color"])

        self._add_aug_control(parent, "Channel Shuffle", "channel_shuffle_enabled", [
            ("Probability", 0, 1, "channel_shuffle_probability", False, "{:.2f}", ["channel"]),
        ], ["channel", "shuffle", "color"])

        self._add_aug_control(parent, "Grayscale", "grayscale_enabled", [
            ("Probability", 0, 1, "grayscale_probability", False, "{:.2f}", ["grayscale"]),
        ], ["grayscale", "color"])

        self._add_aug_control(parent, "Sepia", "sepia_enabled", [
            ("Probability", 0, 1, "sepia_probability", False, "{:.2f}", ["sepia"]),
        ], ["sepia", "color"])

    # ── Blur Controls ─────────────────────────────────────────────────────

    def _build_blur_controls(self, parent):
        self._add_aug_control(parent, "Gaussian Blur", "gaussian_blur_enabled", [
            ("Kernel", 3, 25, "gaussian_blur_kernel", True, "{}", ["gaussian", "blur", "kernel"]),
            ("Probability", 0, 1, "gaussian_blur_probability", False, "{:.2f}", ["gaussian", "blur"]),
        ], ["gaussian", "blur"])

        self._add_aug_control(parent, "Motion Blur", "motion_blur_enabled", [
            ("Kernel", 3, 25, "motion_blur_kernel", True, "{}", ["motion", "blur", "kernel"]),
            ("Probability", 0, 1, "motion_blur_probability", False, "{:.2f}", ["motion", "blur"]),
        ], ["motion", "blur"])

        self._add_aug_control(parent, "Median Blur", "median_blur_enabled", [
            ("Kernel", 3, 25, "median_blur_kernel", True, "{}", ["median", "blur", "kernel"]),
            ("Probability", 0, 1, "median_blur_probability", False, "{:.2f}", ["median", "blur"]),
        ], ["median", "blur"])

        self._add_aug_control(parent, "Glass Blur", "glass_blur_enabled", [
            ("Sigma", 0.1, 2.0, "glass_blur_sigma", False, "{:.2f}", ["glass", "blur"]),
            ("Probability", 0, 1, "glass_blur_probability", False, "{:.2f}", ["glass", "blur"]),
        ], ["glass", "blur"])

        self._add_aug_control(parent, "Defocus Blur", "defocus_blur_enabled", [
            ("Radius", 3, 15, "defocus_blur_radius", True, "{}", ["defocus", "blur"]),
            ("Probability", 0, 1, "defocus_blur_probability", False, "{:.2f}", ["defocus", "blur"]),
        ], ["defocus", "blur"])

        self._add_aug_control(parent, "Zoom Blur", "zoom_blur_enabled", [
            ("Steps", 1, 10, "zoom_blur_steps", True, "{}", ["zoom", "blur"]),
            ("Probability", 0, 1, "zoom_blur_probability", False, "{:.2f}", ["zoom", "blur"]),
        ], ["zoom", "blur"])

    # ── Noise Controls ────────────────────────────────────────────────────

    def _build_noise_controls(self, parent):
        self._add_aug_control(parent, "Gaussian Noise", "gauss_noise_enabled", [
            ("Amount", 0.0, 0.1, "gauss_noise_amount", False, "{:.3f}", ["gaussian", "noise"]),
            ("Probability", 0, 1, "gauss_noise_probability", False, "{:.2f}", ["gaussian", "noise"]),
        ], ["gaussian", "noise"])

        self._add_aug_control(parent, "ISO Noise", "iso_noise_enabled", [
            ("Intensity", 0.01, 1.0, "iso_noise_intensity", False, "{:.2f}", ["iso", "noise"]),
            ("Probability", 0, 1, "iso_noise_probability", False, "{:.2f}", ["iso", "noise"]),
        ], ["iso", "noise"])

        self._add_aug_control(parent, "Salt & Pepper", "salt_pepper_enabled", [
            ("Amount", 0.001, 0.1, "salt_pepper_amount", False, "{:.3f}", ["salt", "pepper"]),
            ("Probability", 0, 1, "salt_pepper_probability", False, "{:.2f}", ["salt", "pepper"]),
        ], ["salt", "pepper", "noise"])

        self._add_aug_control(parent, "Speckle Noise", "speckle_noise_enabled", [
            ("Intensity", 0.01, 1.0, "speckle_noise_intensity", False, "{:.2f}", ["speckle"]),
            ("Probability", 0, 1, "speckle_noise_probability", False, "{:.2f}", ["speckle"]),
        ], ["speckle", "noise"])

    # ── Compression Controls ──────────────────────────────────────────────

    def _build_compression_controls(self, parent):
        self._add_aug_control(parent, "JPEG Compression", "jpeg_compression_enabled", [
            ("Quality Lower", 1, 100, "jpeg_quality_lower", True, "{}", ["jpeg", "compression"]),
            ("Quality Upper", 1, 100, "jpeg_quality_upper", True, "{}", ["jpeg", "compression"]),
            ("Probability", 0, 1, "jpeg_compression_probability", False, "{:.2f}", ["jpeg"]),
        ], ["jpeg", "compression"])

        self._add_aug_control(parent, "Downscale (Low Res)", "downscale_enabled", [
            ("Min Scale", 0.1, 0.9, "downscale_min", False, "{:.2f}", ["downscale"]),
            ("Max Scale", 0.1, 1.0, "downscale_max", False, "{:.2f}", ["downscale"]),
            ("Probability", 0, 1, "downscale_probability", False, "{:.2f}", ["downscale"]),
        ], ["downscale", "low", "resolution", "compression"])

    # ── Advanced Controls ─────────────────────────────────────────────────

    def _build_advanced_controls(self, parent):
        self._add_aug_control(parent, "Random Erasing", "random_erasing_enabled", [
            ("Ratio", 0.05, 0.8, "random_erasing_ratio", False, "{:.2f}", ["erasing"]),
            ("Probability", 0, 1, "random_erasing_probability", False, "{:.2f}", ["erasing"]),
        ], ["random", "erasing", "advanced"])

        self._add_aug_control(parent, "Grid Dropout", "grid_dropout_enabled", [
            ("Ratio", 0.1, 0.9, "grid_dropout_ratio", False, "{:.2f}", ["grid", "dropout"]),
            ("Probability", 0, 1, "grid_dropout_probability", False, "{:.2f}", ["grid", "dropout"]),
        ], ["grid", "dropout", "advanced"])

        self._add_aug_control(parent, "Coarse Dropout", "coarse_dropout_enabled", [
            ("Intensity", 0.05, 1.0, "coarse_dropout_intensity", False, "{:.2f}", ["coarse", "dropout"]),
            ("Probability", 0, 1, "coarse_dropout_probability", False, "{:.2f}", ["coarse", "dropout"]),
        ], ["coarse", "dropout", "advanced"])

    # ── Batch Processing Controls ─────────────────────────────────────────

    def _build_batch_controls(self, parent):
        # Selection mode
        sel_frame = ctk.CTkFrame(parent, fg_color="transparent")
        sel_frame.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(sel_frame, text="Selection Mode", font=FONT_LABEL).pack(anchor="w")
        self.selection_mode_var = ctk.StringVar(value="All Images")
        selection_modes = ["All Images", "Images With Labels", "Images Without Labels", "Random Images", "Filter by Filename", "Recently Added"]
        self.selection_mode_menu = ctk.CTkOptionMenu(sel_frame, values=selection_modes, variable=self.selection_mode_var, width=220)
        self.selection_mode_menu.pack(fill="x", pady=(4, 0))

        # Copies
        self.variants_slider = SliderRow(parent, "Copies per Image", 1, 20, self.variants_per_image, self._set_variants_from_batch, is_int=True, search_tags=["batch", "copies", "variants"])
        self.variants_slider.pack(fill="x", pady=(0, 6))
        self._register_slider("variants_per_image", self.variants_slider)

        # Random percentage
        self.random_pct_slider = SliderRow(parent, "Random %", 10, 100, 100, lambda v: None, is_int=True, search_tags=["random", "percentage"])
        self.random_pct_slider.pack(fill="x", pady=(0, 6))

        # Save mode
        save_frame = ctk.CTkFrame(parent, fg_color="transparent")
        save_frame.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(save_frame, text="Save Mode", font=FONT_LABEL).pack(anchor="w")
        self.save_mode_var = ctk.StringVar(value="Save in New Folder")
        save_modes = ["Overwrite Original", "Save as New", "Save in New Folder", "Save Next to Original"]
        self.save_mode_menu = ctk.CTkOptionMenu(save_frame, values=save_modes, variable=self.save_mode_var, width=220)
        self.save_mode_menu.pack(fill="x", pady=(4, 0))

        # Naming mode
        name_frame = ctk.CTkFrame(parent, fg_color="transparent")
        name_frame.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(name_frame, text="Naming", font=FONT_LABEL).pack(anchor="w")
        self.naming_mode_var = ctk.StringVar(value="Suffix")
        naming_modes = ["Suffix", "Prefix", "Timestamp", "UUID", "Keep Original"]
        self.naming_mode_menu = ctk.CTkOptionMenu(name_frame, values=naming_modes, variable=self.naming_mode_var, width=220)
        self.naming_mode_menu.pack(fill="x", pady=(4, 0))

        # Label handling
        label_frame = ctk.CTkFrame(parent, fg_color="transparent")
        label_frame.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(label_frame, text="Label Handling", font=FONT_LABEL).pack(anchor="w")
        self.label_mode_var = ctk.StringVar(value="Copy Labels")
        label_modes = ["Copy Labels", "Transform Labels", "Verify Labels", "Skip Labels"]
        self.label_mode_menu = ctk.CTkOptionMenu(label_frame, values=label_modes, variable=self.label_mode_var, width=220)
        self.label_mode_menu.pack(fill="x", pady=(4, 0))

        # Max output
        self.max_output_slider = SliderRow(parent, "Max Output Images", 100, 50000, 10000, lambda v: None, is_int=True, search_tags=["max", "output", "limit"])
        self.max_output_slider.pack(fill="x", pady=(0, 6))

        # Checkboxes
        self.skip_existing_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(parent, text="Skip if already augmented", variable=self.skip_existing_var, font=FONT_SMALL).pack(anchor="w", pady=2)
        self.create_backup_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(parent, text="Create backup before processing", variable=self.create_backup_var, font=FONT_SMALL).pack(anchor="w", pady=2)

        # Run / Pause / Cancel buttons
        btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(8, 6))
        self.run_batch_button = ctk.CTkButton(btn_frame, text="▶  Run Batch", fg_color="#2e7d32", hover_color="#1b5e20", command=self.on_run_batch)
        self.run_batch_button.pack(side="left", padx=(0, 6), expand=True, fill="x")
        self.pause_batch_button = ctk.CTkButton(btn_frame, text="⏸", width=40, fg_color="#d97706", hover_color="#b45309", command=self._toggle_pause_batch)
        self.pause_batch_button.pack(side="left", padx=(0, 4))
        self.cancel_batch_button = ctk.CTkButton(btn_frame, text="⏹", width=40, fg_color="#dc2626", hover_color="#b91c1c", command=self._cancel_batch)
        self.cancel_batch_button.pack(side="left")

        # Progress
        self.progress_bar = ctk.CTkProgressBar(parent)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", pady=(0, 4))
        self.batch_status_label = ctk.CTkLabel(parent, text="", font=FONT_SMALL, text_color="#9bb3cf", wraplength=320, justify="left")
        self.batch_status_label.pack(anchor="w", pady=(0, 6))

    # ── Dataset Tools Card ────────────────────────────────────────────────

    def _build_dataset_tools_card(self, parent):
        card = self._section_card(parent, "Dataset Tools", "Clean, balance, split, merge, export, check labels")

        # Export row
        exp_frame = ctk.CTkFrame(card, fg_color="transparent")
        exp_frame.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(exp_frame, text="Export As:", font=FONT_LABEL).pack(side="left")
        for fmt in ["YOLO", "COCO", "VOC", "ZIP"]:
            ctk.CTkButton(exp_frame, text=fmt, width=55, fg_color="#065f46", hover_color="#064e3b",
                          command=lambda f=fmt: self._export_dataset(f)).pack(side="left", padx=(4, 0))

        # Cleaning
        clean_frame = ctk.CTkFrame(card, fg_color="transparent")
        clean_frame.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(clean_frame, text="Clean:", font=FONT_LABEL).pack(side="left")
        ctk.CTkButton(clean_frame, text="Duplicates", width=80, fg_color="#7c3aed", hover_color="#6d28d9",
                      command=self._find_duplicates).pack(side="left", padx=(4, 0))
        ctk.CTkButton(clean_frame, text="Blurry", width=60, fg_color="#7c3aed", hover_color="#6d28d9",
                      command=self._find_blurry).pack(side="left", padx=(4, 0))
        ctk.CTkButton(clean_frame, text="Corrupted", width=75, fg_color="#7c3aed", hover_color="#6d28d9",
                      command=self._find_corrupted).pack(side="left", padx=(4, 0))

        # Split
        split_frame = ctk.CTkFrame(card, fg_color="transparent")
        split_frame.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(split_frame, text="Split:", font=FONT_LABEL).pack(side="left")
        self.train_ratio_var = ctk.StringVar(value="0.7")
        ctk.CTkEntry(split_frame, textvariable=self.train_ratio_var, width=40, placeholder_text="Train").pack(side="left", padx=(4, 0))
        self.val_ratio_var = ctk.StringVar(value="0.2")
        ctk.CTkEntry(split_frame, textvariable=self.val_ratio_var, width=40, placeholder_text="Val").pack(side="left", padx=(4, 0))
        self.test_ratio_var = ctk.StringVar(value="0.1")
        ctk.CTkEntry(split_frame, textvariable=self.test_ratio_var, width=40, placeholder_text="Test").pack(side="left", padx=(4, 0))
        ctk.CTkButton(split_frame, text="Split", width=50, fg_color="#0d9488", hover_color="#0f766e",
                      command=self._split_dataset).pack(side="left", padx=(4, 0))

        # Stats & Label Check
        misc_frame = ctk.CTkFrame(card, fg_color="transparent")
        misc_frame.pack(fill="x", pady=(0, 6))
        ctk.CTkButton(misc_frame, text="📊 Dataset Stats", width=120, fg_color="#0369a1", hover_color="#075985",
                      command=self._show_dataset_stats).pack(side="left", padx=(0, 4))
        ctk.CTkButton(misc_frame, text="🔍 Check Labels", width=120, fg_color="#0369a1", hover_color="#075985",
                      command=self._check_labels).pack(side="left", padx=(0, 4))
        ctk.CTkButton(misc_frame, text="⚖ Balance", width=80, fg_color="#0369a1", hover_color="#075985",
                      command=self._show_balance_info).pack(side="left")

        # Merge
        ctk.CTkButton(card, text="🔗 Merge Datasets", fg_color="#92400e", hover_color="#78350f",
                      command=self._merge_datasets).pack(fill="x", pady=(0, 4))

    # ── Image Utilities Card ─────────────────────────────────────────────

    def _build_image_utils_card(self, parent):
        card = self._section_card(parent, "Image Utilities", "Rename, resize, convert, crop, tiles")

        row1 = ctk.CTkFrame(card, fg_color="transparent")
        row1.pack(fill="x", pady=(0, 4))
        ctk.CTkButton(row1, text="Rename Files", width=100, command=self._rename_files).pack(side="left", padx=(0, 4))
        ctk.CTkButton(row1, text="Resize All", width=90, command=self._resize_all).pack(side="left", padx=(0, 4))
        ctk.CTkButton(row1, text="Remove EXIF", width=95, command=self._remove_exif).pack(side="left")

        row2 = ctk.CTkFrame(card, fg_color="transparent")
        row2.pack(fill="x", pady=(0, 4))
        ctk.CTkButton(row2, text="JPG↔PNG", width=80, command=lambda: self._convert_format(".jpg", ".png")).pack(side="left", padx=(0, 4))
        ctk.CTkButton(row2, text="PNG↔WEBP", width=85, command=lambda: self._convert_format(".png", ".webp")).pack(side="left", padx=(0, 4))
        ctk.CTkButton(row2, text="→ Gray", width=65, command=self._to_grayscale).pack(side="left", padx=(0, 4))
        ctk.CTkButton(row2, text="→ RGB", width=60, command=self._to_rgb).pack(side="left")

        row3 = ctk.CTkFrame(card, fg_color="transparent")
        row3.pack(fill="x", pady=(0, 4))
        ctk.CTkButton(row3, text="Split Tiles", width=90, command=self._split_tiles).pack(side="left", padx=(0, 4))
        ctk.CTkButton(row3, text="Merge Tiles", width=90, command=self._merge_tiles).pack(side="left")

    # ── Output Sidebar (Right) ───────────────────────────────────────────

    def _build_output_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=340, corner_radius=16, fg_color="#121a2b")
        sidebar.grid(row=0, column=2, sticky="nsew", padx=(0, PAD), pady=PAD)
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(sidebar, text="📝 Output Settings", font=FONT_SECTION).pack(anchor="w", padx=PAD, pady=(PAD, 4))
        ctk.CTkLabel(sidebar, text="Output folder, format, quality, and copies", font=FONT_SMALL, text_color="#9bb3cf").pack(anchor="w", padx=PAD, pady=(0, PAD))

        # Output folder
        self.output_dir_var = ctk.StringVar(value="")
        folder_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        folder_frame.pack(fill="x", padx=PAD, pady=(0, 8))
        ctk.CTkLabel(folder_frame, text="Output Folder", font=FONT_LABEL).pack(anchor="w")
        self.output_dir_entry = ctk.CTkEntry(folder_frame, textvariable=self.output_dir_var)
        self.output_dir_entry.pack(fill="x", pady=(4, 6))
        ctk.CTkButton(folder_frame, text="Browse", command=self._browse_output_dir).pack(fill="x")

        # Output format
        self.output_format_var = ctk.StringVar(value=".png")
        format_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        format_frame.pack(fill="x", padx=PAD, pady=(6, 8))
        ctk.CTkLabel(format_frame, text="Output Format", font=FONT_LABEL).pack(anchor="w")
        self.output_format_menu = ctk.CTkOptionMenu(format_frame, values=[".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"], variable=self.output_format_var)
        self.output_format_menu.pack(fill="x", pady=(4, 0))

        # Save quality
        self.jpeg_quality_slider = SliderRow(sidebar, "JPEG Quality", 1, 100, 95, lambda v: None, is_int=True, search_tags=["jpeg", "quality"])
        self.jpeg_quality_slider.pack(fill="x", padx=PAD, pady=(6, 4))

        self.png_compression_slider = SliderRow(sidebar, "PNG Compression", 0, 9, 3, lambda v: None, is_int=True, search_tags=["png", "compression"])
        self.png_compression_slider.pack(fill="x", padx=PAD, pady=(0, 4))

        self.webp_quality_slider = SliderRow(sidebar, "WEBP Quality", 1, 100, 90, lambda v: None, is_int=True, search_tags=["webp", "quality"])
        self.webp_quality_slider.pack(fill="x", padx=PAD, pady=(0, 8))

        # Naming pattern
        self.output_name_var = ctk.StringVar(value="{name}_aug{index}")
        name_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        name_frame.pack(fill="x", padx=PAD, pady=(6, 8))
        ctk.CTkLabel(name_frame, text="Naming Pattern", font=FONT_LABEL).pack(anchor="w")
        ctk.CTkEntry(name_frame, textvariable=self.output_name_var).pack(fill="x", pady=(4, 0))

        # Copies slider
        self.copy_count_slider = SliderRow(sidebar, "Copies to Generate", 1, 20, self.variants_per_image, self._set_variants_from_output, is_int=True, search_tags=["copies", "output", "batch"])
        self.copy_count_slider.pack(fill="x", padx=PAD, pady=(6, 8))

        # Metadata options
        meta_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        meta_frame.pack(fill="x", padx=PAD, pady=(0, 8))
        self.keep_exif_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(meta_frame, text="Keep EXIF metadata", variable=self.keep_exif_var, font=FONT_SMALL).pack(anchor="w")

        # Generate button
        ctk.CTkButton(sidebar, text="Generate Copies", command=self.on_generate_copies, fg_color="#2563eb", hover_color="#1d4ed8").pack(fill="x", padx=PAD, pady=(8, 0))

    # ── Preview Area (Center) ────────────────────────────────────────────

    def _build_preview_area(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=PAD, pady=PAD)
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure((0, 1), weight=1)

        original_frame = ctk.CTkFrame(main, corner_radius=16, fg_color="#121a2b")
        original_frame.grid(row=0, column=0, sticky="nsew", padx=(0, PAD // 2))
        ctk.CTkLabel(original_frame, text="Original", font=FONT_SECTION).pack(pady=(PAD, 4))
        self.original_image_label = ctk.CTkLabel(original_frame, text="No image loaded")
        self.original_image_label.pack(expand=True, fill="both", padx=PAD, pady=(0, PAD))

        augmented_frame = ctk.CTkFrame(main, corner_radius=16, fg_color="#121a2b")
        augmented_frame.grid(row=0, column=1, sticky="nsew", padx=(PAD // 2, 0))
        ctk.CTkLabel(augmented_frame, text="Augmented Preview", font=FONT_SECTION).pack(pady=(PAD, 4))
        self.augmented_image_label = ctk.CTkLabel(augmented_frame, text="No image loaded")
        self.augmented_image_label.pack(expand=True, fill="both", padx=PAD, pady=(0, PAD))

        self.status_bar = ctk.CTkLabel(main, text="Ready.", font=FONT_SMALL, text_color="#9bb3cf", anchor="w")
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(PAD, 0))

    # ── Helper: Add augmentation control block ────────────────────────────

    def _add_aug_control(self, parent, label: str, enabled_attr: str,
                         sliders: list, toggle_tags: list):
        """
        Helper to add a toggle + multiple sliders for an augmentation.
        sliders: list of (label, from_, to, attr, is_int, fmt, tags)
        """
        toggle = ToggleRow(parent, label, self._toggle(enabled_attr), search_tags=toggle_tags)
        toggle.pack(fill="x", pady=(6, 4))
        self._register_toggle(enabled_attr, toggle)

        for s_label, s_from, s_to, s_attr, s_int, s_fmt, s_tags in sliders:
            initial = getattr(self.params, s_attr, s_from)
            slider = SliderRow(parent, s_label, s_from, s_to, initial, self._set(s_attr), is_int=s_int, fmt=s_fmt, search_tags=s_tags)
            slider.pack(fill="x")
            self._register_slider(s_attr, slider)

    # ── Section Card Builders ─────────────────────────────────────────────

    def _build_section_card(self, parent, title: str, subtitle: str, builder: Callable[[ctk.CTkFrame], None]):
        card = self._section_card(parent, title, subtitle)
        builder(card)
        return card

    def _section_card(self, parent, title: str, subtitle: str):
        card = ctk.CTkFrame(parent, fg_color="#162033", corner_radius=14)
        card.pack(fill="x", pady=(0, PAD))
        ctk.CTkLabel(card, text=title, font=FONT_SECTION).pack(anchor="w", padx=PAD, pady=(PAD, 2))
        ctk.CTkLabel(card, text=subtitle, font=FONT_SMALL, text_color="#9bb3cf").pack(anchor="w", padx=PAD, pady=(0, PAD))
        return card

    # ── Registration ──────────────────────────────────────────────────────

    def _register_slider(self, attr_name: str, row: SliderRow):
        self.slider_controls[attr_name] = row
        self.searchable_rows.append(row)

    def _register_toggle(self, attr_name: str, row: ToggleRow):
        self.toggle_controls[attr_name] = row
        self.searchable_rows.append(row)

    # ── Search ────────────────────────────────────────────────────────────

    def _apply_search(self):
        query = self.search_var.get().strip().lower()
        for row in self.searchable_rows:
            if row.matches(query):
                row.pack(fill="x", pady=(0, 4))
            else:
                row.pack_forget()

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _set(self, attr_name: str) -> Callable[[float], None]:
        def _callback(value):
            setattr(self.params, attr_name, value)
            self.update_preview()
        return _callback

    def _toggle(self, attr_name: str) -> Callable[[bool], None]:
        def _callback(value: bool):
            setattr(self.params, attr_name, value)
            self.update_preview()
        return _callback

    def _set_variants_from_batch(self, value: float):
        self._set_variants(value, target="output")

    def _set_variants_from_output(self, value: float):
        self._set_variants(value, target="batch")

    def _set_variants(self, value: float, target: str = "both"):
        self.variants_per_image = int(value)
        if target in {"both", "output"} and hasattr(self, "copy_count_slider"):
            self.copy_count_slider.set_value(float(self.variants_per_image))
        if target in {"both", "batch"} and hasattr(self, "variants_slider"):
            self.variants_slider.set_value(float(self.variants_per_image))

    def _on_seed_toggle(self):
        self.params.lock_seed = bool(self.lock_seed_var.get())
        self.update_preview()

    # ── Undo / Redo ───────────────────────────────────────────────────────

    def _push_undo(self):
        state = copy.deepcopy(self.params)
        self._undo_stack.append(state)
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_undo_label()

    def _undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(copy.deepcopy(self.params))
        self.params = self._undo_stack.pop()
        self.engine = AugmentationEngine(self.params)
        self._sync_controls_from_params()
        self.update_preview()
        self._update_undo_label()

    def _redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(copy.deepcopy(self.params))
        self.params = self._redo_stack.pop()
        self.engine = AugmentationEngine(self.params)
        self._sync_controls_from_params()
        self.update_preview()
        self._update_undo_label()

    def _update_undo_label(self):
        if hasattr(self, "undo_label"):
            text = f"Undo: {len(self._undo_stack)}  Redo: {len(self._redo_stack)}"
            self.undo_label.configure(text=text)

    # ── Presets ───────────────────────────────────────────────────────────

    def _reset_defaults(self):
        self._push_undo()
        self.params = AugmentationParams()
        self.engine = AugmentationEngine(self.params)
        self._sync_controls_from_params()
        self.update_preview()

    def _apply_preset(self, preset_name: str):
        self._push_undo()
        defaults = {
            "Light": {"rotation_enabled": False, "flip_h_enabled": False, "flip_v_enabled": False, "scale_enabled": False, "brightness_enabled": True, "brightness_factor": 1.05, "contrast_enabled": False, "saturation_enabled": False, "hue_enabled": False, "gaussian_blur_enabled": False, "motion_blur_enabled": False, "gauss_noise_enabled": False, "grid_dropout_enabled": False, "coarse_dropout_enabled": False, "rotation_probability": 0.6, "brightness_probability": 0.5, "random_order_enabled": True, "seed_value": 1234, "lock_seed": False},
            "Balanced": {"rotation_enabled": True, "rotation_degrees": 15, "rotation_probability": 0.75, "flip_h_enabled": True, "flip_h_probability": 0.5, "flip_v_enabled": False, "scale_enabled": True, "scale_factor": 1.0, "scale_probability": 0.5, "brightness_enabled": True, "brightness_factor": 1.05, "brightness_probability": 0.5, "contrast_enabled": True, "contrast_factor": 1.02, "contrast_probability": 0.5, "saturation_enabled": True, "saturation_factor": 1.02, "saturation_probability": 0.5, "gaussian_blur_enabled": False, "motion_blur_enabled": False, "gauss_noise_enabled": True, "gauss_noise_amount": 0.01, "gauss_noise_probability": 0.4, "random_order_enabled": True, "seed_value": 1234, "lock_seed": False},
            "Heavy": {"rotation_enabled": True, "rotation_degrees": 25, "rotation_probability": 0.85, "flip_h_enabled": True, "flip_h_probability": 0.7, "flip_v_enabled": True, "flip_v_probability": 0.4, "scale_enabled": True, "scale_factor": 0.92, "scale_probability": 0.7, "brightness_enabled": True, "brightness_factor": 1.12, "brightness_probability": 0.7, "contrast_enabled": True, "contrast_factor": 1.08, "contrast_probability": 0.7, "saturation_enabled": True, "saturation_factor": 1.08, "saturation_probability": 0.7, "gaussian_blur_enabled": True, "gaussian_blur_kernel": 5, "gaussian_blur_probability": 0.6, "motion_blur_enabled": True, "motion_blur_kernel": 7, "motion_blur_probability": 0.5, "gauss_noise_enabled": True, "gauss_noise_amount": 0.03, "gauss_noise_probability": 0.6, "grid_dropout_enabled": True, "grid_dropout_ratio": 0.35, "grid_dropout_probability": 0.5, "coarse_dropout_enabled": True, "coarse_dropout_intensity": 0.4, "coarse_dropout_probability": 0.5, "random_order_enabled": True, "seed_value": 42, "lock_seed": False},
        }
        preset = defaults.get(preset_name, {})
        for key, value in preset.items():
            setattr(self.params, key, value)
        self.engine = AugmentationEngine(self.params)
        self.seed_var.set(str(self.params.seed_value))
        self.lock_seed_var.set(self.params.lock_seed)
        self._sync_controls_from_params()
        self.update_preview()

    def _on_preset_selected(self, name: str):
        if name == "Select preset..." or name == "No presets":
            return
        try:
            self._push_undo()
            self.pipeline_manager.apply_preset(name, self.params)
            self.engine = AugmentationEngine(self.params)
            self._sync_controls_from_params()
            self.update_preview()
            self.status_bar.configure(text=f"Applied preset: {name}")
        except Exception as exc:
            messagebox.showerror("Preset Error", str(exc))

    def _save_custom_preset(self):
        from tkinter import simpledialog
        name = simpledialog.askstring("Save Preset", "Enter preset name:", parent=self)
        if not name:
            return
        try:
            path = self.pipeline_manager.save_preset(name, self.params)
            self._refresh_preset_menu()
            self.status_bar.configure(text=f"Saved preset: {name}")
            messagebox.showinfo("Preset Saved", f"Saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    def _refresh_preset_menu(self):
        names = self.pipeline_manager.get_all_preset_names()
        if hasattr(self, "preset_menu"):
            self.preset_menu.configure(values=names if names else ["No presets"])

    # ── Sync controls ─────────────────────────────────────────────────────

    def _sync_controls_from_params(self):
        for attr_name, row in self.slider_controls.items():
            if attr_name == "variants_per_image":
                continue
            if hasattr(self.params, attr_name):
                row.set_value(getattr(self.params, attr_name))
        for attr_name, row in self.toggle_controls.items():
            if hasattr(self.params, attr_name):
                row.set_value(getattr(self.params, attr_name))
        self.seed_var.set(str(self.params.seed_value))
        self.lock_seed_var.set(self.params.lock_seed)
        if hasattr(self, "random_order_toggle"):
            self.random_order_toggle.set_value(self.params.random_order_enabled)
        if hasattr(self, "copy_count_slider"):
            self.copy_count_slider.set_value(float(self.variants_per_image))

    # ── Browse ────────────────────────────────────────────────────────────

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir_var.set(path)

    # ── Generate Copies ───────────────────────────────────────────────────

    def on_generate_copies(self):
        if self.current_image_np is None or self.current_image_path is None:
            messagebox.showwarning("No image selected", "Please select an image first before generating copies.")
            return
        output_dir = Path(self.output_dir_var.get()).expanduser() if self.output_dir_var.get() else self.current_image_path.parent / "augmented_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        count = max(1, self.variants_per_image)
        suffix = self.output_format_var.get().lstrip(".")
        quality = int(self.jpeg_quality_slider.slider.get())
        png_comp = int(self.png_compression_slider.slider.get())
        webp_q = int(self.webp_quality_slider.slider.get())
        for i in range(count):
            augmented = self.engine.apply(self.current_image_np, deterministic=False)
            name = self.output_name_var.get().format(name=self.current_image_path.stem, index=i + 1)
            out_path = output_dir / f"{name}.{suffix}"
            save_image_rgb(augmented, out_path, quality=quality, png_compression=png_comp, webp_quality=webp_q)
        self.status_bar.configure(text=f"Saved {count} copies to {output_dir}")
        messagebox.showinfo("Copies generated", f"Saved {count} augmented copies to:\n{output_dir}")

    # ── Select Image / Folder ─────────────────────────────────────────────

    def on_select_image(self):
        path_str = filedialog.askopenfilename(title="Select an image", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp")])
        if not path_str:
            return
        path = Path(path_str)
        try:
            self.current_image_np = load_image_rgb(path)
            self.current_image_path = path
            self.selected_folder = None
            self.input_status_label.configure(text=f"Image: {path.name}")
            self.status_bar.configure(text=f"Loaded {path.name}")
            self._show_image(self.original_image_label, self.current_image_np, "_original_ctk_image")
            self.update_preview()
        except Exception as exc:
            messagebox.showerror("Error loading image", str(exc))

    def on_select_folder(self):
        folder_str = filedialog.askdirectory(title="Select a folder of images")
        if not folder_str:
            return
        folder = Path(folder_str)
        try:
            images = find_images_in_folder(folder)
            if not images:
                messagebox.showwarning("No images found", f"No supported image files were found in:\n{folder}")
                return
            self.selected_folder = folder
            self.current_image_path = None
            self.input_status_label.configure(text=f"Folder: {folder.name}  ({len(images)} images found)")
            self.status_bar.configure(text=f"Batch folder selected: {folder}")
            self.current_image_np = load_image_rgb(images[0])
            self._show_image(self.original_image_label, self.current_image_np, "_original_ctk_image")
            self.update_preview()
        except Exception as exc:
            messagebox.showerror("Error reading folder", str(exc))

    # ── Dataset Import ────────────────────────────────────────────────────

    def _import_yolo(self):
        folder = filedialog.askdirectory(title="Select YOLO dataset root (with images/ and labels/)")
        if not folder:
            return
        folder = Path(folder)
        images_dir = folder / "images"
        labels_dir = folder / "labels"
        if not images_dir.exists():
            images_dir = folder
        if not labels_dir.exists():
            labels_dir = folder
        try:
            annotations, class_names = load_yolo_dataset(images_dir, labels_dir)
            self.selected_folder = images_dir
            images = find_images_in_folder(images_dir)
            if images:
                self.current_image_np = load_image_rgb(images[0])
                self._show_image(self.original_image_label, self.current_image_np, "_original_ctk_image")
                self.update_preview()
            self.input_status_label.configure(text=f"YOLO: {len(annotations)} images, {len(class_names)} classes")
            self.status_bar.configure(text=f"Loaded YOLO dataset: {folder.name}")
        except Exception as exc:
            messagebox.showerror("Import Error", str(exc))

    def _import_coco(self):
        json_path = filedialog.askopenfilename(title="Select COCO annotations JSON", filetypes=[("JSON", "*.json")])
        if not json_path:
            return
        json_path = Path(json_path)
        images_dir = filedialog.askdirectory(title="Select COCO images folder")
        if not images_dir:
            return
        images_dir = Path(images_dir)
        try:
            annotations, class_names = load_coco_dataset(json_path, images_dir)
            self.selected_folder = images_dir
            images = find_images_in_folder(images_dir)
            if images:
                self.current_image_np = load_image_rgb(images[0])
                self._show_image(self.original_image_label, self.current_image_np, "_original_ctk_image")
                self.update_preview()
            self.input_status_label.configure(text=f"COCO: {len(annotations)} images, {len(class_names)} classes")
            self.status_bar.configure(text=f"Loaded COCO dataset")
        except Exception as exc:
            messagebox.showerror("Import Error", str(exc))

    def _import_voc(self):
        folder = filedialog.askdirectory(title="Select Pascal VOC dataset root")
        if not folder:
            return
        folder = Path(folder)
        ann_dir = folder / "Annotations"
        img_dir = folder / "JPEGImages"
        if not ann_dir.exists():
            ann_dir = folder
        if not img_dir.exists():
            img_dir = folder
        try:
            annotations, class_names = load_voc_dataset(ann_dir, img_dir)
            self.selected_folder = img_dir
            images = find_images_in_folder(img_dir)
            if images:
                self.current_image_np = load_image_rgb(images[0])
                self._show_image(self.original_image_label, self.current_image_np, "_original_ctk_image")
                self.update_preview()
            self.input_status_label.configure(text=f"VOC: {len(annotations)} images, {len(class_names)} classes")
            self.status_bar.configure(text=f"Loaded VOC dataset")
        except Exception as exc:
            messagebox.showerror("Import Error", str(exc))

    def _import_zip(self):
        zip_path = filedialog.askopenfilename(title="Select ZIP file", filetypes=[("ZIP", "*.zip")])
        if not zip_path:
            return
        zip_path = Path(zip_path)
        extract_to = filedialog.askdirectory(title="Select extraction folder")
        if not extract_to:
            return
        try:
            result = import_zip(zip_path, Path(extract_to))
            self.selected_folder = result
            self.status_bar.configure(text=f"Extracted ZIP to: {result}")
            self.input_status_label.configure(text=f"ZIP extracted to {result.name}")
        except Exception as exc:
            messagebox.showerror("Import Error", str(exc))

    # ── Dataset Export ────────────────────────────────────────────────────

    def _export_dataset(self, fmt: str):
        if self.selected_folder is None:
            messagebox.showwarning("No folder", "Please select a folder or dataset first.")
            return
        output = filedialog.askdirectory(title=f"Select output folder for {fmt} export")
        if not output:
            return
        output = Path(output)
        try:
            images_dir = self.selected_folder
            labels_dir = self.selected_folder.parent / "labels" if (self.selected_folder.parent / "labels").exists() else self.selected_folder
            annotations, class_names = load_yolo_dataset(images_dir, labels_dir)

            if fmt == "YOLO":
                export_yolo_dataset(annotations, class_names, output)
            elif fmt == "COCO":
                export_coco_dataset(annotations, class_names, output)
            elif fmt == "VOC":
                export_voc_dataset(annotations, class_names, output)
            elif fmt == "ZIP":
                export_zip(self.selected_folder, output / "dataset.zip")

            self.status_bar.configure(text=f"Exported {fmt} to {output}")
            messagebox.showinfo("Export Complete", f"Exported {len(annotations)} images as {fmt} to:\n{output}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    # ── Dataset Cleaning ──────────────────────────────────────────────────

    def _find_duplicates(self):
        if self.selected_folder is None:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return
        try:
            images = find_images_in_folder(self.selected_folder)
            duplicates = find_duplicate_images(images)
            total = sum(len(g) - 1 for g in duplicates)
            if duplicates:
                messagebox.showinfo("Duplicates Found", f"Found {total} duplicate images in {len(duplicates)} groups.")
            else:
                messagebox.showinfo("No Duplicates", "No duplicate images found.")
            self.status_bar.configure(text=f"Duplicates: {total}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _find_blurry(self):
        if self.selected_folder is None:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return
        try:
            images = find_images_in_folder(self.selected_folder)
            blurry = find_blurry_images(images)
            messagebox.showinfo("Blurry Images", f"Found {len(blurry)} blurry images (Laplacian variance < 100).")
            self.status_bar.configure(text=f"Blurry images: {len(blurry)}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _find_corrupted(self):
        if self.selected_folder is None:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return
        try:
            images = find_images_in_folder(self.selected_folder)
            corrupted = find_corrupted_images(images)
            messagebox.showinfo("Corrupted Images", f"Found {len(corrupted)} corrupted/unreadable images.")
            self.status_bar.configure(text=f"Corrupted: {len(corrupted)}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ── Dataset Split ─────────────────────────────────────────────────────

    def _split_dataset(self):
        if self.selected_folder is None:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return
        try:
            images = find_images_in_folder(self.selected_folder)
            train_r = float(self.train_ratio_var.get())
            val_r = float(self.val_ratio_var.get())
            test_r = float(self.test_ratio_var.get())
            train, val, test = split_dataset(images, train_r, val_r, test_r)
            output = filedialog.askdirectory(title="Select output folder for split")
            if not output:
                return
            labels_dir = self.selected_folder.parent / "labels" if (self.selected_folder.parent / "labels").exists() else None
            export_split(train, val, test, Path(output), labels_dir)
            messagebox.showinfo("Split Complete", f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
            self.status_bar.configure(text=f"Split: {len(train)}/{len(val)}/{len(test)}")
        except Exception as exc:
            messagebox.showerror("Split Error", str(exc))

    # ── Dataset Stats ─────────────────────────────────────────────────────

    def _show_dataset_stats(self):
        if self.selected_folder is None:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return
        try:
            images_dir = self.selected_folder
            labels_dir = self.selected_folder.parent / "labels" if (self.selected_folder.parent / "labels").exists() else self.selected_folder
            annotations, class_names = load_yolo_dataset(images_dir, labels_dir)
            info = compute_dataset_info(annotations, class_names)
            size_mb = info.dataset_size_bytes / (1024 * 1024)
            stats_text = (
                f"Images: {info.total_images}\n"
                f"Classes: {info.total_classes}\n"
                f"Bounding Boxes: {info.total_bboxes}\n"
                f"Avg Size: {info.avg_width:.0f}x{info.avg_height:.0f}\n"
                f"Size Range: {info.min_width}x{info.min_height} to {info.max_width}x{info.max_height}\n"
                f"Dataset Size: {size_mb:.1f} MB\n"
                f"Empty Labels: {info.empty_labels}\n"
                f"Small/Med/Large: {info.bbox_size_distribution.get('small', 0)}/{info.bbox_size_distribution.get('medium', 0)}/{info.bbox_size_distribution.get('large', 0)}\n"
                f"\nClass Distribution:\n"
            )
            for cls, count in sorted(info.class_distribution.items()):
                stats_text += f"  {cls}: {count}\n"
            messagebox.showinfo("Dataset Statistics", stats_text)
            self.status_bar.configure(text=f"Stats: {info.total_images} images, {info.total_classes} classes")
        except Exception as exc:
            messagebox.showerror("Stats Error", str(exc))

    # ── Label Check ───────────────────────────────────────────────────────

    def _check_labels(self):
        if self.selected_folder is None:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return
        try:
            images_dir = self.selected_folder
            labels_dir = self.selected_folder.parent / "labels" if (self.selected_folder.parent / "labels").exists() else self.selected_folder
            annotations, class_names = load_yolo_dataset(images_dir, labels_dir)
            issues = check_labels(annotations, len(class_names))
            if issues:
                issue_summary = {}
                for issue in issues:
                    issue_summary[issue.issue_type] = issue_summary.get(issue.issue_type, 0) + 1
                text = "Label Issues Found:\n\n"
                for itype, count in sorted(issue_summary.items()):
                    text += f"  {itype}: {count}\n"
                messagebox.showwarning("Label Issues", text)
            else:
                messagebox.showinfo("Labels OK", "No label issues detected!")
            self.status_bar.configure(text=f"Label check: {len(issues)} issues")
        except Exception as exc:
            messagebox.showerror("Check Error", str(exc))

    # ── Balance Info ──────────────────────────────────────────────────────

    def _show_balance_info(self):
        if self.selected_folder is None:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return
        try:
            images_dir = self.selected_folder
            labels_dir = self.selected_folder.parent / "labels" if (self.selected_folder.parent / "labels").exists() else self.selected_folder
            annotations, class_names = load_yolo_dataset(images_dir, labels_dir)
            dist = get_class_distribution(annotations, class_names)
            plan = compute_balance_plan(dist)
            text = "Class Balance:\n\n"
            for cls in class_names:
                count = dist.get(cls, 0)
                needed = plan.get(cls, 0)
                text += f"  {cls}: {count} images"
                if needed > 0:
                    text += f"  (need +{needed})"
                text += "\n"
            messagebox.showinfo("Class Balance", text)
        except Exception as exc:
            messagebox.showerror("Balance Error", str(exc))

    # ── Merge Datasets ────────────────────────────────────────────────────

    def _merge_datasets(self):
        folders = []
        while True:
            folder = filedialog.askdirectory(title=f"Select dataset {len(folders) + 1} (Cancel to finish)")
            if not folder:
                break
            folders.append(Path(folder))
        if len(folders) < 2:
            messagebox.showwarning("Not enough", "Need at least 2 datasets to merge.")
            return
        output = filedialog.askdirectory(title="Select output folder for merged dataset")
        if not output:
            return
        try:
            datasets = []
            for f in folders:
                img_dir = f / "images" if (f / "images").exists() else f
                lbl_dir = f / "labels" if (f / "labels").exists() else f
                datasets.append((img_dir, lbl_dir))
            total, renamed = merge_datasets(datasets, Path(output))
            messagebox.showinfo("Merge Complete", f"Merged {total} images ({renamed} renamed) to:\n{output}")
            self.status_bar.configure(text=f"Merged {total} images")
        except Exception as exc:
            messagebox.showerror("Merge Error", str(exc))

    # ── Image Utilities ───────────────────────────────────────────────────

    def _rename_files(self):
        if self.selected_folder is None:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return
        from tkinter import simpledialog
        pattern = simpledialog.askstring("Rename Pattern", "Enter pattern (use {index}, {original}, {uuid}):", initialvalue="{index:04d}", parent=self)
        if not pattern:
            return
        try:
            renames = rename_files(self.selected_folder, pattern)
            messagebox.showinfo("Renamed", f"Renamed {len(renames)} files.")
            self.status_bar.configure(text=f"Renamed {len(renames)} files")
        except Exception as exc:
            messagebox.showerror("Rename Error", str(exc))

    def _resize_all(self):
        if self.selected_folder is None:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return
        from tkinter import simpledialog
        size_str = simpledialog.askstring("Resize", "Enter size (WxH):", initialvalue="640x640", parent=self)
        if not size_str or "x" not in size_str.lower():
            return
        try:
            w, h = size_str.lower().split("x")
            count = resize_all_images(self.selected_folder, int(w), int(h))
            messagebox.showinfo("Resized", f"Resized {count} images to {w}x{h}.")
            self.status_bar.configure(text=f"Resized {count} images")
        except Exception as exc:
            messagebox.showerror("Resize Error", str(exc))

    def _remove_exif(self):
        if self.selected_folder is None:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return
        try:
            count = remove_exif_batch(self.selected_folder)
            messagebox.showinfo("EXIF Removed", f"Removed EXIF from {count} images.")
            self.status_bar.configure(text=f"Removed EXIF from {count} images")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _convert_format(self, src: str, dst: str):
        if self.selected_folder is None:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return
        try:
            count = convert_format(self.selected_folder, src, dst)
            messagebox.showinfo("Converted", f"Converted {count} images from {src} to {dst}.")
            self.status_bar.configure(text=f"Converted {count} images {src}→{dst}")
        except Exception as exc:
            messagebox.showerror("Convert Error", str(exc))

    def _to_grayscale(self):
        if self.selected_folder is None:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return
        try:
            count = convert_to_grayscale(self.selected_folder)
            messagebox.showinfo("Grayscale", f"Converted {count} images to grayscale.")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _to_rgb(self):
        if self.selected_folder is None:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return
        try:
            count = convert_to_rgb(self.selected_folder)
            messagebox.showinfo("RGB", f"Converted {count} images to RGB.")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _split_tiles(self):
        if self.current_image_path is None:
            messagebox.showwarning("No image", "Please select an image first.")
            return
        from tkinter import simpledialog
        grid = simpledialog.askstring("Tile Grid", "Enter grid (RowsxCols):", initialvalue="3x3", parent=self)
        if not grid or "x" not in grid.lower():
            return
        try:
            r, c = grid.lower().split("x")
            paths = split_into_tiles(self.current_image_path, int(r), int(c))
            messagebox.showinfo("Tiles Created", f"Created {len(paths)} tiles.")
        except Exception as exc:
            messagebox.showerror("Tile Error", str(exc))

    def _merge_tiles(self):
        folder = filedialog.askdirectory(title="Select folder with tiles")
        if not folder:
            return
        from tkinter import simpledialog
        grid = simpledialog.askstring("Merge Grid", "Enter grid (RowsxCols):", initialvalue="3x3", parent=self)
        if not grid or "x" not in grid.lower():
            return
        try:
            r, c = grid.lower().split("x")
            result = merge_tiles(Path(folder), int(r), int(c))
            messagebox.showinfo("Merged", f"Merged tiles to:\n{result}")
        except Exception as exc:
            messagebox.showerror("Merge Error", str(exc))

    # ── Config Save / Load ────────────────────────────────────────────────

    def on_save_config(self):
        out = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], title="Save augmentation config")
        if not out:
            return
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(self._params_to_dict(), handle, indent=2)
        self.status_bar.configure(text=f"Saved config to {Path(out).name}")

    def on_load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")], title="Load augmentation config")
        if not path:
            return
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self._push_undo()
        self._apply_config_dict(payload)
        self.status_bar.configure(text=f"Loaded config from {Path(path).name}")

    def _params_to_dict(self) -> dict:
        payload = self.params.__dict__.copy()
        payload["variants_per_image"] = self.variants_per_image
        return payload

    def _apply_config_dict(self, payload: dict):
        for key, value in payload.items():
            if key == "variants_per_image":
                self.variants_per_image = int(value)
                continue
            if hasattr(self.params, key):
                setattr(self.params, key, value)
        self.engine = AugmentationEngine(self.params)
        self.seed_var.set(str(self.params.seed_value))
        self.lock_seed_var.set(self.params.lock_seed)
        self._sync_controls_from_params()
        self.update_preview()

    # ── Preview ───────────────────────────────────────────────────────────

    def update_preview(self):
        if self.current_image_np is None:
            return
        try:
            result = self.engine.apply(self.current_image_np, deterministic=True)
            self._show_image(self.augmented_image_label, result, "_augmented_ctk_image")
        except Exception as exc:
            self.status_bar.configure(text=f"Preview error: {exc}")

    def _show_image(self, label: ctk.CTkLabel, image_np: np.ndarray, attr_name: str):
        pil_image = Image.fromarray(image_np)
        pil_image.thumbnail((PREVIEW_MAX_SIZE, PREVIEW_MAX_SIZE), Image.LANCZOS)
        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=pil_image.size)
        setattr(self, attr_name, ctk_image)
        label.configure(image=ctk_image, text="")

    # ── Batch Run ─────────────────────────────────────────────────────────

    def on_run_batch(self):
        if self._batch_running:
            return
        if self.selected_folder is None:
            messagebox.showwarning("No folder selected", "Please select a folder first for batch processing.")
            return
        if not self.params.any_enabled():
            proceed = messagebox.askyesno("No augmentations enabled", "No augmentation is currently enabled, so outputs will be plain copies. Continue anyway?")
            if not proceed:
                return

        self._push_undo()
        self._batch_running = True
        self.run_batch_button.configure(state="disabled", text="Running...")
        self.progress_bar.set(0)
        self.batch_status_label.configure(text="Starting batch job...")

        # Build batch config from UI
        config = BatchConfig()
        config.copies_per_image = self.variants_per_image
        config.random_percentage = float(self.random_pct_slider.slider.get())
        config.skip_existing = self.skip_existing_var.get()
        config.create_backup = self.create_backup_var.get()
        config.max_output_images = int(self.max_output_slider.slider.get())
        config.output_format = self.output_format_var.get()
        config.jpeg_quality = int(self.jpeg_quality_slider.slider.get())
        config.png_compression = int(self.png_compression_slider.slider.get())
        config.webp_quality = int(self.webp_quality_slider.slider.get())

        if self.output_dir_var.get():
            config.output_folder = Path(self.output_dir_var.get())

        # Map selection mode
        sel_map = {
            "All Images": SelectionMode.ALL_IMAGES,
            "Images With Labels": SelectionMode.IMAGES_WITH_LABELS,
            "Images Without Labels": SelectionMode.IMAGES_WITHOUT_LABELS,
            "Random Images": SelectionMode.RANDOM_IMAGES,
            "Filter by Filename": SelectionMode.FILTER_BY_FILENAME,
            "Recently Added": SelectionMode.RECENTLY_ADDED,
        }
        config.selection_mode = sel_map.get(self.selection_mode_var.get(), SelectionMode.ALL_IMAGES)

        # Map save mode
        save_map = {
            "Overwrite Original": SaveMode.OVERWRITE_ORIGINAL,
            "Save as New": SaveMode.SAVE_AS_NEW,
            "Save in New Folder": SaveMode.SAVE_IN_NEW_FOLDER,
            "Save Next to Original": SaveMode.SAVE_NEXT_TO_ORIGINAL,
        }
        config.save_mode = save_map.get(self.save_mode_var.get(), SaveMode.SAVE_IN_NEW_FOLDER)

        # Map naming mode
        naming_map = {
            "Suffix": NamingMode.SUFFIX,
            "Prefix": NamingMode.PREFIX,
            "Timestamp": NamingMode.TIMESTAMP,
            "UUID": NamingMode.UUID,
            "Keep Original": NamingMode.KEEP_ORIGINAL,
        }
        config.naming_mode = naming_map.get(self.naming_mode_var.get(), NamingMode.SUFFIX)

        # Map label mode
        label_map = {
            "Copy Labels": LabelSaveMode.COPY,
            "Transform Labels": LabelSaveMode.TRANSFORM,
            "Verify Labels": LabelSaveMode.VERIFY,
            "Skip Labels": LabelSaveMode.SKIP,
        }
        config.label_save_mode = label_map.get(self.label_mode_var.get(), LabelSaveMode.COPY)

        # Check for labels dir
        if (self.selected_folder.parent / "labels").exists():
            config.labels_dir = self.selected_folder.parent / "labels"
        elif (self.selected_folder / "labels").exists():
            config.labels_dir = self.selected_folder / "labels"

        self._batch_processor = BatchProcessor(self.engine, config)
        self._batch_processor.set_progress_callback(self._on_batch_progress)

        thread = threading.Thread(target=self._run_batch_worker, daemon=True)
        thread.start()

    def _run_batch_worker(self):
        try:
            report = self._batch_processor.run(self.selected_folder)
            report_text = (
                f"Done!\n"
                f"Original: {report.original_images}\n"
                f"Generated: {report.generated_images}\n"
                f"Skipped: {report.skipped_images}\n"
                f"Failed: {report.failed_images}\n"
                f"Time: {report.total_time_formatted}\n"
                f"Speed: {report.images_per_second:.1f} img/s\n"
                f"Output: {report.output_folder}"
            )
            self._log_batch_status(report_text)
        except Exception as exc:
            self._log_batch_status(f"Batch failed: {exc}")
            traceback.print_exc()
        finally:
            self._finish_batch()

    def _on_batch_progress(self, fraction: float, text: str):
        self.after(0, lambda: self.progress_bar.set(min(1.0, fraction)))
        self.after(0, lambda: self.batch_status_label.configure(text=text))
        self.after(0, lambda: self.status_bar.configure(text=text.split("|")[0].strip() if "|" in text else text[:60]))

    def _toggle_pause_batch(self):
        if self._batch_processor is None:
            return
        if self._batch_processor.is_paused:
            self._batch_processor.resume()
            self.pause_batch_button.configure(text="⏸")
            self.status_bar.configure(text="Resumed batch processing")
        else:
            self._batch_processor.pause()
            self.pause_batch_button.configure(text="▶")
            self.status_bar.configure(text="Paused batch processing")

    def _cancel_batch(self):
        if self._batch_processor is None:
            return
        self._batch_processor.cancel()
        self.status_bar.configure(text="Cancelling batch...")

    def _log_batch_status(self, text: str):
        self.after(0, lambda: self.batch_status_label.configure(text=text))
        self.after(0, lambda: self.status_bar.configure(text=text.splitlines()[0]))

    def _finish_batch(self):
        def _reset():
            self._batch_running = False
            self._batch_processor = None
            self.run_batch_button.configure(state="normal", text="▶  Run Batch")
            self.pause_batch_button.configure(text="⏸")
        self.after(0, _reset)