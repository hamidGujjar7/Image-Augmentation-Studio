"""
gui_app.py
-----------------------------------------------------------------------------
Modern dark-mode GUI for the Image Augmentation Tool, built with
CustomTkinter. This module only handles presentation and user interaction;
all image-processing logic lives in augmentation_engine.py.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

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

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

FONT_HEADER = ("Segoe UI", 16, "bold")
FONT_SECTION = ("Segoe UI", 13, "bold")
FONT_LABEL = ("Segoe UI", 12)
FONT_SMALL = ("Segoe UI", 11)

PREVIEW_MAX_SIZE = 480
PAD = 12


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


class ImageAugmentationApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Augmentation Studio")
        self.geometry("1360x860")
        self.minsize(1180, 760)
        self.configure(fg_color="#0b1020")

        self.params = AugmentationParams()
        self.engine = AugmentationEngine(self.params)
        self.current_image_path: Optional[Path] = None
        self.current_image_np: Optional[np.ndarray] = None
        self.selected_folder: Optional[Path] = None
        self.variants_per_image = 6
        self._batch_running = False

        self._original_ctk_image: Optional[ctk.CTkImage] = None
        self._augmented_ctk_image: Optional[ctk.CTkImage] = None
        self.searchable_rows = []
        self.slider_controls = {}
        self.toggle_controls = {}
        self.section_cards = {}

        self._build_layout()

    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_preview_area()
        self._build_output_sidebar()

    def _build_sidebar(self):
        sidebar = ctk.CTkScrollableFrame(self, width=360, corner_radius=0, fg_color="#0b1020")
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(sidebar, fg_color="#15233b", corner_radius=16)
        header.pack(fill="x", pady=(0, PAD))
        ctk.CTkLabel(header, text="🎛  Augmentation Studio", font=FONT_HEADER).pack(anchor="w", padx=PAD, pady=(PAD, 4))
        ctk.CTkLabel(header, text="Fast, polished augmentation for images and folders", font=FONT_SMALL, text_color="#9bb3cf").pack(anchor="w", padx=PAD, pady=(0, PAD))

        self.search_var = ctk.StringVar(value="")
        self.search_entry = ctk.CTkEntry(sidebar, placeholder_text="Search controls…", textvariable=self.search_var)
        self.search_entry.pack(fill="x", pady=(0, PAD))
        self.search_entry.bind("<KeyRelease>", lambda _event: self._apply_search())

        self._build_quick_actions_card(sidebar)
        self._build_input_card(sidebar)
        self.section_cards["Geometry"] = self._build_section_card(sidebar, "Geometry", "Rotate, flip, and scale", self._build_geometry_controls)
        self.section_cards["Color"] = self._build_section_card(sidebar, "Color", "Brightness, contrast, saturation, hue", self._build_color_controls)
        self.section_cards["Noise / Blur"] = self._build_section_card(sidebar, "Noise / Blur", "Blur, noise, and occlusion", self._build_noise_controls)
        self.section_cards["Batch"] = self._build_section_card(sidebar, "Batch", "Batch size, seed, and export", self._build_batch_controls)
        self._refresh_section_visibility()

    def _build_quick_actions_card(self, parent):
        card = self._section_card(parent, "Quick Actions", "Presets, seed, and order")
        buttons = ctk.CTkFrame(card, fg_color="transparent")
        buttons.pack(fill="x", pady=(0, 8))
        for label, preset in [("Light", "Light"), ("Balanced", "Balanced"), ("Heavy", "Heavy")]:
            ctk.CTkButton(buttons, text=label, width=90, command=lambda p=preset: self._apply_preset(p)).pack(side="left", padx=(0, 8))
        ctk.CTkButton(buttons, text="Reset", width=90, fg_color="#6b7280", hover_color="#4b5563", command=self._reset_defaults).pack(side="left")

        self.random_order_toggle = ToggleRow(card, "Random Order", self._toggle("random_order_enabled"), search_tags=["random", "order", "shuffle"])
        self.random_order_toggle.pack(fill="x", pady=(4, 6))
        self._register_toggle("random_order_enabled", self.random_order_toggle)

        seed_row = ctk.CTkFrame(card, fg_color="transparent")
        seed_row.pack(fill="x", pady=(2, 6))
        ctk.CTkLabel(seed_row, text="Seed", font=FONT_LABEL).pack(side="left")
        self.seed_var = ctk.StringVar(value=str(self.params.seed_value))
        self.seed_entry = ctk.CTkEntry(seed_row, width=100, textvariable=self.seed_var)
        self.seed_entry.pack(side="left", padx=(8, 0))
        self.lock_seed_var = ctk.BooleanVar(value=self.params.lock_seed)
        self.lock_seed_checkbox = ctk.CTkCheckBox(seed_row, text="Lock", variable=self.lock_seed_var, command=self._on_seed_toggle)
        self.lock_seed_checkbox.pack(side="left", padx=(10, 0))

        action_row = ctk.CTkFrame(card, fg_color="transparent")
        action_row.pack(fill="x", pady=(8, 0))
        ctk.CTkButton(action_row, text="Save Config", command=self.on_save_config).pack(side="left", padx=(0, 8))
        ctk.CTkButton(action_row, text="Load Config", command=self.on_load_config).pack(side="left")

    def _build_input_card(self, parent):
        card = self._section_card(parent, "Input", "Load an image or a folder")
        ctk.CTkButton(card, text="📄  Select Image", command=self.on_select_image).pack(fill="x", pady=4)
        ctk.CTkButton(card, text="📁  Select Folder (Batch)", command=self.on_select_folder).pack(fill="x", pady=4)
        self.input_status_label = ctk.CTkLabel(card, text="No file or folder selected.", font=FONT_SMALL, text_color="#9bb3cf", wraplength=300, justify="left")
        self.input_status_label.pack(anchor="w", pady=(4, 6))

    def _build_geometry_controls(self, parent):
        self.rotation_toggle = ToggleRow(parent, "Rotation", self._toggle("rotation_enabled"), search_tags=["rotation", "geometry"])
        self.rotation_toggle.pack(fill="x", pady=(0, 4))
        self.rotation_slider = SliderRow(parent, "Degrees", -45, 45, self.params.rotation_degrees, self._set("rotation_degrees"), search_tags=["rotation", "degrees", "geometry"])
        self.rotation_slider.pack(fill="x")
        self.rotation_probability_slider = SliderRow(parent, "Probability", 0.0, 1.0, self.params.rotation_probability, self._set("rotation_probability"), fmt="{:.2f}", search_tags=["rotation", "probability", "geometry"])
        self.rotation_probability_slider.pack(fill="x")
        self._register_toggle("rotation_enabled", self.rotation_toggle)
        self._register_slider("rotation_degrees", self.rotation_slider)
        self._register_slider("rotation_probability", self.rotation_probability_slider)

        self.flip_h_toggle = ToggleRow(parent, "Flip Horizontal", self._toggle("flip_h_enabled"), search_tags=["flip", "horizontal", "geometry"])
        self.flip_h_toggle.pack(fill="x", pady=(6, 4))
        self.flip_h_probability_slider = SliderRow(parent, "Horizontal Probability", 0.0, 1.0, self.params.flip_h_probability, self._set("flip_h_probability"), fmt="{:.2f}", search_tags=["flip", "horizontal", "probability", "geometry"])
        self.flip_h_probability_slider.pack(fill="x")
        self._register_toggle("flip_h_enabled", self.flip_h_toggle)
        self._register_slider("flip_h_probability", self.flip_h_probability_slider)

        self.flip_v_toggle = ToggleRow(parent, "Flip Vertical", self._toggle("flip_v_enabled"), search_tags=["flip", "vertical", "geometry"])
        self.flip_v_toggle.pack(fill="x", pady=(6, 4))
        self.flip_v_probability_slider = SliderRow(parent, "Vertical Probability", 0.0, 1.0, self.params.flip_v_probability, self._set("flip_v_probability"), fmt="{:.2f}", search_tags=["flip", "vertical", "probability", "geometry"])
        self.flip_v_probability_slider.pack(fill="x")
        self._register_toggle("flip_v_enabled", self.flip_v_toggle)
        self._register_slider("flip_v_probability", self.flip_v_probability_slider)

        self.scale_toggle = ToggleRow(parent, "Scale", self._toggle("scale_enabled"), search_tags=["scale", "zoom", "geometry"])
        self.scale_toggle.pack(fill="x", pady=(6, 4))
        self.scale_slider = SliderRow(parent, "Factor", 0.5, 1.5, self.params.scale_factor, self._set("scale_factor"), search_tags=["scale", "factor", "geometry"])
        self.scale_slider.pack(fill="x")
        self.scale_probability_slider = SliderRow(parent, "Probability", 0.0, 1.0, self.params.scale_probability, self._set("scale_probability"), fmt="{:.2f}", search_tags=["scale", "probability", "geometry"])
        self.scale_probability_slider.pack(fill="x")
        self._register_toggle("scale_enabled", self.scale_toggle)
        self._register_slider("scale_factor", self.scale_slider)
        self._register_slider("scale_probability", self.scale_probability_slider)

    def _build_color_controls(self, parent):
        self.brightness_toggle = ToggleRow(parent, "Brightness", self._toggle("brightness_enabled"), search_tags=["brightness", "color"])
        self.brightness_toggle.pack(fill="x", pady=(0, 4))
        self.brightness_slider = SliderRow(parent, "Level", 0.5, 1.5, self.params.brightness_factor, self._set("brightness_factor"), search_tags=["brightness", "level", "color"])
        self.brightness_slider.pack(fill="x")
        self.brightness_probability_slider = SliderRow(parent, "Probability", 0.0, 1.0, self.params.brightness_probability, self._set("brightness_probability"), fmt="{:.2f}", search_tags=["brightness", "probability", "color"])
        self.brightness_probability_slider.pack(fill="x")
        self._register_toggle("brightness_enabled", self.brightness_toggle)
        self._register_slider("brightness_factor", self.brightness_slider)
        self._register_slider("brightness_probability", self.brightness_probability_slider)

        self.contrast_toggle = ToggleRow(parent, "Contrast", self._toggle("contrast_enabled"), search_tags=["contrast", "color"])
        self.contrast_toggle.pack(fill="x", pady=(6, 4))
        self.contrast_slider = SliderRow(parent, "Level", 0.5, 1.5, self.params.contrast_factor, self._set("contrast_factor"), search_tags=["contrast", "level", "color"])
        self.contrast_slider.pack(fill="x")
        self.contrast_probability_slider = SliderRow(parent, "Probability", 0.0, 1.0, self.params.contrast_probability, self._set("contrast_probability"), fmt="{:.2f}", search_tags=["contrast", "probability", "color"])
        self.contrast_probability_slider.pack(fill="x")
        self._register_toggle("contrast_enabled", self.contrast_toggle)
        self._register_slider("contrast_factor", self.contrast_slider)
        self._register_slider("contrast_probability", self.contrast_probability_slider)

        self.saturation_toggle = ToggleRow(parent, "Saturation", self._toggle("saturation_enabled"), search_tags=["saturation", "color"])
        self.saturation_toggle.pack(fill="x", pady=(6, 4))
        self.saturation_slider = SliderRow(parent, "Level", 0.5, 1.5, self.params.saturation_factor, self._set("saturation_factor"), search_tags=["saturation", "level", "color"])
        self.saturation_slider.pack(fill="x")
        self.saturation_probability_slider = SliderRow(parent, "Probability", 0.0, 1.0, self.params.saturation_probability, self._set("saturation_probability"), fmt="{:.2f}", search_tags=["saturation", "probability", "color"])
        self.saturation_probability_slider.pack(fill="x")
        self._register_toggle("saturation_enabled", self.saturation_toggle)
        self._register_slider("saturation_factor", self.saturation_slider)
        self._register_slider("saturation_probability", self.saturation_probability_slider)

        self.hue_toggle = ToggleRow(parent, "Hue Shift", self._toggle("hue_enabled"), search_tags=["hue", "color"])
        self.hue_toggle.pack(fill="x", pady=(6, 4))
        self.hue_slider = SliderRow(parent, "Shift", -0.5, 0.5, self.params.hue_shift, self._set("hue_shift"), search_tags=["hue", "shift", "color"])
        self.hue_slider.pack(fill="x")
        self.hue_probability_slider = SliderRow(parent, "Probability", 0.0, 1.0, self.params.hue_probability, self._set("hue_probability"), fmt="{:.2f}", search_tags=["hue", "probability", "color"])
        self.hue_probability_slider.pack(fill="x")
        self._register_toggle("hue_enabled", self.hue_toggle)
        self._register_slider("hue_shift", self.hue_slider)
        self._register_slider("hue_probability", self.hue_probability_slider)

    def _build_noise_controls(self, parent):
        self.gblur_toggle = ToggleRow(parent, "Gaussian Blur", self._toggle("gaussian_blur_enabled"), search_tags=["blur", "gaussian", "noise"])
        self.gblur_toggle.pack(fill="x", pady=(0, 4))
        self.gblur_slider = SliderRow(parent, "Kernel Size", 3, 25, self.params.gaussian_blur_kernel, self._set("gaussian_blur_kernel"), is_int=True, search_tags=["blur", "gaussian", "kernel"])
        self.gblur_slider.pack(fill="x")
        self.gblur_probability_slider = SliderRow(parent, "Probability", 0.0, 1.0, self.params.gaussian_blur_probability, self._set("gaussian_blur_probability"), fmt="{:.2f}", search_tags=["blur", "gaussian", "probability"])
        self.gblur_probability_slider.pack(fill="x")
        self._register_toggle("gaussian_blur_enabled", self.gblur_toggle)
        self._register_slider("gaussian_blur_kernel", self.gblur_slider)
        self._register_slider("gaussian_blur_probability", self.gblur_probability_slider)

        self.mblur_toggle = ToggleRow(parent, "Motion Blur", self._toggle("motion_blur_enabled"), search_tags=["blur", "motion", "noise"])
        self.mblur_toggle.pack(fill="x", pady=(6, 4))
        self.mblur_slider = SliderRow(parent, "Kernel Size", 3, 25, self.params.motion_blur_kernel, self._set("motion_blur_kernel"), is_int=True, search_tags=["blur", "motion", "kernel"])
        self.mblur_slider.pack(fill="x")
        self.mblur_probability_slider = SliderRow(parent, "Probability", 0.0, 1.0, self.params.motion_blur_probability, self._set("motion_blur_probability"), fmt="{:.2f}", search_tags=["blur", "motion", "probability"])
        self.mblur_probability_slider.pack(fill="x")
        self._register_toggle("motion_blur_enabled", self.mblur_toggle)
        self._register_slider("motion_blur_kernel", self.mblur_slider)
        self._register_slider("motion_blur_probability", self.mblur_probability_slider)

        self.noise_toggle = ToggleRow(parent, "Gaussian Noise", self._toggle("gauss_noise_enabled"), search_tags=["noise", "gaussian"])
        self.noise_toggle.pack(fill="x", pady=(6, 4))
        self.noise_slider = SliderRow(parent, "Amount", 0.0, 0.1, self.params.gauss_noise_amount, self._set("gauss_noise_amount"), fmt="{:.3f}", search_tags=["noise", "amount"])
        self.noise_slider.pack(fill="x")
        self.noise_probability_slider = SliderRow(parent, "Probability", 0.0, 1.0, self.params.gauss_noise_probability, self._set("gauss_noise_probability"), fmt="{:.2f}", search_tags=["noise", "probability"])
        self.noise_probability_slider.pack(fill="x")
        self._register_toggle("gauss_noise_enabled", self.noise_toggle)
        self._register_slider("gauss_noise_amount", self.noise_slider)
        self._register_slider("gauss_noise_probability", self.noise_probability_slider)

        self.grid_dropout_toggle = ToggleRow(parent, "Grid Dropout", self._toggle("grid_dropout_enabled"), search_tags=["dropout", "grid", "occlusion"])
        self.grid_dropout_toggle.pack(fill="x", pady=(6, 4))
        self.grid_dropout_slider = SliderRow(parent, "Ratio", 0.1, 0.9, self.params.grid_dropout_ratio, self._set("grid_dropout_ratio"), search_tags=["dropout", "grid", "ratio"])
        self.grid_dropout_slider.pack(fill="x")
        self.grid_dropout_probability_slider = SliderRow(parent, "Probability", 0.0, 1.0, self.params.grid_dropout_probability, self._set("grid_dropout_probability"), fmt="{:.2f}", search_tags=["dropout", "grid", "probability"])
        self.grid_dropout_probability_slider.pack(fill="x")
        self._register_toggle("grid_dropout_enabled", self.grid_dropout_toggle)
        self._register_slider("grid_dropout_ratio", self.grid_dropout_slider)
        self._register_slider("grid_dropout_probability", self.grid_dropout_probability_slider)

        self.coarse_dropout_toggle = ToggleRow(parent, "Coarse Dropout", self._toggle("coarse_dropout_enabled"), search_tags=["dropout", "coarse", "occlusion"])
        self.coarse_dropout_toggle.pack(fill="x", pady=(6, 4))
        self.coarse_dropout_slider = SliderRow(parent, "Intensity", 0.05, 1.0, self.params.coarse_dropout_intensity, self._set("coarse_dropout_intensity"), search_tags=["dropout", "coarse", "intensity"])
        self.coarse_dropout_slider.pack(fill="x")
        self.coarse_dropout_probability_slider = SliderRow(parent, "Probability", 0.0, 1.0, self.params.coarse_dropout_probability, self._set("coarse_dropout_probability"), fmt="{:.2f}", search_tags=["dropout", "coarse", "probability"])
        self.coarse_dropout_probability_slider.pack(fill="x")
        self._register_toggle("coarse_dropout_enabled", self.coarse_dropout_toggle)
        self._register_slider("coarse_dropout_intensity", self.coarse_dropout_slider)
        self._register_slider("coarse_dropout_probability", self.coarse_dropout_probability_slider)

    def _build_batch_controls(self, parent):
        self.variants_slider = SliderRow(parent, "Copies per Image", 1, 20, self.variants_per_image, self._set_variants_from_batch, is_int=True, search_tags=["batch", "copies", "variants"])
        self.variants_slider.pack(fill="x", pady=(0, 6))
        self._register_slider("variants_per_image", self.variants_slider)
        self.run_batch_button = ctk.CTkButton(parent, text="▶  Run Batch Augmentation", fg_color="#2e7d32", hover_color="#1b5e20", command=self.on_run_batch)
        self.run_batch_button.pack(fill="x", pady=(4, 6))
        self.progress_bar = ctk.CTkProgressBar(parent)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", pady=(0, 4))
        self.batch_status_label = ctk.CTkLabel(parent, text="", font=FONT_SMALL, text_color="#9bb3cf", wraplength=300, justify="left")
        self.batch_status_label.pack(anchor="w", pady=(0, 6))

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

    def _build_output_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=320, corner_radius=16, fg_color="#121a2b")
        sidebar.grid(row=0, column=2, sticky="nsew", padx=(0, PAD), pady=PAD)
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(sidebar, text="📝 Output Settings", font=FONT_SECTION).pack(anchor="w", padx=PAD, pady=(PAD, 4))
        ctk.CTkLabel(sidebar, text="Choose output folder, format, and copy count", font=FONT_SMALL, text_color="#9bb3cf").pack(anchor="w", padx=PAD, pady=(0, PAD))

        self.output_dir_var = ctk.StringVar(value="")
        folder_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        folder_frame.pack(fill="x", padx=PAD, pady=(0, 8))
        ctk.CTkLabel(folder_frame, text="Output Folder", font=FONT_LABEL).pack(anchor="w")
        self.output_dir_entry = ctk.CTkEntry(folder_frame, textvariable=self.output_dir_var)
        self.output_dir_entry.pack(fill="x", pady=(4, 6))
        ctk.CTkButton(folder_frame, text="Browse", command=self._browse_output_dir).pack(fill="x")

        self.output_format_var = ctk.StringVar(value=".png")
        format_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        format_frame.pack(fill="x", padx=PAD, pady=(6, 8))
        ctk.CTkLabel(format_frame, text="Output Format", font=FONT_LABEL).pack(anchor="w")
        self.output_format_menu = ctk.CTkOptionMenu(format_frame, values=[".png", ".jpg", ".jpeg", ".bmp", ".tiff"], variable=self.output_format_var)
        self.output_format_menu.pack(fill="x", pady=(4, 0))

        self.output_name_var = ctk.StringVar(value="{name}_aug{index}")
        name_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        name_frame.pack(fill="x", padx=PAD, pady=(6, 8))
        ctk.CTkLabel(name_frame, text="Naming Pattern", font=FONT_LABEL).pack(anchor="w")
        ctk.CTkEntry(name_frame, textvariable=self.output_name_var).pack(fill="x", pady=(4, 0))

        self.copy_count_slider = SliderRow(sidebar, "Copies to Generate", 1, 20, self.variants_per_image, self._set_variants_from_output, is_int=True, search_tags=["copies", "output", "batch"])
        self.copy_count_slider.pack(fill="x", padx=PAD, pady=(6, 8))

        ctk.CTkButton(sidebar, text="Generate Copies", command=self.on_generate_copies, fg_color="#2563eb", hover_color="#1d4ed8").pack(fill="x", padx=PAD, pady=(8, 0))

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

    def _register_slider(self, attr_name: str, row: SliderRow):
        self.slider_controls[attr_name] = row
        self.searchable_rows.append(row)

    def _register_toggle(self, attr_name: str, row: ToggleRow):
        self.toggle_controls[attr_name] = row
        self.searchable_rows.append(row)

    def _apply_search(self):
        query = self.search_var.get().strip().lower()
        for row in self.searchable_rows:
            if row.matches(query):
                row.pack(fill="x", pady=(0, 4))
            else:
                row.pack_forget()

    def _refresh_section_visibility(self):
        section_groups = {
            "Geometry": ["rotation_enabled", "flip_h_enabled", "flip_v_enabled", "scale_enabled"],
            "Color": ["brightness_enabled", "contrast_enabled", "saturation_enabled", "hue_enabled"],
            "Noise / Blur": ["gaussian_blur_enabled", "motion_blur_enabled", "gauss_noise_enabled", "grid_dropout_enabled", "coarse_dropout_enabled"],
        }
        for name, attrs in section_groups.items():
            frame = self.section_cards.get(name)
            if frame is None:
                continue
            visible = any(getattr(self.params, attr) for attr in attrs)
            if visible:
                frame.pack(fill="x", pady=(0, PAD))
            else:
                frame.pack_forget()

    def _set(self, attr_name: str) -> Callable[[float], None]:
        def _callback(value):
            setattr(self.params, attr_name, value)
            self.update_preview()

        return _callback

    def _toggle(self, attr_name: str) -> Callable[[bool], None]:
        def _callback(value: bool):
            setattr(self.params, attr_name, value)
            self._refresh_section_visibility()
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

    def _reset_defaults(self):
        self.params = AugmentationParams()
        self.engine = AugmentationEngine(self.params)
        self._sync_controls_from_params()
        self._refresh_section_visibility()
        self.update_preview()

    def _apply_preset(self, preset_name: str):
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
        self._refresh_section_visibility()
        self.update_preview()

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
        self.random_order_toggle.set_value(self.params.random_order_enabled)
        if hasattr(self, "copy_count_slider"):
            self.copy_count_slider.set_value(float(self.variants_per_image))

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir_var.set(path)

    def on_generate_copies(self):
        if self.current_image_np is None or self.current_image_path is None:
            messagebox.showwarning("No image selected", "Please select an image first before generating copies.")
            return
        output_dir = Path(self.output_dir_var.get()).expanduser() if self.output_dir_var.get() else self.current_image_path.parent / "augmented_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        count = max(1, self.variants_per_image)
        suffix = self.output_format_var.get().lstrip(".")
        for i in range(count):
            augmented = self.engine.apply(self.current_image_np, deterministic=False)
            name = self.output_name_var.get().format(name=self.current_image_path.stem, index=i + 1)
            out_path = output_dir / f"{name}.{suffix}"
            save_image_rgb(augmented, out_path)
        self.status_bar.configure(text=f"Saved {count} copies to {output_dir}")
        messagebox.showinfo("Copies generated", f"Saved {count} augmented copies to:\n{output_dir}")

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
        self._refresh_section_visibility()
        self.update_preview()

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
        self._batch_running = True
        self.run_batch_button.configure(state="disabled", text="Running...")
        self.progress_bar.set(0)
        self.batch_status_label.configure(text="Starting batch job...")
        thread = threading.Thread(target=self._run_batch_worker, daemon=True)
        thread.start()

    def _run_batch_worker(self):
        folder = self.selected_folder
        variants = self.variants_per_image
        output_dir = folder / "augmented_output"
        try:
            images = find_images_in_folder(folder)
            total_ops = max(1, len(images) * variants)
            done = 0
            for image_path in images:
                try:
                    image_np = load_image_rgb(image_path)
                except Exception as exc:
                    self._log_batch_status(f"Skipped {image_path.name}: {exc}")
                    done += variants
                    self._update_progress(done / total_ops)
                    continue
                for i in range(variants):
                    augmented = self.engine.apply(image_np, deterministic=False)
                    out_path = output_dir / f"{image_path.stem}_aug{i + 1}{image_path.suffix}"
                    save_image_rgb(augmented, out_path)
                    done += 1
                    self._update_progress(done / total_ops)
                self._log_batch_status(f"Processed {image_path.name}")
            self._log_batch_status(f"Done. {len(images)} images x {variants} variants saved to:\n{output_dir}")
        except Exception as exc:
            self._log_batch_status(f"Batch failed: {exc}")
            traceback.print_exc()
        finally:
            self._finish_batch()

    def _update_progress(self, fraction: float):
        self.after(0, lambda: self.progress_bar.set(min(1.0, fraction)))

    def _log_batch_status(self, text: str):
        self.after(0, lambda: self.batch_status_label.configure(text=text))
        self.after(0, lambda: self.status_bar.configure(text=text.splitlines()[0]))

    def _finish_batch(self):
        def _reset():
            self._batch_running = False
            self.run_batch_button.configure(state="normal", text="▶  Run Batch Augmentation")
        self.after(0, _reset)