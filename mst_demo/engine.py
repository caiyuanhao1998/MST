"""Sequential RGB → synthetic HSI → coded measurement → pretrained SCI inference."""

import gc
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
import scipy.io as sio
import torch
from PIL import Image, ImageOps

from .models import MODELS, ROOT, build_model, default_weights_dir
from .spectral import rgb_to_hsi


def shift(cube, step=2):
    b, c, h, w = cube.shape
    output = cube.new_zeros((b, c, h, w + (c - 1) * step))
    for band in range(c):
        output[:, band, :, step * band : step * band + w] = cube[:, band]
    return output


def measurement(cube, mask):
    return shift(cube * mask).sum(dim=1)


def model_inputs(y, mask, spec, shifted_mask=None):
    """Match upstream inputs, including the calibrated mask for unfolding models."""
    if spec.setting == "Y":
        inputs = y
    else:
        scaled = y / 28 * 2
        width = mask.shape[-1]
        inputs = torch.stack([scaled[..., 2 * i : 2 * i + width] for i in range(28)], dim=1)
        if spec.setting == "HM":
            inputs = inputs * mask
    if spec.mask == "Phi":
        input_mask = shift(mask)
    elif spec.mask == "Mask":
        input_mask = mask
    elif spec.mask == "Phi_PhiPhiT":
        if shifted_mask is None:
            raise ValueError("This model requires the official mask_3d_shift.mat sensing mask.")
        energy = shifted_mask.square().sum(dim=1)
        input_mask = (shifted_mask, torch.where(energy == 0, torch.ones_like(energy), energy))
    else:
        input_mask = None
    return inputs, input_mask


def prepare_rgb(image):
    if image is None:
        raise ValueError("Upload an RGB image first.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_size = image.size
    image.thumbnail((256, 256), Image.Resampling.LANCZOS)
    width, height = image.size
    left, top = (256 - width) // 2, (256 - height) // 2
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    rgb = np.pad(rgb, ((top, 256 - height - top), (left, 256 - width - left), (0, 0)), mode="edge")
    return rgb, [top, left, height, width], list(original_size)


def checkpoint(model, path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run python scripts/download_demo_assets.py.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    return model.eval()


class Reconstructor:
    def __init__(
        self,
        device="auto",
        weights_dir=None,
        mask_path=ROOT / "datasets/TSA_simu_data/mask.mat",
    ):
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in {"cuda", "cpu"}:
            raise ValueError("Use auto, cpu, cuda, or cuda:N for --device.")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable. Use --device cpu.")
            self.device = torch.device("cuda", self.device.index or 0)
            if self.device.index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {self.device.index} does not exist.")
            torch.cuda.set_device(self.device)
        self.weights_dir = Path(weights_dir or default_weights_dir()).expanduser().resolve()
        self.mask_path = Path(mask_path).expanduser().resolve()
        self.model = None
        self.loaded_key = None
        self.lock = Lock()

    def available_models(self):
        return [
            (spec.label, key)
            for key, spec in MODELS.items()
            if (self.weights_dir / spec.checkpoint).is_file()
        ]

    def load_mask(self):
        if not self.mask_path.is_file():
            raise FileNotFoundError(
                "Sensing mask missing. Run python scripts/download_demo_assets.py."
            )
        mask = np.asarray(sio.loadmat(self.mask_path)["mask"], dtype=np.float32)
        if mask.shape != (256, 256) or not np.isfinite(mask).all() or not np.any(mask):
            raise ValueError("mask.mat must contain a finite, nonempty 256 × 256 array named mask.")
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Sensing mask values must lie within [0, 1].")
        return mask

    def load_shifted_mask(self, mask):
        path = self.mask_path.with_name("mask_3d_shift.mat")
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Download mask_3d_shift.mat from the README's TSA_simu_data "
                "dataset and place it beside mask.mat."
            )
        data = sio.loadmat(path)
        if "mask_3d_shift" not in data:
            raise ValueError("mask_3d_shift.mat must contain an array named mask_3d_shift.")
        shifted = np.asarray(data["mask_3d_shift"], dtype=np.float32)
        if (
            shifted.shape != (256, 310, 28)
            or not np.isfinite(shifted).all()
            or shifted.min() < 0
            or shifted.max() > 1
        ):
            raise ValueError("Expected a finite 256 × 310 × 28 shifted mask in [0, 1].")
        for band in range(28):
            if not np.allclose(shifted[:, 2 * band : 2 * band + 256, band], mask):
                raise ValueError("mask_3d_shift.mat does not match the active region of mask.mat.")
        return shifted

    def prepare(self, image):
        started = perf_counter()
        rgb, bounds, original_size = prepare_rgb(image)
        hsi = rgb_to_hsi(rgb)
        mask = self.load_mask()
        with torch.inference_mode():
            cube = torch.from_numpy(hsi.transpose(2, 0, 1).copy())[None]
            mask_tensor = torch.from_numpy(mask.copy())[None, None].expand(1, 28, -1, -1)
            y = measurement(cube, mask_tensor)[0].numpy().copy()
        top, left, h, w = bounds
        return {
            "synthetic_hsi": hsi[top : top + h, left : left + w].copy(),
            "rgb": rgb[top : top + h, left : left + w].copy(),
            "measurement": y,
            "mask": mask.copy(),
            "bounds": bounds,
            "original_size": original_size,
            "synthesis_seconds": perf_counter() - started,
        }

    def load_model(self, key):
        if key not in MODELS:
            raise ValueError("Choose a reconstruction model from the dropdown.")
        if key != self.loaded_key:
            self.model = None
            self.loaded_key = None
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            model = checkpoint(build_model(key), self.weights_dir / MODELS[key].checkpoint)
            self.model = model.to(self.device)
            self.loaded_key = key
        return self.model

    def reconstruct(self, prepared, key, progress=None):
        if key not in MODELS:
            raise ValueError("No model selected. Download the official model weights first.")
        report = progress or (lambda *args, **kwargs: None)
        context = torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext()
        with self.lock, context, torch.inference_mode():
            started = perf_counter()
            report(0.45, desc=f"Loading {MODELS[key].label}")
            model = self.load_model(key)
            y = torch.from_numpy(prepared["measurement"].copy())[None].to(self.device)
            mask = torch.from_numpy(prepared["mask"].copy())[None, None].to(self.device)
            mask = mask.expand(1, 28, -1, -1)
            shifted_mask = None
            if MODELS[key].mask == "Phi_PhiPhiT":
                calibrated = self.load_shifted_mask(prepared["mask"])
                shifted_mask = torch.from_numpy(calibrated.transpose(2, 0, 1).copy())[None]
                shifted_mask = shifted_mask.to(self.device)
            inputs, input_mask = model_inputs(y, mask, MODELS[key], shifted_mask)
            report(0.65, desc=f"Reconstructing 28 channels with {MODELS[key].label}")
            predicted = model(inputs, input_mask)
            if isinstance(predicted, (tuple, list)):
                predicted = predicted[0]
            if predicted.shape != (1, 28, 256, 256):
                raise RuntimeError(f"Unexpected reconstruction shape: {tuple(predicted.shape)}")
            if not torch.isfinite(predicted).all():
                raise RuntimeError(f"{MODELS[key].label} produced nonfinite values.")
            raw = predicted[0].permute(1, 2, 0).cpu().numpy().copy()
            elapsed = perf_counter() - started
        top, left, h, w = prepared["bounds"]
        record = {
            **prepared,
            "cube": raw[top : top + h, left : left + w].copy(),
            "model": key,
            "inference_seconds": elapsed,
            "device": str(self.device),
        }
        record["range_warning"] = ""
        if np.max(np.abs(raw)) > 4:
            record["range_warning"] = (
                f"{MODELS[key].label} produced intensities far outside [0, 1]. "
                "Previews are clipped; MAT files retain the raw output. "
                "Try another released model for a more usable preview."
            )
        return record
