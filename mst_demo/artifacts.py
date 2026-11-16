"""Session-local raw HSI exports and display-only PNG files."""

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio
from matplotlib.figure import Figure
from PIL import Image

from .spectral import RGB_CENTERS, RGB_SIGMA, WAVELENGTHS, band_image


def cleanup_result(record):
    if record and record.get("artifact_dir"):
        shutil.rmtree(record["artifact_dir"], ignore_errors=True)


def metadata(record, wavelengths, channels, brightness, color):
    return {
        "model": record["model"],
        "device": record["device"],
        "source": "RGB → analytic Gaussian spectral basis → CASSI measurement → SCI model",
        "synthesis": {
            "rgb_centers_nm": RGB_CENTERS.tolist(),
            "sigma_nm": RGB_SIGMA,
            "formula": "H(lambda) = sum_c(RGB_c*g_c(lambda)) / sum_c(g_c(lambda)); "
            "g_c(lambda) = exp(-0.5*((lambda-mu_c)/sigma)^2)",
        },
        "native_wavelengths_nm": WAVELENGTHS.tolist(),
        "display_wavelengths_nm": np.asarray(wavelengths).tolist(),
        "display_channels_1based": [i + 1 for i in channels],
        "export_channels_1based": [i + 1 for i in channels],
        "brightness": float(brightness),
        "pseudocolor": bool(color),
        "cube_shape": list(record["cube"].shape),
        "measurement_shape": list(record["measurement"].shape),
        "input_width_height": record["original_size"],
        "padding_top_left_height_width": record["bounds"],
        "range_warning": record.get("range_warning", ""),
        "note": "Synthetic RGB-derived spectra are not measured HSI ground truth. "
        "Edited wavelengths are display labels; pretrained physical bands remain fixed. "
        "Raw cube values are not clipped or multiplied by display brightness.",
    }


def export_result(
    record, wavelengths, channels, brightness, color, *, include_cube=True, include_channels=True
):
    meta = metadata(record, wavelengths, channels, brightness, color)
    serialized = json.dumps(meta, sort_keys=True)
    key = hashlib.sha256(serialized.encode()).hexdigest()[:16]
    folder = Path(record["artifact_dir"]) / key
    folder.mkdir(parents=True, exist_ok=True)
    mat_path, zip_path = folder / "reconstruction.mat", folder / "spectral_channels.zip"
    if include_cube and not mat_path.is_file():
        sio.savemat(
            mat_path,
            {
                "cube": record["cube"],
                "synthetic_hsi": record["synthetic_hsi"],
                "rgb": record["rgb"],
                "measurement": record["measurement"],
                "mask": record["mask"],
                "wavelengths_nm": WAVELENGTHS,
                "display_wavelengths_nm": wavelengths,
                "metadata_json": serialized,
            },
            do_compression=True,
        )
    if include_channels and not zip_path.is_file():
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("metadata.json", json.dumps(meta, indent=2))
            for i in channels:
                for style, use_color in [("display", color), ("grayscale", False)]:
                    buffer = io.BytesIO()
                    Image.fromarray(
                        band_image(record["cube"], i, wavelengths, brightness, use_color)
                    ).save(buffer, format="PNG")
                    archive.writestr(
                        f"{style}/ch_{i + 1:02d}_{wavelengths[i]:g}nm.png", buffer.getvalue()
                    )
    return (str(mat_path) if include_cube else None, str(zip_path) if include_channels else None)


def spectrum_figure(record, wavelengths, xy=None):
    if xy is None:
        original = record["synthetic_hsi"].mean(axis=(0, 1))
        reconstructed = record["cube"].mean(axis=(0, 1))
        title = "Spatial mean spectrum"
    else:
        x, y = xy
        original, reconstructed = record["synthetic_hsi"][y, x], record["cube"][y, x]
        title = f"Pixel ({x}, {y})"
    figure = Figure(figsize=(8, 3.6), layout="constrained")
    ax = figure.subplots()
    ax.plot(wavelengths, original, color="#6366f1", linewidth=1.8, label="Synthetic HSI")
    ax.plot(wavelengths, reconstructed, color="#f97316", linewidth=1.8, label="Reconstruction")
    ax.set(xlabel="Wavelength (nm)", ylabel="Intensity", title=title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    return figure
