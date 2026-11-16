"""Native spectral grid, formula-based RGB synthesis, and channel visualization."""

import re

import numpy as np

# Exact lam28 from visualization/show_simulation.m (also show_real.m).
WAVELENGTHS = np.array(
    [
        453.5,
        457.5,
        462.0,
        466.0,
        471.5,
        476.5,
        481.5,
        487.0,
        492.5,
        498.0,
        504.0,
        510.0,
        516.0,
        522.5,
        529.5,
        536.5,
        544.0,
        551.5,
        558.5,
        567.5,
        575.5,
        584.5,
        594.5,
        604.0,
        614.5,
        625.0,
        636.5,
        648.0,
    ],
    dtype=np.float32,
)
DEFAULT_CHANNEL = 22  # Native 584.5 nm band: warm yellow pseudocolor.
DEFAULT_WAVELENGTH_TEXT = ", ".join(f"{x:.1f}" for x in WAVELENGTHS)
RGB_CENTERS = np.array([610, 550, 460], dtype=np.float32)
RGB_SIGMA = 40.0


def rgb_to_hsi(rgb):
    """Nonnegative, normalized Gaussian RGB bases; a gray pixel stays spectrally flat.

    This is an explicit synthetic assumption, not a measured physical spectrum.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("RGB input must be a finite H × W × 3 array.")
    if rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("RGB intensities must be in [0, 1].")
    basis = np.exp(-0.5 * ((WAVELENGTHS[:, None] - RGB_CENTERS[None]) / RGB_SIGMA) ** 2)
    basis /= basis.sum(axis=1, keepdims=True)
    return np.einsum("hwc,lc->hwl", rgb, basis, optimize=True).astype(np.float32)


def parse_wavelengths(text):
    try:
        wave = np.array([float(x) for x in re.split(r"[,;，\s]+", text.strip()) if x])
    except (ValueError, AttributeError) as exc:
        raise ValueError("Enter 28 wavelength labels separated by commas.") from exc
    if wave.shape != (28,) or not np.isfinite(wave).all():
        raise ValueError("Exactly 28 finite wavelength labels are required.")
    if np.any(np.diff(wave) <= 0) or wave[0] < 380 or wave[-1] > 780:
        raise ValueError("Wavelength labels must increase strictly within 380–780 nm.")
    return wave


def parse_channels(text):
    """Return zero-based indices from human band numbers, e.g. 1, 4, 8-12, 28."""
    if not text or text.strip().lower() in {"all", "全部"}:
        return list(range(28))
    bands = []
    for token in re.split(r"[,;，\s]+", text.strip()):
        if not token:
            continue
        if not re.fullmatch(r"\d+(?:-\d+)?", token):
            raise ValueError("Use channel numbers 1–28, e.g. 1, 4, 8-12, 28.")
        limits = [int(x) for x in token.split("-")]
        start, end = limits[0], limits[-1]
        if not 1 <= start <= end <= 28:
            raise ValueError("Channels must lie between 1 and 28, with increasing ranges.")
        bands.extend(range(start - 1, end))
    if not bands:
        raise ValueError("Choose at least one channel from 1–28.")
    return list(dict.fromkeys(bands))


def wavelength_color(wavelength):
    """Approximate spectral hue; this is a display aid, not calibrated colorimetry."""
    w = float(wavelength)
    if w < 440:
        rgb = [(440 - w) / 60, 0, 1]
    elif w < 490:
        rgb = [0, (w - 440) / 50, 1]
    elif w < 510:
        rgb = [0, 1, (510 - w) / 20]
    elif w < 580:
        rgb = [(w - 510) / 70, 1, 0]
    elif w < 645:
        rgb = [1, (645 - w) / 65, 0]
    else:
        rgb = [1, 0, 0]
    return np.clip(rgb, 0, 1).astype(np.float32)


def band_image(cube, band, wavelengths=WAVELENGTHS, brightness=1, color=True):
    intensity = np.clip(cube[..., band] * float(brightness), 0, 1)
    hue = wavelength_color(wavelengths[band]) if color else np.ones(3)
    return np.rint(intensity[..., None] * hue * 255).astype(np.uint8)


def channel_gallery(cube, channels, wavelengths, brightness=1, color=True):
    return [
        (
            band_image(cube, i, wavelengths, brightness, color),
            f"Ch {i + 1:02d} · {wavelengths[i]:g} nm",
        )
        for i in channels
    ]
