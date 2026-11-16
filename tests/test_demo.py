"""Check sensing physics, upstream model contracts, and raw artifact correctness."""

import ast
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io as sio
import torch
from PIL import Image

from mst_demo.artifacts import export_result
from mst_demo.engine import measurement, model_inputs, prepare_rgb
from mst_demo.models import MODELS, ROOT, load_module
from mst_demo.spectral import (
    DEFAULT_WAVELENGTH_TEXT,
    WAVELENGTHS,
    parse_channels,
    parse_wavelengths,
    rgb_to_hsi,
)


class SensingTests(unittest.TestCase):
    def test_gray_has_a_flat_spectrum_without_brightness_normalization(self):
        rgb = np.repeat(np.array([[0, 0.25, 1]], np.float32)[..., None], 3, axis=2)
        output = rgb_to_hsi(rgb)
        np.testing.assert_allclose(output, np.repeat(rgb[..., :1], 28, axis=2), atol=1e-7)

    def test_primary_colors_have_distinct_ordered_spectra(self):
        hsi = rgb_to_hsi(np.eye(3, dtype=np.float32)[None])
        peaks = WAVELENGTHS[hsi[0].argmax(axis=1)]
        self.assertGreater(peaks[0], peaks[1])
        self.assertGreater(peaks[1], peaks[2])
        self.assertTrue(((hsi >= 0) & (hsi <= 1)).all())

    def test_rgb_synthesis_rejects_invalid_inputs(self):
        for value in [np.zeros((2, 3)), np.full((2, 3, 3), np.nan), np.ones((2, 3, 3)) * 2]:
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                rgb_to_hsi(value)

    def test_measurement_impulses_shift_two_pixels_per_band(self):
        cube = torch.zeros(1, 28, 4, 8)
        cube[0, 0, 1, 2], cube[0, 3, 1, 2] = 2, 5
        y = measurement(cube, torch.full_like(cube, 0.5))
        self.assertEqual(y.shape, (1, 4, 62))
        self.assertEqual(float(y[0, 1, 2]), 1)
        self.assertEqual(float(y[0, 1, 8]), 2.5)
        self.assertEqual(float(y.sum()), 3.5)

    def test_all_twenty_input_contracts_match_upstream_templates(self):
        template = load_module(ROOT / "simulation/test_code/template.py")
        self.assertEqual(len(MODELS), 20)
        for key, spec in MODELS.items():
            args = SimpleNamespace(template=key)
            template.set_template(args)
            with self.subTest(method=key):
                self.assertEqual((spec.setting, spec.mask), (args.input_setting, args.input_mask))

    def test_backprojection_scaling_and_mask_energy(self):
        mask = torch.ones(1, 28, 3, 256) * 0.5
        y = torch.arange(310, dtype=torch.float32)[None, None].expand(1, 3, 310)
        back, phi = model_inputs(y, mask, MODELS["mst_s"])
        torch.testing.assert_close(back[0, 27], y[0, :, 54:310] / 28 * 2)
        self.assertEqual(phi.shape, (1, 28, 3, 310))
        hm, _ = model_inputs(y, mask, MODELS["tsa_net"])
        torch.testing.assert_close(hm, back * mask)
        raw, (shifted, energy) = model_inputs(y, mask, MODELS["gap_net"])
        self.assertTrue(torch.equal(raw, y))
        torch.testing.assert_close(energy, shifted.square().sum(1))
        _, (_, safe) = model_inputs(y, torch.zeros_like(mask), MODELS["dauhst_2stg"])
        self.assertTrue(torch.equal(safe, torch.ones_like(safe)))

    def test_sensing_matches_original_helpers_on_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("Original upstream helper requires CUDA")
        parsed = ast.parse((ROOT / "simulation/test_code/utils.py").read_text())
        selected = ast.Module(
            body=[
                n
                for n in parsed.body
                if isinstance(n, ast.FunctionDef)
                and n.name in {"gen_meas_torch", "shift", "shift_back"}
            ],
            type_ignores=[],
        )
        namespace = {"torch": torch}
        exec(compile(selected, "original_mst_physics", "exec"), namespace)
        with torch.inference_mode():
            cube = torch.rand(1, 28, 256, 256, device="cuda")
            mask = torch.rand_like(cube)
            y = measurement(cube, mask)
            original = namespace["gen_meas_torch"](cube, mask, Y2H=False)
            torch.testing.assert_close(y, original, rtol=0, atol=0)
            for key in ["mst_s", "tsa_net"]:
                actual, _ = model_inputs(y, mask, MODELS[key])
                expected = namespace["gen_meas_torch"](cube, mask, mul_mask=key == "tsa_net")
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_resize_preserves_aspect_ratio_and_absolute_rgb_intensity(self):
        rgb, bounds, original = prepare_rgb(Image.new("RGB", (1000, 500), (50, 75, 100)))
        self.assertEqual(rgb.shape, (256, 256, 3))
        self.assertEqual(bounds[2:], [128, 256])
        self.assertEqual(original, [1000, 500])
        np.testing.assert_allclose(rgb[0, 0], np.array([50, 75, 100]) / 255, atol=1e-7)
        tiny, tiny_bounds, _ = prepare_rgb(Image.new("RGB", (1, 1), (0, 0, 0)))
        self.assertEqual(tiny_bounds[2:], [1, 1])
        self.assertTrue(np.isfinite(tiny).all())


class DisplayTests(unittest.TestCase):
    def test_native_grid_exactly_matches_repository_matlab(self):
        source = (ROOT / "visualization/show_simulation.m").read_text()
        wave = source.split("lam28 = [", 1)[1].split("];", 1)[0].replace("...", " ")
        np.testing.assert_array_equal(parse_wavelengths(wave), WAVELENGTHS)
        np.testing.assert_array_equal(parse_wavelengths(DEFAULT_WAVELENGTH_TEXT), WAVELENGTHS)

    def test_channel_ranges_and_invalid_selections(self):
        self.assertEqual(parse_channels("1, 4, 8-12, 28, 4"), [0, 3, 7, 8, 9, 10, 11, 27])
        self.assertEqual(parse_channels("all"), list(range(28)))
        for text in ["0", "29", "3-1", "x", ",", "-1", "1.5"]:
            with self.subTest(value=text), self.assertRaises(ValueError):
                parse_channels(text)

    def test_invalid_wavelengths_are_rejected(self):
        for text in ["1,2", "nan," * 28, ",".join(["500"] * 28), "inf," * 28]:
            with self.subTest(value=text), self.assertRaises(ValueError):
                parse_wavelengths(text)

    def test_exports_keep_raw_values_and_use_distinct_display_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            cube = np.linspace(-0.2, 1.2, 3 * 4 * 28, dtype=np.float32).reshape(3, 4, 28)
            record = {
                "cube": cube,
                "synthetic_hsi": cube * 0.5,
                "rgb": np.zeros((3, 4, 3)),
                "measurement": np.ones((256, 310), np.float32),
                "mask": np.ones((256, 256)),
                "bounds": [0, 0, 3, 4],
                "original_size": [4, 3],
                "model": "mst_s",
                "device": "cpu",
                "artifact_dir": directory,
            }
            edited = WAVELENGTHS + 1
            mat, archive = export_result(record, edited, [0, 27], 3, True)
            saved = sio.loadmat(mat)
            np.testing.assert_array_equal(saved["cube"], cube)
            np.testing.assert_array_equal(saved["wavelengths_nm"].ravel(), WAVELENGTHS)
            np.testing.assert_array_equal(saved["display_wavelengths_nm"].ravel(), edited)
            with zipfile.ZipFile(archive) as zipped:
                self.assertEqual(len([n for n in zipped.namelist() if n.endswith(".png")]), 4)
                meta = json.loads(zipped.read("metadata.json"))
                self.assertEqual(meta["display_channels_1based"], [1, 28])
                self.assertEqual(meta["brightness"], 3)
                self.assertIn("analytic Gaussian", meta["source"])
            changed_mat, _ = export_result(record, WAVELENGTHS, [0, 27], 1, False)
            self.assertNotEqual(Path(mat).parent, Path(changed_mat).parent)


if __name__ == "__main__":
    unittest.main()
