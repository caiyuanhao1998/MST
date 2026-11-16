"""Regression checks for failures exposed by running the README commands."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from mst_demo.engine import Reconstructor, model_inputs
from mst_demo.models import MODELS, ROOT, load_module


class ReadmeRegressionTests(unittest.TestCase):
    def test_cst_hash_extension_preserves_vectors_and_norms(self):
        x = torch.linspace(-2, 3, 2 * 7 * 4, dtype=torch.float64).reshape(2, 7, 4)
        for mode in ["simulation", "real"]:
            for phase in ["train_code", "test_code"]:
                with self.subTest(mode=mode, phase=phase):
                    module = load_module(ROOT / mode / phase / "architecture/CST.py")
                    transform = module.XBOXPLUS()
                    transform.set_norms(x)
                    extended = transform.X(x)
                    self.assertEqual(extended.shape, (2, 7, 6))
                    torch.testing.assert_close(extended[..., :4], x)
                    torch.testing.assert_close(extended[..., -1], torch.zeros_like(x[..., 0]))
                    expected = transform.MX.square().expand(2, 7, 1)
                    torch.testing.assert_close(extended.square().sum(-1, keepdim=True), expected)

    def test_explicit_training_settings_override_templates(self):
        for template in ["cst_s", "dauhst_2stg", "birnat"]:
            with self.subTest(template=template):
                result = subprocess.check_output(
                    [
                        sys.executable,
                        "-c",
                        "import json;from option import opt;"
                        "print(json.dumps([opt.max_epoch,opt.batch_size,opt.epoch_sam_num]))",
                        "--template",
                        template,
                        "--max_epoch",
                        "1",
                        "--batch_size",
                        "1",
                        "--epoch_sam_num",
                        "1",
                    ],
                    cwd=ROOT / "simulation/train_code",
                    text=True,
                )
                self.assertEqual(json.loads(result), [1, 1, 1])

    @unittest.skipUnless(torch.cuda.is_available(), "Native crop helper uses CUDA")
    def test_single_image_crop_and_smaller_patch(self):
        directory = str(ROOT / "simulation/train_code")
        sys.path.insert(0, directory)
        try:
            utils = load_module(ROOT / "simulation/train_code/utils.py")
        finally:
            sys.path.remove(directory)
        image = np.full((256, 256, 28), 0.25, np.float32)
        for size in [128, 256]:
            with self.subTest(size=size):
                batch = utils.shuffle_crop([image], 1, crop_size=size)
                self.assertEqual(batch.shape, (1, 28, size, size))
                torch.testing.assert_close(batch, torch.full_like(batch, 0.25))

    def test_unfolding_models_keep_calibrated_mask_borders(self):
        mask = np.full((256, 256), 0.25, np.float32)
        calibrated = np.full((256, 310, 28), 0.75, np.float32)
        for band in range(28):
            calibrated[:, 2 * band : 2 * band + 256, band] = mask
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.mat"
            engine = Reconstructor("cpu", mask_path=path)
            with self.assertRaises(FileNotFoundError):
                engine.load_shifted_mask(mask)
            sio.savemat(
                path.with_name("mask_3d_shift.mat"),
                {"mask_3d_shift": calibrated},
                do_compression=True,
            )
            loaded = engine.load_shifted_mask(mask)
            np.testing.assert_array_equal(loaded, calibrated)
        phi = torch.from_numpy(loaded.transpose(2, 0, 1).copy())[None]
        mask_tensor = torch.from_numpy(mask)[None, None].expand(1, 28, -1, -1)
        y = torch.zeros((1, 256, 310))
        for name in ["admm_net", "gap_net", "dauhst_2stg"]:
            with self.subTest(model=name):
                with self.assertRaises(ValueError):
                    model_inputs(y, mask_tensor, MODELS[name])
                _, (actual, energy) = model_inputs(y, mask_tensor, MODELS[name], phi)
                torch.testing.assert_close(actual, phi)
                torch.testing.assert_close(energy, phi.square().sum(dim=1))
                self.assertEqual(float(actual[0, 0, 0, -1]), 0.75)

    def test_simulation_reader_ignores_unrelated_downloads(self):
        directory = str(ROOT / "simulation/train_code")
        sys.path.insert(0, directory)
        try:
            utils = load_module(ROOT / "simulation/train_code/utils.py")
        finally:
            sys.path.remove(directory)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cube = np.full((8, 8, 28), 16384, np.uint16)
            for name in ["scene01.mat", "scene2.MAT", "scene206.mat"]:
                sio.savemat(root / name, {"img_expand": cube})
            (root / "notes.txt").write_text("Not a scene")
            (root / "scene1_guest.mat").write_text("Not a training scene")
            data = utils.LoadTraining(str(root))
            self.assertEqual(len(data), 2)
            for image in data:
                self.assertEqual(image.dtype, np.float32)
                np.testing.assert_array_equal(image, 0.25)

    def test_real_cave_reader_handles_directory_without_trailing_slash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sio.savemat(
                root / "scene01.mat",
                {"data_slice": np.full((512, 512, 28), 32767.5, np.float32)},
                do_compression=True,
            )
            (root / "notes.txt").write_text("Not a scene")
            for phase in ["train_code", "test_code"]:
                with self.subTest(phase=phase):
                    utils = load_module(ROOT / "real" / phase / "utils.py")
                    data = utils.prepare_data_cave(str(root), 1)
                    self.assertEqual(data.shape, (512, 512, 28, 1))
                    np.testing.assert_array_equal(data, 0.5)


if __name__ == "__main__":
    unittest.main()
