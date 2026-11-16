"""Verify prepared resources work offline and older demo installations still load."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.io as sio
import torch

from mst_demo import models
from mst_demo.engine import Reconstructor
from mst_demo.models import MODELS, ROOT, load_module


class PreparedSetupTests(unittest.TestCase):
    def test_prepared_model_zoo_takes_priority_over_older_demo_weights(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(models, "ROOT", Path(directory)),
        ):
            root = Path(directory)
            relative = MODELS["mst_s"].checkpoint
            legacy = root / "pretrained_weights"
            (legacy / relative).parent.mkdir(parents=True)
            (legacy / relative).touch()
            self.assertEqual(Reconstructor("cpu").weights_dir, legacy)
            prepared = root / "simulation/test_code/model_zoo"
            (prepared / relative).parent.mkdir(parents=True)
            (prepared / relative).touch()
            self.assertEqual(Reconstructor("cpu").weights_dir, prepared)

    def test_explicit_model_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            chosen = Path(directory)
            self.assertEqual(Reconstructor("cpu", weights_dir=chosen).weights_dir, chosen)

    def test_hdnet_training_frequency_loss_works_with_shared_torch(self):
        for mode in ["simulation", "real"]:
            for phase in ["train_code", "test_code"]:
                with self.subTest(mode=mode, phase=phase):
                    module = load_module(ROOT / mode / phase / "architecture/HDNet.py")
                    loss = module.FDL()
                    image = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) / 64
                    actual = loss.tensor2freq(image)[0, 0].detach().numpy()
                    expected = np.fft.fft2(image.numpy()[0], norm="ortho")
                    np.testing.assert_allclose(actual[..., 0], expected.real, atol=1e-6)
                    np.testing.assert_allclose(actual[..., 1], expected.imag, atol=1e-6)
                    image.requires_grad_()
                    value = loss(image, torch.zeros_like(image))
                    self.assertTrue(torch.isfinite(value).all())
                    value.backward()
                    self.assertTrue(torch.isfinite(image.grad).all())
                    self.assertGreater(float(image.grad.abs().sum()), 0)

    def test_prepared_assets_need_no_downloads(self):
        assets = load_module(ROOT / "scripts/download_demo_assets.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "simulation/test_code/model_zoo"
            for spec in MODELS.values():
                target = weights / spec.checkpoint
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"existing checkpoint")
            mask = root / "datasets/TSA_simu_data/mask.mat"
            mask.parent.mkdir(parents=True)
            sio.savemat(mask, {"mask": np.ones((256, 256), dtype=np.float32)})
            sio.savemat(
                mask.with_name("mask_3d_shift.mat"),
                {"mask_3d_shift": np.ones((256, 310, 28), dtype=np.float32)},
                do_compression=True,
            )
            shutil.copytree(ROOT / "demo/examples", root / "demo/examples")
            (root / "docs").mkdir()
            with (
                patch.object(models, "ROOT", root),
                patch.object(assets, "ROOT", root),
                patch.object(assets, "snapshot_download") as snapshot,
                patch.object(assets.urllib.request, "urlopen") as request,
                patch("sys.argv", ["download_demo_assets.py"]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                assets.main()
            snapshot.assert_not_called()
            request.assert_not_called()
            self.assertFalse((root / "pretrained_weights").exists())
            manifest = json.loads((root / "docs/asset_manifest.json").read_text())
            self.assertIn("simulation/test_code/model_zoo/mst/mst_s.pth", manifest["sha256"])
            self.assertEqual(len(manifest["sha256"]), 32)


if __name__ == "__main__":
    unittest.main()
