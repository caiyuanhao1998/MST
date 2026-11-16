"""Download only public assets; this script never uploads or publishes anything."""

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.io as sio
from huggingface_hub import snapshot_download

from mst_demo.models import MODELS, ROOT, default_weights_dir

MODEL_REVISION = "7575c2b1c6a4299849ebae759a9f41f9497c915a"
MASK_URL = "https://raw.githubusercontent.com/TaoHuang95/DGSMP/master/Simulation/Data/mask.mat"
MASK_SHA256 = "29f528668225484a833ddaacbe071166236f430277e000f876bc4ceda92b74c7"


def download(url, target):
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path)
    args = parser.parse_args()
    weights_dir = (args.weights_dir or default_weights_dir()).expanduser().resolve()
    missing = [
        spec.checkpoint for spec in MODELS.values() if not (weights_dir / spec.checkpoint).is_file()
    ]
    if missing:
        snapshot_download(
            "CaiYuanhao/MST",
            revision=MODEL_REVISION,
            allow_patterns=missing,
            local_dir=weights_dir,
            max_workers=4,
        )
    mask_path = ROOT / "datasets/TSA_simu_data/mask.mat"
    existing_mask = mask_path.is_file()
    download(MASK_URL, mask_path)
    data = mask_path.read_bytes()
    if not existing_mask and hashlib.sha256(data).hexdigest() != MASK_SHA256:
        raise ValueError("The default sensing mask does not match the verified public mask.")
    mask = sio.loadmat(io.BytesIO(data))["mask"]
    if (
        mask.shape != (256, 256)
        or not np.isfinite(mask).all()
        or not np.any(mask)
        or mask.min() < 0
        or mask.max() > 1
    ):
        raise ValueError("Expected a finite, nonempty 256 × 256 sensing mask in [0, 1].")
    example_manifest = json.loads((ROOT / "demo/examples/simulation/manifest.json").read_text())
    examples = []
    for item in example_manifest["images"]:
        target = ROOT / item["file"]
        download(f"https://drive.google.com/uc?export=download&id={item['drive_id']}", target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Official SCI example hash mismatch: {target.name}")
        examples.append(target)
    paths = [weights_dir / spec.checkpoint for spec in MODELS.values()] + [mask_path] + examples
    manifest = {
        "models": {"repo": "CaiYuanhao/MST", "revision": MODEL_REVISION},
        "mask_url": MASK_URL,
        "examples_folder": example_manifest["folder_url"],
        "sha256": {
            str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in paths
        },
    }
    (ROOT / "docs/asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Ready: all 20 model checkpoints, the sensing mask, and 10 official SCI RGB examples.")


if __name__ == "__main__":
    main()
