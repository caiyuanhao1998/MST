"""Run actual pretrained inference for every installed reconstruction checkpoint."""

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image

from mst_demo.engine import Reconstructor
from mst_demo.models import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/model_validation.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    engine = Reconstructor(args.device)
    prepared = engine.prepare(Image.open(ROOT / "demo/examples/simulation/scene_04.png"))
    assert prepared["measurement"].shape == (256, 310)
    results = []
    for _, key in engine.available_models():
        if args.models and key not in args.models:
            continue
        print(f"Testing {key}...", flush=True)
        try:
            result = engine.reconstruct(prepared, key)
            assert result["cube"].shape == prepared["synthetic_hsi"].shape
            assert np.isfinite(result["cube"]).all()
            row = {
                "model": key,
                "passed": True,
                "cube_shape": list(result["cube"].shape),
                "seconds": round(result["inference_seconds"], 3),
                "minimum": float(result["cube"].min()),
                "maximum": float(result["cube"].max()),
                "range_warning": result["range_warning"],
            }
        except Exception as exc:
            traceback.print_exc()
            row = {"model": key, "passed": False, "error": str(exc)}
        results.append(row)
        print(json.dumps(row), flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2) + "\n")
    if not results or not all(row["passed"] for row in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
