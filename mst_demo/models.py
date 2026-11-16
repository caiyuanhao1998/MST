"""Network constructors and input contracts from the upstream simulation code."""

import importlib.util
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    label: str
    module: str
    constructor: str
    checkpoint: str
    setting: str = "H"
    mask: str | None = "Mask"
    kwargs: dict = field(default_factory=dict)


MODELS = {}
for size, blocks in [("s", [2, 2, 2]), ("m", [2, 4, 4]), ("l", [4, 7, 5])]:
    MODELS[f"mst_{size}"] = ModelSpec(
        f"MST-{size.upper()}",
        "MST",
        "MST",
        f"mst/mst_{size}.pth",
        mask="Phi",
        kwargs={"dim": 28, "stage": 2, "num_blocks": blocks},
    )
for size, blocks in [("s", [1, 1, 2]), ("m", [2, 2, 2]), ("l", [2, 4, 6]), ("l_plus", [2, 4, 6])]:
    MODELS[f"cst_{size}"] = ModelSpec(
        f"CST-{size.upper().replace('_PLUS', '+')}",
        "CST",
        "CST",
        f"cst/cst_{size}.pth",
        kwargs={"num_blocks": blocks, "sparse": size != "l_plus"},
    )
for stages in (2, 3, 5, 9):
    key = f"dauhst_{stages}stg"
    MODELS[key] = ModelSpec(
        f"DAUHST-{stages}stg",
        "DAUHST",
        "DAUHST",
        f"dauhst/{key}.pth",
        "Y",
        "Phi_PhiPhiT",
        {"num_iterations": stages},
    )
MODELS.update(
    {
        "bisrnet": ModelSpec(
            "BiSRNet",
            "BiSRNet",
            "BiSRNet",
            "bisrnet/bisrnet.pth",
            kwargs={
                "in_channels": 28,
                "out_channels": 28,
                "n_feat": 28,
                "stage": 1,
                "num_blocks": [1, 1, 1],
            },
        ),
        "mst_plus_plus": ModelSpec(
            "MST++ (SCI)",
            "MST_Plus_Plus",
            "MST_Plus_Plus",
            "mst_plus_plus/mst_plus_plus.pth",
            kwargs={"in_channels": 28, "out_channels": 28, "n_feat": 28, "stage": 3},
        ),
        "hdnet": ModelSpec("HDNet", "HDNet", "HDNet", "hdnet/hdnet.pth", mask=None),
        "tsa_net": ModelSpec("TSA-Net", "TSA_Net", "TSA_Net", "tsa_net/tsa_net.pth", "HM", None),
        "dgsmp": ModelSpec(
            "DGSMP", "DGSMP", "HSI_CS", "dgsmp/dgsmp.pth", "Y", None, {"Ch": 28, "stages": 4}
        ),
        "gap_net": ModelSpec(
            "GAP-Net", "GAP_Net", "GAP_net", "gap_net/gap_net.pth", "Y", "Phi_PhiPhiT"
        ),
        "admm_net": ModelSpec(
            "ADMM-Net", "ADMM_Net", "ADMM_net", "admm_net/admm_net.pth", "Y", "Phi_PhiPhiT"
        ),
        "lambda_net": ModelSpec(
            "λ-Net",
            "Lambda_Net",
            "Lambda_Net",
            "lambda_net/lambda_net.pth",
            "Y",
            "Phi",
            {"out_ch": 28},
        ),
        "birnat": ModelSpec("BIRNAT", "BIRNAT", "BIRNAT", "birnat/birnat.pth", "Y", "Phi"),
    }
)


@cache
def load_module(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"mst_demo_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(key):
    spec = MODELS[key]
    module = load_module(ROOT / "simulation/test_code/architecture" / f"{spec.module}.py")
    return getattr(module, spec.constructor)(**spec.kwargs)
