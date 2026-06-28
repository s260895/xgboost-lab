"""Verify the XGBoost GPU path works on this machine.

This is the Stage 0 gate. On an RTX 5060 Ti (Blackwell, compute capability
sm_120) a stock ``pip install xgboost`` is *expected* to "just work": the
default CUDA-12 wheel ships native sm_120 cubin kernels (per the build matrix in
``dmlc/xgboost`` ``cmake/Utils.cmake``), unlike PyTorch which needed special
nightly wheels for Blackwell.

What this script proves:
  * ``xgb.build_info()`` reports ``USE_CUDA: True`` (GPU support compiled in).
  * A short training loop with ``device="cuda"`` completes with **no**
    "no kernel image is available for execution on the device" error (the
    tell-tale sign of a missing-architecture wheel).

Run:  python scripts/check_gpu.py   (or `make check-gpu`)

Exit code is 0 on success, 1 on failure, 2 if CUDA support isn't compiled in.
A CPU-only machine is fully supported by the rest of the repo — this check is
skippable there.
"""

from __future__ import annotations

import sys

import numpy as np
import xgboost as xgb


def main() -> int:
    print(f"xgboost            : {xgb.__version__}")  # expect 3.3.0

    info = xgb.build_info()
    use_cuda = bool(info.get("USE_CUDA", False))
    print(f"USE_CUDA           : {use_cuda}")
    # build_info also reports the CUDA runtime version bundled in the wheel.
    if "CUDA_VERSION" in info:
        print(f"bundled CUDA       : {info['CUDA_VERSION']}")

    if not use_cuda:
        print(
            "\nThis wheel was built WITHOUT CUDA (likely the xgboost-cpu variant).\n"
            "That's fine for CPU work; reinstall the default `xgboost` wheel for GPU."
        )
        return 2

    # --- short GPU training loop --------------------------------------------
    # QuantileDMatrix pre-bins the data (ideal on GPU; lower memory). Using it
    # also keeps data + booster on the same device, avoiding the benign
    # "Falling back to prediction using DMatrix due to mismatched devices" warning.
    rng = np.random.default_rng(42)
    X = rng.random((200_000, 50), dtype=np.float32)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(np.int32)

    dtrain = xgb.QuantileDMatrix(X, label=y)
    params = {
        "device": "cuda",
        "tree_method": "hist",
        "objective": "binary:logistic",
    }

    try:
        xgb.train(params, dtrain, num_boost_round=50)
    except xgb.core.XGBoostError as exc:
        msg = str(exc)
        print(f"\nGPU training FAILED:\n{msg}\n")
        if "no kernel image" in msg.lower():
            print(
                "This is the missing-architecture signature: the installed wheel\n"
                "has no kernel compiled for this GPU's compute capability. On\n"
                "Blackwell (sm_120) the stock CUDA-12 wheel should include it — if\n"
                "you see this, check the wheel variant and NVIDIA driver (R570+)."
            )
        return 1

    print("\nOK - GPU training completed (device=cuda, no kernel-image error).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
