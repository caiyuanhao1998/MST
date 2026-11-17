# README reproduction check — 2026-11-16

Checked GitHub source `9896222849e0214eed7acdea6533dc766c8d6ec5` in a new clone and a Python 3.11 virtual environment without system packages. Installed with the README's `pip install -r requirements.txt`; downloaded the data and weights again from its links.

Environment: Python 3.11.7, PyTorch 2.15.0+cu132, torchvision 0.29.2+cu132, NumPy 2.4.6, SciPy 1.17.1, Gradio 5.50.0; NVIDIA RTX A5000 with 24 GB VRAM. CUDA computation and `pip check` passed.

## Scope and results

- Data: 205 simulation CAVE scenes, 30 real-training CAVE scenes, 30 KAIST scenes, 10 simulation test scenes, five real measurements, and three sensing-mask files. All downloaded scene fields and shapes checked.
- Weights: all 20 simulation checkpoints from Hugging Face revision `7575c2b1c6a4299849ebae759a9f41f9497c915a`, plus the separately released real BiSRNet checkpoint.
- Simulation testing: all 20 README commands passed on all 10 scenes each.
- Real testing: BiSRNet passed on all five scenes. The other 19 real models are outside the agreed scope because corresponding real-scene weights are not linked.
- Training: all 20 simulation entry points and the real BiSRNet entry point completed one optimizer batch with finite parameters and a state-dict save/load check. Simulation smoke runs used one downloaded CAVE scene and validated on all 10 test scenes; real BiSRNet loaded all 30 CAVE and 30 KAIST scenes. Real BiSRNet's native saved checkpoint also loaded through `real/test_code/test.py` and processed all five measurements. These checks do not establish full training convergence.
- Demo: all 20 models passed actual RGB → synthetic HSI → measurement → reconstruction inference with no intensity-range warnings. Browser checks passed preview sizing, all examples, thumbnail selection, exports, spectrum, progress timing, mobile layout, and an additional ADMM-Net reconstruction.
- Both Params/FLOPs example calls completed. FVCore reports unsupported operations, so these are execution checks rather than a new validation of paper FLOP totals.
- Regression suite: 22 tests passed. All changed Python files compile; updated demo helpers and tests pass Ruff; whitespace checks passed.

Training smoke runs added `--max_epoch 1 --epoch_sam_num 1 --batch_size 1`. BIRNAT used `--size 128` after a reproducible 256-pixel training OOM on the 24 GB GPU. Real BiSRNet used the README's supported `--size 128`. Simulation inference retained the full 256-pixel test resolution.

## Bugs fixed

1. CST's hash extension called `Tensor.repeat()` with an invalid signature. All four architecture copies now preserve vector shape, norms, dtype, and device.
2. Four DAUHST testing commands pointed to nonexistent per-stage weight directories; they now use the downloaded `model_zoo/dauhst/` directory.
3. HDNet training called `.cuda()` on the `(model, loss)` tuple. BIRNAT's one-image training batch also reached an uninitialized crop dimension. Both training failures are fixed.
4. Training templates overwrote explicit CLI settings. Explicit arguments now take precedence; real training also honors `--epoch_sam_num`. Simulation crop size is configurable, including its sensing masks and mosaic augmentation.
5. Real testing parsed CLI arguments twice, saved files beside the requested directory when it lacked a trailing slash, and omitted the combined `pred` array consumed by visualization. It now uses one parser, joins paths correctly, and writes both individual results and `Test_result.mat`.
6. Real training now saves state dictionaries compatible with the test loader. Its general training loop includes the requested final epoch and uses consistent epoch numbers. Only the BiSRNet real training path was exercised end to end.
7. Native dataset readers now join directory paths, ignore unrelated files, and avoid repeatedly clipping the entire real-training dataset after every scene. Real arrays store each scene contiguously. Host RAM requirements remain about 60 GiB plus temporary arrays.
8. The demo generated zero-padded shifted masks for GAP-Net, ADMM-Net, and DAUHST, while their official input contract uses the supplied `mask_3d_shift.mat`, including its nonzero borders. The demo now preserves that calibration and checks it against `mask.mat`. On the same RGB example, ADMM-Net's raw range changed from roughly −5321…6798 to 0.016…1.118; the exported cube retains raw model outputs. Prepared assets remain reusable without downloads.
9. MATLAB entry points used Windows-only paths, a filename with casing inconsistent with the README, an undocumented `truth.mat`, and a real-result variable layout unavailable from the test script. Paths and inputs now match generated MAT files; quality metrics are stored separately from the predicted cube.
10. The Params/FLOPs snippet now imports its model classes explicitly.

## Comparison with README metrics

Passing the execution checks above does not establish exact reproduction of the README benchmark table. [The full comparison](readme_metrics_comparison.md) reports both native Python and MATLAB-formula results on the same predictions. Lambda-Net SSIM and TSA-Net PSNR retain unexplained differences from the README values.

## MATLAB verification

All four MATLAB command blocks in the README are byte-for-byte unchanged. NumPy/SciPy checked the original metrics on 200 reconstructed simulation cubes, the original 28-channel wavelength grid, MAT variable layouts, 280 simulation channel images, 140 real channel images, and six spectral curves. The Python rendering uses the repository's color-matching table and interpolation; MATLAB figures were not asserted pixel-identical.

A native MATLAB engine was not executed. Image Processing Toolbox and any other toolbox requirements of the original MATLAB plotting functions are therefore not validated by this run.

The following metrics follow `cal_quality_assessment.m`: clipped outputs converted to 8-bit values, per-band PSNR averaged across 28 bands, Gaussian-window SSIM, then averages across all 10 scenes. They are results from the released weights, independent of the short training checks.

| Simulation model | PSNR (dB) | SSIM | Training patch |
| --- | ---: | ---: | ---: |
| MST-S | 34.319 | 0.9328 | 256 |
| MST-M | 34.974 | 0.9439 | 256 |
| MST-L | 35.277 | 0.9464 | 256 |
| CST-S | 34.717 | 0.9402 | 256 |
| CST-M | 35.307 | 0.9473 | 256 |
| CST-L | 35.876 | 0.9540 | 256 |
| CST-L+ | 36.113 | 0.9563 | 256 |
| GAP-Net | 33.284 | 0.9167 | 256 |
| ADMM-Net | 33.607 | 0.9177 | 256 |
| TSA-Net | 31.645 | 0.8922 | 256 |
| HDNet | 35.001 | 0.9431 | 256 |
| DGSMP | 32.636 | 0.9157 | 256 |
| BIRNAT | 37.619 | 0.9605 | 128 |
| MST++ (SCI) | 36.019 | 0.9513 | 256 |
| λ-Net | 28.634 | 0.7985 | 256 |
| DAUHST-2stg | 36.380 | 0.9522 | 256 |
| DAUHST-3stg | 37.253 | 0.9592 | 256 |
| DAUHST-5stg | 37.787 | 0.9624 | 256 |
| DAUHST-9stg | 38.407 | 0.9672 | 256 |
| BiSRNet | 29.753 | 0.8335 | 256 |
