# Comparison with README metrics

Computed on all 10 official simulation test scenes using the released weights and the same saved predictions evaluated in [the reproduction check](readme_validation.md).

Execution checks passed for all 20 simulation models. The measurements below show that exact reproduction of every README metric remains unresolved. Lambda-Net SSIM is 0.79687 with the native Python helper and 0.79853 with the MATLAB formulas, versus 0.841 in the README. TSA-Net PSNR is 31.6107 dB with the Python helper and 31.6449 dB with the MATLAB formulas, versus 31.46 dB in the README. The causes of these differences have not been established.

The two evaluation definitions are reported separately:

- Native Python uses `torch_psnr` and `torch_ssim` from `simulation/train_code/utils.py`: PSNR rounds values multiplied by 256; SSIM uses floating-point inputs and padded convolution windows.
- MATLAB formulas use the quantization and valid-window SSIM in `simulation/test_code/Quality_Metrics/cal_quality_assessment.m`. These formulas were checked with NumPy/SciPy; a native MATLAB runtime was not executed.

Each cell below is PSNR (dB) / SSIM, averaged across all 10 scenes.

| Model | README | Native Python | MATLAB formulas via Python |
| --- | ---: | ---: | ---: |
| MST-S | 34.26 / 0.935 | 34.2827 / 0.93804 | 34.3195 / 0.93277 |
| MST-M | 34.94 / 0.943 | 34.9387 / 0.94891 | 34.9745 / 0.94395 |
| MST-L | 35.18 / 0.948 | 35.2416 / 0.95121 | 35.2767 / 0.94642 |
| CST-S | 34.71 / 0.940 | 34.6826 / 0.94551 | 34.7168 / 0.94017 |
| CST-M | 35.31 / 0.947 | 35.2727 / 0.95225 | 35.3068 / 0.94735 |
| CST-L | 35.85 / 0.954 | 35.8410 / 0.95875 | 35.8757 / 0.95402 |
| CST-L-Plus | 36.12 / 0.957 | 36.0788 / 0.96093 | 36.1130 / 0.95626 |
| GAP-Net | 33.26 / 0.917 | 33.2447 / 0.92215 | 33.2839 / 0.91672 |
| ADMM-Net | 33.58 / 0.918 | 33.5644 / 0.92286 | 33.6071 / 0.91769 |
| TSA-Net | 31.46 / 0.894 | 31.6107 / 0.89846 | 31.6449 / 0.89220 |
| HDNet | 34.97 / 0.943 | 34.9657 / 0.94838 | 35.0006 / 0.94306 |
| DGSMP | 32.63 / 0.917 | 32.6019 / 0.92263 | 32.6360 / 0.91569 |
| BIRNAT | 37.58 / 0.960 | 37.5851 / 0.96492 | 37.6188 / 0.96049 |
| MST++ | 35.99 / 0.951 | 35.9845 / 0.95618 | 36.0190 / 0.95131 |
| λ-Net | 28.53 / 0.841 | 28.5824 / 0.79687 | 28.6341 / 0.79853 |
| DAUHST-2stg | 36.34 / 0.952 | 36.3451 / 0.95690 | 36.3795 / 0.95220 |
| DAUHST-3stg | 37.21 / 0.959 | 37.2192 / 0.96359 | 37.2531 / 0.95916 |
| DAUHST-5stg | 37.75 / 0.962 | 37.7525 / 0.96692 | 37.7872 / 0.96240 |
| DAUHST-9stg | 38.36 / 0.967 | 38.3727 / 0.97131 | 38.4070 / 0.96717 |
| BiSRNet | 29.76 / 0.837 | 29.7068 / 0.83696 | 29.7531 / 0.83352 |
