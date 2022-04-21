# Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction (CVPR 2022)
[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2111.07910)
![visitors](https://visitor-badge.glitch.me/badge?page_id=caiyuanhao1998/MST)

[Yuanhao Cai](caiyuanhao1998.github.io), [Jing Lin](https://scholar.google.com/citations?hl=zh-CN&user=SvaU2GMAAAAJ), Xiaowan Hu, [Haoqian Wang](https://scholar.google.com.hk/citations?user=eldgnIYAAAAJ&hl=zh-CN), [Xin Yuan](https://xygroup6.github.io/xygroup/), [Yulun Zhang](yulunzhang.com), [Radu Timofte](https://people.ee.ethz.ch/~timofter/), and [Luc Van Gool](https://ee.ethz.ch/the-department/faculty/professors/person-detail.OTAyMzM=.TGlzdC80MTEsMTA1ODA0MjU5.html)

*The first two authors contribute equally to this work*

Code and models are coming soon.

#### News
- **2022.04.02 :** Our further work [MST++](https://github.com/caiyuanhao1998/MST-plus-plus/) won the **First** place of NTIRE 2022 Challenge on Spectral Reconstruction from RGB. :trophy: 
- **2022.03.02 :** Our paper has been accepted by CVPR 2022, code and models are coming soon. :rocket: 

|            *Scene 2*             |             *Scene 3*             |             *Scene 4*             |             *Scene 7*             |
| :------------------------------: | :-------------------------------: | :-------------------------------: | :-------------------------------: |
| <img src="./figure/frame2channel12.gif"  height=170 width=170> | <img src="./figure/frame3channel21.gif" width=170 height=170> | <img src="./figure/frame4channel28.gif" width=170 height=170> |  <img src="./figure/frame7channel4.gif" width=170 height=170> |

<hr />

> **Abstract:** *Hyperspectral image (HSI) reconstruction aims to recover the 3D spatial-spectral signal from a 2D measurement in the coded aperture snapshot spectral imaging  (CASSI) system. The HSI  representations are highly similar and correlated across the spectral dimension. Modeling the inter-spectra interactions is beneficial for HSI reconstruction. However, existing CNN-based methods show limitations in capturing spectral-wise similarity and long-range dependencies. Besides, the HSI information is modulated by a coded aperture (physical mask) in CASSI. Nonetheless, current algorithms have not fully explored the guidance effect of the mask for HSI restoration. In this paper, we propose a novel framework, Mask-guided Spectral-wise Transformer (MST), for HSI reconstruction. Specifically, we present a Spectral-wise Multi-head Self-Attention (S-MSA) that treats each spectral feature as a token and calculates self-attention along the spectral dimension. In addition, we customize a Mask-guided Mechanism (MM) that directs S-MSA to pay attention to spatial regions with high-fidelity spectral representations. Extensive experiments show that our MST significantly outperforms state-of-the-art (SOTA) methods on simulation and real HSI datasets while requiring dramatically cheaper computational and memory costs.* 
<hr />


![Illustration of MST](/figure/MST.png)

![Pipeline of MST](/figure/pipeline.png)

# Quantitative Results
![Main Results of PNGAN](/figure/main.png)

# Qualitative Results
## Simulation HSI Reconstruction
![Simulation](/figure/simulation.png)

## Real HSI Reconstruction
![Real](/figure/real.png)

# Citation
If this repo helps you, please consider citing
```
@inproceedings{mst,
	title={Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction},
	author={Yuanhao Cai and Jing Lin and Xiaowan Hu and Haoqian Wang and Xin Yuan and Yulun Zhang and Radu Timofte and Luc Van Gool},
	booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	year={2022}
}
```
