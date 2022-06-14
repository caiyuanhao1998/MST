# Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction (CVPR 2022)
[![winner](https://img.shields.io/badge/MST++-Winner_of_NTIRE_2022_Challenge_on_Spectral_Reconstruction_from_RGB-179bd3)](https://github.com/caiyuanhao1998/MST-plus-plus/)
[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2111.07910)
[![zhihu](https://img.shields.io/badge/zhihu-知乎中文解读-179bd3)](https://zhuanlan.zhihu.com/p/501101943)
![visitors](https://visitor-badge.glitch.me/badge?page_id=caiyuanhao1998/MST)

[Yuanhao Cai](caiyuanhao1998.github.io), [Jing Lin](https://scholar.google.com/citations?hl=zh-CN&user=SvaU2GMAAAAJ), Xiaowan Hu, [Haoqian Wang](https://scholar.google.com.hk/citations?user=eldgnIYAAAAJ&hl=zh-CN), [Xin Yuan](https://xygroup6.github.io/xygroup/), [Yulun Zhang](yulunzhang.com), [Radu Timofte](https://people.ee.ethz.ch/~timofter/), and [Luc Van Gool](https://ee.ethz.ch/the-department/faculty/professors/person-detail.OTAyMzM=.TGlzdC80MTEsMTA1ODA0MjU5.html)

*The first two authors contribute equally to this work*

Code and models are coming soon.

#### News
- **2022.04.02 :** Further work [MST++](https://github.com/caiyuanhao1998/MST-plus-plus/) has won the NTIRE 2022 Spectral Reconstruction Challenge. :trophy: 
- **2022.03.02 :** Our paper has been accepted by CVPR 2022, code and models are coming soon. :rocket: 

|            *Scene 2*             |             *Scene 3*             |             *Scene 4*             |             *Scene 7*             |
| :------------------------------: | :-------------------------------: | :-------------------------------: | :-------------------------------: |
| <img src="./figure/frame2channel12.gif"  height=170 width=170> | <img src="./figure/frame3channel21.gif" width=170 height=170> | <img src="./figure/frame4channel28.gif" width=170 height=170> |  <img src="./figure/frame7channel4.gif" width=170 height=170> |

<hr />

> **Abstract:** *Hyperspectral image (HSI) reconstruction aims to recover the 3D spatial-spectral signal from a 2D measurement in the coded aperture snapshot spectral imaging  (CASSI) system. The HSI  representations are highly similar and correlated across the spectral dimension. Modeling the inter-spectra interactions is beneficial for HSI reconstruction. However, existing CNN-based methods show limitations in capturing spectral-wise similarity and long-range dependencies. Besides, the HSI information is modulated by a coded aperture (physical mask) in CASSI. Nonetheless, current algorithms have not fully explored the guidance effect of the mask for HSI restoration. In this paper, we propose a novel framework, Mask-guided Spectral-wise Transformer (MST), for HSI reconstruction. Specifically, we present a Spectral-wise Multi-head Self-Attention (S-MSA) that treats each spectral feature as a token and calculates self-attention along the spectral dimension. In addition, we customize a Mask-guided Mechanism (MM) that directs S-MSA to pay attention to spatial regions with high-fidelity spectral representations. Extensive experiments show that our MST significantly outperforms state-of-the-art (SOTA) methods on simulation and real HSI datasets while requiring dramatically cheaper computational and memory costs.* 
<hr />

## Illustration of Our Method
![Illustration of MST](/figure/MST.png)


## Comparison with State-of-the-art Methods
This repo is a baseline and toolbox containing 11 learning-based algorithms for spectral compressive imaging.

We are going to enlarge our model zoo in the future.


<details close>
<summary><b>Supported algorithms:</b></summary>

* [x] [MST](https://arxiv.org/abs/2111.07910) (CVPR 2022)
* [x] [MST++](https://arxiv.org/abs/2111.07910) (CVPRW 2022)
* [x] [HDNet](https://arxiv.org/abs/2203.02149) (CVPR 2022)
* [x] [BIRNAT](https://ieeexplore.ieee.org/abstract/document/9741335/) (TPAMI 2022)
* [x] [DGSMP](https://arxiv.org/abs/2103.07152) (CVPR 2021)
* [x] [GAP-Net](https://arxiv.org/abs/2012.08364) (Arxiv 2020)
* [x] [TSA-Net](https://link.springer.com/chapter/10.1007/978-3-030-58592-1_12) (ECCV 2020)
* [x] [ADMM-Net](https://openaccess.thecvf.com/content_ICCV_2019/html/Ma_Deep_Tensor_ADMM-Net_for_Snapshot_Compressive_Imaging_ICCV_2019_paper.html) (ICCV 2019)
* [x] [Lambda-Net](https://ieeexplore.ieee.org/document/9010044) (ICCV 2019)


</details>

![comparison_fig](/figure/compare_fig.png)

### Results on NTIRE 2022 HSI Dataset - Validation
|  Method   | Params (M) | FLOPS (G) |    PSNR    |    SSIM    |  Model Zoo   |  Simulation  Result  |  Real  Result  |
| :-------: | :--------: | :-------: | :--------: | :--------: | :----------: | :------------------: | :------------: |
|   [Lambda-Net](https://ieeexplore.ieee.org/document/9010044)   |    28.53    |  0.841   |   0.3814   |   0.0588   | [Google Drive](https://drive.google.com/file/d/1DqeTNNYaIKodCQWrhclVO3bXw1r4OAxJ/view?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/1VebfZFZCNxeT44IE5GtLsg)| -/- |
|   [ADMM-Net](https://openaccess.thecvf.com/content_ICCV_2019/html/Ma_Deep_Tensor_ADMM-Net_for_Snapshot_Compressive_Imaging_ICCV_2019_paper.html)   |   31.70    |  163.81   |   0.3476   |   0.0550   | [Google Drive](https://drive.google.com/file/d/1RZXtCj7q_80xUT59UmdoFPP3g1YDND02/view?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/1gf0jtZfTNgMG7u-Jm3WQDw) | -/- |
|   [TSA-Net](https://link.springer.com/chapter/10.1007/978-3-030-58592-1_12)    |    2.42    |  158.32   |   0.3277   |   0.0437   | [Google Drive](https://drive.google.com/file/d/1b2DyuxEr8u2_3mnM-dOWjz_YH8XK8ir3/view?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/1R1xpN2dzm31eMR5Fw-SL1A) | -/- |
|   [GAP-Net](https://arxiv.org/abs/2012.08364)    |    4.04    |  270.61   |   0.2500   |   0.0367   | [Google Drive](https://drive.google.com/file/d/1M16NrkHcGdzh9fu0NAdRXIMfCK3xQyy0/view?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/12U8F9YAh9HzykPbCuOJWBA) | -/- |
|   [DGSMP](https://arxiv.org/abs/2103.07152)   |    2.66    |  173.81   |   0.2048   |   0.0317   | [Google Drive](https://drive.google.com/file/d/1HYWUplVkCgHznuInVeghYisdJuNqSr4z/view?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/1G51Q4Fp9YdW17dvadAJGPQ) | -/- |
|   [BIRNAT](https://ieeexplore.ieee.org/abstract/document/9741335/)   |    5.21    |   31.04   |   0.2032   |   0.0303   | [Google Drive](https://drive.google.com/file/d/1dOczPBwz6ZCcBCuRG7L7aJdfT4nCb1nx/view?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/16QMyRxnXjgx6wCg8kZr4Ww) | -/- |
|   [HDNet](https://arxiv.org/abs/2203.02149)   |    3.75    |   42.95   |   0.1890   |   0.0274   | [Google Drive](https://drive.google.com/file/d/1Vndgr3csKZp624NtYWiSp8FTikDy8O6Q/view?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/1W63FiCaP1VDfwcKBIWB55w) | -/- |
|   [MST++](https://arxiv.org/abs/2111.07910) |   15.11    |   93.77   |   0.1833   |   0.0274   | [Google Drive](https://drive.google.com/file/d/1tDwe9X46bfaRrdTnQi4q2ZPnwssA5B2A/view?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/1Y9uggzjOGXcuPj92QnRRnw) | -/- |
|   [MST-S](https://arxiv.org/abs/2111.07910)   |    3.62    |  101.59   |   0.1817   |   0.0270   | [Google Drive](https://drive.google.com/file/d/1amYEpxlBnT1pmk7JmKWYT_bi7WAJH-ad/view?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/1CZYtmNj2tE30KmWM8em7gg) | -/- |
|   [MST-M](https://arxiv.org/abs/2111.07910)   |    2.45    |   32.07   |   0.1772   |   0.0256   | [Google Drive](https://drive.google.com/file/d/17dMffhghFf7nZIqR_f8fNY7Kqh_Xm4Vw/view?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/1X1ICPhJuX91RpEkAQykYGQ) | -/- |
|   [MST-L](https://arxiv.org/abs/2111.07910)   |    2.45    |   32.07   |   0.1772   |   0.0256   | [Google Drive](https://drive.google.com/file/d/17dMffhghFf7nZIqR_f8fNY7Kqh_Xm4Vw/view?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/1X1ICPhJuX91RpEkAQykYGQ) | -/- |

Note: access code for `Baidu Disk` is `mst1`.

## 1. Create Envirement:

------

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

  ```shell
  pip install -r requirements.txt
  ```

## 2. Prepare Dataset:

Download simulation and real dataset from https://github.com/mengziyi64/TSA-Net and https://github.com/TaoHuang95/DGSMP, and then put them into the corresponding folders of 'datasets/' and recollect them as the following form:

```shell
|--MST-plus-plus
    |--test_challenge_code
    |--test_develop_code
    |--train_code  
    |--datasets
        |--cave_1024_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene205.mat
        |--CAVE_512_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene30.mat
        |--KAIST_CVPR2021  
            |--1.mat
            |--2.mat
            ： 
            |--30.mat
        |--TSA_simu_data  
            |--mask.mat   
            |--Truth
                |--scene01.mat
                |--scene02.mat
                ： 
                |--scene10.mat
        |--TSA_real_data  
            |--mask.mat   
            |--Measurements
                |--scene1.mat
                |--scene2.mat
                ： 
                |--scene5.mat
```
Following the setting of TSA-Net and DGSMP, we use the CAVE dataset (cave_1024_28) as the simulation training set. And we use both the CAVE (CAVE_512_28) and KAIST (KAIST_CVPR2021) datasets as the real training set. 

## 3. Simulation Experiement:

(1)  Training:	

```shell
cd MST/simulation/train_code/

# MST_S
python train.py --template mst_s --outf ./exp/mst_s/ --method mst_s 

# MST_M
python train.py --template mst_m --outf ./exp/mst_m/ --method mst_m  

# MST_L
python train.py --template mst_l --outf ./exp/mst_l/ --method mst_l 

# GAP-Net
python train.py --template gap_net --outf ./exp/gap_net/ --method gap_net 

# ADMM-Net
python train.py --template admm_net --outf ./exp/admm_net/ --method admm_net 

# TSA-Net
python train.py --template tsa_net --outf ./exp/tsa_net/ --method tsa_net 

# HDNet
python train.py --template hdnet --outf ./exp/hdnet/ --method hdnet 

# DGSMP
python train.py --template dgsmp --outf ./exp/dgsmp/ --method dgsmp 

# BIRNAT
python train.py --template birnat --outf ./exp/birnat/ --method birnat 

# MST_Plus_Plus
python train.py --template mst_plus_plus --outf ./exp/mst_plus_plus/ --method mst_plus_plus 

# λ-Net
python train.py --template lambda_net --outf ./exp/lambda_net/ --method lambda_net
```

The training log, trained model, and reconstrcuted HSI will be available in "MST/simulation/test_code/exp/" . 

(2)  Testing :	

```python
cd MST/simulation/test_code/

# MST_S
python test.py --template mst_s --outf ./exp/mst_s/ --method mst_s --pretrained_model_path ./model_zoo/mst/mst_s.pth

# MST_M
python test.py --template mst_m --outf ./exp/mst_m/ --method mst_m --pretrained_model_path ./model_zoo/mst/mst_m.pth

# MST_L
python test.py --template mst_l --outf ./exp/mst_l/ --method mst_l --pretrained_model_path ./model_zoo/mst/mst_l.pth

# GAP_Net
python test.py --template gap_net --outf ./exp/gap_net/ --method gap_net --pretrained_model_path ./model_zoo/gap_net/gap_net.pth

# ADMM_Net
python test.py --template admm_net --outf ./exp/admm_net/ --method admm_net --pretrained_model_path ./model_zoo/admm_net/admm_net.pth

# TSA_Net
python test.py --template tsa_net --outf ./exp/tsa_net/ --method tsa_net --pretrained_model_path ./model_zoo/tsa_net/tsa_net.pth

# HDNet
python test.py --template hdnet --outf ./exp/hdnet/ --method hdnet --pretrained_model_path ./model_zoo/hdnet/hdnet.pth

# DGSMP
python test.py --template dgsmp --outf ./exp/dgsmp/ --method dgsmp --pretrained_model_path ./model_zoo/dgsmp/dgsmp.pth

# BIRNAT
python test.py --template birnat --outf ./exp/birnat/ --method birnat --pretrained_model_path ./model_zoo/birnat/birnat.pth

# MST_Plus_Plus
python test.py --template mst_plus_plus --outf ./exp/mst_plus_plus/ --method mst_plus_plus --pretrained_model_path ./model_zoo/mst_plus_plus/mst_plus_plus.pth

# λ-Net
python test.py --template lambda_net --outf ./exp/lambda_net/ --method lambda_net --pretrained_model_path ./model_zoo/lambda_net/lambda_net.pth
```

- The reconstrcuted HSI will be available in "MST/simulation/test_code/exp/" .  

- Put "MST/simulation/test_code/exp/" to "MST/simulation/test_code/Quality_Metrics/results" and run "cal_quality_assessment.m" to calculate the PSNR and SSIM of the reconstructed HSIs.

(3)  Visualization :	

- Run "MST/visualization/show_simulation.m" to generate the RGB images of the reconstructed HSI.
- Run "MST/visualization/show_line.m" to draw the spetra density lines.

## 4. Real Experiement:

(1)  Training:	

```shell
cd MST/real/train_code/

# MST_S
python train.py --template mst_s --outf ./exp/mst_s/ --method mst_s 

# MST_M
python train.py --template mst_m --outf ./exp/mst_m/ --method mst_m  

# MST_L
python train.py --template mst_l --outf ./exp/mst_l/ --method mst_l 

# GAP-Net
python train.py --template gap_net --outf ./exp/gap_net/ --method gap_net 

# ADMM-Net
python train.py --template admm_net --outf ./exp/admm_net/ --method admm_net 

# TSA-Net
python train.py --template tsa_net --outf ./exp/tsa_net/ --method tsa_net 

# HDNet
python train.py --template hdnet --outf ./exp/hdnet/ --method hdnet 

# DGSMP
python train.py --template dgsmp --outf ./exp/dgsmp/ --method dgsmp 

# BIRNAT
python train.py --template birnat --outf ./exp/birnat/ --method birnat 

# MST_Plus_Plus
python train.py --template mst_plus_plus --outf ./exp/mst_plus_plus/ --method mst_plus_plus 

# λ-Net
python train.py --template lambda_net --outf ./exp/lambda_net/ --method lambda_net
```

The training log, trained model, and reconstrcuted HSI will be available in "MST/real/test_code/exp/" . 

(2)  Testing :	

```python
cd MST/real/test_code/

# MST_S
python test.py --template mst_s --outf ./exp/mst_s/ --method mst_s --pretrained_model_path ./model_zoo/mst/mst_s.pth

# MST_M
python test.py --template mst_m --outf ./exp/mst_m/ --method mst_m --pretrained_model_path ./model_zoo/mst/mst_m.pth

# MST_L
python test.py --template mst_l --outf ./exp/mst_l/ --method mst_l --pretrained_model_path ./model_zoo/mst/mst_l.pth

# GAP_Net
python test.py --template gap_net --outf ./exp/gap_net/ --method gap_net --pretrained_model_path ./model_zoo/gap_net/gap_net.pth

# ADMM_Net
python test.py --template admm_net --outf ./exp/admm_net/ --method admm_net --pretrained_model_path ./model_zoo/admm_net/admm_net.pth

# TSA_Net
python test.py --template tsa_net --outf ./exp/tsa_net/ --method tsa_net --pretrained_model_path ./model_zoo/tsa_net/tsa_net.pth

# HDNet
python test.py --template hdnet --outf ./exp/hdnet/ --method hdnet --pretrained_model_path ./model_zoo/hdnet/hdnet.pth

# DGSMP
python test.py --template dgsmp --outf ./exp/dgsmp/ --method dgsmp --pretrained_model_path ./model_zoo/dgsmp/dgsmp.pth

# BIRNAT
python test.py --template birnat --outf ./exp/birnat/ --method birnat --pretrained_model_path ./model_zoo/birnat/birnat.pth

# MST_Plus_Plus
python test.py --template mst_plus_plus --outf ./exp/mst_plus_plus/ --method mst_plus_plus --pretrained_model_path ./model_zoo/mst_plus_plus/mst_plus_plus.pth

# λ-Net
python test.py --template lambda_net --outf ./exp/lambda_net/ --method lambda_net --pretrained_model_path ./model_zoo/lambda_net/lambda_net.pth
```

- The reconstrcuted HSI will be available in "MST/real/test_code/exp/" .  

(3)  Visualization :	

- Run "MST/visualization/show_real.m" to generate the RGB images of the reconstructed HSI.

## Citation
If this repo helps you, please consider citing our works:


```
@inproceedings{mst,
	title={Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction},
	author={Yuanhao Cai and Jing Lin and Xiaowan Hu and Haoqian Wang and Xin Yuan and Yulun Zhang and Radu Timofte and Luc Van Gool},
	booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	year={2022}
}

@inproceedings{mst_pp,
  title={MST++: Multi-stage Spectral-wise Transformer for Efficient Spectral Reconstruction},
  author={Yuanhao Cai and Jing Lin and Zudi Lin and Haoqian Wang and Yulun Zhang and Hanspeter Pfister and Radu Timofte and Luc Van Gool},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2022}
}

@inproceedings{hdnet,
	title={HDNet: High-resolution Dual-domain Learning for Spectral Compressive Imaging},
	author={Xiaowan Hu and Yuanhao Cai and Jing Lin and  Haoqian Wang and Xin Yuan and Yulun Zhang and Radu Timofte and Luc Van Gool},
	booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	year={2022}
}
```
