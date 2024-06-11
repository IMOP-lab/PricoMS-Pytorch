# PricoMS: Prior-coordinated Multiscale Synthesis Framework for Vessel Reconstruction in Intravascular Ultrasound Image Amidst Label Scarcity

PricoMS: Prior-coordinated Multiscale Synthesis Framework for Vessel Reconstruction in Intravascular Ultrasound Image Amidst Label Scarcity

Xingru Huang, Huawei Wang, Shuaibin Chen, Yihao Guo, Francesca Pugliese, Anantharaman Ramasamy, Ryo Torii, Jouke Dijkstra, Huiyu Zhou, Christos V. Bourantas,Qianni Zhang

Hangzhou Dianzi University IMOP-lab

<div align=center>
  <img src="https://github.com/IMOP-lab/PricoMS-Pytorch/blob/main/figures/main.png"> 
</div>
<p align=center>
** Figure 1: Detailed network structure of the PricoMS.**
</p>

We propose PricoMS,  a framework that leverages prior-coordinated multiscale synthesis approaches for the segmentation of cerebral ischemia preventive vessels in IVUS images amidst the challenge of label scarcity. PricoMS achieves state-of-the-art performance over 13 previous methods on the IVUS datasets.

We will first introduce our method and principles, then introduce the experimental environment and provide Github links to previous methods we have compared. Finally, we will present the experimental results.

## Methods
### PCP Module

<div align=center>
  <img src="https://github.com/IMOP-lab/PricoMS-Pytorch/blob/main/figures/prior.png">
</div>
<p align=center>
  Figure 2: Structure of the PCP module.
</p>

We propose the PCP to address label scarcity, consisting of the prior encoder and calibration module . This approach integrates spatial feature extraction with a novel temporal coherence strategy, enabling precise input of rich spatial and temporally coordinated data into the segmentation framework.

### HCS Module

<div align=center>
  <img src="https://github.com/IMOP-lab/PricoMS-Pytorch/blob/main/figures/HCS.png">
</div>
<p align=center>
  Figure 3: Structure of the HCS module.
</p>

We propose the HCS Module , which enhances feature information by refining latent subspace features and focusing attention to attenuate less beneficial elements

### CSE-AMF Module

<div align=center>
  <img src="https://github.com/IMOP-lab/PricoMS-Pytorch/blob/main/figures/AMF.png">
</div>
<p align=center>
  Figure 3: Structure of the AMF module.
</p>

We propose the AMF-CSE Module to tackle decoders’ inability to accurately recover detailed information and the issue of using features in isolation. This module includes two units: adaptive morphological fusion for global multiscale feature fusion, and contextual space encoding for organizing image context sequentially

## Installation

To ensure an equitable comparison, all experimental procedures were conducted within a consistent hardware and software milieu: an assembly of four servers, each equipped with dual NVIDIA GeForce RTX 3080 10GB graphics cards, and augmented by 128GB of system memory. Our project is anchored in Python 3.9.0, harnesses PyTorch 1.13.1 for deep learning operations, and operates atop CUDA 11.7.64, employing
distributed training methodologies for both training and evaluation phases. The optimizer of choice was AdamW, with a learning rate meticulously set to 0.0001. Model weights were initially set to random values to ensure a fair start, and the training regimen was designed to span 100 epochs.

## Experiment
### Baselines

We provide GitHub links pointing to the PyTorch implementation code for all networks compared in this experiment here, so you can easily reproduce all these projects.

[U-Net+AttGate](https://github.com/tjboise/APCGAN-AttuNet); [Bisenet](https://github.com/ooooverflow/BiSeNet); [Dunet](https://github.com/RanSuLab/DUNet-retinal-vessel-detection); [DeepLab](https://github.com/fregu856/deeplabv3); [FCN](https://github.com/shelhamer/fcn.berkeleyvision.org);[GCN](https://github.com/sungyongs/graph-based-nn); [ICNet](https://github.com/hszhao/ICNet); [LEDNe](https://github.com/xiaoyufenfei/LEDNet); [SegResNet](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/segresnet.py); [HighRes3DNet](https://github.com/fepegar/highresnet);
[TransBTS](https://github.com/Rubics-Xuan/TransBTS); [nnFormer](https://github.com/282857341/nnFormer); [SETR](https://github.com/fudan-zvg/SETR)



























