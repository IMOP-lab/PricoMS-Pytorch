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
  Figure 4: Structure of the AMF module.
</p>

We propose the AMF-CSE Module to tackle decoders’ inability to accurately recover detailed information and the issue of using features in isolation. This module includes two units: adaptive morphological fusion for global multiscale feature fusion, and contextual space encoding for organizing image context sequentially

## Installation

To ensure an equitable comparison, all experimental procedures were conducted within a consistent hardware and software milieu: an assembly of four servers, each equipped with dual NVIDIA GeForce RTX 3080 10GB graphics cards, and augmented by 128GB of system memory. Our project is anchored in Python 3.9.0, harnesses PyTorch 1.13.1 for deep learning operations, and operates atop CUDA 11.7.64, employing
distributed training methodologies for both training and evaluation phases. The optimizer of choice was AdamW, with a learning rate meticulously set to 0.0001. Model weights were initially set to random values to ensure a fair start, and the training regimen was designed to span 100 epochs.

## Experiment
### Baselines

We provide GitHub links pointing to the PyTorch implementation code for all networks compared in this experiment here, so you can easily reproduce all these projects.

[U-Net+AttGate](https://github.com/tjboise/APCGAN-AttuNet);[Mask2Former](https://bowenc0221.github.io/mask2former); [Dunet](https://github.com/RanSuLab/DUNet-retinal-vessel-detection); [DeepLab](https://github.com/fregu856/deeplabv3); [FCN](https://github.com/shelhamer/fcn.berkeleyvision.org);[GCN](https://github.com/sungyongs/graph-based-nn); [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet); [MedT](https://github.com/jeya-maria-jose/Medical-Transformer); [OCNet](https://github.com/thuyngch/Fast-LightWeight-SemSeg-Papers); [PSPNet](https://github.com/hszhao/PSPNet);[R2U-Net+AttGate](https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-); [R2U-Net](https://github.com/LeeJae-hoon/Dense-Recurrent-Residual-U-Net-with-for-Video-Quality-Mapping); [MaxViT](https://github.com/google-research/maxvit)

### Compare with others on the IVUS dataset

<div align=center>
  <img src="https://github.com/IMOP-lab/PricoMS-Pytorch/blob/main/figures/base.png">
</div>
<p align=center>
  Figure 5: Comparison experiments between our method and 13 previous segmentation methods on the IVUS dataset.
</p>

<div align=center>
  <img src="https://github.com/IMOP-lab/PricoMS-Pytorch/blob/main/figures/Diagram 2.png">
</div>
<p align=center>
  Figure 6: The visual results of our method compared to the existing 13 segmentation methods on the IVUS dataset. (a) U-Net+Attention Gate; (b) Mask2Former; (c) DUNet; (d) DeepLab3+; (e) FCN-8s; (f) GCN; (g) Swin-Unet; (h) MedT; (i) OCNet; (j) PSPNet; (k) R2U-Net+Attention Gate; (l) R2U-Net; (m) MaxViT; (n) PricoMS;
</p>

### Ablation study

<div align=center>
  <img src="https://github.com/IMOP-lab/PricoMS-Pytorch/blob/main/figures/ablation.png">
</div>
<p align=center>
  Figure 7: Ablation experiments on key components of PricoMS on the IVUS dataset.
</p>

### Checkpoint

[PricoMS](https://pan.baidu.com/s/1RRrl1Yxu_meTA2JmVGrcpQ?pwd=1234) 提取码: 1234
















