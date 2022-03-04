# RSTT (Real-time Spatial Temporal Transformer)
Official pytorch implementation of paper "RSTT: Real-time Spatial Temporal Transformer for Space-Time Video Super-Resolution"

[Zhicheng Geng*](https://zhichenggeng.com/), [Luming Liang*](https://scholar.google.com/citations?user=vTgdAS4AAAAJ&hl=en), [Tianyu Ding](https://www.tianyuding.com) and Ilya Zharkov

IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**), 2022

Video: https://www.youtube.com/watch?v=UItUdbLEPHM

Arxiv: https://arxiv.org/abs/2203.14186

## Introduction
Space-time video super-resolution (STVSR) is the task of interpolating videos with both Low Frame Rate (LFR) and Low Resolution (LR) to produce a High-Frame-Rate (HFR) and also High-Resolution (HR) counterpart. The existing methods based on Convolutional Neural Network (CNN) succeed in achieving visually satisfied results while suffer from slow inference speed due to their heavy architectures. 
We propose to resolve this issue by using a spatial-temporal transformer that naturally incorporates the spatial and temporal super resolution modules into a single model. Unlike CNN-based methods, we do not explicitly use separated building blocks for temporal interpolations and spatial super-resolutions; instead, we only use a single end-to-end transformer architecture. Specifically, a reusable dictionary is built by encoders based on the input LFR and LR frames, which is then utilized in the decoder part to synthesize the HFR and HR frames.

## Dataset Preparation
```
python prepare_vimeo.py --path /path/to/vimeo/
```

## Evaluation
### Vid4:
```
python eval_vid4.py --config ./configs/RSTT-S-eval-vid4.yml
```
### Vimeo90k:
```
python eval_vimeo90k.py --config ./configs/RSTT-S-eval-vimeo90k.yml
```
