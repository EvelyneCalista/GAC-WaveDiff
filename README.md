<div align="center">

# GAC-WaveDiff

### Geometry-Aware Conditional 3D Wavelet Diffusion for Pseudo-Healthy Brain MRI Inpainting

**Evelyne Calista<sup>1</sup> &nbsp; · &nbsp;
Yong-Sheng Chen<sup>1</sup> &nbsp; · &nbsp;
**

<sup>1</sup>National Yang Ming Chiao Tung University, Taiwan
</div>

## Overview

This repository is for Geometry-Aware Conditional 3D Wavelet Diffusion for Pseudo-Healthy Brain MRI Inpainting, in part of BraTS 2026 Inpainting Challenge

## Installation
Install dependencies:
```bash
conda env create -f environment.yml
```

## Data Preparation

Describe where the dataset can be obtained and how it should be organized.

```text
data/
├── train/
├── validation/
└── test/
```

## Training

```bash
bash run.sh
```

## Inference

```bash
bash run_inference.sh
```



## Citation

When using this repository, please cite:


## Acknowledgements
This repository uses dataset [BraTS 2026 Inpainting Challenge](https://challenges.synapse.org/Challenges/DetailsPage/Task4?id=syn74274097).
Thanks to Durrer et al. for releasing their code [fastWDM3D](https://github.com/AliciaDurrer/fastWDM3D).



## License

This project is licensed under the [LICENSE NAME](LICENSE).



```
```
