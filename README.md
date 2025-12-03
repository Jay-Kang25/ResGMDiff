# ResGMDiff:Residual-Guided Multiscale Diffusion Model for Low-Dose CT Denoising
This is the official implementation of the paper "ResGMDiff:Residual-Guided Multiscale Diffusion Model for Low-Dose CT Denoising"

# Requirements
To install requirements: (If an error occurs, you may need to install the packages one by one.)
```bash
conda env create -f install.yaml
```

# Data Preparation
The AAPM-Mayo dataset can be found from: [Mayo 2016](https://ctcicblog.mayo.edu/).\
The QIN Lung CT dataset can be found from:[QIN Lung CT](https://www.cancerimagingarchive.net/collection/qin-lung-ct/)

#Training and Inference
To train ResGMDiff, run this command:
```bash
PYTHONPATH=$(pwd) python ResGMDiff-main/train.py
```
To evaluate LDCT image, run:
```bash
PYTHONPATH=$(pwd) python ResGMDiff-main/test.py
```
