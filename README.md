# Learned Reduced-Rank Sharpening Method (LRRSM)

This repository contains the official implementation of ["A Learned Reduced-Rank Sharpening Method for Multiresolution Satellite Imagery"](https://doi.org/10.3390/rs17030432) [Remote Sensing, 2025].

## Overview

LRRSM is an unsupervised single-image sharpening method designed specifically for enhancing Sentinel-2 imagery. The method uniquely combines traditional model-based approaches with neural network optimization techniques, requiring no external training data beyond the image being processed. Through its compact, interpretable network model, LRRSM achieves fast training times while adapting to different input images without extensive parameter tuning. The method demonstrates consistent performance across both 20m and 60m bands and can be extended to multi-sensor fusion tasks, including Landsat 8 data.

## Installation

LRRSM requires Python 3.11 or later and depends on PyTorch 2.4+, PyTorch Lightning, NumPy, h5py, and einops (using GPU aware implementations is highly recommended). Experiment tracking through wandb is supported but optional. To install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

For Sentinel-2 image sharpening, run:
```bash
python S2_LRRSM.py --dataset your_dataset --datadir path_to_data --lamb 0.0025 --train --test
```

For Sentinel-2 and Landsat 8 fusion tasks, use:
```bash
python S2L8_LRRSM.py --dataset your_dataset --datadir path_to_data --lamb 0.001 --train --test
```

The method can be configured through several parameters. The regularization strength `lamb` defaults to 0.0025 for S2 sharpening and 0.001 for S2/L8 fusion. The learning rate `lr` defaults to 8e-3. 
Training typically runs for 20 epochs with an initial PReLU activation slope of 0.75.

## Model Architecture

The LRRSM architecture combines a customized U-Net for generating low-rank spatial representations with a learnable linear transformation to the final image. The model employs resolution-matched pooling layers along with instance normalization and PReLU activation. A specialized loss function operates in the low-rank subspace to maintain image fidelity while enhancing spatial resolution.

## Citation

```bibtex
@article{armannsson2025learned,
  title={A Learned Reduced-Rank Sharpening Method for Multiresolution Satellite Imagery},
  author={Armannsson, Sveinn E. and Ulfarsson, Magnus O. and Sigurdsson, Jakob},
  journal={Remote Sensing},
  volume={17},
  number={3},
  pages={432},
  year={2025},
  publisher={MDPI}
}
```

## License and Contact

For questions about the code or paper, please open an issue or contact the authors through email: Sveinn E. Armannsson (sea2@hi.is).