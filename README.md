# AnySat: An Earth Observation Model for Any Resolutions, Scales, and Modalities

[![python](https://img.shields.io/badge/-Python_3.8+-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.2+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.2+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

[//]: # ([![Paper]&#40;https://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg&#41;]&#40;https://www.nature.com/articles/nature14539&#41;)
[//]: # ([![Conference]&#40;https://img.shields.io/badge/AnyConference-year-4b44ce.svg&#41;]&#40;https://papers.nips.cc/paper/2020&#41;)


Official implementation for
<br>
<br>
[_AnySat: An Earth Observation Model for Any Resolutions, Scales, and Modalities_](https://arxiv.org)
<br>

<p align="center">
  <img src=".media/image.png" alt="AnySat Architecture">
</p>


# Abstract

We introduce AnySat, a novel architecture that exploits the spatial alignment between multiple Earth Observation (EO) modalities to learn expressive multimodal representations without labels. To demonstrate the advantages of combining modalities of different natures, we augment two existing datasets with new modalities. Through extensive experiments on three downstream tasks - forestry, land cover classification, and crop mapping - AnySat demonstrates its ability to learn rich representations in an unsupervised manner, leading to improved performance in both semi- and fully-supervised settings, even when only one modality is available for inference.

<p align="center">
  <img src=".media/teaser.png" alt="AnySat Teaser" width="800">
</p>

# Key Features

- **Multimodal Training**: Trains simultaneously on a collection of 5 multimodal datasets with 11 distinct sensors.
- **Adaptive Patch Encoding**: Processes images with various patch sizes depending on the application, ensuring adaptability to different data scales and resolutions.
- **Efficient Architecture**: Features modality-specific projectors and a shared transformer, with over 75% of learnable parameters shared across all modalities and resolutions.
- **Dense Maps and Linear Probing**: Produces dense maps and allows for linear probing in segmentation tasks without needing complex heads like UPerNet.

# Datasets

## GeoPlex Datasets

GeoPlex is composed of five distinct datasets, each offering a rich combination of data types, modalities, and resolutions. Below is a summary of each dataset:

1. **TreeSatAI-TS**
   - **Description**: Multimodal dataset for tree species identification.
   - **Extent**: 50,381 tiles covering 180 km² with multi-label annotations across 20 classes.
   - **Modalities**: VHR images (0.2 m), Sentinel-2 time series, Sentinel-1 time series.
   - **Tasks**: Tree species classification.

2. **PASTIS-HD**
   - **Description**: Crop mapping dataset with delineated agricultural parcels.
   - **Extent**: 2,433 tiles covering 3986 km² with annotations across 18 crop types.
   - **Modalities**: SPOT6/7 VHR imagery (1.5 m), Sentinel-2 time series, Sentinel-1 time series.
   - **Tasks**: Classification, semantic segmentation, panoptic segmentation.

3. **FLAIR**
   - **Description**: Land cover dataset combining VHR aerial imagery with Sentinel-2 time series.
   - **Extent**: 77,762 tiles covering 815 km² with annotations across 13 land cover classes.
   - **Modalities**: VHR images (0.2 m), Sentinel-2 time series.
   - **Tasks**: Land cover mapping.

4. **PLANTED**
   - **Description**: Global forest dataset for tree species identification.
   - **Extent**: 1,346,662 tiles covering 33,120 km² with annotations across 40 classes.
   - **Modalities**: Sentinel-2, Landsat-7, MODIS, Sentinel-1, ALOS-2.
   - **Tasks**: Tree species classification.

5. **S2NAIP-URBAN**
   - **Description**: Urban dataset with high-resolution imagery and time series data.
   - **Extent**: 515,270 tiles covering 211,063 km² with NAIP, Sentinel-2, Sentinel-1, and Landsat-8/9 data.
   - **Modalities**: NAIP (1.25 m), Sentinel-2 time series, Sentinel-1 time series, Landsat-8/9.
   - **Tasks**: Pretraining only (no official labels).

## External Evaluation Datasets

In addition to GeoPlex, AnySat was evaluated on the following external datasets:

1. **BraDD-S1TS**
   - **Description**: Change detection dataset for deforestation in the Amazon rainforest.
   - **Extent**: 13,234 tiles with Sentinel-1 time series.
   - **Tasks**: Change detection (deforestation segmentation).

2. **SICKLE**
   - **Description**: Multimodal crop mapping dataset from India.
   - **Extent**: 34,848 tiles with Sentinel-1, Sentinel-2, and Landsat-8 time series.
   - **Tasks**: Crop type classification (paddy/non-paddy).

3. **TimeSen2Crop**
   - **Description**: Crop mapping dataset from Slovenia.
   - **Extent**: 1,212,224 single-pixel Sentinel-2 time series.
   - **Tasks**: Crop type classification.


<p align="center">
  <img src=".media/datasets.png" alt="AnySat Datasets">
</p>

# Results

Achieves state-of-the-art or near state-of-the-art performance on multiple datasets and tasks, demonstrating robustness across different sensor configurations and geographical regions.

<p align="center">
  <img src=".media/results.png" alt="AnySat Results">
</p>

# 🚀  Quickstart

# Basic usage of the model

See [huggingface page](https://huggingface.co/gastruc/anysat) for more details.

```python

from anysat import AnySat

anysat = AnySat.from_pretrained("gastruc/anysat")

features = anysat(features, scale=scale)

```

## Repo installation

```bash
# clone project
git clone https://github.com/gastruc/anysat
cd anysat

# [OPTIONAL] create conda environment
conda create -n anysat python=3.9
conda activate anysat

# install requirements
pip install -r requirements.txt

# Create data folder where you can put your datasets
mkdir data
# Create logs folder
mkdir logs
```

# Usage

Every experience of the paper has its own config. Feel free to explore configs/exp folder.

```bash
# Run AnySat pretraining on GeoPlex
python src/train.py exp=GeoPlex_AnySAT

# Run AnySat finetuning on BraDD-S1TS
python src/train.py exp=BraDD_AnySAT_FT

# Run AnySat linear probing on BraDD-S1TS
python src/train.py exp=BraDD_AnySAT_LP
```

# Acknowledgements
- The code is conducted on the same base as [OmniSat](https://github.com/gastruc/OmniSat)
- The JEPA implementation comes from [JEPA](https://github.com/facebookresearch/ijepa)
- The code from Pangaea datasets comes from [Pangaea](https://github.com/VMarsocci/pangaea-bench)
<br>