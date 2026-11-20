# MSAU-Net-AAA

Source code for three-dimensional aortic segmentation using a
multi-scale attention U-Net (MSAU-Net) and complementary quantitative
morphological analysis of abdominal aortic aneurysms (AAA).

## Overview

This project investigates multi-scale feature extraction and
attention-guided feature refinement for 3D aortic segmentation.

The study includes:
- Segmentation experiments with MSAU-Net, 3D U-Net, and Attention U-Net.
- Five-fold cross-validation and component ablation experiments.
- Supplementary automated localization-to-segmentation experiments.
- Morphological analysis using aortic surface meshes and centerlines.

## Data

Raw imaging data, segmentation annotations, surface meshes, and
centerlines are not distributed in this repository.

Segmentation experiments use AortaSeg-60. The complementary
morphological analysis uses surface meshes and centerlines from
AAA-100. Users must obtain these datasets separately from their
original providers and comply with the applicable terms of use.

The AAA-100 morphological analysis is independent of the segmentation
evaluation and does not constitute external validation of MSAU-Net
segmentation performance.

## Experimental Settings

The primary segmentation experiments use reference-mask-guided
region-of-interest (ROI) cropping. Supplementary experiments examine
automated localization followed by fine segmentation from complete
image volumes.

These experimental settings should be distinguished when interpreting
or reproducing the results.

## Morphological Analysis

The analysis includes cross-sectional and whole-model geometric
descriptors, such as:
- Maximum cross-sectional Feret diameter
- Cross-sectional area and equivalent diameter
- Volume
- Centerline length and tortuosity
- Sphericity

The composite morphological score is exploratory and has not been
validated as a predictor of clinical outcomes.

## Software and Execution

The segmentation implementation uses Python and PyTorch.
Additional dependencies depend on the preprocessing, evaluation,
and visualization scripts.

Some scripts may contain paths specific to the original server.
Review and update dataset paths, output directories, and checkpoint
paths before execution.

Use the scripts corresponding to the experimental setting being
reproduced. The presence of a script in this repository does not
indicate that all scripts should be run sequentially.

## Model Weights

Trained model weights are not included in this repository.

## Intended Use

This code is provided for research purposes. It is not a clinically
validated diagnostic or treatment-planning system.
