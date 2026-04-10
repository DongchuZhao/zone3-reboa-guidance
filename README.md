# Cross-Modal Knowledge Distillation for Fluoroscopy-Free Patient-Specific Zone 3 REBOA Guidance

This repository contains the code for a cross-modal teacher-student framework for fluoroscopy-free, patient-specific Zone 3 REBOA guidance.

## Overview

Fluoroscopy-free Zone 3 REBOA requires accurate patient-specific estimation of access-path geometry and insertion depth. Fixed-length and landmark-based approaches are often unreliable because the effective iliofemoral path varies substantially across patients.

To address this problem, we developed a cross-modal knowledge distillation framework:

- **Teacher model:** learns from 3D CT volumes and demographic covariates
- **Student model:** operates only on anterior/lateral 2D body-surface projections and demographic covariates
- **Goal:** predict patient-specific aorto-iliac morphometrics and support accurate Zone 3 REBOA placement without fluoroscopy

## Method Summary

The framework predicts a **1799-dimensional morphometric representation** of the aorto-iliac system, consisting of:

- **7 anatomy-defined vascular segments**
  - Zone 2
  - Zone 3a
  - Zone 3b
  - Right iliac artery
  - Left iliac artery
  - Right common femoral artery (CFA)
  - Left common femoral artery (CFA)

- **64 ordered sample points per segment**
- **4 values per point**
  - x
  - y
  - z
  - local diameter

- **1 scalar segment length per segment**

This representation encodes 3D trajectory, local caliber, and segment length, enabling downstream computation of insertion depth, balloon positioning, and device sizing.

## Inputs

### Teacher model
- 3D CT volume
- demographic covariates:
  - sex
  - age
  - height
  - weight
  - BMI

### Student model
- anterior and lateral body-surface projections
- 8-channel 2D projection tensor derived from:
  - binary silhouette
  - thickness map
  - depth map
  - Euclidean distance transform
- demographic covariates

## Outputs

The model outputs:

- predicted 1799-dimensional morphometric vector
- centerline-related geometric quantities
- insertion-depth-related quantities
- simulated Zone 3 placement assessment

## Training Strategy

Training is performed in two stages:

1. **Teacher pretraining**
   - input: 3D CT + demographics
   - objective: supervised geometric regression

2. **Student distillation**
   - input: 2D body-surface projections + demographics
   - objective: supervised regression + knowledge distillation from the frozen teacher

Key settings reported in the manuscript:

- PyTorch 2.x
- AdamW optimizer
- initial learning rate: 1e-4
- batch size: 16
- ReduceLROnPlateau scheduler
- gradient clipping: 1.0
- mixed precision (AMP)
- patient-level 5-fold cross-validation

## Main Results

Reported manuscript results include:

- **length RMSE:** approximately **11–12 mm**
- **mean Zone 3 malpositioning:** **7.5%**
- **in-zone classification AUC**
  - left: **0.972**
  - right: **0.970**
  - overall: **0.971**

Compared with published formula-based and fixed-length methods, the personalized model substantially reduced simulated malpositioning.

## Data

Raw clinical CT data are **not included** in this repository.

Expected data elements include:

- contrast-enhanced abdominopelvic CT with groin coverage
- de-identified demographic metadata
- vessel segmentation / centerline-derived morphometric data
- anterior and lateral body-surface projection features

Please use de-identified data only.

## Installation

```bash
pip install -r requirements.txt
