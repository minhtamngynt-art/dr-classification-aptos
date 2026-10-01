# Diabetic Retinopathy Classification (APTOS 2019)

Deep learning pipeline for multi-class Diabetic Retinopathy (DR) grading on the APTOS 2019 Blindness Detection dataset using DenseNet121 and ResNet50.

## Overview

Diabetic Retinopathy (DR) severity classification into 5 stages (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR). 
The project includes image preprocessing (black border cropping), fine-tuning pretrained CNN backbones, and custom threshold optimization for imbalanced classes.

## Dataset Setup

Download the APTOS 2019 dataset from Kaggle and place it under `./data/aptos2019-blindness-detection/`:

```text
data/aptos2019-blindness-detection/
├── train.csv
└── train_images/
    ├── 000c1434d8d7.png
    └── ...
```

## Requirements

```bash
pip install tensorflow opencv-python pandas numpy matplotlib seaborn scikit-learn
```

## Workflow

1. **Preprocessing**: Automatically crops uninformative black borders around fundus images and resizes inputs to 224x224.
2. **Class Imbalance Handling**: Utilizes weighted cross-entropy loss with increased penalty weights for severe/proliferative stages.
3. **Training & Thresholding**:
   - Stage 1: Freeze base model, train classification head (Adam, lr=1e-2).
   - Stage 2: Unfreeze base model, fine-tune end-to-end (Adam, lr=1e-4).
   - Dynamic threshold optimization per class using Precision-Recall curves.

## Results

Comparison of evaluation metrics across models:

| Model | Accuracy | QWK (Quadratic Weighted Kappa) | Macro AUC |
| :--- | :---: | :---: | :---: |
| DenseNet121 | 0.812 | 0.884 | 0.941 |
| ResNet50 | 0.798 | 0.867 | 0.928 |

## Usage

Open and execute `dr_classification_aptos.ipynb` in Jupyter or VS Code to run preprocessing, model fine-tuning, and evaluation.
