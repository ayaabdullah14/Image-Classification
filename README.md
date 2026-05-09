# ENCS3340 – Project #2: Image Classification
## Comparative Study: Decision Tree, Naive Bayes, and Feedforward Neural Networks



**Team Members:**
- Aya Abdullah
- Lana Daramna 

**Instructor:** Samah Alaydi | **Section:** 2
**Submission Date:** June 24, 2025

---

## Overview

This project compares three machine learning classifiers for animal image classification:

| Model | Accuracy |
|---|---|
| Naive Bayes (NB) | 47% |
| Decision Tree (DT) | 59% |
| MLP – 1 Hidden Layer | **79%** |
| MLP – 2 Hidden Layers | 78% |

---

## Dataset

- **Total images:** 1,500 (sourced) / 1,512 (with augmentation)
- **Classes:** Chicken, Horse, Sheep — 500 images each
- **Image size:** Resized to 32×32 pixels, then flattened into feature vectors
- **Split:** 400 training / 100 testing per class → 300 total test samples

```
dataset/
├── chicken/    # 500 images
├── horse/      # 500 images
└── sheep/      # 500 images
```

---

## Models

### 1. Naive Bayes
A probabilistic classifier based on Bayes' Theorem, assuming conditional independence between features.

- **Features:** Mean and variance of pixel intensities per image
- **Accuracy:** 47%
- **Strengths:** Fast, simple, good baseline
- **Weakness:** Statistical summary loses spatial detail; sheep recall only 15%

### 2. Decision Tree
A non-parametric tree-based classifier that recursively splits the dataset by feature thresholds.

- **Features:** Raw pixel intensity values
- **Max depth:** 15 (to control overfitting)
- **Accuracy:** 59%
- **Strengths:** Interpretable; visual decision paths; improved sheep recall (61%)
- **Weakness:** Axis-aligned splits limit expressiveness; prone to overfitting

### 3. Feedforward Neural Network (MLP)
A multi-layer perceptron implemented with Scikit-learn's `MLPClassifier`.

- **Preprocessing:** Pixel values normalized to [0, 1], then standardized (zero mean, unit variance)
- **Activation:** ReLU
- **1 Hidden Layer:** 79% accuracy — best overall performance
- **2 Hidden Layers:** 78% accuracy — slight drop, likely due to overfitting on limited data
- **Strengths:** Learns non-linear patterns; strong precision/recall balance across all classes

---

## Results Summary

| Model | Precision | Recall | F1-Score | Accuracy |
|---|---|---|---|---|
| Naive Bayes | ~50% | ~47% | ~46% | 47% |
| Decision Tree | ~58–62% | ~56–62% | ~58% | 59% |
| MLP (1 layer) | 73–85% | 75–83% | ~79% | **79%** |
| MLP (2 layers) | ~73–83% | ~74–82% | ~78% | 78% |

The MLP with one hidden layer achieved the best performance across all metrics.

---

## Requirements

```bash
pip install scikit-learn numpy pillow matplotlib
```

**Python version:** 3.x

**Libraries used:**
- `scikit-learn` — all three classifiers (`GaussianNB`, `DecisionTreeClassifier`, `MLPClassifier`)
- `numpy` — image array manipulation
- `Pillow` — image loading and resizing
- `matplotlib` — decision tree visualization and confusion matrices

---

## Running the Code

```bash
python classify.py
```

The script will:
1. Load and resize all images to 32×32
2. Extract features and split into train/test sets
3. Train all three models
4. Print classification reports (accuracy, precision, recall, F1)
5. Display confusion matrices and the decision tree visualization

---

## Key Findings

- **Naive Bayes** is too simple for image data; mean/variance features discard spatial structure.
- **Decision Tree** improves on NB by using raw pixels but is constrained by axis-aligned splits.
- **MLP (1 hidden layer)** is the best model — it learns non-linear features and generalizes well.
- Adding a second hidden layer did not help here; more data or tuning would be needed to benefit from added depth.

---

## Future Improvements

- Use **Convolutional Neural Networks (CNNs)** for spatially-aware feature extraction
- Apply **data augmentation** (flipping, rotation, brightness) to expand the training set
- Increase image resolution beyond 32×32 to preserve more detail
- Tune MLP hyperparameters (learning rate, hidden layer size, regularization)

---

## References

- Scikit-learn documentation: https://scikit-learn.org
- Naive Bayes: Bayes' Theorem and probabilistic classification
- Decision Trees: CART algorithm
- MLP: Multi-Layer Perceptron with backpropagation
