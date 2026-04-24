# Self-Pruning Neural Network

A PyTorch implementation of a neural network that learns sparse connectivity during training using trainable gate parameters and sparsity regularization.
Notebook can be opened directly in Google Colab.
---

## Overview

Traditional neural network pruning is usually performed after training by removing low-importance weights. This project integrates pruning directly into the learning process.

Each connection is assigned a learnable gate value between **0 and 1**:

* Gate near **1** → connection remains active
* Gate near **0** → connection is effectively suppressed

This allows the model to jointly learn:

* CIFAR-10 image classification
* Important parameters
* Sparse network structure

---

## Features

* Custom `PrunableLinear` layer in PyTorch
* Learnable gating mechanism
* Dynamic weight suppression during training
* Sparsity-aware loss function
* CNN + prunable dense architecture
* Lambda tradeoff experiments
* Saved model, metrics, and plots

---

## Architecture

```text
Input (32x32x3)
↓
Conv2D(32) → ReLU → MaxPool
↓
Conv2D(64) → ReLU → MaxPool
↓
Flatten
↓
PrunableLinear(4096→256) → ReLU
↓
PrunableLinear(256→128) → ReLU
↓
PrunableLinear(128→10)
```

---

## Core Formulation

```python
gate = sigmoid(gate_scores)
pruned_weight = weight * gate
```

```python
Total Loss = CrossEntropyLoss + λ * SparsityLoss
```

---

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
| ------ | ------------ | ------------ |
| 1e-05  | 72.44        | 8.42         |
| 1e-04  | 70.14        | 11.21        |
| 1e-03  | 67.41        | 9.47         |

Best tradeoff achieved at:

```text
λ = 1e-05
```

---

## Installation

```bash
pip install torch torchvision matplotlib pandas numpy tqdm
```

---

## Run Project

```bash
python main.py
```

---

## Project Structure

```text
project/
├── main.py
├── README.md
├── Report.md
└── outputs/
```

---

## Outputs

* `best_model.pth`
* `results.csv`
* `training_curve.png`
* `gate_histogram.png`

---

## Author

Shreya Shahi
