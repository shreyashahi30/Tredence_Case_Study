# Self-Pruning Neural Network

A PyTorch implementation of a neural network that learns to prune unnecessary weights during training using trainable gate parameters and sparsity regularization.

---

## Overview

Traditional neural network pruning is usually performed after training by removing low-importance weights. This project integrates pruning directly into the training process.

Each learnable connection is assigned a trainable gate value between **0 and 1**:

* Gate near **1** → connection remains active
* Gate near **0** → connection is effectively suppressed

This allows the model to jointly learn:

* CIFAR-10 image classification
* Important parameters
* Sparse network connectivity

The result is a more efficient network with reduced unnecessary weights while maintaining strong predictive performance.

---

## Features

* Custom `PrunableLinear` layer built from scratch in PyTorch
* Learnable gating mechanism for dynamic weight pruning
* End-to-end training with sparsity regularization
* CNN feature extractor for image classification
* Evaluation across multiple regularization strengths (`λ`)
* Accuracy and sparsity tradeoff analysis
* Training curve and gate distribution visualization
* Saved trained model and experiment outputs

---

## Model Architecture

```text
Input Image (32x32x3)
↓
Conv2D (32 filters, 3x3)
↓
ReLU
↓
MaxPool (2x2)
↓
Conv2D (64 filters, 3x3)
↓
ReLU
↓
MaxPool (2x2)
↓
Flatten
↓
PrunableLinear (4096 → 256)
↓
ReLU
↓
PrunableLinear (256 → 128)
↓
ReLU
↓
PrunableLinear (128 → 10)
↓
Output Classes
```

---

## Core Idea

Each custom fully connected layer uses learnable gate scores.

```python
gate = sigmoid(gate_scores)
pruned_weight = weight * gate
```

Where:

* `weight` = trainable layer weights
* `gate_scores` = trainable pruning parameters
* `gate` = values between 0 and 1 after sigmoid activation
* `pruned_weight` = effective weights used during forward pass

Connections with smaller gate values contribute less and can be considered pruned.

---

## Loss Function

The total objective combines classification performance and sparsity pressure:

```python
Total Loss = CrossEntropyLoss + λ * SparsityLoss
```

Where:

```python
SparsityLoss = sum(all gate values)
```

### Why This Works

Minimizing the sum of gate values encourages many gates to move toward zero, creating sparse connectivity while preserving useful parameters.

---

## Dataset

* **Dataset:** CIFAR-10
* **Classes:** 10 image categories
* **Image Size:** 32 × 32 RGB images

---

## Training Configuration

* **Framework:** PyTorch
* **Optimizer:** Adam
* **Learning Rate:** 0.001
* **Batch Size:** 128
* **Epochs:** 15

### Lambda Values Tested

* `1e-05`
* `1e-04`
* `1e-03`

---

## Results

| Lambda  | Test Accuracy (%) | Sparsity Level (%) |
| ------- | ----------------- | ------------------ |
| 0.00001 | 72.44             | 8.42               |
| 0.00010 | 70.14             | 11.21              |
| 0.00100 | 67.41             | 9.47               |

---

## Analysis

### λ = 1e-05

* Highest test accuracy
* Mild pruning pressure
* Best balance between performance and sparsity

### λ = 1e-04

* Increased sparsity
* Slight reduction in accuracy
* Stronger pruning effect

### λ = 1e-03

* More aggressive regularization
* Lower model capacity
* Reduced classification accuracy

### Key Observation

As regularization strength increases:

* Sparsity generally increases
* Accuracy may decrease

This demonstrates the expected tradeoff between compression and predictive performance.

---

## Generated Outputs

The project saves the following outputs:

```text
outputs/
├── best_model.pth
├── results.csv
├── training_curve.png
├── gate_histogram.png
```

### Visualizations

* **training_curve.png** → validation/test accuracy across epochs
* **gate_histogram.png** → final distribution of gate values

---

## Installation

```bash
pip install torch torchvision matplotlib pandas numpy tqdm
```

---

## Usage

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

## Future Improvements

* Hard threshold pruning after training
* Structured neuron or channel pruning
* Quantization for additional compression
* Deployment benchmarking (latency / memory)
* Applying the method to larger architectures such as ResNet

---

## Conclusion

This project demonstrates that neural networks can learn sparse connectivity during training through trainable gates and sparsity-aware optimization.

By combining learnable pruning with classification objectives, the model reduces unnecessary parameters while maintaining strong performance on CIFAR-10.

The best overall balance in this experiment was achieved with:

```text
λ = 1e-05
```

---

## Author

Shreya Shahi
