# Tredence AI Engineering Internship Case Study

## Self-Pruning Neural Network

A PyTorch implementation of a neural network that learns to prune unnecessary weights during training using trainable gate parameters and sparsity regularization.

---

## Project Summary

Traditional pruning removes weak neural network connections after training.
This project integrates pruning directly into the training process.

Each connection receives a learnable gate value between **0 and 1**:

* Gate near **1** → connection remains active
* Gate near **0** → connection is effectively removed

This enables the model to jointly learn:

* CIFAR-10 image classification
* Important parameters
* Sparse network connectivity

---

## Model Architecture

```text id="jsk1po"
Input Image (32x32x3)
↓
Conv2D (32 filters)
↓
ReLU
↓
MaxPool
↓
Conv2D (64 filters)
↓
ReLU
↓
MaxPool
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

Each custom dense layer uses:

```python id="59w0wt"
gate = sigmoid(gate_scores)
pruned_weight = weight * gate
```

The effective weights are scaled by trainable gates, allowing the model to suppress unnecessary connections automatically.

---

## Loss Function

```python id="ypo52e"
Total Loss = CrossEntropyLoss + λ * SparsityLoss
```

Where:

```python id="0c5l4k"
SparsityLoss = sum(all gate values)
```

This encourages sparse connectivity.

---

## Training Configuration

* Dataset: CIFAR-10
* Optimizer: Adam
* Learning Rate: 0.001
* Batch Size: 128
* Epochs: 15

Lambda values tested:

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

## Key Findings

* Lower λ preserves accuracy
* Higher λ increases pruning pressure
* Excessive regularization lowers model capacity

Best tradeoff achieved at:

```text id="r8v3r3"
λ = 1e-05
```

---

## Files

* `Report.md` → Detailed case study report
* `Tredence.py` → Full source code implementation

---

## Conclusion

This project successfully demonstrates that neural networks can learn sparse connectivity during training using trainable gates and regularization, reducing unnecessary parameters while maintaining strong classification performance.

---

## Author

Shreya Shahi
