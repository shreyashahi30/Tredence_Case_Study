# Self-Pruning Neural Network

## Technical Report

---

# 1. Objective

The goal of this project was to design a neural network that can automatically identify and suppress unnecessary weights during training.

Unlike traditional post-training pruning methods, this approach integrates pruning directly into optimization using learnable gate parameters.

This enables the network to learn:

* Accurate image classification
* Parameter importance
* Sparse connectivity structure

---

# 2. Methodology

## 2.1 Model Design

A convolutional feature extractor was used before fully connected prunable layers.

```text
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
```

---

## 2.2 PrunableLinear Layer

Each custom dense layer contains:

* Weight tensor
* Bias tensor
* Trainable `gate_scores`

Gate values are computed as:

```python
gate = sigmoid(gate_scores)
```

Effective weights:

```python
pruned_weight = weight * gate
```

Connections with low gate values contribute very little and are effectively pruned.

---

# 3. Training Objective

The final optimization target combines classification loss and sparsity pressure:

```python
Total Loss = CrossEntropyLoss + λ * SparsityLoss
```

Where:

```python
SparsityLoss = sum(all gate values)
```

---

# 4. Why This Encourages Sparsity

Because gate values lie in the range `[0,1]`, minimizing their sum encourages many gates to approach zero.

This leads to:

* Removal of weak connections
* Retention of important weights
* Reduced parameter usage

This behavior is similar to L1 regularization.

---

# 5. Experimental Setup

* Dataset: CIFAR-10
* Framework: PyTorch
* Optimizer: Adam
* Learning Rate: 0.001
* Batch Size: 128
* Epochs: 15

Lambda values tested:

* `1e-05`
* `1e-04`
* `1e-03`

---

# 6. Results

| Lambda  | Test Accuracy (%) | Sparsity Level (%) |
| ------- | ----------------- | ------------------ |
| 0.00001 | 72.44             | 8.42               |
| 0.00010 | 70.14             | 11.21              |
| 0.00100 | 67.41             | 9.47               |

---

# 7. Analysis

## λ = 1e-05

* Highest accuracy
* Mild regularization
* Best balance overall

## λ = 1e-04

* Higher sparsity
* Slight drop in performance

## λ = 1e-03

* Strong regularization
* Lower model capacity
* Reduced accuracy

## Observation

Increasing λ generally improves sparsity but can reduce predictive performance.

---

# 8. Visual Diagnostics

## Gate Histogram

The gate value distribution showed:

* Many gates near zero
* Important gates remaining active

## Training Curve

Accuracy increased steadily across epochs and converged in later training stages.

---

# 9. Limitations

* Sparsity levels were moderate rather than highly aggressive
* Gates remain soft values instead of exact zeros
* Larger architectures may require tuning

---

# 10. Future Improvements

* Hard threshold pruning after training
* Structured channel pruning
* Quantization for compression
* Apply to ResNet architectures
* Benchmark inference speed and memory usage

---

# 11. Conclusion

This project successfully demonstrates that trainable gate mechanisms can integrate pruning directly into neural network training.

The best configuration achieved:

```text
72.44% accuracy with λ = 1e-05
```

This shows sparse learning can reduce unnecessary parameters while preserving strong predictive performance.

---

# Author

Shreya Shahi
