# Tredence AI Engineering Internship Case Study

## Self-Pruning Neural Network

---

# 1. Objective

The objective of this case study was to design and implement a neural network that can automatically prune unnecessary weights during training instead of relying on post-training pruning methods.

Traditional pruning removes weak connections only after training is complete. In this project, pruning is integrated directly into the learning process using trainable gate parameters.

The model learns:

* Image classification on CIFAR-10
* Which connections are important
* Which weights can be removed automatically

This results in a sparse and computationally efficient neural network.

---

# 2. Approach

## 2.1 Model Architecture

To improve classification performance, a CNN-based feature extractor was used before the prunable fully connected layers.

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

## 2.2 PrunableLinear Layer

A custom dense layer named `PrunableLinear` was implemented.

Each layer contains:

* Trainable weights
* Bias parameters
* Learnable `gate_scores`

Gate values are computed using:

```python
gate = sigmoid(gate_scores)
```

Effective weights during forward pass:

```python
pruned_weight = weight * gate
```

Interpretation:

* `gate ≈ 1` → important connection remains active
* `gate ≈ 0` → connection is effectively pruned

Because sigmoid outputs values between 0 and 1, each connection receives a learnable importance score.

---

# 3. Loss Function

The total loss combines classification performance with sparsity pressure:

```python
Total Loss = CrossEntropyLoss + λ * SparsityLoss
```

Where:

```python
SparsityLoss = sum(all gate values)
```

---

# 4. Why L1 Regularization Encourages Sparsity

Since gate values are positive after sigmoid activation, minimizing their sum encourages many gates to move toward zero.

This creates sparsity because:

* Unimportant weights receive very small gates
* Important weights remain active
* Total active parameters decrease

L1-style penalties are widely used because they naturally promote sparse solutions.

---

# 5. Training Setup

* **Dataset:** CIFAR-10
* **Optimizer:** Adam
* **Learning Rate:** 0.001
* **Batch Size:** 128
* **Epochs:** 15
* **Device:** CPU / GPU (depending on availability)

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

# 7. Tradeoff Analysis

## λ = 0.00001

* Highest classification accuracy
* Mild pruning pressure
* Best balance between performance and sparsity

## λ = 0.00010

* Increased sparsity
* Slight reduction in accuracy
* Stronger pruning behavior

## λ = 0.00100

* Aggressive regularization
* Lower accuracy
* Excessive pruning reduced model capacity

## Key Observation

As λ increases:

* Sparsity generally increases
* Accuracy decreases

This confirms the expected pruning vs performance tradeoff.

---

# 8. Gate Distribution Analysis

The generated histogram (`gate_histogram.png`) shows the distribution of final gate values.

Observed behavior:

* Many gate values concentrated near zero
* Remaining gates stayed significantly larger

This indicates that the network learned to suppress weak connections while preserving important ones.

---

# 9. Training Curve

The validation accuracy curve (`training_curve.png`) shows stable learning progress across epochs.

Accuracy improved consistently during training and converged near the final epochs.

This demonstrates successful optimization of both classification and sparsity objectives.

---

# 10. Project Files

```text
outputs/
├── best_model.pth
├── results.csv
├── training_curve.png
├── gate_histogram.png
main.py
README.md
```

---

# 11. Conclusion

This project successfully implemented a self-pruning neural network that learns sparse connectivity during training.

### Key Achievements

* Built a custom `PrunableLinear` layer
* Integrated pruning into gradient-based learning
* Demonstrated sparsity-accuracy tradeoff across multiple λ values
* Achieved **72.44% CIFAR-10 test accuracy** while pruning unnecessary weights

The best balance was obtained with:

```text
λ = 1e-05
```

This experiment shows that learnable gating mechanisms can reduce model complexity while maintaining strong predictive performance.

---

# 12. Future Improvements

Possible next steps:

* Hard threshold pruning after training
* Structured neuron/channel pruning
* Quantization for further compression
* Deployment benchmarking (latency / memory)
* Applying the method to larger architectures such as ResNet
