# Machine Learning Equations Cheat Sheet

---

## I. Linear Regression

### **Equation:**

```math
y = \mathbf{w}^\top \mathbf{x} + b
```

**Symbols:**

* $\mathbf{w}$: Weight vector
* $\mathbf{x}$: Input feature vector
* $b$: Bias term
* $y$: Predicted output

**Use-case:**
Used in **Linear Regression** for predicting continuous outcomes (e.g., house prices, stock returns).

---

### **Loss Function (Mean Squared Error):**

```math
J(\mathbf{w}) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\mathbf{w}}(x^{(i)}) - y^{(i)})^2
```

**Symbols:**

* $m$: Number of training examples
* $h_{\mathbf{w}}(x)$: Model prediction
* $y^{(i)}$: True label

**Use-case:**
Minimized to fit the linear regression model to training data.

---

## II. Logistic Regression

### **Sigmoid Activation:**

```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

**Use-case:**
Used to map predictions to probabilities (between 0 and 1) in **binary classification**.

---

### **Log Loss (Binary Cross-Entropy):**

```math
J(\mathbf{w}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
```

**Symbols:**

* $\hat{y}^{(i)}$: Predicted probability
* $y^{(i)}$: Ground truth label

**Use-case:**
Loss function for **logistic regression** and **binary classifiers**.

---

## III. Optimization

### **Gradient Descent:**

```math
\theta := \theta - \eta \nabla J(\theta)
```

**Symbols:**

* $\theta$: Parameters (weights/bias)
* $\eta$: Learning rate
* $\nabla J(\theta)$: Gradient of cost function

**Use-case:**
Iteratively updates model parameters to minimize the loss function.

---

## IV. Regularization

### **L2 Regularization (Ridge):**

```math
J(\mathbf{w}) = \text{Loss} + \lambda \|\mathbf{w}\|_2^2
```

### **L1 Regularization (Lasso):**

```math
J(\mathbf{w}) = \text{Loss} + \lambda \|\mathbf{w}\|_1
```

**Symbols:**

* $\lambda$: Regularization strength
* $\|\mathbf{w}\|$: Norm of weight vector

**Use-case:**
Prevents overfitting; L1 also enables **feature selection**.

---

## V. Neural Networks

### **Neuron Computation:**

```math
z = \mathbf{w}^\top \mathbf{x} + b, \quad a = \phi(z)
```

**Activation Functions:**

* Sigmoid:

```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

* ReLU:

```math
\text{ReLU}(z) = \max(0, z)
```

* Tanh:

```math
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
```

**Use-case:**
Basic unit in **deep learning models** for nonlinear transformation.

---

### **Softmax Function:**

```math
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
```

**Use-case:**
Used in **multi-class classification** (e.g., image classification with 10 classes).

---

## VI. Convolutional Neural Networks

### **2D Convolution:**

```math
(I * K)(x, y) = \sum_m \sum_n I(x+m, y+n) \cdot K(m, n)
```

**Symbols:**

* $I$: Input image matrix
* $K$: Kernel/filter
* $*$: Convolution operation

**Use-case:**
Used in CNNs for **feature extraction from images**.

---

## VII. Recurrent Neural Networks

### **RNN Hidden State Update:**

```math
h_t = \phi(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
```

**Use-case:**
Used for **sequence modeling** (e.g., text, time series).

---

## VIII. Support Vector Machines

### **SVM Optimization Problem:**

```math
\min_{\mathbf{w}} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{s.t. } y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1
```

**Use-case:**
Finds the **maximum margin hyperplane** for classification.

---

## IX. Decision Trees

### **Gini Impurity:**

```math
\text{Gini}(D) = 1 - \sum_{i=1}^{C} p_i^2
```

### **Entropy:**

```math
\text{Entropy}(D) = - \sum_{i=1}^{C} p_i \log_2(p_i)
```

**Use-case:**
Used to choose the best feature in **decision tree splits**.

---

## X. Naive Bayes

### **Bayes' Theorem:**

```math
P(C_k | x) = \frac{P(x | C_k) P(C_k)}{P(x)}
```

**Use-case:**
Classify documents/emails using **probabilistic models**.

---

## XI. PCA

### **Covariance Matrix:**

```math
\Sigma = \frac{1}{n} X^T X
```

### **Eigen Decomposition:**

```math
\Sigma v = \lambda v
```

**Use-case:**
Used in **dimensionality reduction** for high-dimensional data.

---

## XII. Attention Mechanism

### **Scaled Dot-Product Attention:**

```math
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V
```

**Symbols:**

* $Q$: Query matrix
* $K$: Key matrix
* $V$: Value matrix
* $d_k$: Key dimension

**Use-case:**
Core of **transformer models** like BERT and GPT.