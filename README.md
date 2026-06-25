# Regularized Logistic Regression for Microchip Quality Assurance

This repository contains a modular Python implementation of **Regularized Logistic Regression** to automate the Quality Assurance (QA) process for microchip manufacturing. Using test metrics from two distinct automated diagnostics, the model learns a non-linear decision boundary to classify microchips as *Accepted* or *Rejected*.

---

## 📌 Problem Statement

As a product manager at a fabrication plant, you want to determine whether microchips should be accepted or rejected based on the results of two diagnostic tests. 

Historical test data reveals that the functional microchips form a central cluster, while defective chips lie on the outer edges. Because the dataset is **non-linearly separable**, a standard linear decision boundary will fail. This project showcases how **feature mapping** combined with **$L_2$ regularization (Ridge)** enables a logistic regression model to learn complex, non-linear boundaries while preventing overfitting.

---

## 📊 Dataset & Feature Mapping

The dataset (`ex2data2.txt`) contains 118 historical microchip logs with three columns: `Test 1`, `Test 2`, and `Status` (1 = Accepted, 0 = Rejected).

### Feature Mapping to Polynomials
To capture the circular/elliptical shape of the dataset, the 2 original continuous features are mapped into all polynomial terms up to the $6^{\text{th}}$ degree:

$$\text{map\_feature}(X_1, X_2) = \begin{bmatrix} X_1 & X_2 & X_1^2 & X_1X_2 & X_2^2 & X_1^3 & \dots & X_1X_2^5 & X_2^6 \end{bmatrix}^T$$

This transforms the 2-dimensional input space into a **28-dimensional feature space**, giving the logistic model the freedom to construct an intricate polynomial contour.

---

## 🧮 Mathematical Formulation

### 1. Regularized Cost Function
To penalize overly complex parameters ($\mathbf{w}$) resulting from the 28 polynomial degrees, a regularization parameter ($\lambda$) is introduced to the binary cross-entropy loss (excluding the bias term $b$):

$$J(\mathbf{w},b) = \frac{1}{m} \sum_{i=0}^{m-1} \left[ -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \right] + \frac{\lambda}{2m} \sum_{j=0}^{n-1} w_j^2$$

Where the model prediction utilizes the sigmoid activation function:
$$f_{\mathbf{w},b}(\mathbf{x}) = g(\mathbf{w} \cdot \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}}$$

### 2. Gradient Updates
During simultaneously updates via Gradient Descent, the partial derivatives include the regularized penalty for weight matrices:

$$\frac{\partial J(\mathbf{w},b)}{\partial b} = \frac{1}{m} \sum_{i=0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})$$

$$\frac{\partial J(\mathbf{w},b)}{\partial w_j} = \left( \frac{1}{m} \sum_{i=0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m} w_j \quad \text{for } j \geq 0$$

---

## 📉 Results and Model Evaluation

- **Training Hyperparameters:**
  - Learning Rate ($\alpha$): `0.01`
  - Regularization Term ($\lambda$): `0.01`
  - Iterations: `10,000`
- **Convergence Summary:** Binary cross-entropy loss minimized steadily from `0.72` down to `0.45`.
- **Classification Accuracy:** The final model achieves a classification accuracy of **82.20%** on the training data, successfully plotting a perfect elliptical decision boundary that generalizes well to the underlying dataset distribution.

---

## 📂 Repository Layout

```bash
├── README.md           # Professional project documentation
├── code.ipynb          # Interactive Jupyter notebook tracking implementation steps
├── ex2data2.txt        # Raw QA historical dataset (comma-separated entries)
├── plot.png            # Final Decision boundary curve
└── utils.py            # Helper library (Data parser, feature mapper, and plot generator)
