# Data Science Interview Tasks

**Author:** Supreeth Mysore

---

## Overview

This repository contains solutions to four data science tasks demonstrating statistical analysis, machine learning, and time series processing capabilities.

---

## Task 1: Reverse Engineering the DataAPI

### Objective
Reverse-engineer the underlying distribution of 20 variables provided by the DataAPI.get() function.

### Methodology
1. **Extensive Sampling:** Collected 100,000 samples from the API treating it as a black box
2. **Marginal Statistics:** Computed mean, std, skewness, and kurtosis for each variable
3. **Correlation Analysis:** Identified relationships between variables
4. **PCA Analysis:** Determined latent dimensionality
5. **ICA Analysis:** Extracted independent components

### Findings

**Latent Structure:**
- 7 latent Gaussian dimensions explain 97.85% of variance
- Strong correlations found: x_4 <-> x_15 (rho=0.78), x_14 <-> x_15 (rho=-0.57)

**Discovered Model:**
```
Latent Space: z = [z_0, ..., z_6] where each z_i ~ N(0, 1)

Observation Model:
  x_i = sum_{k=0}^{12} A[k,i] * f[J[k]](W[k,:,i] dot z + B[k,i] + S[k,i])

Activation Functions f[j]:
  - sin(x), cos(x), tanh(x)
  - sigmoid(x) = 1/(1+exp(-x))
  - x * tanh(x)
  - sin(x^2)
  - cosh(x) - 1
  - exp(-x^2)
  - tanh(x) + 0.1*sin(3x)
  - cos(x) * sin(x)
```

---

## Task 1b: Model Scaling Laws

### Objective
Characterize scaling laws for a model family with respect to data size, model size, and compute.

### Methodology
- **Model Family:** Gradient Boosting Regressor
- **Task:** Predict x_0 from x_1...x_19
- **Variables:** Data size (500 to 30,000), Model size (20 to 100 trees)

### Results

| Data Size | Trees | MSE    | Time (s) |
|-----------|-------|--------|----------|
| 500       | 50    | 0.6716 | 0.34     |
| 2,000     | 50    | 0.5006 | 0.73     |
| 10,000    | 50    | 0.7046 | 5.20     |
| 30,000    | 50    | 0.6519 | 15.05    |

### Key Insights
1. MSE does not consistently decrease with more data
2. Scaling laws break down when noise dominates signal
3. The DataAPI generates i.i.d. samples with limited learnable structure
4. Epoch-to-accuracy and FLOP-to-accuracy do NOT share the same distribution due to varying compute efficiency across training phases

---

## Task 2: Fast Multivariate Distribution Sampler

### Objective
Create a fast sampler for the Iris dataset with unconditional and conditional sampling capabilities.

### Algorithm

**Multivariate Gaussian with Cholesky Decomposition:**

```python
# Fit: O(n*d + d^3)
mean = X.mean(axis=0)
cov = (X - mean).T @ (X - mean) / n
chol = cholesky(cov)

# Sample: O(n_samples * d^2)
samples = mean + random_normal(n_samples, d) @ chol.T

# Conditional Sample using Gaussian conditionals:
# For partition [x_1 (known) | x_2 (unknown)]:
#   mu_2|1 = mu_2 + Sigma_21 @ Sigma_11^{-1} @ (x_1 - mu_1)
#   Sigma_2|1 = Sigma_22 - Sigma_21 @ Sigma_11^{-1} @ Sigma_12
```

### Performance

| Metric           | Result    | Target   | Status |
|------------------|-----------|----------|--------|
| fit_time         | 0.10 ms   | < 0.5 ms | PASS   |
| sample_time      | 1.34 ms   | < 0.5 ms | -      |
| cond_sample_time | 0.86 ms   | < 1.0 ms | PASS   |
| Mean error       | 0.0252    | -        | -      |
| Covariance error | 0.0338    | -        | -      |

---

## Task 3: Cluster Count Comparison for K-Means

### Objective
Compare methods for determining optimal number of clusters and provide scaling recommendations.

### Methods Evaluated

1. **Elbow Method (Inertia/WCSS):** Plot inertia vs k, find elbow point
2. **Silhouette Score:** Measure cluster cohesion vs separation [-1, 1]
3. **Calinski-Harabasz Index:** Ratio of between-cluster to within-cluster variance
4. **Davies-Bouldin Index:** Average similarity between clusters (lower is better)

### Results on Iris Dataset

| Method            | Time (s) | Memory (MB) | Optimal k | Correct |
|-------------------|----------|-------------|-----------|---------|
| Elbow             | 1.39     | 0.22        | 3         | YES     |
| Silhouette        | 0.88     | 0.43        | 2         | NO      |
| Calinski-Harabasz | 1.16     | 0.06        | 2         | NO      |
| Davies-Bouldin    | 0.89     | 0.09        | 2         | NO      |

**Note:** Many methods prefer k=2 because Iris versicolor and virginica species overlap significantly.

### Scaling Analysis

| Method            | Time Complexity | Memory Complexity | Scalable |
|-------------------|-----------------|-------------------|----------|
| Elbow (Inertia)   | O(N * K * D)    | O(N * D)          | YES      |
| Silhouette        | O(N^2 * K)      | O(N^2)            | NO       |
| Calinski-Harabasz | O(N * K * D)    | O(N * D)          | YES      |
| Davies-Bouldin    | O(N * K * D)    | O(N * D)          | YES      |

### Recommendation
For large-scale clustering (N > 100k samples):
- **USE:** Calinski-Harabasz or Davies-Bouldin (linear complexity)
- **AVOID:** Silhouette (quadratic memory is prohibitive)

---

## Task 4: Time Series Tokenization and Next-Token Prediction

### Objective
Tokenize time series data and train a next-token predictor model.

### Data Investigation
- Shape: 5000 timesteps x 20 dimensions
- Range: [-289.54, 943.36]
- Low autocorrelation (~0.02) indicates limited temporal structure

### Tokenization Algorithm

**Quantile-based Binning:**
```python
# For each dimension d:
bin_edges[d] = percentile(data[:, d], linspace(0, 100, n_bins+1))
bin_centers[d, b] = (bin_edges[d, b] + bin_edges[d, b+1]) / 2

# Tokenize: continuous -> discrete
tokens[:, d] = clip(digitize(data[:, d], bin_edges[d]) - 1, 0, n_bins-1)

# Detokenize: discrete -> continuous
data[:, d] = bin_centers[d, tokens[:, d]]
```

### Next-Token Predictor

**Architecture:** 2-layer MLP
- Input: Flattened context window (5 timesteps x 20 dims = 100 features)
- Hidden: 128 units with ReLU activation
- Output: 20 dims x 64 bins = 1280 logits (softmax per dimension)

**Training:**
- Optimizer: SGD with learning rate 0.01
- Loss: Cross-entropy per dimension
- Epochs: 30
- Batch size: 256

### Results

| Component         | Metric              | Value     |
|-------------------|---------------------|-----------|
| Tokenizer         | Vocab size          | 1280      |
| Tokenizer         | Reconstruction RMSE | 11.30     |
| Predictor         | Token accuracy      | 1.70%     |
| Predictor         | Prediction RMSE     | 11.40     |
| Predictor         | Inference time      | 0.12 ms   |

**Note:** Low token accuracy is expected given:
- 64^20 possible state combinations
- Near-zero autocorrelation in the data
- For production, transformer architecture would improve sequence modeling

### Scaling with Polars

The implementation includes Polars integration for large-scale processing:
- Lazy evaluation for query optimization
- Automatic multi-core parallelization
- Memory-efficient columnar storage
- Streaming mode for out-of-core processing

---

## Repository Structure

```
deeter-interview/
├── .vscode/
│   ├── settings.json       # VSCode Python/Jupyter settings
│   ├── extensions.json     # Recommended extensions
│   └── launch.json         # Debug configurations
├── API.py                  # DataAPI and TimeSeriesAPI implementations
├── Interview.ipynb         # Complete Jupyter notebook with all tasks
├── README.md               # This documentation file
├── Deeter_Interview_Prep.md    # Deep-dive interview preparation
├── Interview_Cheat_Sheet.md    # Quick reference cheat sheet
├── requirements.txt        # Python dependencies
└── setup.sh               # Environment setup script
```

---

## Quick Start

### Option 1: Using setup script (Recommended)

```bash
# Clone the repository
git clone https://github.com/drmysore/deeter-interview.git
cd deeter-interview

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate environment
source venv/bin/activate

# Start Jupyter
jupyter notebook Interview.ipynb
```

### Option 2: Manual setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Jupyter kernel
python -m ipykernel install --user --name=deeter-interview

# Start Jupyter
jupyter notebook Interview.ipynb
```

### Option 3: VSCode (Recommended for development)

1. Open the folder in VSCode
2. Install recommended extensions when prompted
3. Select Python interpreter: `./venv/bin/python`
4. Open `Interview.ipynb`
5. Select kernel: `deeter-interview`

---

## Dependencies

All dependencies are listed in `requirements.txt`:

- numpy >= 1.24.0
- scipy >= 1.10.0
- scikit-learn >= 1.2.0
- pandas >= 2.0.0
- polars >= 0.18.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- jupyter >= 1.0.0
- ipykernel >= 6.0.0

---

## Key Algorithms Summary

| Task | Algorithm | Complexity |
|------|-----------|------------|
| 1 | PCA + ICA for latent structure discovery | O(n * d^2) |
| 1b | Gradient Boosting with scaling analysis | O(n * trees * depth) |
| 2 | MVN with Cholesky decomposition | O(d^3) fit, O(n * d^2) sample |
| 3 | K-Means with multiple validity indices | O(n * k * d) |
| 4 | Quantile tokenization + MLP predictor | O(n * d) tokenize, O(d * h) predict |

---

## Contact

**Author:** Supreeth Mysore
