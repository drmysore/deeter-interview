# Deeter Analytics Technical Interview Preparation
## 2-Hour Deep Dive + Pair Programming

---

## PART 1: DEEP DIVE ON YOUR ANSWERS (45-60 minutes)

### Question 1: Reverse Engineering Distributions

#### What They'll Ask:
1. **"Walk me through your approach to identifying the functional relationships"**
   
   **Your Answer:**
   ```
   "I started with dimensionality reduction using PCA, which confirmed the 7 latent 
   dimensions through explained variance analysis. Then I analyzed:
   
   1. Marginal distributions - identified bounded vs unbounded patterns
   2. Correlation structure - found which outputs share latent factors
   3. Distribution shapes - skewness/kurtosis suggested transformation types
   4. ICA - attempted to unmix the sources
   
   The challenge is this is an underdetermined system: 7D latent → 20D observation 
   through 13 non-linear functions. Without symbolic regression or gradient-based 
   optimization, exact recovery requires testing all possible function combinations."
   ```

2. **"Why didn't you complete the exact function forms?"**
   
   **Honest Answer:**
   ```
   "In the 2-hour constraint, I prioritized demonstrating systematic analysis over 
   potentially incorrect guesses. The ground truth uses base64-encoded numpy functions 
   (sin, cos, tanh, exp, cosh) that I identified through distribution analysis, but 
   mapping each of the 20 outputs to exact weighted combinations of 13 transformations 
   of 7 latent variables would require:
   
   - Symbolic regression (PySR, gplearn) - 30+ min to setup and run
   - Or gradient-based optimization to fit all parameters
   - Or exhaustive search through function combinations
   
   I showed I understand the problem structure and could complete this with more time."
   ```

3. **"What about Question 1b - the scaling laws?"**
   
   **Your Answer:**
   ```
   "I skipped 1b to focus on completing all other questions. Scaling laws research 
   would require:
   - Training models of varying sizes (10-1000+ params)
   - Varying training data (100 to 100k samples)
   - Computing FLOPs for each configuration
   - Fitting power laws to loss curves
   
   This is essentially a mini research project that would take the full 2 hours alone. 
   Given the time constraint, I prioritized breadth over depth."
   ```

#### How to Improve During Interview:

**Approach 1: Symbolic Regression**
```python
# They might ask you to implement this live
from gplearn.genetic import SymbolicRegressor

# For a single output dimension
def fit_symbolic_regression(data, latent_sources, output_idx):
    """
    Try to discover functional form for one output dimension.
    """
    X = latent_sources  # (n_samples, 7)
    y = data[:, output_idx]  # (n_samples,)
    
    est = SymbolicRegressor(
        population_size=5000,
        generations=20,
        tournament_size=20,
        function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'tan'),
        metric='mean absolute error',
        parsimony_coefficient=0.01,
        random_state=2025
    )
    
    est.fit(X, y)
    return str(est._program)
```

**Approach 2: Manual Pattern Matching**
```python
def identify_transformation_type(data_col):
    """
    Heuristic classification of transformation based on distribution.
    """
    # Normalize
    normalized = (data_col - data_col.mean()) / data_col.std()
    
    # Check bounds
    data_min, data_max = data_col.min(), data_col.max()
    range_span = data_max - data_min
    
    # Skewness and kurtosis
    skew = stats.skew(data_col)
    kurt = stats.kurtosis(data_col)
    
    # Classification rules
    if data_min > -1.1 and data_max < 1.1:
        if abs(skew) < 0.3:
            return "tanh or bounded linear"
        else:
            return "sin or cos"
    elif data_min > -0.1 and data_max < 1.1:
        return "sigmoid (1/(1+exp(-x)))"
    elif kurt > 3:
        return "exp-based (heavy tails)"
    else:
        return "linear combination or product"
```

**Approach 3: Gradient-Based Fitting**
```python
import torch
import torch.nn as nn

class ParametricMixingModel(nn.Module):
    """
    Learn the mixing matrices W, B, A and function indices.
    """
    def __init__(self, h=7, d=20, t=13):
        super().__init__()
        self.W = nn.Parameter(torch.randn(t, h, d) * 0.7)
        self.B = nn.Parameter(torch.randn(t, 1, d) * 0.3)
        self.A = nn.Parameter(torch.randn(t, 1, d) * 0.4)
        self.S = nn.Parameter(torch.randn(t, 1, d) * 0.15)
        
        # Function selection (learned weights over function set)
        self.func_weights = nn.Parameter(torch.randn(t, 10))
        
    def apply_function_mix(self, x, k):
        """Apply weighted mixture of basis functions."""
        funcs = [
            torch.sin(x),
            torch.cos(x),
            torch.tanh(x),
            1.0 / (1.0 + torch.exp(-x)),
            x * torch.tanh(x),
            torch.sin(x * x),
            torch.cosh(x) - 1.0,
            torch.exp(-(x * x)),
            torch.tanh(x) + 0.1 * torch.sin(3.0 * x),
            torch.cos(x) * torch.sin(x)
        ]
        
        weights = torch.softmax(self.func_weights[k], dim=0)
        result = sum(w * f for w, f in zip(weights, funcs))
        return result
    
    def forward(self, z):
        """z: (batch, 7) latent variables"""
        y = torch.zeros(z.size(0), 20)
        for k in range(13):
            x = z @ self.W[k] + self.B[k]
            h = self.apply_function_mix(x + self.S[k], k)
            y += self.A[k] * h
        return y
```

---

### Question 2: Fast Multi-Variate Sampling

#### What They'll Ask:

1. **"Why did you choose diagonal covariance for unconditional sampling?"**
   
   **Your Answer:**
   ```
   "Speed-accuracy tradeoff. Diagonal covariance:
   - Fit: O(nd) vs O(nd²) for full covariance
   - Sample: O(nd) vs O(nd²) + Cholesky decomposition
   - For 4D Iris data, diagonal is ~4x faster
   - Loses inter-dimension correlations but preserves marginals
   
   For the speed requirement (<0.0005s), this was necessary. If accuracy was 
   paramount, I'd use full covariance or Gaussian Mixture Model."
   ```

2. **"Explain your conditional sampling implementation"**
   
   **Your Answer:**
   ```
   "I used the analytical formula for Gaussian conditionals:
   
   For partition: X = [X_known, X_unknown]
   
   Conditional mean: 
   μ_{unknown|known} = μ_u + Σ_{uk} Σ_{kk}^{-1} (x_k - μ_k)
   
   Conditional covariance:
   Σ_{unknown|known} = Σ_{uu} - Σ_{uk} Σ_{kk}^{-1} Σ_{ku}
   
   This is exact for Gaussians and faster than MCMC. I used Cholesky decomposition 
   for efficient multivariate normal sampling."
   ```

3. **"What if the speed requirements were even stricter?"**

#### How to Improve During Interview:

**Improvement 1: Pre-computation**
```python
class FastIrisSampler:
    """Optimized with pre-computed decompositions."""
    
    def __init__(self):
        self.mu = None
        self.L = None  # Pre-computed Cholesky factor
        self.cov_inv = None  # Pre-computed inverse
        
    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.cov = np.cov(X, rowvar=False)
        
        # Pre-compute for unconditional sampling
        self.L = np.linalg.cholesky(self.cov + 1e-6 * np.eye(len(self.mu)))
        
        # Pre-compute inverse for conditional
        self.cov_inv = np.linalg.inv(self.cov + 1e-6 * np.eye(len(self.mu)))
    
    def sample(self, n_samples, random_state=None):
        """Ultra-fast sampling using pre-computed Cholesky."""
        if random_state is not None:
            np.random.seed(random_state)
        
        # z ~ N(0, I), then x = μ + L*z ~ N(μ, Σ)
        z = np.random.randn(n_samples, len(self.mu))
        return self.mu + z @ self.L.T
```

**Improvement 2: Gaussian Mixture Model**
```python
from sklearn.mixture import GaussianMixture

class GMMSampler:
    """Better accuracy with minimal speed penalty."""
    
    def __init__(self, n_components=3):
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='diag',  # Still fast
            random_state=42
        )
        
    def fit(self, X):
        self.gmm.fit(X)
        
    def sample(self, n_samples, random_state=None):
        if random_state is not None:
            self.gmm.random_state = random_state
        samples, _ = self.gmm.sample(n_samples)
        return samples
    
    def conditional_sample(self, template, n_samples, random_state=None):
        """
        Gibbs sampling for GMM conditionals.
        Still fast if we limit iterations.
        """
        # Implementation using Gibbs sampling or rejection sampling
        pass
```

**Improvement 3: Numba JIT Compilation**
```python
from numba import jit

@jit(nopython=True)
def fast_mvn_sample(mu, L, n_samples):
    """JIT-compiled sampling - 10x speedup."""
    d = len(mu)
    z = np.random.randn(n_samples, d)
    samples = np.zeros((n_samples, d))
    
    for i in range(n_samples):
        samples[i] = mu + L @ z[i]
    
    return samples
```

---

### Question 3: Cluster Count Comparison

#### What They'll Ask:

1. **"Why did you recommend Calinski-Harabasz over Silhouette?"**
   
   **Your Answer:**
   ```
   "Computational complexity and scalability:
   
   | Method | Complexity | 1M points | 100M points |
   |--------|-----------|-----------|-------------|
   | Silhouette | O(n²k) | 16 minutes | 11 days |
   | Calinski-H | O(nk) | 1 second | 100 seconds |
   
   For large-scale production:
   - Silhouette requires computing all pairwise distances: n(n-1)/2 operations
   - Calinski-H only needs cluster means and variances: linear in n
   - Both correctly identified k=3 for Iris
   - Silhouette is 100-1000x slower
   
   The only advantage of Silhouette is interpretability ([-1,1] range), but 
   Calinski-H values can be normalized if needed."
   ```

2. **"What about the Gap Statistic? Why is it so slow?"**
   
   **Your Answer:**
   ```
   "Gap Statistic has O(Bnk) complexity where B is bootstrap samples (typically 10-50).
   
   For each value of k, it:
   1. Fits k-means on real data
   2. Generates B reference datasets (uniform random)
   3. Fits k-means on each reference dataset
   4. Computes log(WCSS) difference
   
   So it's essentially running k-means B+1 times per k value. For B=10, that's 
   10x slower than other methods. The theoretical soundness doesn't justify the 
   computational cost in production."
   ```

3. **"How would you implement this at scale - say 100 million points?"**

#### How to Improve During Interview:

**Improvement 1: Mini-batch for Large Data**
```python
def scalable_cluster_evaluation(X, k_range, method='calinski', sample_size=10000):
    """
    For datasets too large to fit in memory.
    Use stratified sampling for metric computation.
    """
    from sklearn.utils import resample
    
    results = {}
    
    for k in k_range:
        # Fit on full data using MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1000, random_state=42)
        
        # Streaming fit
        for chunk in np.array_split(X, 100):
            kmeans.partial_fit(chunk)
        
        # Evaluate on sample
        X_sample = resample(X, n_samples=sample_size, random_state=42)
        labels_sample = kmeans.predict(X_sample)
        
        if method == 'calinski':
            score = calinski_harabasz_score(X_sample, labels_sample)
        elif method == 'davies':
            score = davies_bouldin_score(X_sample, labels_sample)
        
        results[k] = score
    
    return results
```

**Improvement 2: Parallel Processing**
```python
from joblib import Parallel, delayed

def parallel_cluster_evaluation(X, k_range, n_jobs=-1):
    """
    Parallelize across k values.
    """
    def evaluate_k(k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_jobs=1)
        labels = kmeans.fit_predict(X)
        
        ch_score = calinski_harabasz_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        
        return k, ch_score, db_score
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_k)(k) for k in k_range
    )
    
    return results
```

**Improvement 3: Distributed with PySpark**
```python
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

def distributed_cluster_evaluation(spark_df, k_range):
    """
    For 100M+ points, use Spark.
    """
    results = {}
    
    for k in k_range:
        # Fit KMeans
        kmeans = SparkKMeans(k=k, seed=42)
        model = kmeans.fit(spark_df)
        
        # Predict
        predictions = model.transform(spark_df)
        
        # Evaluate (Spark has silhouette)
        evaluator = ClusteringEvaluator(
            predictionCol='prediction',
            featuresCol='features',
            metricName='silhouette'
        )
        
        score = evaluator.evaluate(predictions)
        results[k] = score
    
    return results
```

**Improvement 4: Elbow Method with Automatic Detection**
```python
from kneed import KneeLocator

def automatic_elbow_detection(X, k_range):
    """
    Automate the subjective elbow method.
    """
    wcss = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    # Automatic knee detection
    kl = KneeLocator(
        k_range, wcss, 
        curve='convex', 
        direction='decreasing',
        interp_method='polynomial'
    )
    
    optimal_k = kl.knee
    
    return optimal_k, wcss
```

---

### Question 4: Time Series Tokenization

#### What They'll Ask:

1. **"Why quantile-based tokenization instead of learned embeddings?"**
   
   **Your Answer:**
   ```
   "Three reasons:
   
   1. Data constraint: 2k samples isn't enough to train VQ-VAE or autoencoder 
      reliably. You need 10-100k for learned embeddings.
   
   2. Interpretability: Quantile bins have clear meaning - each token represents 
      a specific percentile range. Learned embeddings are black boxes.
   
   3. Speed: Quantile binning is O(n log n) one-time cost. VQ-VAE requires 
      training epochs and forward passes.
   
   For 10k+ samples, I'd use Vector Quantization with a learned codebook."
   ```

2. **"Why Random Forest instead of LSTM/Transformer?"**
   
   **Your Answer:**
   ```
   "Sample efficiency and overfitting risk:
   
   - With window_size=5 and 2k samples, I have ~1600 training sequences
   - LSTM/Transformer have 10k-1M parameters
   - Rule of thumb: need 10-100 samples per parameter
   - Random Forest is an ensemble (built-in regularization)
   - RF can handle multivariate output naturally
   
   At 10k+ samples, I'd switch to a small Transformer (2-4 layers, 256 hidden dim).
   At 100k+ samples, full-scale Transformer with attention mechanisms."
   ```

3. **"Your prediction time is ~0.01s. How would you optimize for real-time?"**

#### How to Improve During Interview:

**Improvement 1: Better Tokenization**
```python
class HierarchicalTokenizer:
    """
    Coarse + fine tokenization for better reconstruction.
    """
    def __init__(self, n_coarse=16, n_fine=16):
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.total_bins = n_coarse * n_fine  # 256
        
    def fit(self, X):
        self.n_dims = X.shape[1]
        self.coarse_boundaries = []
        self.fine_boundaries = []
        
        for d in range(self.n_dims):
            # Coarse quantiles
            coarse_q = np.linspace(0, 100, self.n_coarse + 1)
            coarse_b = np.percentile(X[:, d], coarse_q)
            self.coarse_boundaries.append(coarse_b)
            
            # Fine quantiles within each coarse bin
            fine_b_all = []
            for i in range(self.n_coarse):
                mask = (X[:, d] >= coarse_b[i]) & (X[:, d] < coarse_b[i+1])
                if mask.sum() > 0:
                    fine_q = np.linspace(0, 100, self.n_fine + 1)
                    fine_b = np.percentile(X[mask, d], fine_q)
                else:
                    fine_b = np.linspace(coarse_b[i], coarse_b[i+1], self.n_fine + 1)
                fine_b_all.append(fine_b)
            
            self.fine_boundaries.append(fine_b_all)
    
    def tokenize(self, X):
        """Returns (coarse_token, fine_token) pairs."""
        coarse_tokens = np.zeros_like(X, dtype=np.int32)
        fine_tokens = np.zeros_like(X, dtype=np.int32)
        
        for d in range(self.n_dims):
            # Coarse binning
            coarse = np.searchsorted(self.coarse_boundaries[d], X[:, d]) - 1
            coarse = np.clip(coarse, 0, self.n_coarse - 1)
            coarse_tokens[:, d] = coarse
            
            # Fine binning within coarse
            for i in range(len(X)):
                c = coarse[i]
                fine = np.searchsorted(self.fine_boundaries[d][c], X[i, d]) - 1
                fine = np.clip(fine, 0, self.n_fine - 1)
                fine_tokens[i, d] = fine
        
        return coarse_tokens, fine_tokens
```

**Improvement 2: Transformer Model (if they want you to implement)**
```python
import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    """
    Small transformer for time series prediction.
    """
    def __init__(self, vocab_size=256, n_dims=20, d_model=128, nhead=4, 
                 num_layers=2, window_size=10):
        super().__init__()
        
        self.n_dims = n_dims
        self.window_size = window_size
        
        # Embedding for each dimension
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model // n_dims)
            for _ in range(n_dims)
        ])
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, window_size, d_model)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output heads (one per dimension)
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size)
            for _ in range(n_dims)
        ])
    
    def forward(self, x):
        """
        x: (batch, window_size, n_dims) of token indices
        """
        batch_size = x.size(0)
        
        # Embed each dimension and concatenate
        embeddings = []
        for d in range(self.n_dims):
            emb = self.embeddings[d](x[:, :, d])  # (batch, window, d_model//n_dims)
            embeddings.append(emb)
        
        x = torch.cat(embeddings, dim=-1)  # (batch, window, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Take last time step
        x_last = x[:, -1, :]  # (batch, d_model)
        
        # Predict next token for each dimension
        outputs = []
        for d in range(self.n_dims):
            logits = self.output_heads[d](x_last)  # (batch, vocab_size)
            outputs.append(logits)
        
        return torch.stack(outputs, dim=1)  # (batch, n_dims, vocab_size)

# Training loop
def train_transformer(model, train_data, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward
            outputs = model(batch_x)  # (batch, n_dims, vocab_size)
            
            # Compute loss for each dimension
            loss = 0
            for d in range(model.n_dims):
                loss += criterion(outputs[:, d, :], batch_y[:, d])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
```

**Improvement 3: Cross-Dimensional Attention**
```python
class CrossDimensionalTokenizer:
    """
    Capture joint distributions across dimensions.
    """
    def __init__(self, n_bins_per_dim=16):
        self.n_bins = n_bins_per_dim
        self.kmeans_models = []
        
    def fit(self, X):
        """
        Learn joint codebook for correlated dimension pairs.
        """
        self.n_dims = X.shape[1]
        
        # Find highly correlated pairs
        corr = np.corrcoef(X.T)
        self.high_corr_pairs = []
        
        for i in range(self.n_dims):
            for j in range(i+1, self.n_dims):
                if abs(corr[i, j]) > 0.7:
                    self.high_corr_pairs.append((i, j))
        
        # Learn joint codebook for each pair
        for (i, j) in self.high_corr_pairs:
            from sklearn.cluster import KMeans
            X_pair = X[:, [i, j]]
            kmeans = KMeans(n_clusters=self.n_bins**2, random_state=42)
            kmeans.fit(X_pair)
            self.kmeans_models.append((i, j, kmeans))
```

**Improvement 4: Production Optimization**
```python
# Use ONNX for fast inference
def export_to_onnx(model, dummy_input, filename='model.onnx'):
    """
    Export PyTorch model to ONNX for 10x faster inference.
    """
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )

# Inference with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession('model.onnx')
def predict_fast(x):
    outputs = session.run(None, {'input': x})
    return outputs[0]
```

---

## PART 2: PAIR PROGRAMMING IMPROVEMENTS (45-60 minutes)

### Likely Tasks They'll Give You:

#### Task 1: "Implement the exact function recovery for Question 1"
```python
def recover_exact_functions(data, n_samples=10000):
    """
    Full implementation of function recovery.
    They'll likely give you hints or partial code.
    """
    
    # Step 1: PCA to get latent estimates
    from sklearn.decomposition import PCA
    pca = PCA(n_components=7)
    Z_est = pca.fit_transform(data)
    
    # Step 2: For each output dimension, try different function combinations
    from itertools import product
    
    functions = {
        'sin': np.sin,
        'cos': np.cos,
        'tanh': np.tanh,
        'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-x)),
        'x_tanh': lambda x: x * np.tanh(x),
        'sin_sq': lambda x: np.sin(x * x),
        'cosh_1': lambda x: np.cosh(x) - 1.0,
        'exp_sq': lambda x: np.exp(-(x * x)),
    }
    
    best_functions = {}
    
    for dim in range(20):
        y_true = data[:, dim]
        best_error = float('inf')
        best_func = None
        
        # Try each function
        for func_name, func in functions.items():
            # Try simple linear combination: y = func(w @ Z + b)
            from sklearn.linear_model import Ridge
            
            # Create features: apply function to linear combinations
            X_transformed = []
            for i in range(7):
                X_transformed.append(func(Z_est[:, i]))
            X_transformed = np.column_stack(X_transformed)
            
            # Fit linear model
            model = Ridge(alpha=0.1)
            model.fit(X_transformed, y_true)
            
            # Evaluate
            y_pred = model.predict(X_transformed)
            error = np.mean((y_true - y_pred) ** 2)
            
            if error < best_error:
                best_error = error
                best_func = (func_name, model.coef_, model.intercept_)
        
        best_functions[dim] = best_func
        print(f"Dim {dim}: {best_func[0]}, error={best_error:.4f}")
    
    return best_functions
```

#### Task 2: "Optimize the sampling code for better speed"
```python
# They might ask you to profile and optimize

import cProfile
import pstats

def profile_sampler():
    sampler = IrisSampler()
    X = load_iris_data()
    
    # Profile fit
    profiler = cProfile.Profile()
    profiler.enable()
    sampler.fit(X)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')
    stats.print_stats(10)
    
    # Profile sample
    profiler = cProfile.Profile()
    profiler.enable()
    samples = sampler.sample(10000)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')
    stats.print_stats(10)

# Then optimize based on bottlenecks
```

#### Task 3: "Add Davies-Bouldin implementation from scratch"
```python
def davies_bouldin_index(X, labels):
    """
    Implement DB index from scratch.
    Lower is better.
    """
    n_clusters = len(np.unique(labels))
    
    # Compute cluster centers
    centers = []
    for k in range(n_clusters):
        mask = labels == k
        center = X[mask].mean(axis=0)
        centers.append(center)
    centers = np.array(centers)
    
    # Compute average within-cluster distance
    S = np.zeros(n_clusters)
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            distances = np.linalg.norm(X[mask] - centers[k], axis=1)
            S[k] = distances.mean()
    
    # Compute Davies-Bouldin index
    DB = 0
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                M_ij = np.linalg.norm(centers[i] - centers[j])
                ratio = (S[i] + S[j]) / M_ij
                if ratio > max_ratio:
                    max_ratio = ratio
        DB += max_ratio
    
    DB /= n_clusters
    return DB
```

#### Task 4: "Implement streaming tokenization for large datasets"
```python
def streaming_tokenizer(data_stream, n_bins=256, chunk_size=1000):
    """
    Tokenize data that doesn't fit in memory.
    Use online quantile estimation.
    """
    from scipy.stats import rankdata
    
    # First pass: estimate quantiles using reservoir sampling
    reservoir = []
    n_seen = 0
    
    for chunk in data_stream:
        for row in chunk:
            n_seen += 1
            
            if len(reservoir) < 10000:
                reservoir.append(row)
            else:
                # Reservoir sampling
                j = np.random.randint(0, n_seen)
                if j < 10000:
                    reservoir[j] = row
    
    # Compute quantiles from reservoir
    reservoir = np.array(reservoir)
    quantiles = []
    for d in range(reservoir.shape[1]):
        q = np.percentile(reservoir[:, d], np.linspace(0, 100, n_bins+1))
        quantiles.append(q)
    
    # Second pass: tokenize
    # (reset data_stream)
    tokens_list = []
    for chunk in data_stream:
        chunk_tokens = np.zeros_like(chunk, dtype=np.int32)
        for d in range(chunk.shape[1]):
            bins = np.searchsorted(quantiles[d], chunk[:, d]) - 1
            bins = np.clip(bins, 0, n_bins-1)
            chunk_tokens[:, d] = bins
        tokens_list.append(chunk_tokens)
    
    return np.vstack(tokens_list)
```

---

## PART 3: END-TO-END SOLUTIONS (15-30 minutes)

### Likely Scenarios:

#### Scenario 1: "Build a complete pipeline for Question 4"
```python
class TimeSeriesPredictionPipeline:
    """
    Complete end-to-end pipeline:
    Data loading → Tokenization → Training → Prediction → Evaluation
    """
    
    def __init__(self, window_size=10, n_bins=256):
        self.window_size = window_size
        self.tokenizer = TimeSeriesTokenizer(n_bins=n_bins)
        self.model = None
        
    def fit(self, data):
        """
        Train the complete pipeline.
        """
        print("Step 1: Fitting tokenizer...")
        self.tokenizer.fit(data)
        
        print("Step 2: Tokenizing data...")
        tokens = self.tokenizer.tokenize(data)
        
        print("Step 3: Training prediction model...")
        self.model = SimpleNextTokenPredictor(window_size=self.window_size)
        self.model.fit(tokens)
        
        print("Pipeline training complete!")
        
    def predict(self, history_data, n_steps=1):
        """
        Predict n_steps into the future.
        """
        # Tokenize history
        history_tokens = self.tokenizer.tokenize(history_data)
        
        # Predict next tokens
        predictions = []
        current_history = history_tokens[-self.window_size:]
        
        for _ in range(n_steps):
            next_token = self.model.predict_next(current_history)
            predictions.append(next_token)
            
            # Update history (rolling window)
            current_history = np.vstack([current_history[1:], next_token])
        
        # Detokenize
        predictions = np.array(predictions)
        predictions_continuous = self.tokenizer.detokenize(predictions)
        
        return predictions_continuous
    
    def evaluate(self, test_data, n_steps=100):
        """
        Comprehensive evaluation.
        """
        mae_per_step = []
        
        for i in range(self.window_size, len(test_data) - n_steps):
            history = test_data[i-self.window_size:i]
            true_future = test_data[i:i+n_steps]
            
            pred_future = self.predict(history, n_steps=n_steps)
            
            mae = np.abs(pred_future - true_future).mean(axis=1)
            mae_per_step.append(mae)
        
        mae_per_step = np.array(mae_per_step)
        
        return {
            'mae_mean': mae_per_step.mean(axis=0),
            'mae_std': mae_per_step.std(axis=0),
            'mae_overall': mae_per_step.mean()
        }

# Usage
pipeline = TimeSeriesPredictionPipeline(window_size=10, n_bins=256)

# Generate data
train_data = TimeSeriesAPI.get(n=1600)
test_data = TimeSeriesAPI.get(n=400)

# Train
pipeline.fit(train_data)

# Evaluate
results = pipeline.evaluate(test_data, n_steps=10)
print(f"Overall MAE: {results['mae_overall']:.4f}")
```

#### Scenario 2: "Production deployment considerations"
```python
class ProductionTimeSeriesService:
    """
    Production-ready service with monitoring and error handling.
    """
    
    def __init__(self, model_path='model.pkl'):
        self.model_path = model_path
        self.pipeline = None
        self.metrics = {
            'predictions_made': 0,
            'avg_latency': 0,
            'errors': 0
        }
        
    def load_model(self):
        """Load trained pipeline."""
        import pickle
        with open(self.model_path, 'rb') as f:
            self.pipeline = pickle.load(f)
        print("Model loaded successfully")
        
    def predict_with_monitoring(self, history_data, n_steps=1):
        """
        Prediction with latency tracking and error handling.
        """
        import time
        
        try:
            start_time = time.time()
            
            # Input validation
            if len(history_data) < self.pipeline.window_size:
                raise ValueError(f"Need at least {self.pipeline.window_size} historical points")
            
            # Predict
            predictions = self.pipeline.predict(history_data, n_steps=n_steps)
            
            # Update metrics
            latency = time.time() - start_time
            self.metrics['predictions_made'] += 1
            self.metrics['avg_latency'] = (
                (self.metrics['avg_latency'] * (self.metrics['predictions_made'] - 1) + latency)
                / self.metrics['predictions_made']
            )
            
            return {
                'predictions': predictions.tolist(),
                'latency_ms': latency * 1000,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.metrics['errors'] += 1
            return {
                'error': str(e),
                'timestamp': time.time()
            }
    
    def health_check(self):
        """Service health metrics."""
        return {
            'status': 'healthy' if self.pipeline is not None else 'unhealthy',
            'predictions_made': self.metrics['predictions_made'],
            'avg_latency_ms': self.metrics['avg_latency'] * 1000,
            'error_rate': self.metrics['errors'] / max(1, self.metrics['predictions_made'])
        }

# FastAPI service
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
service = ProductionTimeSeriesService()

@app.on_event("startup")
def startup():
    service.load_model()

class PredictionRequest(BaseModel):
    history: list
    n_steps: int = 1

@app.post("/predict")
def predict(request: PredictionRequest):
    result = service.predict_with_monitoring(
        np.array(request.history),
        n_steps=request.n_steps
    )
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return result

@app.get("/health")
def health():
    return service.health_check()
```

---

## PART 4: COMMON FOLLOW-UP QUESTIONS

### Architecture & System Design

**Q: "How would you deploy this in production at scale?"**
```
A: Multi-tier architecture:

1. Data Ingestion Layer:
   - Apache Kafka for streaming time series
   - Polars/DuckDB for batch processing
   - Schema validation (Pydantic)

2. Preprocessing Layer:
   - Tokenization microservice (horizontal scaling)
   - Redis cache for tokenizer state
   - Batch processing for historical data

3. Prediction Layer:
   - Model served via TorchServe or TensorFlow Serving
   - Load balancer across GPU instances
   - Batch predictions for efficiency (latency vs throughput tradeoff)

4. Monitoring:
   - Prometheus metrics (latency, throughput, error rate)
   - Grafana dashboards
   - Alerting on model drift (compare prediction distributions)

5. CI/CD:
   - A/B testing for model updates
   - Canary deployments
   - Automated retraining pipeline (Airflow/Kubeflow)
```

**Q: "What about model versioning and reproducibility?"**
```
A: MLOps best practices:

1. Version Control:
   - Git for code
   - DVC for data and models
   - Track hyperparameters (MLflow, Weights & Biases)

2. Experiment Tracking:
   - Log all experiments with metrics
   - Track data lineage
   - Reproducible training (fixed seeds, Docker containers)

3. Model Registry:
   - Central model store (MLflow Model Registry)
   - Metadata: accuracy, training time, data version
   - Promotion workflow: dev → staging → production

4. Monitoring:
   - Track prediction drift (KL divergence from training distribution)
   - Retrain triggers (accuracy drop, distribution shift)
   - Rollback mechanism
```

### Performance & Optimization

**Q: "How would you reduce latency from 10ms to 1ms?"**
```
A: Multiple optimization strategies:

1. Model Optimization:
   - Quantization (FP32 → INT8): 4x speedup
   - Pruning (remove 30-50% of weights)
   - Knowledge distillation (smaller student model)
   - ONNX Runtime: 2-10x faster than PyTorch

2. Infrastructure:
   - Move to GPU inference (T4, A10)
   - Batch predictions (trade latency for throughput)
   - Edge deployment (reduce network latency)

3. Algorithmic:
   - Approximate nearest neighbors for similarity search
   - Caching frequent predictions
   - Precompute embeddings

4. System-Level:
   - C++ inference (vs Python)
   - TensorRT optimization
   - Reduce I/O (memory-mapped files)
```

**Q: "Memory constraints - how to handle 1TB dataset?"**
```
A: Out-of-core and distributed processing:

1. Streaming:
   - Process in chunks (Polars lazy evaluation)
   - Online learning (incremental updates)
   - Reservoir sampling for statistics

2. Distributed:
   - Spark for preprocessing
   - Dask for pandas-like API at scale
   - Ray for distributed training

3. Compression:
   - Parquet/Arrow columnar formats
   - Tokenization reduces size (float64 → int8: 8x compression)
   - Delta encoding for time series

4. Smart Sampling:
   - Train on representative sample
   - Importance sampling (weight difficult examples)
   - Active learning (select informative points)
```

### Error Handling & Edge Cases

**Q: "What if the time series has missing values?"**
```python
def handle_missing_values(data, strategy='interpolate'):
    """
    Robust missing value handling.
    """
    if strategy == 'interpolate':
        # Linear interpolation
        df = pd.DataFrame(data)
        return df.interpolate(method='linear').fillna(method='bfill').values
    
    elif strategy == 'forward_fill':
        return pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values
    
    elif strategy == 'model_imputation':
        # Use model to predict missing values
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        return imputer.fit_transform(data)
    
    elif strategy == 'token':
        # Add special [MISSING] token
        # Model learns to handle it
        pass
```

**Q: "What if distribution shifts over time?"**
```python
def detect_distribution_shift(new_data, reference_data, threshold=0.1):
    """
    Monitor for concept drift.
    """
    # KL divergence for each dimension
    from scipy.stats import entropy
    
    shifts = []
    for d in range(new_data.shape[1]):
        # Create histograms
        ref_hist, bins = np.histogram(reference_data[:, d], bins=50, density=True)
        new_hist, _ = np.histogram(new_data[:, d], bins=bins, density=True)
        
        # KL divergence (add smoothing)
        kl_div = entropy(ref_hist + 1e-10, new_hist + 1e-10)
        shifts.append(kl_div)
    
    if max(shifts) > threshold:
        print(f"WARNING: Distribution shift detected! Max KL={max(shifts):.3f}")
        print(f"Affected dimension: {np.argmax(shifts)}")
        return True
    
    return False

# In production
if detect_distribution_shift(new_batch, training_data):
    # Trigger retraining
    # Or switch to ensemble with recent model
    pass
```

---

## PART 5: QUESTIONS YOU SHOULD ASK THEM

### Show Technical Depth:

1. **"What's the typical data volume you work with? This affects tokenization strategy"**
   - Shows you think about production scale
   
2. **"What's the latency requirement for predictions? Sub-second vs batch?"**
   - Shows you understand real-time vs batch trade-offs

3. **"Do you have labeled data for supervised learning or is this unsupervised?"**
   - Shows you think about problem formulation

4. **"What's more important - interpretability or accuracy?"**
   - Shows you understand business constraints

### Show Business Acumen:

5. **"What downstream decisions depend on these predictions?"**
   - Shows you care about impact

6. **"What's the cost of false positives vs false negatives?"**
   - Shows you understand error types matter

### Show You're a Team Player:

7. **"How does the data science team collaborate with engineering?"**
   - MLOps, deployment, monitoring

8. **"What tools and frameworks does the team currently use?"**
   - Shows willingness to adapt

---

## FINAL PREPARATION CHECKLIST

### Before the Call:

- [ ] Review your notebook thoroughly - be ready to explain every line
- [ ] Practice explaining trade-offs (speed vs accuracy, complexity vs interpretability)
- [ ] Prepare to write code from scratch (no copy-paste)
- [ ] Test your screen sharing setup
- [ ] Have documentation ready (NumPy, Pandas, Scikit-learn)
- [ ] Prepare questions about Deeter's tech stack and problems

### During the Call:

- [ ] **Think aloud** - explain your reasoning as you code
- [ ] **Ask clarifying questions** before jumping to code
- [ ] **Test incrementally** - run code frequently
- [ ] **Acknowledge mistakes** - "Good catch, let me fix that"
- [ ] **Suggest alternatives** - "We could also try X, but Y is better because..."

### Key Mindsets:

1. **Systematic Problem Solver**: "Let me break this down..."
2. **Production-Minded**: "At scale, this would need..."
3. **Collaborative**: "What do you think about this approach?"
4. **Humble**: "I haven't used this exact method, but here's my understanding..."
5. **Curious**: "Why did you choose this architecture?"

---

## REMEMBER:

- They're evaluating **problem-solving process**, not just correct answers
- It's okay to not know something - show how you'd figure it out
- **Communication > Code perfection**
- They want someone who can **teach and learn** from the team
- Your enthusiasm for the problem matters as much as technical skill

Good luck! You've got this! 🚀
