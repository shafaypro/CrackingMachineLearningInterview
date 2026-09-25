# Mathematics for ML: Linear Algebra, Calculus, and Optimization

Interviewers use math questions to separate people who can call `model.fit()` from people who can debug a model when `fit()` misbehaves. You will rarely be asked to prove a theorem, but you will be asked to derive the OLS solution on a whiteboard, explain why L1 gives sparse weights, say what the Hessian tells you, explain why Adam converges faster than SGD on a badly scaled problem, or explain why `softmax` returned `nan`. This guide covers the linear algebra, calculus, optimization, and information theory those questions rely on. Probability and statistics have their own guide: [Statistics & Probability](./intro_statistics_probability.md).

---

## Table of Contents
1. [Vectors, Dot Products, and Norms](#vectors-dot-products-and-norms)
2. [Matrices as Linear Maps](#matrices-as-linear-maps)
3. [Eigendecomposition and SVD](#eigendecomposition-and-svd)
4. [Positive Definite Matrices and Covariance](#positive-definite-matrices-and-covariance)
5. [Gradients, Jacobians, and Hessians](#gradients-jacobians-and-hessians)
6. [Chain Rule and Backpropagation](#chain-rule-and-backpropagation)
7. [Matrix Calculus Identities](#matrix-calculus-identities)
8. [Worked Derivations: OLS and Logistic Regression](#worked-derivations-ols-and-logistic-regression)
9. [Convexity](#convexity)
10. [First-Order Optimization](#first-order-optimization)
11. [Second-Order Methods](#second-order-methods)
12. [Constrained Optimization: Lagrange and KKT](#constrained-optimization-lagrange-and-kkt)
13. [Regularization: Constraint and Prior Views](#regularization-constraint-and-prior-views)
14. [Information Theory](#information-theory)
15. [Numerical Stability](#numerical-stability)
16. [Gradient Checking](#gradient-checking)
17. [Interview Q&A](#interview-qa)
18. [Common Pitfalls](#common-pitfalls)
19. [Related Topics](#related-topics)

---

## Vectors, Dot Products, and Norms

A feature vector `x ∈ R^d` is a point in `d`-dimensional space. Most of ML is computing distances and angles between such points. **Dot product**: `x · y = Σ_i x_i y_i = ||x|| ||y|| cos θ`. It measures alignment. A linear model's score `w · x + b` is a dot product. So is an attention logit `q · k`, and so is a single neuron's pre-activation.

**Cosine similarity**: `cos θ = (x · y) / (||x|| ||y||)`. This is the dot product after normalizing both vectors to unit length, so it ignores magnitude. It is the standard choice for comparing embeddings. **Norms** measure vector size:

| Norm | Formula | Unit ball shape | Where it shows up |
|---|---|---|---|
| L1 | `Σ \|x_i\|` | Diamond (cross-polytope) | Lasso, sparse coding, MAE loss |
| L2 | `sqrt(Σ x_i²)` | Sphere | Ridge / weight decay, Euclidean distance, MSE |
| L∞ | `max_i \|x_i\|` | Cube | Adversarial robustness (FGSM/PGD budgets) |
| L0 (not a true norm) | count of non-zeros | Axes only | What L1 approximates; NP-hard to optimize directly |

**Why L1 induces sparsity.** There are two ways to see it:

1. **Geometry.** Minimize the loss subject to `||w||_1 ≤ t`. The loss contours are ellipses, and they usually first touch the L1 diamond at a corner, and corners lie on the axes, where some coordinates are exactly zero. The L2 ball is round and has no corners, so the contact point is generically off-axis. Weights get small but not zero.
2. **Gradients.** The L2 penalty's gradient is `2λw`, which shrinks as `w → 0`, so the pull toward zero weakens and the weight never gets there exactly. The L1 penalty's (sub)gradient is `λ · sign(w)`, a constant push regardless of how small `w` is. If a feature's loss gradient is smaller than `λ` in magnitude, the optimum is exactly `w = 0`. That is the soft-thresholding operator: `w ← sign(z) · max(|z| - λ, 0)`.

```python
import numpy as np

x = np.array([3.0, -4.0, 0.0])
print(np.linalg.norm(x, 1), np.linalg.norm(x, 2), np.linalg.norm(x, np.inf))  # 7.0 5.0 4.0

def soft_threshold(z, lam):
    """Proximal operator of lam * ||w||_1: the step that produces exact zeros."""
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)

print(soft_threshold(np.array([2.5, 0.3, -0.8, -3.0]), lam=1.0))  # [ 1.5  0.  -0.  -2. ]
```

---

## Matrices as Linear Maps

A matrix `A ∈ R^{m×n}` is a function that maps `R^n → R^m` linearly. A linear layer `y = Wx + b` is a linear map followed by a shift. Stacking linear layers without nonlinearities gives `W_2 W_1 x`, which is still a single linear map. That is why activations are necessary.

| Concept | Meaning | ML relevance |
|---|---|---|
| Column space | All outputs `Ax` can reach | Predictions of a linear model live here |
| Null space | All `x` with `Ax = 0` | Directions the model cannot see; causes non-unique solutions |
| Rank | Dimension of the column space (= number of non-zero singular values) | Perfect multicollinearity → rank-deficient `XᵀX` |
| Inverse `A⁻¹` | Exists iff `A` is square and full rank | Rarely computed explicitly in practice |
| Determinant | Volume scaling factor; 0 iff singular | Normalizing flows (log-det Jacobian), Gaussian densities |
| Trace | Sum of diagonal = sum of eigenvalues | Matrix calculus tricks, `E[xᵀAx]` |

**Rank-nullity**: `rank(A) + dim(null(A)) = n`. If you have more features than samples (`d > n`), `XᵀX` (size `d×d`) has rank at most `n < d`, so it is singular and OLS has infinitely many solutions.

**Pseudo-inverse.** For any `A`, the Moore-Penrose pseudo-inverse `A⁺` is computed from the SVD `A = U Σ Vᵀ` as `A⁺ = V Σ⁺ Uᵀ`, where `Σ⁺` inverts the non-zero singular values and leaves zeros as zeros. `x = A⁺b` gives the least-squares solution. When that solution is not unique, it gives the one with minimum `||x||_2`.

**Practical rule:** never write `np.linalg.inv(A) @ b`. Use `np.linalg.solve(A, b)` for square systems and `np.linalg.lstsq(A, b)` for least squares. They are faster and far more accurate when `A` is ill-conditioned.

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(5, 8))           # more features than samples: underdetermined
y = rng.normal(size=5)

print(np.linalg.matrix_rank(X.T @ X))  # 5, not 8 -> X^T X is singular
w_pinv = np.linalg.pinv(X) @ y         # minimum-norm solution
w_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)
print(np.allclose(w_pinv, w_lstsq), np.allclose(X @ w_pinv, y))  # True True
```

---

## Eigendecomposition and SVD

**Eigendecomposition.** For a square matrix `A`, an eigenvector `v` satisfies `Av = λv`: `A` only stretches it by `λ`. If `A` is symmetric (covariance matrices, Hessians, `XᵀX`), the spectral theorem guarantees `A = Q Λ Qᵀ`, with real eigenvalues and orthonormal eigenvectors. Non-symmetric matrices may have complex eigenvalues or no full eigenbasis at all.

**SVD.** Every matrix `A ∈ R^{m×n}` (any shape, any rank) factors as `A = U Σ Vᵀ`, where `V` (`n×n`, orthogonal) holds input directions, `Σ` (diagonal, `σ_1 ≥ σ_2 ≥ ... ≥ 0`) holds stretch factors, and `U` (`m×m`, orthogonal) holds output directions. Geometrically: rotate, scale along the axes, rotate again.

**The connection:** `AᵀA = V Σ² Vᵀ`. The right singular vectors of `A` are the eigenvectors of `AᵀA`, and the eigenvalues are `σ_i²`.

| Application | How it uses the decomposition |
|---|---|
| **PCA** | Center `X`, then take the SVD `X = U Σ Vᵀ`. Rows of `Vᵀ` are principal directions; explained variance is `σ_i² / (n-1)`. Equivalent to eigendecomposing the covariance matrix, but numerically better because you never form `XᵀX` |
| **Low-rank approximation** | Eckart-Young: keeping the top `k` singular triplets gives the best rank-`k` approximation in Frobenius and spectral norm |
| **LoRA** | Assumes the fine-tuning *update* `ΔW` is low rank and parameterizes it as `ΔW = BA`, with `B ∈ R^{d×r}`, `A ∈ R^{r×k}`, `r ≪ min(d,k)`. That trains `r(d+k)` parameters instead of `dk` |
| **Recommenders** | Matrix factorization of the user-item matrix (SVD-style latent factors) |
| **Condition number** | `κ(A) = σ_max / σ_min`. Large `κ` means solving is numerically unstable and gradient descent is slow |
| **Spectral norm** | `\|\|A\|\|_2 = σ_max`, the largest possible stretch. Used in spectral normalization for GAN discriminators |

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5))  # correlated features
Xc = X - X.mean(axis=0)

U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
explained_var = S**2 / (len(Xc) - 1)
eigvals = np.linalg.eigvalsh(np.cov(Xc, rowvar=False))[::-1]   # eigh returns ascending
print(np.allclose(explained_var, eigvals))                     # True: PCA via SVD == eig of cov

k = 2
X_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k]                       # best rank-2 approximation
print(np.linalg.norm(Xc - X_k), np.sqrt((S[k:]**2).sum()))     # equal (Eckart-Young)
```

More on PCA in practice: [Dimensionality Reduction](./intro_dimensionality_reduction.md).

---

## Positive Definite Matrices and Covariance

A symmetric matrix `A` is **positive semidefinite (PSD)** if `xᵀAx ≥ 0` for all `x` (all eigenvalues `≥ 0`), and **positive definite (PD)** if `xᵀAx > 0` for all `x ≠ 0` (all eigenvalues `> 0`).

**Why you care:**
- **Covariance matrices are always PSD.** `Var(aᵀx) = aᵀ Σ a ≥ 0`, because a variance cannot be negative. `Σ` is PD unless some linear combination of features is constant, i.e. there is perfect collinearity.
- **`XᵀX` is always PSD.** `vᵀXᵀXv = ||Xv||² ≥ 0`. Adding `λI` (ridge) makes it PD for any `λ > 0`, which is why ridge always has a unique solution.
- **Hessian PSD everywhere ⇔ convex function.** A PD Hessian at a critical point means a strict local minimum.
- **Kernels.** A valid kernel must produce a PSD Gram matrix (Mercer's condition).
- **Cholesky.** A PD matrix factors as `A = LLᵀ`. This is the fast way to solve PD systems and to sample from `N(μ, Σ)` via `μ + Lz` with `z ~ N(0, I)`.

```python
import numpy as np

rng = np.random.default_rng(1)
X = rng.normal(size=(100, 3))
cov = np.cov(X, rowvar=False)
print(np.linalg.eigvalsh(cov).min() > 0)          # True: PD

L = np.linalg.cholesky(cov)                        # fails with LinAlgError if not PD
samples = rng.normal(size=(10000, 3)) @ L.T        # samples with covariance ~= cov
print(np.allclose(np.cov(samples, rowvar=False), cov, atol=0.05))
```

---

## Gradients, Jacobians, and Hessians

| Object | Function type | Shape | Meaning |
|---|---|---|---|
| Derivative | `R → R` | scalar | Slope |
| Gradient `∇f` | `R^n → R` | `n` vector | Direction of steepest ascent; `-∇f` is steepest descent |
| Jacobian `J` | `R^n → R^m` | `m×n` | `J_ij = ∂f_i/∂x_j`; best linear approximation of `f` |
| Hessian `H` | `R^n → R` | `n×n` | `H_ij = ∂²f/∂x_i∂x_j`; curvature. Symmetric if `f` is twice continuously differentiable |

**Taylor expansion** ties them together, and most optimizers are built from it:

```
f(x + δ) ≈ f(x) + ∇f(x)ᵀ δ + ½ δᵀ H(x) δ
```

Gradient descent uses the first-order term; Newton's method minimizes the full quadratic.

**Reading the Hessian at a critical point (`∇f = 0`):** all eigenvalues `> 0` means a local minimum, all `< 0` a local maximum, and mixed signs a saddle point. In high dimensions, saddles are by far the most common critical points in neural network losses. Zero eigenvalues mean flat directions, and the test is inconclusive.

The size of the Hessian is the practical problem: a model with 1B parameters would have a `10^18`-entry Hessian. That is why deep learning optimizers stay first-order or use cheap diagonal/Hessian-vector-product approximations.

---

## Chain Rule and Backpropagation

For a composition `L = f(g(h(x)))`, the chain rule gives `dL/dx = f'(g(h(x))) · g'(h(x)) · h'(x)`. With vectors, each factor is a Jacobian and the product is a chain of matrix multiplications.

**Backpropagation is reverse-mode automatic differentiation.** The two orders of evaluating that Jacobian product differ enormously in cost:

| Mode | Order | Cost for `f: R^n → R^m` | Best when |
|---|---|---|---|
| Forward mode | Input → output, carries `∂/∂x_j` for one input | One pass per **input** (`n` passes) | Few inputs, many outputs |
| Reverse mode | Output → input, carries `∂L/∂(node)` | One pass per **output** (`m` passes) | Many inputs, scalar output: **every loss function** |

A neural network has millions of inputs (parameters) and one output (scalar loss), so reverse mode gets the entire gradient in one backward pass, at roughly 2-3x the cost of the forward pass. The price is memory: every intermediate activation from the forward pass has to be stored for the backward pass. Gradient checkpointing trades recomputation for that memory.

**Tiny worked example.** `L = (σ(w·x + b) - y)²` with `w=0.5, x=2, b=-1, y=1`.

```
Forward:
  z = w·x + b      = 0.5·2 - 1 = 0
  a = σ(z)         = 0.5
  L = (a - y)²     = 0.25

Backward (each local derivative times the upstream gradient):
  dL/da = 2(a - y)          = -1.0
  da/dz = σ(z)(1 - σ(z))    = 0.25      -> dL/dz = -1.0 · 0.25 = -0.25
  dz/dw = x = 2             -> dL/dw = -0.25 · 2 = -0.5
  dz/db = 1                 -> dL/db = -0.25
```

```python
import numpy as np

def forward_backward(w, b, x, y):
    z = w * x + b
    a = 1 / (1 + np.exp(-z))
    L = (a - y) ** 2
    dL_da = 2 * (a - y)
    dL_dz = dL_da * a * (1 - a)
    return L, dL_dz * x, dL_dz          # loss, dL/dw, dL/db

print(forward_backward(0.5, -1.0, 2.0, 1.0))   # (0.25, -0.5, -0.25)
```

Note that `σ'(z) ≤ 0.25`. Chain ten sigmoids together and the gradient shrinks by at least `0.25^10 ≈ 10⁻⁶`. That is the vanishing-gradient problem in one line, and it is why ReLU (derivative 1 on the active side), residual connections, and careful initialization matter. See [Neural Network Training](../deep_learning/intro_neural_network_training.md).

---

## Matrix Calculus Identities

These cover almost every derivation you will be asked to do. Convention: gradient of a scalar with respect to a column vector is a column vector (denominator layout).

| Expression `f(x)` | `∇_x f` | Notes |
|---|---|---|
| `aᵀx` | `a` | Linear |
| `xᵀx = \|\|x\|\|²` | `2x` | |
| `xᵀAx` | `(A + Aᵀ)x` | `= 2Ax` if `A` is symmetric |
| `\|\|Ax - b\|\|²` | `2Aᵀ(Ax - b)` | Least squares |
| `Ax` (vector-valued) | Jacobian `A` | |
| `log(1 + e^{x})` (softplus) | `σ(x)` | Logistic loss building block |
| `σ(x)` | `σ(x)(1 - σ(x))` | |
| `log Σ_j e^{x_j}` (log-sum-exp) | `softmax(x)` | Why softmax + cross-entropy has a clean gradient |

**Matrix arguments:**

| Expression `f(W)` | `∇_W f` |
|---|---|
| `aᵀWb` | `abᵀ` |
| `tr(AW)` | `Aᵀ` |
| `\|\|W\|\|_F²` | `2W` |
| `log det W` | `W⁻ᵀ` |

**Linear layer rule** (the one you use constantly): for `Y = XW` with upstream gradient `G = ∂L/∂Y`:
```
∂L/∂W = Xᵀ G        ∂L/∂X = G Wᵀ
```

Sanity-check with shapes: `X` is `n×d`, `W` is `d×k`, `G` is `n×k`. The only way to get a `d×k` result from `X` and `G` is `XᵀG`. When you forget a transpose, matching shapes usually tells you where it goes.

---

## Worked Derivations: OLS and Logistic Regression

### OLS normal equations

Model `ŷ = Xw`, with `X ∈ R^{n×d}`. Loss:

```
L(w) = ||Xw - y||² = wᵀXᵀXw - 2yᵀXw + yᵀy
∇L   = 2XᵀXw - 2Xᵀy
Set to 0:  XᵀX w = Xᵀy                    (normal equations)
           w*    = (XᵀX)⁻¹ Xᵀy            (if XᵀX is invertible)
Ridge:     w*    = (XᵀX + λI)⁻¹ Xᵀy       (always invertible for λ > 0)
```

The Hessian is `2XᵀX`, which is PSD, so the loss is convex and any stationary point is a global minimum. Geometrically, `Xw*` is the orthogonal projection of `y` onto the column space of `X`: the residual `y - Xw*` is orthogonal to every column (`Xᵀ(y - Xw*) = 0` is literally the normal equations). Cost: forming `XᵀX` is `O(nd²)` and solving is `O(d³)`. That is fine for `d` in the thousands, but not for millions of features. Past that point, use (stochastic) gradient descent.

### Logistic regression gradient

Model `p_i = σ(x_iᵀw)`, labels `y_i ∈ {0, 1}`. Negative log-likelihood:

```
L(w) = -Σ_i [ y_i log p_i + (1 - y_i) log(1 - p_i) ]
```

Using `dσ/dz = σ(1-σ)`, per example `∂L_i/∂z_i = p_i - y_i` (the `σ(1-σ)` factors cancel), so:

```
∇L = Xᵀ(p - y)                 (same form as OLS: features times residual)
H  = Xᵀ S X,  S = diag(p_i(1 - p_i))   PSD -> convex
```

There is no closed-form solution because `p` is nonlinear in `w`. Solve with gradient descent or Newton's method (called IRLS, iteratively reweighted least squares, in this setting). If the data is linearly separable, the unregularized MLE does not exist: `||w|| → ∞`. Add L2 regularization.

```python
import numpy as np

rng = np.random.default_rng(0)
n, d = 500, 3
X = np.c_[np.ones(n), rng.normal(size=(n, d))]
w_true = np.array([0.5, 2.0, -1.0, 0.0])

# OLS closed form vs lstsq
y = X @ w_true + 0.1 * rng.normal(size=n)
w_ols = np.linalg.solve(X.T @ X, X.T @ y)
print(np.round(w_ols, 2))                          # ~[0.5, 2.0, -1.0, 0.0]

# Logistic regression by gradient descent
y_cls = (rng.random(n) < 1 / (1 + np.exp(-X @ w_true))).astype(float)
w = np.zeros(d + 1)
for _ in range(2000):
    p = 1 / (1 + np.exp(-X @ w))
    w -= 0.5 * X.T @ (p - y_cls) / n               # mean gradient
print(np.round(w, 2))                              # close to w_true (noisy)
```

---

## Convexity

A function is **convex** if the chord between any two points lies on or above the graph: `f(θx + (1-θ)y) ≤ θf(x) + (1-θ)f(y)`. Equivalently (if twice differentiable), `H ⪰ 0` everywhere.

**Why it matters:** for a convex function every local minimum is a global minimum, and gradient methods come with convergence guarantees. With strong convexity (`H ⪰ μI`, `μ > 0`) the minimizer is unique and gradient descent converges linearly.

| Convex | Not convex |
|---|---|
| Linear / ridge / lasso regression | Neural networks (any hidden layer) |
| Logistic regression, softmax regression | Matrix factorization `\|\|R - UVᵀ\|\|²` (convex in `U` or `V` alone, not jointly) |
| SVM (hinge loss + L2) | k-means objective |
| Norms, max of convex functions, log-sum-exp | Gaussian mixture log-likelihood |

**Composition rules:** sums with non-negative weights, pointwise maxima, and affine pre-compositions (`f(Ax + b)`) preserve convexity.

**Neural networks are non-convex, and they still train.** Two facts explain most of this. In high dimensions most critical points are saddle points rather than bad local minima, and SGD noise escapes saddles. Heavily overparameterized networks also tend to have many global-quality minima connected by low-loss paths. Nothing is guaranteed, but in practice it works.

---

## First-Order Optimization

**Gradient descent:** `w ← w - η ∇L(w)`.

**Learning rate and the condition number.** Near a minimum the loss is roughly quadratic with Hessian `H`, whose eigenvalues run from `λ_min` to `λ_max`. GD is stable only if `η < 2/λ_max`. The steepest direction sets the ceiling. Progress along the flattest direction per step is then about `η · λ_min ≤ 2λ_min/λ_max = 2/κ`. With condition number `κ = λ_max/λ_min = 10⁴`, you get oscillation across the narrow valley and a crawl along it, and you need `O(κ)` iterations. This is the reason for feature standardization (a rounder loss surface, smaller `κ`), for batch/layer normalization (partly the same role inside networks), and for momentum and adaptive methods (which cope with ill-conditioning).

| Optimizer | Update (simplified) | What it fixes | Caveat |
|---|---|---|---|
| **SGD** | `w -= η g` on a minibatch gradient `g` | Full-batch cost; noise helps escape saddles and may aid generalization | Sensitive to `η`; slow on ill-conditioned problems |
| **Momentum** | `v = βv + g;  w -= η v` | Averages out oscillation, accelerates along consistent directions (`~1/(1-β)` speedup) | Another hyperparameter (`β ≈ 0.9`) |
| **AdaGrad** | `w -= η g / sqrt(Σ g²)` | Per-parameter rates; good for sparse features | Accumulator only grows, so the LR decays to zero |
| **RMSProp** | EMA of `g²` instead of a sum | Fixes AdaGrad's decay | |
| **Adam** | Momentum (`m`) + RMSProp (`v`) + bias correction | Robust default; handles badly scaled gradients | Can generalize slightly worse than tuned SGD on vision |
| **AdamW** | Adam with weight decay decoupled from the gradient | L2 inside Adam is scaled by `1/sqrt(v)`, which is not true weight decay; AdamW fixes that | Standard for transformers |

```python
import numpy as np

def adam_step(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g**2
    m_hat = m / (1 - b1**t)          # bias correction: m, v start at 0
    v_hat = v / (1 - b2**t)
    w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    return w, m, v

# Ill-conditioned quadratic: L = 0.5 * (100 * w0^2 + 1 * w1^2), kappa = 100
grad = lambda w: np.array([100 * w[0], w[1]])
w_gd = np.array([1.0, 1.0])
for _ in range(100):
    w_gd = w_gd - 0.019 * grad(w_gd)          # eta just under 2/lambda_max = 0.02
print(w_gd)   # [~3e-05, ~0.15]: w0 flips sign every step (factor -0.9); w1 still crawls
```

**Learning rate schedules.** Warmup avoids early instability while Adam's second-moment estimates are still noisy and the network is at a sharp initial point. Cosine or linear decay then lets the iterate settle into a minimum instead of bouncing around at the SGD noise floor.

---

## Second-Order Methods

**Newton's method:** minimize the local quadratic model and you get `w ← w - H⁻¹ ∇L`. It rescales every direction by its curvature, so it is immune to conditioning. It converges quadratically near the optimum and solves a quadratic exactly in one step.

| Method | Idea | Cost per step | Used in |
|---|---|---|---|
| Newton | Exact `H⁻¹∇L` | `O(d²)` memory, `O(d³)` solve | Small convex problems, logistic regression (IRLS) |
| Gauss-Newton / Levenberg-Marquardt | Approximates `H ≈ JᵀJ` for least squares | Same order as Newton | Curve fitting, bundle adjustment |
| BFGS | Builds a `H⁻¹` estimate from gradient differences | `O(d²)` memory | Medium-sized smooth problems |
| **L-BFGS** | Keeps only the last `m` (≈10) gradient/step pairs | `O(md)` | scikit-learn's default `LogisticRegression` solver, CRFs, full-batch convex problems |
| K-FAC, Shampoo | Block/Kronecker-factored curvature | Moderate overhead | Some large-scale DL research and production |

**Why they are rare in deep learning:**
1. **Size.** `d` is in the millions to billions. An `O(d²)` Hessian cannot be stored and an `O(d³)` solve cannot be run.
2. **Noise.** Minibatch curvature estimates are noisy, and quasi-Newton methods like L-BFGS assume consistent gradients and break under stochastic ones.
3. **Non-convexity.** `H` has negative eigenvalues near saddles, so the raw Newton step moves *toward* saddles and maxima. It needs damping or trust regions.
4. **Adequate substitutes.** Adam's `1/sqrt(v)` is a cheap diagonal preconditioner, and it captures a good share of the benefit at `O(d)` cost.

---

## Constrained Optimization: Lagrange and KKT

**Equality constraints.** To minimize `f(x)` subject to `h(x) = 0`, form the Lagrangian `ℒ(x, ν) = f(x) + ν h(x)` and set `∇_x ℒ = 0`, `h(x) = 0`. At the optimum `∇f` is parallel to `∇h`: you cannot decrease `f` without leaving the constraint surface.

Classic example: PCA as `max_w wᵀΣw` subject to `wᵀw = 1`. Setting `∇ℒ = 2Σw - 2νw = 0` gives `Σw = νw`, so `w` is an eigenvector, and the objective equals `ν`, so you take the largest eigenvalue.

**KKT conditions.** For `min f(x)` subject to `g_i(x) ≤ 0` and `h_j(x) = 0`, a solution satisfies:

```
1. Stationarity:          ∇f + Σ μ_i ∇g_i + Σ ν_j ∇h_j = 0
2. Primal feasibility:    g_i(x) ≤ 0,  h_j(x) = 0
3. Dual feasibility:      μ_i ≥ 0
4. Complementary slackness: μ_i g_i(x) = 0
```

For convex problems satisfying a constraint qualification (e.g. Slater's condition), KKT is necessary and sufficient, and strong duality holds.

**SVM link.** The hard-margin SVM is `min ½||w||²` subject to `y_i(wᵀx_i + b) ≥ 1`. The dual is:

```
max_α  Σ α_i - ½ Σ_i Σ_j α_i α_j y_i y_j x_iᵀx_j     s.t. α_i ≥ 0, Σ α_i y_i = 0
w = Σ α_i y_i x_i
```

Two consequences interviewers look for:

- **Complementary slackness** gives `α_i > 0` only for points with `y_i(wᵀx_i + b) = 1`, i.e. points on the margin. Those are the **support vectors**. All other points have `α_i = 0` and do not affect the solution.
- The data enters the dual only through the dot products `x_iᵀx_j`. Replace them with `k(x_i, x_j)` and you have the **kernel trick**.

---

## Regularization: Constraint and Prior Views

Three equivalent views of `min L(w) + λ R(w)`:

| View | Statement | What it explains |
|---|---|---|
| Penalty | Add `λ\|\|w\|\|²` or `λ\|\|w\|\|_1` to the loss | The implementation |
| Constraint | `min L(w)` s.t. `R(w) ≤ t`; `λ` is the Lagrange multiplier for `t` | The L1-diamond sparsity picture; larger `λ` ↔ smaller `t` |
| Bayesian MAP | `argmax log p(D\|w) + log p(w)` | L2 ↔ Gaussian prior `N(0, σ²I)` with `λ ∝ 1/σ²`; L1 ↔ Laplace prior |

MAP derivation for L2: `-log p(w) = ||w||²/(2σ²) + const`. Add that to the negative log-likelihood and you get exactly the ridge penalty. More detail in [Statistics & Probability](./intro_statistics_probability.md).

**Weight decay vs L2.** In plain SGD, `w ← w - η(∇L + λw) = (1 - ηλ)w - η∇L`, so L2 and weight decay are identical. In Adam they are not, because the L2 gradient gets divided by `sqrt(v)`. That is why AdamW exists.

---

## Information Theory

| Quantity | Formula | Meaning |
|---|---|---|
| Entropy `H(p)` | `-Σ p(x) log p(x)` | Average surprise; minimum bits needed to encode samples from `p` |
| Cross-entropy `H(p, q)` | `-Σ p(x) log q(x)` | Bits needed to encode samples from `p` using a code built for `q` |
| KL divergence `D_KL(p‖q)` | `Σ p(x) log(p(x)/q(x)) = H(p,q) - H(p)` | Extra bits from using `q` instead of `p`. `≥ 0`, `= 0` iff `p = q`, **not symmetric** |
| Mutual information `I(X;Y)` | `D_KL(p(x,y) ‖ p(x)p(y))` | How much knowing `Y` reduces uncertainty about `X`. Used in feature selection and decision-tree information gain |

**Why cross-entropy loss is MLE.** Let `p` be the empirical data distribution (one-hot on the true label for each example) and `q_θ` the model. Then

```
H(p, q_θ) = -(1/n) Σ_i log q_θ(y_i | x_i) = average negative log-likelihood
```

Minimizing cross-entropy is exactly maximizing likelihood. And since `H(p)` does not depend on `θ`, minimizing cross-entropy also minimizes `D_KL(p‖q_θ)`. The same argument turns MSE into MLE under Gaussian noise.

**KL asymmetry matters in practice.** Forward KL `D_KL(p‖q)` (which MLE minimizes) is *mean-seeking*: `q` must put mass wherever `p` does, so it spreads out to cover all modes. Reverse KL `D_KL(q‖p)` (used in variational inference) is *mode-seeking*: `q` is punished for putting mass where `p` is small, so it locks onto one mode.

**Softmax + cross-entropy gradient**: with logits `z` and one-hot `y`, `∂L/∂z = softmax(z) - y`. It is bounded and never saturates the way `MSE ∘ sigmoid` does.

---

## Numerical Stability

Floating point has limited range: float32 overflows near `3.4e38`, so `exp(89)` is already `inf`. Float16 overflows at 65504, which means `exp(12)` is too big. The fixes all rely on algebraic identities that keep exponent arguments bounded.

**Log-sum-exp trick:** `log Σ exp(x_i) = m + log Σ exp(x_i - m)` with `m = max(x)`. The largest term becomes `exp(0) = 1`, so nothing overflows and at least one term does not underflow.

**Stable softmax:** subtract the max before exponentiating. The result is mathematically identical because softmax is shift-invariant.

**Stable sigmoid:** `1/(1+exp(-z))` overflows for very negative `z`. Branch on sign so the exponent argument is always `≤ 0`.

**Fused losses:** compute log-probabilities directly (`log_softmax`, `BCEWithLogitsLoss`, `softmax_cross_entropy_with_logits`) instead of `log(softmax(z))`. Otherwise tiny probabilities round to 0 and `log(0) = -inf`.

```python
import numpy as np

def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def log_softmax(x):
    return x - logsumexp(x)

def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])                      # z < 0 -> exp(z) in (0, 1), no overflow
    out[~pos] = ez / (1 + ez)
    return out

def bce_with_logits(z, y):
    """-[y log σ(z) + (1-y) log(1-σ(z))] rewritten to be stable for any z."""
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))

x = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    print(np.exp(x) / np.exp(x).sum())        # [nan nan nan]
print(softmax(x))                             # [0.090 0.245 0.665]
print(logsumexp(x))                           # 1002.4076
print(sigmoid(np.array([-1000.0, 0.0, 1000.0])))   # [0.  0.5 1. ] with no warnings
print(bce_with_logits(np.array([-50.0, 50.0]), np.array([1.0, 0.0])))  # [50. 50.] not inf
```

Other stability habits: add `eps` inside `log` and `sqrt` (Adam's denominator, normalization layers); use `np.log1p`/`np.expm1` for small arguments; keep float32 master weights and use loss scaling in fp16 mixed precision so small gradients do not underflow; clip gradients by global norm.

---

## Gradient Checking

When you write a gradient by hand (custom layer, custom loss, interview coding round), verify it against central finite differences:

```
∂f/∂x_i ≈ (f(x + h e_i) - f(x - h e_i)) / (2h)      error O(h²)
```

Use central rather than forward differences, whose error is `O(h)`. Use float64, and pick `h ≈ 1e-5`: much smaller and floating-point cancellation dominates. Compare with a **relative** error.

```python
import numpy as np

def numerical_grad(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(x.size):
        old = x.flat[i]
        x.flat[i] = old + h; f_plus = f(x)
        x.flat[i] = old - h; f_minus = f(x)
        x.flat[i] = old
        grad.flat[i] = (f_plus - f_minus) / (2 * h)
    return grad

def rel_error(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12)

# Check the analytic logistic-regression gradient X^T (p - y)
rng = np.random.default_rng(0)
X, y = rng.normal(size=(20, 4)), rng.integers(0, 2, 20).astype(float)
w = rng.normal(size=4)

def loss(w):
    z = X @ w
    return np.sum(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))

analytic = X.T @ (1 / (1 + np.exp(-X @ w)) - y)
print(rel_error(analytic, numerical_grad(loss, w.copy())))   # ~1e-10: correct
```

Rules of thumb for relative error: below `1e-7` is correct; around `1e-4` is suspicious unless there are kinks (ReLU, max, hinge, L1) near the evaluation point; above `1e-2` is a bug. Turn off dropout and other randomness during the check, and check a few coordinates rather than all of them for big models.

---

## Interview Q&A

#### Derive the closed-form solution for linear regression. When wouldn't you use it?

Write the loss as `||Xw - y||²`, expand, and take the gradient: `2XᵀXw - 2Xᵀy`. Setting it to zero gives the normal equations `XᵀXw = Xᵀy`, so `w = (XᵀX)⁻¹Xᵀy`. The Hessian `2XᵀX` is PSD, so this is a global minimum.

Don't use it when `d` is large, since forming and solving `XᵀX` is `O(nd² + d³)`. Don't use it when `XᵀX` is singular or badly conditioned (collinear features, `d > n`); use ridge, `lstsq`, or the pseudo-inverse instead. And don't use it when the data does not fit in memory, where SGD streams over it. Even when the closed form is appropriate, solve the system with `solve`/Cholesky/QR rather than computing the inverse.

#### Why does L1 regularization produce sparse solutions and L2 does not?

Geometrically, the L1 constraint region is a diamond whose corners sit on the axes. The loss contours typically touch it first at a corner, where some weights are exactly zero. The L2 ball is smooth, so the touching point is generically off-axis.

From the optimization side, the L1 subgradient `λ·sign(w)` pushes toward zero with constant force no matter how small `w` is, and any weight whose loss gradient is weaker than `λ` sits exactly at zero (soft thresholding). The L2 gradient `2λw` fades as `w` shrinks, so weights get small but not zero. The Bayesian version: a Laplace prior has a sharp peak at zero, while a Gaussian prior is flat there.

#### What is the relationship between SVD, eigendecomposition, and PCA?

For centered data `X = UΣVᵀ`, the covariance is `XᵀX/(n-1) = VΣ²Vᵀ/(n-1)`. The right singular vectors `V` are the eigenvectors of the covariance (the principal directions), and the eigenvalues are `σ_i²/(n-1)` (the variance along each direction). PCA can therefore be done either way. SVD is preferred because forming `XᵀX` squares the condition number and loses precision. Keeping the top `k` components is the best rank-`k` approximation (Eckart-Young).

#### What is the condition number and why does it matter for training?

For a matrix it is `σ_max/σ_min`. For a loss near its minimum it is `λ_max/λ_min` of the Hessian. Gradient descent needs `η < 2/λ_max` for stability, so progress along the flattest direction is limited to about `1/κ` per step. With high `κ` you zig-zag across a narrow valley and crawl along it. The fixes are to precondition the problem: standardize features, normalize activations, use momentum or Adam (a diagonal preconditioner), or go second-order. In linear systems, high `κ` also means small input perturbations cause large solution changes.

#### Explain backpropagation. Why reverse mode and not forward mode?

Backprop applies the chain rule from the output back toward the inputs, reusing each node's upstream gradient `∂L/∂node` and multiplying it by local Jacobians. Reverse mode costs one backward pass per *output*, and forward mode costs one pass per *input*. A loss is a single scalar and a network has millions of parameters, so reverse mode gets the full gradient in roughly the cost of one extra forward pass. Forward mode would need millions of passes. The tradeoff is memory: reverse mode must store forward activations. Gradient checkpointing recomputes some of them to save memory.

#### Derive the gradient of logistic regression. Is the problem convex?

With `p = σ(Xw)` and the negative log-likelihood, per example `∂L/∂z = p - y`, because the `σ(1-σ)` derivative cancels against the `1/p` and `1/(1-p)` terms of the log loss. So `∇L = Xᵀ(p - y)`. The Hessian is `XᵀSX` with `S = diag(p(1-p)) ≥ 0`, which is PSD, so the problem is convex. There is no closed form because `p` is nonlinear in `w`. On linearly separable data the loss keeps decreasing as `||w|| → ∞`, so the MLE does not exist without regularization.

#### Why is cross-entropy the standard classification loss rather than MSE?

First, it is maximum likelihood under a categorical/Bernoulli model: minimizing cross-entropy equals minimizing NLL and equals minimizing `D_KL(data ‖ model)`. Second, the gradient with respect to the logits is `p - y`, which stays large when the model is confidently wrong. MSE on a sigmoid output has gradient `(p - y)·σ'(z)`, and `σ'(z)` vanishes exactly when the model is confidently wrong, so learning stalls. Cross-entropy also gives a convex problem for logistic regression, while MSE on a sigmoid does not.

#### What does the Hessian tell you, and why don't we use Newton's method for neural networks?

The Hessian describes local curvature. Its eigenvalues classify critical points (all positive: minimum; mixed: saddle), its condition number predicts how hard first-order optimization will be, and the magnitude of its top eigenvalue relates to "sharpness" of minima. Newton's method `w -= H⁻¹∇L` fixes conditioning and converges quadratically. But for `d` parameters it needs `O(d²)` memory and an `O(d³)` solve, minibatch curvature estimates are noisy, and in non-convex regions it gets attracted to saddles. L-BFGS works well for full-batch convex problems such as scikit-learn's logistic regression. Deep learning uses Adam's diagonal preconditioning instead.

#### Why does Adam often converge faster than SGD, and when might you still prefer SGD?

Adam divides each coordinate's step by a running RMS of its gradient. That makes the step size roughly invariant to gradient scale per parameter, so it handles ill-conditioned or badly scaled problems and sparse gradients without per-parameter tuning, and its momentum term smooths noise. SGD with momentum and a well-tuned schedule sometimes generalizes slightly better, historically on image classification with CNNs, and it uses less optimizer memory (Adam stores two extra tensors per parameter). For transformers, AdamW is the default. If you use Adam, use AdamW so weight decay is not distorted by the adaptive scaling.

#### Explain KKT conditions and how they show up in SVMs.

KKT generalizes Lagrange multipliers to inequality constraints. The conditions are stationarity of the Lagrangian, primal feasibility, non-negative multipliers, and complementary slackness (`μ_i g_i(x) = 0`: a multiplier can be non-zero only if its constraint is active). For convex problems with a constraint qualification they characterize the optimum exactly. In the SVM dual, complementary slackness means only points on the margin have `α_i > 0`. These are the support vectors, and the solution `w = Σα_i y_i x_i` depends only on them. Because the dual touches the data only through dot products, you can replace them with a kernel.

#### What is KL divergence and why is it asymmetric?

`D_KL(p‖q) = E_p[log p/q]` is the expected extra code length from using `q` to encode data from `p`. It is non-negative and zero iff `p = q`, but it is not a metric, because the expectation is taken under `p`. Forward KL `D_KL(p‖q)` blows up where `p > 0` and `q ≈ 0`, so `q` must cover all of `p`'s modes (mean-seeking; this is what MLE does). Reverse KL `D_KL(q‖p)` blows up where `q > 0` and `p ≈ 0`, so `q` hides inside a single mode (mode-seeking; this is what variational inference and many RL policy constraints use).

#### Your training loss suddenly becomes NaN. What do you check?

Work from the math outward. Check for exponential overflow: unstable softmax/sigmoid, or `exp` of unbounded logits. Check for `log(0)` or division by zero: probabilities computed and then logged separately instead of through a fused `log_softmax` or with-logits loss, or a missing `eps` in normalization or the Adam denominator. Check for exploding gradients from too high a learning rate relative to curvature (`η > 2/λ_max`), which you can fix with gradient clipping, warmup, or a lower LR. In fp16, check for overflow above 65504 (use loss scaling or bf16). And check for bad inputs: NaN or inf in the data, or unnormalized features with huge scale. Use `torch.autograd.detect_anomaly()` or per-layer gradient-norm logging to find the first bad op.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| `np.linalg.inv(X.T @ X) @ X.T @ y` | Explicit inverse is slow; forming `XᵀX` squares the condition number | `np.linalg.lstsq`, `solve`, or QR/Cholesky |
| Not standardizing features before GD | Different feature scales inflate the Hessian's condition number; training is slow or unstable | Standardize; use normalization layers |
| Regularizing the bias term | Pulls the intercept toward zero for no reason and shifts predictions | Exclude the bias from the penalty |
| `np.exp(logits) / sum(...)` | Overflows to `inf/inf = nan` for large logits | Subtract the max; use `log_softmax` |
| `log(sigmoid(z))` or `log(softmax(z))` computed separately | Probabilities round to 0, giving `-inf` loss and NaN gradients | Fused with-logits losses |
| Using L2 penalty inside Adam as "weight decay" | Adaptive scaling weakens decay on high-variance parameters | AdamW (decoupled decay) |
| Assuming a local minimum is global | Only true for convex problems | Know which models are convex; use multiple restarts for non-convex ones |
| Treating KL as symmetric / a distance | Forward and reverse KL give very different fits | Pick the direction deliberately; use JS divergence if you need symmetry |
| Forward differences in gradient checks, or float32 | `O(h)` error plus rounding give false alarms | Central differences, float64, `h ≈ 1e-5`, relative error |
| Gradient-checking with dropout on or near ReLU kinks | Randomness or non-differentiable points produce spurious mismatches | Fix seeds / disable dropout; nudge away from kinks |
| Confusing eigen- and singular values for non-symmetric matrices | They differ unless the matrix is symmetric PSD | Use SVD for general matrices, `eigh` for symmetric ones |
| Picking the learning rate without regard to curvature | `η > 2/λ_max` diverges; far below it crawls | LR range test, warmup, schedules |

---

## Related Topics

- [Statistics & Probability](./intro_statistics_probability.md)
- [Dimensionality Reduction](./intro_dimensionality_reduction.md)
- [Model Evaluation](./intro_model_evaluation.md)
- [Feature Engineering](./intro_feature_engineering.md)
- [Recommender Systems](./intro_recommender_systems.md)
- [Neural Network Training](../deep_learning/intro_neural_network_training.md)
- [Fine-Tuning (LoRA)](../deep_learning/intro_fine_tuning.md)
- [Transformers](../deep_learning/intro_transformers.md)
- [Generative Models](../deep_learning/intro_generative_models.md)
- [ML Coding Challenges](../coding_challenges/ml_coding_challenges.md)
- [Classical ML Overview](./README.md)
