"""
ICA Algorithm Implementations
==============================
Author: Ratanakmuny NOUV
Course: M2 Data Science — Université d'Évry, 2026

Implements 4 ICA algorithms:
    1. InfomaxBatch   — full-batch gradient ascent (Bell & Sejnowski, 1995)
    2. FastICAWrapper  — fixed-point baseline via sklearn (Hyvärinen, 1999)
    3. SGD_ICA         — stochastic gradient with Robbins-Monro decay
    4. EM_ICA          — stochastic EM with learned MoG score (creative contribution)

Gradient Formula (shared by algorithms 1, 3, 4):
    On whitened data, V is orthogonal and V^{-T} = V (course Section 5.6.3).
    The gradient ascent update is:

        V ← V + η (V + (1/N) g(Y) X^T)

    followed by symmetric decorrelation to enforce VV^T = I.

Score Function Convention:
    We use the true score g(y) = d/dy log p(y):
        - Super-Gaussian: g(y) = -tanh(y)   [from p ∝ 1/cosh(y)]
        - Sub-Gaussian:   g(y) = -y + tanh(y) [Extended Infomax]

    Note: the course writes g = +tanh because it uses the contrast
    derivative G' instead of the score. Since G ≈ -log p, we have
    G' = -score. Both conventions yield valid ICA solutions up to
    the sign ambiguity (course Section 5.3.2).

Data Convention:
    X : (D, N) — D signals, N samples (columns = observations)
    V : (D, D) — unmixing matrix, so Y = V @ X recovers sources
    W : (D, D) — true mixing matrix (for evaluation only)
"""

import numpy as np
from typing import Optional, Dict
from sklearn.decomposition import FastICA as SklearnFastICA
from src.utils import amari_index


# ================================================================
#  SHARED UTILITY: Symmetric Decorrelation
# ================================================================

def symmetric_decorrelation(V: np.ndarray) -> np.ndarray:
    """
    Project V onto the orthogonal group O(D).

    Formula: V ← (V V^T)^{-1/2} V

    On whitened data, V must satisfy VV^T = I (course Section 5.6.5).
    Each gradient step breaks this constraint slightly. This projection
    restores orthogonality, analogous to how FastICA's parallel mode
    decorrelates all components simultaneously.

    Implementation: eigendecomposition of VV^T, then apply D^{-1/2}.

    Parameters
    ----------
    V : ndarray (D, D)
        Matrix to orthogonalize.

    Returns
    -------
    V_orth : ndarray (D, D)
        Orthogonal matrix closest to V.
    """
    VVt = V @ V.T
    eigvals, eigvecs = np.linalg.eigh(VVt)
    eigvals = np.maximum(eigvals, 1e-10)  # numerical safety
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    return eigvecs @ D_inv_sqrt @ eigvecs.T @ V


# ================================================================
#  SHARED UTILITY: Adaptive Score Function
# ================================================================

def adaptive_score(Y: np.ndarray) -> np.ndarray:
    """
    Compute the score function g(y) = d/dy log p(y) per source,
    adapting to each source's distribution type via kurtosis.

    This implements the Extended Infomax idea (Lee, Girolami,
    Sejnowski, 1999): detect whether each source is sub- or
    super-Gaussian using the excess kurtosis, and apply the
    appropriate score function.

    Score functions:
        Super-Gaussian (kurtosis > 0):
            g(y) = -tanh(y)
            Derived from: p(y) ∝ 1/cosh(y), log p = -log cosh + C

        Sub-Gaussian (kurtosis ≤ 0):
            g(y) = -y + tanh(y)
            From Extended Infomax (Lee et al., 1999)

    Parameters
    ----------
    Y : ndarray (D, N)
        Current source estimates (Y = V @ X).

    Returns
    -------
    gY : ndarray (D, N)
        Score function values, same shape as Y.
    """
    D, N = Y.shape
    gY = np.zeros_like(Y)
    tanh_Y = np.tanh(Y)  # precompute once for efficiency

    for j in range(D):
        y = Y[j]
        v = y.var()

        # Compute excess kurtosis: E[(y-μ)^4]/σ^4 - 3
        # Gaussian has kurtosis = 0
        if v > 1e-10:
            k = ((y - y.mean()) ** 4).mean() / (v ** 2) - 3.0
        else:
            k = 0.0

        # Select score based on source type
        if k > 0:
            gY[j] = -tanh_Y[j]            # super-Gaussian score
        else:
            gY[j] = -Y[j] + tanh_Y[j]     # sub-Gaussian score

    return gY


# ================================================================
#  ALGORITHM 1: Infomax Batch (Bell & Sejnowski, 1995)
# ================================================================

class InfomaxBatch:
    """
    Infomax ICA with full-batch gradient ascent.

    Reference: Bell & Sejnowski (1995), "An information-maximization
    approach to blind separation", Neural Computation.

    Update rule (course Section 5.6.3, on whitened data):
        V ← V + η (V + (1/N) g(Y) X^T)
        V ← symmetric_decorrelation(V)

    Uses adaptive kurtosis switching (Extended Infomax) to handle
    both sub- and super-Gaussian sources.

    Parameters
    ----------
    n_components : int
        Number of sources to extract (D).
    lr : float
        Learning rate η.
    max_iter : int
        Maximum number of gradient steps.
    tol : float
        Convergence tolerance (measured on Stiefel manifold).
    """

    def __init__(self, n_components: int = 3, lr: float = 0.1,
                 max_iter: int = 1000, tol: float = 1e-7):
        self.n_components = n_components
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.V_ = None       # Learned unmixing matrix
        self.history_ = []   # Amari index per iteration (if W_true given)

    def fit(self, X: np.ndarray, W_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the Infomax model.

        Parameters
        ----------
        X : ndarray (D, N)
            Whitened observed data.
        W_true : ndarray (D, D), optional
            True mixing matrix (for tracking Amari index during training).

        Returns
        -------
        V : ndarray (D, D)
            Estimated unmixing matrix.
        """
        D, N = X.shape
        self.history_ = []

        # Initialize V near identity + small random perturbation
        V = np.eye(D) + 0.01 * np.random.randn(D, D)
        V = symmetric_decorrelation(V)

        for iteration in range(self.max_iter):
            # 1. Compute estimated sources
            Y = V @ X

            # 2. Compute adaptive score function
            gY = adaptive_score(Y)

            # 3. Gradient: V + g(Y) X^T / N  (since V^{-T} = V on whitened data)
            grad = V + (gY @ X.T) / N

            # 4. Gradient ascent step
            V_new = V + self.lr * grad

            # 5. Enforce orthogonality
            V_new = symmetric_decorrelation(V_new)

            # 6. NaN protection (reduce lr if numerically unstable)
            if np.any(np.isnan(V_new)):
                self.lr *= 0.5
                continue

            # 7. Track convergence (optional)
            if W_true is not None:
                self.history_.append(amari_index(V_new, W_true))

            # 8. Check convergence on Stiefel manifold
            # Measures how much V rotated since last step
            delta = np.max(np.abs(np.abs(V_new @ V.T) - np.eye(D)))
            V = V_new

            if delta < self.tol and iteration > 10:
                break

        self.V_ = V
        return V

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Recover sources: Y = V @ X."""
        return self.V_ @ X


# ================================================================
#  ALGORITHM 2: FastICA Wrapper (Hyvärinen, 1999)
# ================================================================

class FastICAWrapper:
    """
    Wrapper around sklearn FastICA for consistent interface.

    Reference: Hyvärinen (1999), "Fast and robust fixed-point
    algorithms for ICA", IEEE Trans. Neural Networks.

    FastICA uses a fixed-point iteration (approximate Newton method)
    to maximize negentropy. Unlike gradient-based methods, it has
    cubic convergence and requires no learning rate tuning.

    Key difference from Infomax: FastICA solves the stationarity
    condition directly (course Section 5.8), rather than following
    the gradient.

    Parameters
    ----------
    n_components : int
        Number of sources.
    max_iter : int
        Maximum fixed-point iterations.
    tol : float
        Convergence tolerance.
    fun : str
        Contrast function ('logcosh', 'exp', 'cube').
    """

    def __init__(self, n_components: int = 3, max_iter: int = 500,
                 tol: float = 1e-4, fun: str = 'logcosh'):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.fun = fun
        self.model_ = None   # sklearn FastICA instance
        self.V_ = None       # Unmixing matrix

    def fit(self, X: np.ndarray, W_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit FastICA on pre-whitened data.

        Parameters
        ----------
        X : ndarray (D, N)
            Whitened data.
        W_true : optional
            Not used (included for consistent interface).

        Returns
        -------
        V : ndarray (D, D)
            Unmixing matrix.
        """
        self.model_ = SklearnFastICA(
            n_components=self.n_components,
            algorithm='parallel',   # Parallel extraction (all components at once)
            whiten=False,           # We pre-whiten ourselves
            fun=self.fun,           # Contrast: G(u) = log cosh(u)
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.model_.fit_transform(X.T)  # sklearn expects (N, D)
        self.V_ = self.model_.components_
        return self.V_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Recover sources: Y = V @ X."""
        return self.V_ @ X


# ================================================================
#  ALGORITHM 3: SGD-ICA (Stochastic Gradient Descent)
# ================================================================

class SGD_ICA:
    """
    Stochastic Gradient Descent on the Infomax criterion.

    Mini-batch version of the batch gradient (course Section 5.6.4):
        V ← V + η_t (V + (1/B) g(Y_batch) X_batch^T)
        V ← symmetric_decorrelation(V)

    The learning rate decays as η_t = η_0 / (1 + t/τ), satisfying
    the Robbins-Monro conditions for stochastic approximation:
        Σ η_t = ∞  and  Σ η_t² < ∞

    This enables online learning: data can be processed in streaming
    fashion without storing the full dataset.

    Parameters
    ----------
    n_components : int
        Number of sources (D).
    lr : float
        Initial learning rate η_0.
    decay : float
        Decay timescale τ.
    batch_size : int
        Mini-batch size B.
    n_epochs : int
        Number of passes over the full dataset.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_components: int = 3, lr: float = 0.1,
                 decay: float = 2000, batch_size: int = 64,
                 n_epochs: int = 50, seed: int = 42):
        self.n_components = n_components
        self.lr = lr
        self.decay = decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.seed = seed
        self.V_ = None       # Learned unmixing matrix
        self.history_ = []   # Amari index per epoch

    def fit(self, X: np.ndarray, W_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit SGD-ICA.

        Parameters
        ----------
        X : ndarray (D, N)
            Whitened data.
        W_true : optional
            True mixing matrix (for Amari tracking).

        Returns
        -------
        V : ndarray (D, D)
            Estimated unmixing matrix.
        """
        rng = np.random.default_rng(self.seed)
        D, N = X.shape
        self.history_ = []

        # Initialize near identity
        V = np.eye(D) + 0.01 * rng.standard_normal((D, D))
        V = symmetric_decorrelation(V)

        step = 0
        for epoch in range(self.n_epochs):
            # Shuffle data indices each epoch
            indices = rng.permutation(N)

            # Process mini-batches
            for start in range(0, N - self.batch_size + 1, self.batch_size):
                X_batch = X[:, indices[start:start + self.batch_size]]
                B = X_batch.shape[1]

                # 1. Estimated sources for this batch
                Y_batch = V @ X_batch

                # 2. Adaptive score
                gY = adaptive_score(Y_batch)

                # 3. Stochastic gradient (same formula, mini-batch average)
                grad = V + (gY @ X_batch.T) / B

                # 4. Decaying learning rate (Robbins-Monro)
                lr_t = self.lr / (1 + step / self.decay)

                # 5. Gradient step + orthogonality projection
                V_new = V + lr_t * grad
                V_new = symmetric_decorrelation(V_new)

                # 6. NaN protection
                if not np.any(np.isnan(V_new)):
                    V = V_new
                step += 1

            # Track Amari index at end of each epoch
            if W_true is not None:
                self.history_.append(amari_index(V, W_true))

        self.V_ = V
        return V

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Recover sources: Y = V @ X."""
        return self.V_ @ X


# ================================================================
#  ALGORITHM 4: EM-ICA (Creative Contribution)
# ================================================================

class EM_ICA:
    """
    Stochastic EM for ICA with learned source densities.

    MOTIVATION:
        Algorithms 1 and 3 use a fixed score (kurtosis switching between
        -tanh and -y+tanh). This is limited to 2 density shapes and fails
        when sources have similar kurtosis but different density shapes
        (e.g., real photographs). EM-ICA learns the density of each source.

    METHOD:
        Model each source density as a Mixture of Gaussians (MoG):
            p_j(z_j) = Σ_k π_{jk} N(z_j | μ_{jk}, σ²_{jk})

        The MoG score (true derivative of log p) is:
            g_j(y) = d/dy log p_j(y) = -Σ_k r_{jk} (y - μ_{jk}) / σ²_{jk}

        where r_{jk} are posterior responsibilities from the E-step.

    ALGORITHM (per mini-batch):
        E-step:  Compute responsibilities r_{jk} for each source
        M-step:  1. Update MoG parameters (π, μ, σ²) via EMA
                 2. Update V using gradient with learned score:
                    V ← V + η_t (V + (1/B) g_MoG(Y) X^T)
                    V ← symmetric_decorrelation(V)

    KEY ADVANTAGE:
        The score adapts continuously to the data — no binary kurtosis
        switching. Particularly effective when sources have similar
        kurtosis but different density shapes (demonstrated on real photos:
        Amari 0.034 vs 0.086 for kurtosis-switching methods).

    Parameters
    ----------
    n_components : int
        Number of sources (D).
    n_gaussians : int
        Number of Gaussian components K per source in the MoG.
    lr : float
        Initial learning rate.
    decay : float
        Learning rate decay timescale.
    batch_size : int
        Mini-batch size.
    n_epochs : int
        Number of passes over the dataset.
    seed : int
        Random seed.
    """

    def __init__(self, n_components: int = 3, n_gaussians: int = 4,
                 lr: float = 0.1, decay: float = 2000,
                 batch_size: int = 64, n_epochs: int = 50,
                 seed: int = 42):
        self.n_components = n_components
        self.n_gaussians = n_gaussians
        self.lr = lr
        self.decay = decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.seed = seed
        self.V_ = None         # Learned unmixing matrix
        self.history_ = []     # Amari index per epoch
        self.mog_weights_ = None  # π_{jk}: (D, K) mixture weights
        self.mog_means_ = None    # μ_{jk}: (D, K) component means
        self.mog_vars_ = None     # σ²_{jk}: (D, K) component variances

    # --- MoG Initialization ---

    def _init_mog(self, D: int):
        """Initialize MoG parameters for each source."""
        K = self.n_gaussians
        rng = np.random.default_rng(self.seed + 100)

        self.mog_weights_ = np.ones((D, K)) / K              # uniform weights
        self.mog_means_ = rng.standard_normal((D, K)) * 0.5  # small random means
        self.mog_vars_ = np.ones((D, K)) * 1.0               # unit variances

    # --- E-step ---

    def _e_step(self, Y: np.ndarray) -> np.ndarray:
        """
        Compute posterior responsibilities for each source's MoG.

        For source j, component k, sample i:
            r_{jk,i} = π_{jk} N(y_{j,i} | μ_{jk}, σ²_{jk})
                       / Σ_{k'} π_{jk'} N(y_{j,i} | μ_{jk'}, σ²_{jk'})

        Uses log-sum-exp trick for numerical stability.

        Parameters
        ----------
        Y : ndarray (D, B)
            Current source estimates for the mini-batch.

        Returns
        -------
        R : ndarray (D, K, B)
            Posterior responsibilities.
        """
        D, B = Y.shape
        K = self.n_gaussians
        R = np.zeros((D, K, B))

        for j in range(D):
            # Compute log responsibilities (unnormalized)
            for k in range(K):
                diff = Y[j, :] - self.mog_means_[j, k]
                var = self.mog_vars_[j, k] + 1e-8
                R[j, k, :] = (np.log(self.mog_weights_[j, k] + 1e-10)
                               - 0.5 * np.log(2 * np.pi * var)
                               - 0.5 * diff ** 2 / var)

            # Normalize via log-sum-exp (prevents over/underflow)
            log_max = np.max(R[j], axis=0)
            R[j] = np.exp(R[j] - log_max[np.newaxis, :])
            R[j] /= (R[j].sum(axis=0, keepdims=True) + 1e-10)

        return R

    # --- M-step: MoG parameter update ---

    def _m_step_mog(self, Y: np.ndarray, R: np.ndarray):
        """
        Update MoG parameters using exponential moving average (EMA).

        EMA is used instead of full M-step for stochastic stability:
            θ ← (1 - α) θ_old + α θ_batch

        where α = 0.1 is the EMA coefficient.

        Parameters
        ----------
        Y : ndarray (D, B)
            Current source estimates.
        R : ndarray (D, K, B)
            Posterior responsibilities from E-step.
        """
        D, B = Y.shape
        alpha = 0.1  # EMA coefficient (controls update speed)

        for j in range(D):
            for k in range(self.n_gaussians):
                r = R[j, k, :]           # responsibilities for component k
                N_jk = r.sum() + 1e-10   # effective count

                # Batch estimates
                new_weight = N_jk / B
                new_mean = (r * Y[j]).sum() / N_jk
                new_var = np.clip((r * (Y[j] - new_mean) ** 2).sum() / N_jk,
                                  0.01, 10.0)  # clip for stability

                # EMA update
                self.mog_weights_[j, k] = (1 - alpha) * self.mog_weights_[j, k] + alpha * new_weight
                self.mog_means_[j, k] = (1 - alpha) * self.mog_means_[j, k] + alpha * new_mean
                self.mog_vars_[j, k] = (1 - alpha) * self.mog_vars_[j, k] + alpha * new_var

            # Re-normalize mixture weights to sum to 1
            self.mog_weights_[j] /= (self.mog_weights_[j].sum() + 1e-10)

    # --- M-step: Learned score function ---

    def _learned_score(self, Y: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Compute the MoG-derived score function.

        The score of a MoG density is:
            g_j(y) = d/dy log p_j(y) = -Σ_k r_{jk} (y - μ_{jk}) / σ²_{jk}

        This replaces the fixed -tanh / kurtosis-switching score used
        in Infomax and SGD-ICA.

        Parameters
        ----------
        Y : ndarray (D, B)
            Current source estimates.
        R : ndarray (D, K, B)
            Posterior responsibilities.

        Returns
        -------
        gY : ndarray (D, B)
            Learned score function values.
        """
        D, B = Y.shape
        gY = np.zeros_like(Y)

        for j in range(D):
            for k in range(self.n_gaussians):
                r = R[j, k, :]
                var = self.mog_vars_[j, k] + 1e-8
                gY[j] += -r * (Y[j] - self.mog_means_[j, k]) / var

        return gY

    # --- Main fit method ---

    def fit(self, X: np.ndarray, W_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit EM-ICA using stochastic EM.

        Parameters
        ----------
        X : ndarray (D, N)
            Whitened data.
        W_true : optional
            True mixing matrix (for Amari tracking).

        Returns
        -------
        V : ndarray (D, D)
            Estimated unmixing matrix.
        """
        rng = np.random.default_rng(self.seed)
        D, N = X.shape
        self.history_ = []

        # Initialize unmixing matrix and MoG parameters
        V = np.eye(D) + 0.01 * rng.standard_normal((D, D))
        V = symmetric_decorrelation(V)
        self._init_mog(D)

        step = 0
        for epoch in range(self.n_epochs):
            indices = rng.permutation(N)

            for start in range(0, N - self.batch_size + 1, self.batch_size):
                X_batch = X[:, indices[start:start + self.batch_size]]
                B = X_batch.shape[1]

                # Current source estimates
                Y_batch = V @ X_batch

                # === E-step: compute responsibilities ===
                R = self._e_step(Y_batch)

                # === M-step part 1: update MoG parameters ===
                self._m_step_mog(Y_batch, R)

                # === M-step part 2: update V with learned score ===
                gY = self._learned_score(Y_batch, R)

                # Same gradient formula as Infomax/SGD
                grad = V + (gY @ X_batch.T) / B

                # Gradient clipping for stability
                gn = np.linalg.norm(grad)
                if gn > 10.0:
                    grad *= 10.0 / gn

                # Decaying learning rate
                lr_t = self.lr / (1 + step / self.decay)

                # Gradient step + orthogonality projection
                V_new = V + lr_t * grad
                V_new = symmetric_decorrelation(V_new)

                # NaN protection
                if not np.any(np.isnan(V_new)):
                    V = V_new
                step += 1

            # Track Amari index at end of each epoch
            if W_true is not None:
                self.history_.append(amari_index(V, W_true))

        self.V_ = V
        return V

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Recover sources: Y = V @ X."""
        return self.V_ @ X

    def get_learned_densities(self) -> Dict:
        """
        Return the learned MoG parameters for visualization.

        Returns
        -------
        params : dict with keys 'weights', 'means', 'vars'
            Each is an ndarray of shape (D, K).
        """
        return {
            'weights': self.mog_weights_.copy(),
            'means': self.mog_means_.copy(),
            'vars': self.mog_vars_.copy(),
        }


# ================================================================
#  CONVENIENCE: Multi-run Experiment
# ================================================================

def run_experiment(X: np.ndarray, W_true: np.ndarray,
                   algorithms: dict, n_runs: int = 10,
                   seed_base: int = 0) -> Dict:
    """
    Run multiple algorithms over multiple random seeds and
    collect Amari indices for statistical comparison.

    Parameters
    ----------
    X : ndarray (D, N)
        Whitened data.
    W_true : ndarray (D, D)
        True mixing matrix.
    algorithms : dict
        {name: algorithm_instance}
    n_runs : int
        Number of repetitions with different seeds.
    seed_base : int
        Starting seed.

    Returns
    -------
    results : dict
        {name: {'amari_scores': list, 'histories': list}}
    """
    results = {}
    for name, algo in algorithms.items():
        scores, histories = [], []
        for run in range(n_runs):
            # Re-seed for reproducibility
            if hasattr(algo, 'seed'):
                algo.seed = seed_base + run
            np.random.seed(seed_base + run)

            # Fit and evaluate
            V = algo.fit(X, W_true)
            scores.append(amari_index(V, W_true))
            if hasattr(algo, 'history_') and algo.history_:
                histories.append(algo.history_.copy())

        results[name] = {'amari_scores': scores, 'histories': histories}
        print(f"{name}: Amari = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    return results
