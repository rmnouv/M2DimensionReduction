"""
Utility Functions for ICA Project
=================================
Author: Ratanakmuny NOUV
Course: M2 Data Science — Université d'Évry, 2026

This module provides:
    1. Synthetic data generation (multiple source distributions)
    2. Data preprocessing (whitening)
    3. Evaluation metric (Amari index)
    4. Source matching (permutation + sign correction)
    5. Plotting helpers (convergence, bar plots, source comparison)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


# ================================================================
#  1. DATA GENERATION
# ================================================================

def generate_sources(n_samples: int, n_sources: int = 3,
                     source_types: Optional[List[str]] = None,
                     seed: int = 42) -> np.ndarray:
    """
    Generate independent non-Gaussian source signals.

    Each source is drawn from a different distribution to test ICA's
    ability to separate signals with varying statistical properties.

    Parameters
    ----------
    n_samples : int
        Number of observations (time steps).
    n_sources : int
        Number of independent sources (D).
    source_types : list of str, optional
        Distribution for each source. Supported types:
            'laplace'   — super-Gaussian (heavy tails, kurtosis > 0)
            'uniform'   — sub-Gaussian (flat, kurtosis < 0)
            'sawtooth'  — periodic, sub-Gaussian
            'binary'    — ±1 random, super-Gaussian
            'student_t' — heavy-tailed, super-Gaussian (df=5)
            'exp_mix'   — asymmetric mixture of exponentials
        Defaults to cycling through all types.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    S : ndarray of shape (n_sources, n_samples)
        Source matrix. Each row is one source, standardized to
        zero mean and unit variance.
    """
    rng = np.random.default_rng(seed)

    # Default: cycle through available distributions
    all_types = ['laplace', 'uniform', 'sawtooth', 'binary', 'student_t', 'exp_mix']
    if source_types is None:
        source_types = [all_types[i % len(all_types)] for i in range(n_sources)]

    S = np.zeros((n_sources, n_samples))

    for j, stype in enumerate(source_types):
        if stype == 'laplace':
            # Laplace distribution: super-Gaussian, heavy tails
            S[j] = rng.laplace(0, 1, n_samples)

        elif stype == 'uniform':
            # Uniform on [-√3, √3]: sub-Gaussian, unit variance
            S[j] = rng.uniform(-np.sqrt(3), np.sqrt(3), n_samples)

        elif stype == 'sawtooth':
            # Periodic sawtooth wave: sub-Gaussian
            t = np.linspace(0, 8 * np.pi, n_samples)
            freq = 1.0 + 0.3 * j  # slightly different frequency per source
            S[j] = 2 * (t * freq / (2 * np.pi) - np.floor(0.5 + t * freq / (2 * np.pi)))

        elif stype == 'binary':
            # Binary ±1: extremely super-Gaussian
            S[j] = rng.choice([-1.0, 1.0], size=n_samples)

        elif stype == 'student_t':
            # Student-t with df=5: super-Gaussian, heavy tails
            S[j] = rng.standard_t(df=5, size=n_samples)

        elif stype == 'exp_mix':
            # Symmetric mixture of exponentials: asymmetric, super-Gaussian
            signs = rng.choice([-1, 1], size=n_samples)
            S[j] = signs * rng.exponential(1, n_samples)

        else:
            raise ValueError(f"Unknown source type: '{stype}'. "
                             f"Supported: {all_types}")

    # Standardize: zero mean, unit variance (ICA convention)
    S = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
    return S


def generate_mixing_matrix(n_sources: int, condition_number: float = None,
                           seed: int = 42) -> np.ndarray:
    """
    Generate a random invertible mixing matrix W.

    The matrix is checked to be well-conditioned (cond < 20) to ensure
    the ICA problem is solvable. Poorly conditioned matrices make the
    problem numerically unstable.

    Parameters
    ----------
    n_sources : int
        Dimension D (number of sources = number of sensors).
    condition_number : float, optional
        If set, enforces a specific condition number via SVD.
    seed : int
        Random seed.

    Returns
    -------
    W : ndarray of shape (D, D)
        Invertible mixing matrix.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n_sources, n_sources))

    # Optionally enforce a specific condition number
    if condition_number is not None:
        U, s, Vt = np.linalg.svd(W)
        s = np.linspace(condition_number, 1, n_sources)
        W = U @ np.diag(s) @ Vt

    # Reject ill-conditioned matrices (would cause numerical issues)
    while np.linalg.cond(W) > 20:
        W = rng.standard_normal((n_sources, n_sources))

    return W


def mix_sources(S: np.ndarray, W: np.ndarray,
                noise_std: float = 0.0, seed: int = 42) -> np.ndarray:
    """
    Create observed mixtures: X = W @ S + noise.

    This simulates the ICA generative model x = Wz + ε.

    Parameters
    ----------
    S : ndarray (D, N)
        Source signals.
    W : ndarray (D, D)
        Mixing matrix.
    noise_std : float
        Standard deviation of additive Gaussian noise (0 = noiseless).
    seed : int
        Random seed for noise generation.

    Returns
    -------
    X : ndarray (D, N)
        Observed mixture signals.
    """
    X = W @ S
    if noise_std > 0:
        rng = np.random.default_rng(seed + 999)
        X += noise_std * rng.standard_normal(X.shape)
    return X


# ================================================================
#  2. PREPROCESSING
# ================================================================

def whiten(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center and whiten the data (course Section 5.4).

    Whitening transforms X so that:
        E[X_white @ X_white^T] = I  (identity covariance)

    This simplifies ICA because the unmixing matrix V becomes
    orthogonal, reducing the search space from D² to D(D-1)/2
    parameters.

    Method: eigenvalue decomposition of covariance matrix.
        X_white = D^{-1/2} E^T (X - mean)

    Parameters
    ----------
    X : ndarray (D, N)
        Raw observed data.

    Returns
    -------
    X_white : ndarray (D, N)
        Whitened data with identity covariance.
    whiten_mat : ndarray (D, D)
        Whitening matrix (for computing effective mixing matrix).
    """
    # Step 1: Center (subtract mean)
    X_centered = X - X.mean(axis=1, keepdims=True)

    # Step 2: Eigendecomposition of covariance
    cov = np.cov(X_centered)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Step 3: Whitening matrix = D^{-1/2} @ E^T
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + 1e-10))  # +eps for stability
    whiten_mat = D_inv_sqrt @ eigvecs.T

    # Step 4: Apply whitening
    X_white = whiten_mat @ X_centered

    return X_white, whiten_mat


# ================================================================
#  3. EVALUATION METRIC
# ================================================================

def amari_index(V: np.ndarray, W: np.ndarray) -> float:
    """
    Compute the Amari index (Amari et al., 1996).

    Measures how close the performance matrix C = V @ W is to a
    permutation-scaling matrix. This is the standard metric for
    evaluating ICA separation quality.

    The formula (project PDF page 3):
        r(C) = 1/(2D(D-1)) * [Σ_i (Σ_j |c_ij|/max_k|c_ik| - 1)
                              + Σ_j (Σ_i |c_ij|/max_k|c_kj| - 1)]

    Parameters
    ----------
    V : ndarray (D, D)
        Estimated unmixing matrix.
    W : ndarray (D, D)
        True mixing matrix.

    Returns
    -------
    r : float
        Amari index in [0, 1].
        0 = perfect separation, 1 = complete failure.
    """
    C = V @ W
    D = C.shape[0]
    C_abs = np.abs(C)

    # Term 1: row-wise normalization
    row_max = C_abs.max(axis=1, keepdims=True)
    term1 = (C_abs / (row_max + 1e-15)).sum(axis=1)  # sum over columns
    term1 = (term1 - 1).sum()                         # sum over rows

    # Term 2: column-wise normalization
    col_max = C_abs.max(axis=0, keepdims=True)
    term2 = (C_abs / (col_max + 1e-15)).sum(axis=0)  # sum over rows
    term2 = (term2 - 1).sum()                         # sum over columns

    # Normalize to [0, 1]
    r = (term1 + term2) / (2 * D * (D - 1))
    return r


# ================================================================
#  4. SOURCE MATCHING
# ================================================================

def match_sources(S_true: np.ndarray, S_est: np.ndarray) -> np.ndarray:
    """
    Reorder and sign-flip estimated sources to best match true sources.

    ICA has two inherent ambiguities (course Section 5.3.2):
        1. Permutation: sources can be in any order
        2. Sign: each source can be flipped (multiplied by -1)

    This function resolves both by greedy correlation-based matching:
    each estimated source is paired with the true source that has
    the highest absolute correlation.

    Parameters
    ----------
    S_true : ndarray (D, N)
        True source signals.
    S_est : ndarray (D, N)
        Estimated source signals (from ICA).

    Returns
    -------
    S_matched : ndarray (D, N)
        Estimated sources reordered and sign-corrected to match S_true.
    """
    D = S_true.shape[0]

    # Compute absolute correlation matrix between true and estimated
    corr = np.abs(np.corrcoef(S_true, S_est)[:D, D:])

    # Greedy matching: assign each true source to best available estimate
    S_matched = np.zeros_like(S_est)
    used = set()

    for i in range(D):
        # Find the best unmatched estimated source for true source i
        best_j = -1
        best_val = -1
        for j in range(D):
            if j not in used and corr[i, j] > best_val:
                best_val = corr[i, j]
                best_j = j
        used.add(best_j)

        # Correct sign ambiguity
        sign = np.sign(np.corrcoef(S_true[i], S_est[best_j])[0, 1])
        S_matched[i] = sign * S_est[best_j]

    return S_matched


# ================================================================
#  5. PLOTTING HELPERS
# ================================================================

def plot_sources_comparison(S_true: np.ndarray, S_est: np.ndarray,
                            title: str = "Source Recovery",
                            max_display: int = 500) -> plt.Figure:
    """
    Plot true sources vs. estimated sources side by side.

    Automatically matches sources using correlation-based assignment
    and corrects sign ambiguity before plotting.

    Parameters
    ----------
    S_true : ndarray (D, N)
        True source signals.
    S_est : ndarray (D, N)
        Estimated source signals.
    title : str
        Figure title.
    max_display : int
        Number of samples to display (for readability).

    Returns
    -------
    fig : matplotlib Figure
    """
    D = S_true.shape[0]
    S_est_matched = match_sources(S_true, S_est)

    fig, axes = plt.subplots(D, 2, figsize=(12, 2.5 * D))
    if D == 1:
        axes = axes.reshape(1, -1)

    for j in range(D):
        # Left: true source
        axes[j, 0].plot(S_true[j, :max_display], linewidth=0.8)
        axes[j, 0].set_title(f"True source {j+1}")
        axes[j, 0].set_ylabel("Amplitude")

        # Right: estimated source
        axes[j, 1].plot(S_est_matched[j, :max_display], linewidth=0.8, color='tab:orange')
        axes[j, 1].set_title(f"Estimated source {j+1}")

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_convergence(histories: dict, ylabel: str = "Amari Index",
                     title: str = "Convergence Comparison") -> plt.Figure:
    """
    Plot convergence curves for multiple algorithms on a log scale.

    Parameters
    ----------
    histories : dict
        {algorithm_name: list_of_metric_values_per_iteration}
    ylabel : str
        Y-axis label.
    title : str
        Figure title.

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, values in histories.items():
        ax.plot(values, label=name, linewidth=1.5)

    ax.set_xlabel("Iteration / Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_amari_barplot(results: dict,
                       title: str = "Final Amari Index Comparison") -> plt.Figure:
    """
    Bar plot of Amari indices with error bars (mean ± std over runs).

    Parameters
    ----------
    results : dict
        {algorithm_name: list_of_amari_indices_over_runs}
    title : str
        Figure title.

    Returns
    -------
    fig : matplotlib Figure
    """
    names = list(results.keys())
    means = [np.mean(v) for v in results.values()]
    stds = [np.std(v) for v in results.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, means, yerr=stds, capsize=5,
           color=plt.cm.Set2(np.linspace(0, 1, len(names))),
           edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Amari Index (lower = better)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig
