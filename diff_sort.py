"""
Differentiable sorting and ranking.

Two algorithms, with honest documentation of where each breaks.

NeuralSort (Grover et al. 2019)
    Replaces the sort permutation with a soft permutation matrix.
    Temperature tau controls the accuracy-vs-gradient tradeoff:
        tau → 0 : exact sort, but gradients vanish
        tau → ∞ : large gradients, but output is just the mean
    Cannot represent exact discrete sort with non-zero gradients at ties —
    that's a fundamental impossibility, not a bug.

SoftRank
    Computes a differentiable approximate rank for each element.
    Useful for ranking losses (NDCG optimisation, listwise learning-to-rank).
"""

import numpy as np
from autograd import Tensor, gradient_check


# ------------------------------------------------------------------ #
# NeuralSort                                                           #
# ------------------------------------------------------------------ #

def neuralsort(x, tau=1.0):
    """
    Differentiable sort via the NeuralSort relaxation (Grover et al. 2019).

    Parameters
    ----------
    x   : Tensor (n,)
    tau : temperature

    Returns
    -------
    Tensor (n,) approximately sorted descending.

    Internally builds a soft n×n permutation matrix P where
    P[i,j] ≈ P(x[j] has rank i). Then y = P @ x.
    """
    n = x.shape[0]
    x2d = x.reshape(n, 1)                         # (n, 1)

    # Differentiable |x_i - x_j|  =  sqrt((x_i-x_j)^2 + ε)
    diff   = x2d - x2d.T                           # (n, n)
    abs_ij = (diff ** 2 + 1e-9) ** 0.5             # (n, n)

    # sum_k |x_j - x_k|  for each j  →  (n,)
    sum_abs = abs_ij.sum(axis=0)                   # (n,)

    # NeuralSort score matrix:
    #   M[i, j] = (n+1-2i) * x[j]  -  Σ_k |x[j]-x[k]|
    # where i = 1..n is the target rank position (1 = largest).
    # As tau→0, softmax picks the true rank-i element.
    coeff = Tensor((n + 1) - 2 * np.arange(1, n + 1, dtype=float))   # (n,)
    M = coeff.reshape(n, 1) * x.reshape(1, n) - sum_abs.reshape(1, n) # (n, n)

    P = (M * (1.0 / tau)).softmax(axis=1)          # (n, n)  rows sum to 1

    return (P @ x.reshape(n, 1)).reshape(n)        # (n,)


def soft_rank(x, tau=1.0):
    """
    Differentiable rank (0-indexed: 0 = smallest, n-1 = largest).

    rank[i] = Σ_j  sigmoid((x[i] - x[j]) / tau)
            ≈ number of elements smaller than x[i]

    Useful as a smooth surrogate for ranking metrics (NDCG, AP).
    """
    n = x.shape[0]
    xi = x.reshape(n, 1)
    xj = x.reshape(1, n)
    return ((xi - xj) * (1.0 / tau)).sigmoid().sum(axis=1)  # (n,)


# ------------------------------------------------------------------ #
# Tradeoff analysis                                                    #
# ------------------------------------------------------------------ #

def _sort_error(x_np, tau):
    """Mean absolute error between neuralsort and true sort."""
    true_sorted = np.sort(x_np)[::-1]
    soft = neuralsort(Tensor(x_np), tau=tau).data
    return np.abs(soft - true_sorted).mean()


def _grad_magnitude(x_np, tau):
    """L2 norm of ∂(neuralsort output sum)/∂x."""
    x_t = Tensor(x_np.copy())
    neuralsort(x_t, tau=tau).sum().backward()
    return np.linalg.norm(x_t.grad)


if __name__ == "__main__":
    import sys

    np.random.seed(0)
    x_np = np.array([3.1, 1.4, 2.7, 0.5, 4.2])

    print("=" * 60)
    print("NeuralSort temperature tradeoff")
    print(f"{'tau':>8}  {'sort_err':>10}  {'|grad|':>10}  note")
    print("-" * 60)
    notes = {1e-3: "exact sort, ∇≈0", 0.1: "good sort, small ∇",
             1.0: "balanced", 5.0: "large ∇, poor sort",
             50.0: "output→mean, ∇ large but useless"}
    for tau in [1e-3, 0.1, 1.0, 5.0, 50.0]:
        err  = _sort_error(x_np, tau)
        gmag = _grad_magnitude(x_np, tau)
        print(f"{tau:>8.3f}  {err:>10.4f}  {gmag:>10.4f}  {notes[tau]}")

    print()
    print("Gradient check (tau=1.0):")
    x_t = Tensor(x_np.copy())
    gradient_check(lambda t: neuralsort(t, tau=1.0), x_t)

    print()
    print("SoftRank vs true rank:")
    x_t2 = Tensor(x_np.copy())
    sr = soft_rank(x_t2, tau=0.3)
    true_rank = np.argsort(np.argsort(x_np)).astype(float)
    print(f"  soft_rank  : {sr.data.round(2)}")
    print(f"  true_rank  : {true_rank}")

    print()
    print("The fundamental limit:")
    print("  At x[i]=x[j] exactly, sort is discontinuous.")
    print("  No smooth function can be both exactly sorted and have non-zero")
    print("  gradients at ties. NeuralSort chooses gradient over exactness.")
