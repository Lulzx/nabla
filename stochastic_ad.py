"""
Differentiating through stochastic nodes: REINFORCE vs reparameterization.

When a computation graph includes a stochastic sample z ~ p(z; θ), the
sample operation itself has no gradient — p is a distribution, not a
differentiable function. Two estimators give ∂/∂θ E_{z~p}[f(z)]:

REINFORCE (Williams 1992) — score function estimator
------------------------------------------------------
    ∂/∂θ E[f(z)] = E[ f(z) · ∂log p(z; θ)/∂θ ]

Works for any f (doesn't need to be differentiable w.r.t. z), any p
(doesn't need a reparameterization). Unbiased but high variance.

Variance reduction: subtract a baseline b(θ) that doesn't depend on z:
    ∂/∂θ E[f(z)] = E[ (f(z) − b) · ∂log p(z; θ)/∂θ ]
A common choice: b = mean(f(z)) over the current batch.

Reparameterization trick (Kingma & Welling 2013)
--------------------------------------------------
When z can be written as z = g(ε, θ) for ε ~ p(ε) independent of θ:
    ∂/∂θ E_{z~p(z;θ)}[f(z)] = ∂/∂θ E_{ε~p(ε)}[f(g(ε, θ))]
                             = E[∂f/∂z · ∂g/∂θ]

The expectation is now over a fixed distribution p(ε); we differentiate
through g(ε, θ) using ordinary reverse-mode AD.

Requires f to be differentiable w.r.t. z. Much lower variance than
REINFORCE in practice — this is why VAEs use reparameterization.

Variance comparison
--------------------
    ·  Reparameterization variance ≈ O(1/n)  (like i.i.d. sampling)
    ·  REINFORCE variance ≈ O(Var[f²]/n)  — can be much larger
    ·  REINFORCE variance → 0 as f becomes constant (good baseline choice)

Connection to AD type system (smooth_types.py)
-----------------------------------------------
REINFORCE treats f as a black box (no gradient through z) — escape hatch
analogous to @custom_vjp, but for stochastic nodes.

Reparameterization requires f to be PSMOOTH or better (differentiable a.e.)
and a known reparameterization g. This is the typed version: stronger
requirements in exchange for lower variance.
"""

import numpy as np
from autograd import Tensor


# ── REINFORCE ─────────────────────────────────────────────────────────────────

def reinforce(f_np, score_fn, theta_np, n_samples=1000, baseline=True, seed=None):
    """
    REINFORCE gradient estimator: ∂/∂θ E_{z~p(z;θ)}[f(z)].

    ≈ mean_i[ (f(zᵢ) − b) · score(zᵢ, θ) ]  where zᵢ ~ p(z; θ)

    Parameters
    ----------
    f_np      : callable(ndarray) → float  — objective function (black-box)
    score_fn  : callable(z, θ) → ndarray   — ∂log p(z; θ)/∂θ
    theta_np  : ndarray (m,)               — current parameters
    n_samples : int                        — Monte Carlo sample count
    baseline  : bool                       — subtract mean baseline (reduces variance)

    Returns
    -------
    grad_est  : ndarray (m,) — gradient estimate
    f_mean    : float        — mean objective value
    f_std     : float        — std of objective values (diagnostic)
    """
    rng = np.random.default_rng(seed)
    raise_fn = getattr(rng, '_sample_fn', None)

    # Caller provides a sample_fn via score_fn closure or we use the default.
    # For simplicity, the sample distribution is encoded in score_fn — the
    # user generates samples externally and passes them to f_np / score_fn.
    # Here we follow the convention: score_fn(theta) returns (samples, scores).
    samples, scores = score_fn(theta_np, n_samples, rng)

    fvals = np.array([f_np(z) for z in samples])        # (n_samples,)
    b     = np.mean(fvals) if baseline else 0.0
    grad_est = np.mean((fvals - b)[:, None] * scores, axis=0)
    return grad_est, float(np.mean(fvals)), float(np.std(fvals))


# ── Reparameterization ────────────────────────────────────────────────────────

def reparam(f_tensor, reparam_fn, theta, n_samples=1000, seed=None):
    """
    Reparameterization gradient: ∂/∂θ E_{z=g(ε,θ), ε~p(ε)}[f(z)].

    Uses reverse-mode AD through g(ε, θ) to propagate the gradient.

    Parameters
    ----------
    f_tensor   : callable(Tensor) → scalar Tensor  — objective (must be diff.)
    reparam_fn : callable(eps_np, Tensor) → Tensor  — z = g(ε, θ)
    theta      : Tensor (m,)                        — parameters (leaf)
    n_samples  : int

    Returns
    -------
    grad_est : ndarray (m,)
    f_mean   : float
    f_std    : float
    """
    rng = np.random.default_rng(seed)
    theta_leaf = theta if isinstance(theta, Tensor) else Tensor(np.asarray(theta, dtype=np.float64))

    total_grad = np.zeros_like(theta_leaf.data)
    fvals = []
    for _ in range(n_samples):
        theta_leaf.grad = np.zeros_like(theta_leaf.data)   # reset each time
        eps = rng.standard_normal(theta_leaf.data.shape)
        z   = reparam_fn(eps, theta_leaf)
        fv  = f_tensor(z)
        fv.backward()
        total_grad += theta_leaf.grad.copy()
        fvals.append(float(fv.data))

    theta_leaf.grad = np.zeros_like(theta_leaf.data)
    return total_grad / n_samples, float(np.mean(fvals)), float(np.std(fvals))


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "=" * 60

    # ── 1. E_{z~N(μ,1)}[z²] = μ² + 1  →  ∂/∂μ = 2μ ─────────────────
    print(SEP)
    print("1. Estimating ∂/∂μ E_{z~N(μ,1)}[z²]")
    print("   Exact answer: 2μ")
    print()

    # --- REINFORCE setup ---
    # p(z; μ) = N(z; μ, 1)
    # log p(z; μ) = −½(z−μ)² + const
    # ∂log p/∂μ  = z − μ

    def score_normal(theta, n_samples, rng):
        mu = theta[0]
        z  = rng.normal(mu, 1.0, n_samples)
        score = (z - mu)[:, None]              # (n, 1)
        return z, score

    def f_sq(z):
        return float(z) ** 2

    # --- Reparameterization setup ---
    # z = μ + ε,  ε ~ N(0, 1)

    def reparam_normal(eps_np, theta):
        return theta + Tensor(eps_np)

    def f_sq_tensor(z):
        return (z * z).sum()

    mu_val  = 1.0
    n_trials = 60    # independent gradient estimates per method

    print(f"  μ=1.0, exact ∂/∂μ = 2.0")
    print(f"  (empirical std computed over {n_trials} independent runs)")
    print()
    print(f"  {'n':>8}  {'RF mean':>10}  {'RF std':>10}  {'RP mean':>10}  {'RP std':>10}  {'var ratio':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for n in [100, 500, 2000]:
        rf_ests, rp_ests = [], []
        for trial in range(n_trials):
            theta_rf = np.array([mu_val])
            g_rf, _, _ = reinforce(f_sq, score_normal, theta_rf, n_samples=n,
                                   baseline=True, seed=trial * 997)
            rf_ests.append(g_rf[0])

            theta_rp = Tensor(np.array([mu_val]))
            g_rp, _, _ = reparam(f_sq_tensor, reparam_normal, theta_rp,
                                 n_samples=n, seed=trial * 997)
            rp_ests.append(g_rp[0])

        rf_mean, rf_std = np.mean(rf_ests), np.std(rf_ests)
        rp_mean, rp_std = np.std(rp_ests), np.std(rp_ests)
        ratio = (np.std(rf_ests) / np.std(rp_ests)) ** 2

        print(f"  {n:8d}  {rf_mean:10.4f}  {np.std(rf_ests):10.4f}"
              f"  {np.mean(rp_ests):10.4f}  {np.std(rp_ests):10.4f}  {ratio:10.1f}×")

    print()
    print(f"  Exact: 2μ = {2*mu_val:.4f}")
    print()
    print(f"  var ratio = Var[RF] / Var[RP] >> 1: reparam gradient is much")
    print(f"  lower variance.  For z² with z~N(μ,1): Var[RF] = Var[z²(z-μ)],")
    print(f"  Var[RP] = Var[2z] = 4.  The ratio grows as μ increases.")

    # ── 2. Variance comparison across μ ──────────────────────────────
    print()
    print(SEP)
    print("2. Gradient estimates across μ ∈ {0.5, 1.0, 2.0}  (n=2000)")
    print()
    print(f"  {'μ':>6}  {'exact':>8}  {'REINFORCE':>12}  {'reparam':>12}  {'RF error':>10}  {'RP error':>10}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}")

    for mu_val in [0.5, 1.0, 2.0]:
        exact = 2 * mu_val

        theta_rf = np.array([mu_val])
        g_rf, _, _ = reinforce(f_sq, score_normal, theta_rf, n_samples=2000, baseline=True, seed=42)

        theta_rp = Tensor(np.array([mu_val]))
        g_rp, _, _ = reparam(f_sq_tensor, reparam_normal, theta_rp, n_samples=2000, seed=42)

        print(f"  {mu_val:6.2f}  {exact:8.4f}  {g_rf[0]:12.4f}  {g_rp[0]:12.4f}"
              f"  {abs(g_rf[0]-exact):10.4f}  {abs(g_rp[0]-exact):10.4f}")

    # ── 3. Discrete distributions: reparameterization doesn't apply ───
    print()
    print(SEP)
    print("3. Discrete z ~ Bernoulli(p): REINFORCE only")
    print("   E_{z~Bern(p)}[z] = p  →  ∂/∂p = 1")
    print()
    print("   Reparameterization: no continuous reparameterization exists.")
    print("   (Gumbel-softmax / STRAIGHT-THROUGH are approximations.)")
    print()

    # log p(z; p) = z·log(p) + (1-z)·log(1-p)
    # ∂log p/∂p   = z/p − (1−z)/(1−p)

    def score_bernoulli(theta, n_samples, rng):
        p = np.clip(theta[0], 1e-6, 1 - 1e-6)
        z = (rng.random(n_samples) < p).astype(float)
        score = (z / p - (1 - z) / (1 - p))[:, None]
        return z, score

    def f_identity(z):
        return float(z)

    for p_val in [0.3, 0.5, 0.8]:
        g_rf, mean_f, _ = reinforce(f_identity, score_bernoulli,
                                    np.array([p_val]), n_samples=5000,
                                    baseline=True, seed=123)
        print(f"  p={p_val:.1f}:  E[z]≈{mean_f:.4f} (exact {p_val:.4f})"
              f"  ∂E[z]/∂p ≈ {g_rf[0]:.4f} (exact 1.0000)")

    print()
    print("  ✓  REINFORCE works for discrete z — reparameterization cannot.")
    print("  ✗  REINFORCE variance is high for discrete distributions;")
    print("     control variates / NVIL / VIMCO are needed for stability.")
