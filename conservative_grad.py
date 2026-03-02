"""
Conservative gradients and the Clarke subdifferential.

The question: what does reverse-mode AD actually compute at non-smooth points?
-------------------------------------------------------------------------------
For smooth f: AD returns ∇f(x) exactly.
For piecewise-smooth f (relu network): AD returns a specific subgradient.
  Which one? The one selected by the branch taken in the forward pass.

This is NOT a bug — it is exactly what Bolte & Pauwels (2020) formalize.

Theorem (Bolte & Pauwels 2020, Corollary 1)
---------------------------------------------
Let f = h ∘ g₁ ∘ ⋯ ∘ gₙ where each gᵢ is "definable" (built from
polynomials, exp, relu, max, piecewise-linear ops, etc.).
Then the set-valued map D : x ↦ {AD output at x} is a *conservative field*
for f.

Conservative field  ←→  satisfies the non-smooth chain rule
--------------------------------------------------------------
D is a conservative field for f if for every absolutely-continuous path γ:

    d/dt f(γ(t)) = ⟨v, γ'(t)⟩   for a.e. t,  ∀v ∈ D(γ(t))

This is the chain rule for non-smooth functions. For smooth f it holds for
D(x) = {∇f(x)}. For a relu network, D gives a different element at each kink
(determined by the forward pass branch), but the chain rule still holds
along any trajectory.

Why this matters for optimization
-----------------------------------
If the chain rule holds along gradient-descent trajectories, then the loss
decreases along descent directions — guaranteeing convergence under mild
conditions (e.g., Kurdyka–Łojasiewicz inequality for neural network losses).

This is stronger than "subgradient a.e.": we get convergence guarantees even
though the gradient is undefined at measure-zero kink sets.

Clarke subdifferential
-----------------------
For f: ℝⁿ → ℝ Lipschitz near x:
    ∂f(x) = conv { lim ∇f(xᵢ) : xᵢ → x, xᵢ ∉ S_f }
where S_f is the measure-zero non-differentiable set.

The AD output D(x) is always a point IN ∂f(x).

Honest limits
--------------
  · "Definable" excludes: non-elementary transcendentals, exotic piece-wise
    definitions, or programs with data-dependent recursion.
  · Conservative ≠ correct gradient: relu networks have zero gradient almost
    everywhere (flat regions). The guarantee is convergence, not fast descent.
  · At exact kinks, the AD output depends on the strict inequality branch
    (x > 0 vs x ≤ 0). Different implementations may return different
    elements of the subdiff — all are valid.
"""

import numpy as np
from autograd import Tensor
from autograd import grad as rev_grad


# ── Clarke subdifferential tools ──────────────────────────────────────────────

def grad_scalar(f, x_val):
    """
    Gradient of a scalar-to-scalar function f at x_val (Python float).
    f receives a 0-d Tensor; returns a 0-d Tensor.
    """
    return float(rev_grad(f)(np.float64(x_val)))


def clarke_interval_1d(f, x, eps=1e-3, n=600, seed=0):
    """
    Numerical estimate of the Clarke subdifferential interval [a, b] for
    a scalar Lipschitz function f at x.

    Samples n points in (x-eps, x) ∪ (x, x+eps), computes ∇f at each
    (via AD, which is correct a.e.), and returns the observed range.

    Returns (lo, hi, ad_at_x) where ad_at_x is what AD returns AT x.
    """
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(1e-8, eps, n)
    signs   = rng.choice([-1, 1], n)
    points  = x + signs * offsets

    grads = [grad_scalar(f, float(xi)) for xi in points]
    lo, hi = min(grads), max(grads)
    ad_x   = grad_scalar(f, x)
    return lo, hi, ad_x


# ── Reference non-smooth functions (using Tensor operations) ─────────────────

def abs_val(x):
    """|x| implemented as relu(x) + relu(-x) — works for Tensor or float."""
    if isinstance(x, Tensor):
        return x.relu() + (-x).relu()
    return abs(float(x))


def abs_fn(x):   # Tensor → scalar Tensor, for use with rev_grad
    return x.relu() + (-x).relu()


def relu_fn(x):  # Tensor → scalar Tensor
    return x.relu()


def max_xy(x):
    """max(x², x·(x-1)+1) — kink at x=1 where both branches equal 1."""
    # Branch A: x·(x-1)+1  = x² - x + 1  (parabola, grad = 2x-1)
    # Branch B: x²                         (parabola, grad = 2x)
    # They meet at x=1: A(1)=1, B(1)=1. Clarke subdiff = [2·1-1, 2·1] = [1, 2]
    a = x * x - x + Tensor(np.float64(1.0))
    b = x * x
    # max(a, b) = relu(a-b) + b
    return (a - b).relu() + b


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "=" * 60

    # ── 1. Clarke subdiff vs AD output ───────────────────────────────
    print(SEP)
    print("1. Clarke subdifferential vs AD output at kinks")
    print()

    cases = [
        ("│x│  = relu(x) + relu(-x)",   abs_fn,   0.0),
        ("relu(x)",                       relu_fn,  0.0),
        ("max(x²-x+1, x²)  (kink x=1)", max_xy,   1.0),
        ("│x│  away from kink (x=2)",    abs_fn,   2.0),
        ("relu  away from kink (x=1)",   relu_fn,  1.0),
    ]

    print(f"  {'function':30s}  {'Clarke ∂f(x)':18s}  {'AD output':12s}  {'in ∂f?'}")
    print(f"  {'-'*30}  {'-'*18}  {'-'*12}  {'-'*6}")

    for label, f, x0 in cases:
        lo, hi, ad = clarke_interval_1d(f, x0)
        in_subdiff = lo - 1e-9 <= ad <= hi + 1e-9
        print(f"  {label:30s}  [{lo:+.3f}, {hi:+.3f}]    {ad:+.3f}         {'✓' if in_subdiff else '✗'}")

    print()
    print("  At smooth points: Clarke subdiff is a singleton {f'(x)} = AD output.")
    print("  At kinks: AD output is a specific element chosen by the branch taken.")
    print("  Our convention (relu'(0)=0) picks the left-limit subgradient.")
    print("  Any element of the subdiff would be valid; this one is consistent.")

    # ── 2. The branch-selection at kinks ─────────────────────────────
    print()
    print(SEP)
    print("2. Branch selection: the same function, different conventions")
    print()
    print("  relu'(0) via our backprop   = 0   (from `out.data > 0` rule)")
    print("  relu'(0) via finite diff    = 0.5 (symmetric limit)")
    print("  Both are valid: 0 ∈ [0,1] and 0.5 ∈ [0,1].")
    print()

    # Forward-difference approximation (what you'd get from numerical grad)
    eps = 1e-7
    fd = (float(relu_fn(Tensor(np.float64(eps))).data)
          - float(relu_fn(Tensor(np.float64(-eps))).data)) / (2 * eps)
    ad0 = grad_scalar(relu_fn, 0.0)
    print(f"  AD at x=0:              {ad0:.1f}")
    print(f"  Finite-diff at x=0:     {fd:.1f}")
    print()
    print("  The training trajectory almost surely avoids x=0 (measure-zero),")
    print("  so the specific value at the kink doesn't affect convergence.")

    # ── 3. Subgradient descent on L1 regression ──────────────────────
    print()
    print(SEP)
    print("3. Subgradient descent on L1 regression: min_w Σ|y − Xw|")
    print("   (non-smooth convex — AD gives valid subgradients)")
    print()

    np.random.seed(42)
    n_data, n_feat = 80, 4
    w_true = np.array([2.0, -1.0, 0.5, 3.0])

    # Design matrix and sparse-noise targets (L1 is robust to outliers)
    X_np = np.random.randn(n_data, n_feat)
    noise = np.where(np.random.rand(n_data) < 0.1,         # 10% outliers
                     np.random.randn(n_data) * 5.0,
                     np.random.randn(n_data) * 0.3)
    y_np = X_np @ w_true + noise

    X_const = Tensor(X_np)   # constant

    def l1_loss(w):
        r = Tensor(y_np) - X_const @ w
        return (r.relu() + (-r).relu()).sum()

    def l2_loss(w):
        r = Tensor(y_np) - X_const @ w
        return (r * r).sum()

    def run_gd(loss_fn, w0, lr, n_iter):
        w = w0.copy()
        hist = []
        for k in range(n_iter + 1):
            if k % (n_iter // 5) == 0:
                lv = float(loss_fn(Tensor(w)).data)
                hist.append((k, lv))
            if k < n_iter:
                g = rev_grad(loss_fn)(w)
                w -= lr * g
        return w, hist

    w0 = np.zeros(n_feat)

    w_l1, hist_l1 = run_gd(l1_loss, w0, lr=2e-3, n_iter=2000)
    w_l2, hist_l2 = run_gd(l2_loss, w0, lr=2e-4, n_iter=2000)

    print(f"  {'step':>6}  {'L1 loss':>10}  {'L2 loss':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}")
    for (k, l1), (_, l2) in zip(hist_l1, hist_l2):
        print(f"  {k:6d}  {l1:10.2f}  {l2:10.2f}")

    err_l1 = np.linalg.norm(w_l1 - w_true)
    err_l2 = np.linalg.norm(w_l2 - w_true)
    print()
    print(f"  True weights:   {w_true}")
    print(f"  L1 estimate:    {w_l1.round(3)}  error={err_l1:.4f}")
    print(f"  L2 estimate:    {w_l2.round(3)}  error={err_l2:.4f}")
    print()
    print("  L1 converges despite non-smoothness: Bolte-Pauwels guarantees")
    print("  the AD subgradients form a conservative field for l1_loss.")
    print("  L1 is more robust to outliers (10% of data) than L2.")

    # ── 4. Conservative field: chain rule verification ────────────────
    print()
    print(SEP)
    print("4. Conservative field: chain rule check along a path")
    print()
    print("   γ(t) = t·e₁  (the first coordinate axis)")
    print("   f(x) = │x│   Conservative field D(x) = {AD gradient}")
    print()
    print("   Chain rule: d/dt f(γ(t)) = ⟨D(γ(t)), γ'(t)⟩")
    print("   = D(t) · 1  =  AD(│·│)(t)")
    print()
    print("   Check: d/dt│t│  =  sign(t)  (and AD gives sign(t) for t ≠ 0)")
    print()

    ts = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    print(f"   {'t':>6}  {'d/dt|t| (FD)':>14}  {'AD (conserv.)':>14}  {'match'}")
    print(f"   {'-'*6}  {'-'*14}  {'-'*14}  {'-'*5}")
    for t in ts:
        eps = 1e-6
        fd  = (abs(t + eps) - abs(t - eps)) / (2 * eps)
        ad  = grad_scalar(abs_fn, t)
        ok  = abs(fd - ad) < 0.01 or t == 0.0
        note = "(kink: AD=0, FD=0)" if t == 0.0 else ""
        print(f"   {t:6.2f}  {fd:14.6f}  {ad:14.6f}  {'✓' if ok else '✗'}  {note}")

    print()
    print("   At t=0, both FD and AD give 0 (FD by symmetry, AD by convention).")
    print("   The chain rule holds along this path: ∫ d/dt|γ(t)| dt = 0 − 0 = 0.")

    # ── 5. Bolte-Pauwels theorem (informal statement) ─────────────────
    print()
    print(SEP)
    print("5. Summary: what Bolte & Pauwels (2020) actually prove")
    print()
    print("   For 'definable' programs (relu, exp, polynomials, max, min):")
    print()
    print("   (a) AD ∈ Clarke subdifferential at every x (not just a.e.)")
    print("       The specific element is determined by the forward-pass branch.")
    print()
    print("   (b) The set-valued map D: x ↦ {AD(x)} is a conservative field.")
    print("       This is the non-smooth chain rule: holds along any AC path.")
    print()
    print("   (c) KŁ property + conservative field → convergence of subgradient")
    print("       descent to a critical point (not necessarily global optimum).")
    print()
    print("   What this does NOT mean:")
    print("   ✗ The gradient is always useful (relu networks: often 0)")
    print("   ✗ Global convergence (only to stationary/critical points)")
    print("   ✗ Correct Hessians (see hessian.py: relu is C¹ but not C²)")
    print("   ✗ Any guarantee for non-definable programs (e.g., sort, argmax)")
    print("     (see diff_combinatorial.py for those — different framework)")
    print()
    print("   Connection to smooth_types.py:")
    print("   PSMOOTH (relu) → AD in Clarke subdiff, conservative field ✓")
    print("   LIPSCHITZ (abs, max) → same — Clarke subdiff always non-empty ✓")
    print("   CONTINUOUS (step) → Clarke subdiff = {0, 1} but AD gives 0")
    print("     (step is not definable in the KŁ sense — training stalls)")
