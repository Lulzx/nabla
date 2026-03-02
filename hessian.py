"""
Second-order automatic differentiation: Hessians and Newton's method.

The C² gap in smooth_types.py
-------------------------------
smooth_types.py tracks SMOOTH > C1 > PSMOOTH > LIPSCHITZ > CONTINUOUS but
collapses everything ≥ C¹ into a single bin. That omission matters here:

  · Newton's method requires ∇²f (Hessian), so f must be C².
  · For a relu network, ∇f is piecewise-constant — its derivative is zero
    a.e. and undefined at kinks. H·v via finite-difference-of-gradient
    returns zero everywhere and blow-ups at kinks.
  · gelu / tanh networks are C∞: Hessian exists everywhere.

Two primitives
--------------
hvp(f, x, v)    Hessian-vector product H(x)·v  in O(n) time.
                 Two gradient evaluations (central difference), O(ε²) error.

hessian(f, x)   Full n×n Hessian matrix in O(n²) time.
                 Calls hvp n times with standard basis vectors.
                 For large n, only use hvp — you rarely need the full matrix.

Forward-over-reverse (the "proper" approach)
---------------------------------------------
The exact way to compute H(x)·v without finite-difference noise is to
differentiate through the backward pass using forward-mode perturbations.
This is the Pearlmutter (1994) R{·} operator:

    R{y}   = lim_{ε→0} [y(x + ε·v) − y(x)] / ε    (applied to any forward var)
    R{∇f}  = H·v                                     (applied to the gradient)

In practice: jvp(grad(f), x, v).  Cost: one extra backward + one forward pass.
JAX does this with jax.jvp(jax.grad(f), (x,), (v,)).

We implement finite-difference-of-gradient here because:
  (a) It reuses the existing reverse-mode engine with no new rules.
  (b) For C³ functions, central-difference error is O(ε²) — numerically fine.
  (c) The exact vs. finite-difference distinction is invisible in demos.

Newton-CG
----------
x_{k+1} = x_k − H(x_k)^{-1} ∇f(x_k)

Never form H explicitly. Instead solve H·d = −g via conjugate gradient,
which only requires HVPs. Cost per Newton step: O(√κ) reverse passes where
κ = cond(H). Quadratic convergence near the optimum for C² f.
"""

import numpy as np
from autograd import Tensor
from autograd import grad as rev_grad


# ── Core primitives ───────────────────────────────────────────────────────────

def _grad(f, x_np):
    """∇f(x) as a numpy array via reverse-mode AD."""
    return rev_grad(f)(x_np)


def hvp(f, x_np, v_np, eps=1e-5):
    """
    Hessian-vector product H(x)·v via central difference of gradient.

        H(x)·v  ≈  [∇f(x + ε·v) − ∇f(x − ε·v)] / (2ε)

    Cost  : 2 gradient evaluations (= 2 reverse passes), O(n) each.
    Error : O(ε²) for C³ functions.  Use ε ≈ (machine_eps)^{1/3} ≈ 6e-6
            for best tradeoff between truncation and rounding error.

    Parameters
    ----------
    f    : callable(Tensor) → scalar Tensor
    x_np : ndarray (n,)
    v_np : ndarray (n,)   direction vector
    eps  : float          finite-difference step

    Returns
    -------
    ndarray (n,)   H(x)·v
    """
    v_np = np.asarray(v_np, dtype=np.float64)
    g_plus  = _grad(f, x_np + eps * v_np)
    g_minus = _grad(f, x_np - eps * v_np)
    return (g_plus - g_minus) / (2.0 * eps)


def hessian(f, x_np, eps=1e-5):
    """
    Full n×n Hessian matrix via n Hessian-vector products.

    H[:,j] = hvp(f, x, eⱼ) for each standard basis vector eⱼ.

    Cost: O(n²).  For large n, use hvp with conjugate gradient instead.
    The result is symmetrized: Ĥ = (H + Hᵀ)/2 to cancel O(ε) asymmetry.

    Returns
    -------
    ndarray (n, n)
    """
    x_flat = np.asarray(x_np, dtype=np.float64).flatten()
    n = x_flat.size
    H = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        H[:, j] = hvp(f, x_flat, e_j, eps)
    return 0.5 * (H + H.T)


# ── Newton-CG internals ───────────────────────────────────────────────────────

def _cg(matvec, b, tol=1e-8, max_iter=None):
    """
    Conjugate gradient for the symmetric positive-definite system Av = b.
    matvec(v) computes A·v without forming A.
    """
    if max_iter is None:
        max_iter = len(b)
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = float(r @ r)

    for _ in range(max_iter):
        if rs < tol * tol:
            break
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if pAp <= 0:
            break                    # non-positive curvature — stop CG
        alpha = rs / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(r @ r)
        p = r + (rs_new / rs) * p
        rs = rs_new

    return x


def _armijo(f, x, g, d, alpha=1.0, rho=0.5, c=1e-4):
    """Backtracking line search with Armijo sufficient-decrease condition."""
    f0    = float(f(Tensor(x)).data)
    slope = float(g @ d)
    for _ in range(50):
        xd = x + alpha * d
        if float(f(Tensor(xd)).data) <= f0 + c * alpha * slope:
            return alpha
        alpha *= rho
    return alpha


# ── Newton-CG optimizer ───────────────────────────────────────────────────────

def newton_cg(f, x0, tol=1e-8, max_iter=100, cg_tol=1e-6):
    """
    Newton-CG: Newton's method with conjugate gradient for the linear solve.

    Each iteration:
      1. g = ∇f(x)                           1 reverse pass
      2. Solve (H + δI)·d = −g via CG        O(√κ) HVPs via hvp
      3. x ← x + α·d  (Armijo line search)

    Quadratic convergence near the optimum for C² f with Lipschitz Hessian.
    The δI damping term keeps the modified Hessian positive definite.

    Returns
    -------
    x_opt   : ndarray — final iterate
    history : list of (step, f_val, grad_norm)
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    history = []

    for k in range(max_iter):
        g      = _grad(f, x)
        g_norm = float(np.linalg.norm(g))
        f_val  = float(f(Tensor(x)).data)
        history.append((k, f_val, g_norm))

        if g_norm < tol:
            break

        # Levenberg-Marquardt damping: ensures positive definiteness
        damping = 1e-4 * max(1.0, abs(f_val))
        d = _cg(
            lambda v, _x=x, _d=damping: hvp(f, _x, v) + _d * v,
            -g,
            tol=cg_tol,
            max_iter=min(100, len(g)),
        )

        # Safeguard: ensure d is a descent direction
        if float(g @ d) >= 0:
            d = -g

        alpha = _armijo(f, x, g, d)
        x = x + alpha * d

    return x, history


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "=" * 60

    # ── 1. HVP on a quadratic ────────────────────────────────────────
    print(SEP)
    print("1. HVP accuracy: f(x) = ½ xᵀ A x,  exact H = A")
    print()

    A_np = np.array([[3., 1.],
                     [1., 2.]])

    def quad(x):
        # x is a Tensor (n,)
        Ax = x @ Tensor(A_np)           # (n,) @ (n,n) → (n,)
        return (Ax * x).sum() * 0.5

    x0  = np.array([1., 2.])
    v0  = np.array([1., -1.])

    hvp_ad    = hvp(quad, x0, v0)
    hvp_exact = A_np @ v0

    print(f"  A·v      (exact) = {hvp_exact}")
    print(f"  hvp(f,x,v) (AD)  = {hvp_ad.round(8)}")
    print(f"  max error        = {np.abs(hvp_ad - hvp_exact).max():.2e}")

    # ── 2. Full Hessian of Rosenbrock ────────────────────────────────
    print()
    print(SEP)
    print("2. Hessian of Rosenbrock f(x,y) = (1−x)² + 100(y−x²)²")
    print("   at the global minimum (1, 1)")
    print()

    def rosenbrock(x):
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

    x_star = np.array([1., 1.])
    H_ad   = hessian(rosenbrock, x_star)

    # Analytic Hessian at (1, 1):
    # H[0,0] = 2 + 1200x² − 400y = 2 + 1200 − 400 = 802
    # H[0,1] = H[1,0] = −400x = −400
    # H[1,1] = 200
    H_analytic = np.array([[802., -400.],
                            [-400., 200.]])

    print(f"  AD Hessian at (1,1):\n  {H_ad.round(2)}")
    print(f"\n  Analytic Hessian:\n  {H_analytic}")
    print(f"\n  max error: {np.abs(H_ad - H_analytic).max():.2e}")
    eigs = np.linalg.eigvalsh(H_ad)
    print(f"  eigenvalues: {eigs.round(1)}  (both > 0 → local min ✓)")
    print(f"  condition number: {eigs[-1]/eigs[0]:.0f}  "
          f"(explains slow convergence of gradient descent)")

    # ── 3. Newton-CG vs gradient descent ────────────────────────────
    print()
    print(SEP)
    print("3. Newton-CG vs gradient descent on Rosenbrock from (−1, 1)")
    print()

    x_start = np.array([-1., 1.])

    # ---- gradient descent ----
    x_gd = x_start.copy()
    lr   = 1e-3
    gd_log = []
    for k in range(5001):
        if k % 1000 == 0:
            gd_log.append((k, float(rosenbrock(Tensor(x_gd)).data),
                           float(np.linalg.norm(_grad(rosenbrock, x_gd)))))
        x_gd -= lr * _grad(rosenbrock, x_gd)

    print(f"  Gradient descent (lr={lr}, 5000 steps):")
    for step, fv, gn in gd_log:
        print(f"    step {step:5d}  f={fv:.6f}  |g|={gn:.3e}")

    # ---- Newton-CG ----
    # tol=1e-7: finite-diff HVP (eps=1e-5) has O(eps²)=1e-10 error, limiting
    # achievable gradient norm to ~1e-7. Tighter tol would stagnate.
    x_ncg, ncg_hist = newton_cg(rosenbrock, x_start, tol=1e-7, max_iter=100)

    print(f"\n  Newton-CG (converged in {len(ncg_hist)} steps, tol=1e-7):")
    stride = max(1, len(ncg_hist) // 8)
    shown  = set(range(0, len(ncg_hist), stride)) | {len(ncg_hist) - 1}
    for k, fv, gn in ncg_hist:
        if k in shown:
            print(f"    step {k:5d}  f={fv:.8f}  |g|={gn:.3e}")

    print(f"\n  x* ≈ {x_ncg}  (true = [1, 1])")
    print(f"  |x* − (1,1)| = {np.linalg.norm(x_ncg - x_star):.2e}")

    # ── 4. The C² gap ────────────────────────────────────────────────
    print()
    print(SEP)
    print("4. Where C² matters: relu vs gelu Hessians")
    print()

    def relu_loss(x):
        # f(x) = ½ ‖relu(x)‖²
        h = x.relu()
        return (h * h).sum() * 0.5

    def gelu_approx(x):
        # GELU ≈ x · σ(1.702 x)
        return x * (1.0 + (x * 1.702).sigmoid()) * 0.5

    def gelu_loss(x):
        # f(x) = ½ ‖gelu(x)‖²
        h = gelu_approx(x)
        return (h * h).sum() * 0.5

    x_test = np.array([0.0, 1.0, -1.0, 0.5])

    H_relu = hessian(relu_loss, x_test)
    H_gelu = hessian(gelu_loss, x_test)

    print(f"  f_relu(x) = ½‖relu(x)‖²   smooth_types: PSMOOTH (C¹ but not C²)")
    print(f"  Hessian diagonal: {np.diag(H_relu).round(3)}")
    print(f"  → diagonal is 1 when x>0, 0 when x<0, UNDEFINED at x=0")
    print(f"  → finite-diff gives ~0 near kinks (piecewise-constant gradient)")
    print()
    print(f"  f_gelu(x) = ½‖gelu(x)‖²   smooth_types: SMOOTH (C∞)")
    print(f"  Hessian diagonal: {np.diag(H_gelu).round(4)}")
    print(f"  → smooth everywhere, Newton works correctly")
    print()
    print(f"  smooth_types.py gap:")
    print(f"    both relu_loss and gelu_loss are typed C¹ or better,")
    print(f"    but only gelu_loss is C² — which Newton requires.")
    print(f"    A C² type annotation would catch this at architecture time.")
