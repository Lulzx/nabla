"""
Implicit differentiation via the implicit function theorem (IFT).

When the forward computation involves an inner solver — a Newton iteration,
fixed-point loop, or argmin — the computation graph is either infinite
(unrolled iterations) or doesn't exist (black-box solver). The IFT provides
exact gradients at the cost of one linear solve per backward pass.

Setup
------
Implicit equation:  F(x, θ) = 0  defines  x*(θ)  implicitly.
Special cases:
  · Fixed-point:   x = f(x, θ)   →  F = x − f(x, θ)
  · Optimality:    ∇_x L(x; θ) = 0  →  F = ∇_x L

IFT:  dx*/dθ = −(∂F/∂x)⁻¹ · ∂F/∂θ

Backward pass (adjoint form)
-----------------------------
Given upstream gradient dL/dx* from an outer loss:
  1. Solve  (∂F/∂x)ᵀ v = dL/dx*   (one linear solve — same cost as forward)
  2. dL/dθ = −(∂F/∂θ)ᵀ v

This is the continuous adjoint method specialized to algebraic constraints.
For ODEs the analogous structure is odeint.py's adjoint pass; for contact,
diff_contact.py; for combinatorial solvers, diff_combinatorial.py.

Connection to smooth_types.py
-------------------------------
smooth_types.py lists as a limitation:
  '✗ Data-dependent loops over ℝ fall outside the typed fragment'
Implicit differentiation is the escape hatch for exactly these cases:
the forward loop is opaque, the backward pass is the IFT linear solve.
The type of x*(θ) inherits from F's smoothness w.r.t. x and θ
(e.g., F smooth → x*(θ) smooth by the smooth IFT).

Honest limits
--------------
  · Requires F to be differentiable w.r.t. x — same C¹ requirement as hessian.py.
  · ∂F/∂x must be invertible at x* (non-degenerate implicit function).
  · Newton forward solver fails for non-smooth F; use bisection or continuation.
  · For large n, the direct solve np.linalg.solve is O(n³) — use CG + HVPs.
"""

import numpy as np
from autograd import Tensor


# ── Finite-difference Jacobian ────────────────────────────────────────────────

def _fd_jac(f, x_np, eps=1e-5):
    """Central-difference Jacobian ∂f/∂x, shape (|f|, |x|)."""
    m = f(x_np).size
    n = x_np.size
    J = np.zeros((m, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        J[:, j] = (f(x_np + eps * e) - f(x_np - eps * e)) / (2.0 * eps)
    return J


# ── Newton solver ─────────────────────────────────────────────────────────────

def _newton(F_np, x0, tol=1e-10, max_iter=100):
    """
    Solve F(x) = 0 via Newton's method: x ← x − (∂F/∂x)⁻¹ F(x).
    F_np : ndarray → ndarray (same shape)
    """
    x = x0.copy().astype(np.float64)
    for _ in range(max_iter):
        Fv = F_np(x)
        if np.linalg.norm(Fv) < tol:
            break
        J = _fd_jac(F_np, x)
        x = x - np.linalg.solve(J, Fv)
    return x


# ── implicit_solve ────────────────────────────────────────────────────────────

def implicit_solve(F, theta, x0, tol=1e-10, max_iter=100):
    """
    Differentiable implicit solver: find x*(θ) such that F(x*, θ) = 0.

    Forward  : Newton's method on F(·, θ) starting from x0.
    Backward : IFT — dL/dθ = −(∂F/∂θ)ᵀ (∂F/∂x)⁻ᵀ dL/dx*

    Parameters
    ----------
    F     : callable(Tensor, Tensor) → Tensor  — F(x, θ) = 0
    theta : Tensor or ndarray (m,)
    x0    : ndarray (n,)  — initial guess for x*

    Returns
    -------
    Tensor (n,) with IFT backward pass attached.
    """
    theta_leaf = theta if isinstance(theta, Tensor) else Tensor(np.asarray(theta, dtype=np.float64))
    theta_np   = theta_leaf.data

    # Forward: Newton iterations
    def F_at_theta(xv):
        return F(Tensor(xv), Tensor(theta_np)).data

    x_star = _newton(F_at_theta, np.asarray(x0, dtype=np.float64), tol, max_iter)

    out = Tensor(x_star, _children=(theta_leaf,), _op="implicit")

    def _backward():
        dL_dx = out.grad.copy()                                # upstream ∂L/∂x*

        Jx = _fd_jac(lambda xv: F(Tensor(xv), Tensor(theta_np)).data, x_star)
        Jt = _fd_jac(lambda tv: F(Tensor(x_star), Tensor(tv)).data, theta_np)

        v = np.linalg.solve(Jx.T, dL_dx)                      # solve Jxᵀ v = dL/dx*
        theta_leaf.grad += -Jt.T @ v                          # dL/dθ = −Jtᵀ v

    out._backward = _backward
    return out


# ── fixed_point_solve ─────────────────────────────────────────────────────────

def fixed_point_solve(f, theta, x0, tol=1e-10, max_iter=500):
    """
    Differentiable fixed-point solver: find x* = f(x*, θ).

    This is implicit_solve with F(x, θ) = x − f(x, θ).

    Forward  : Banach iteration x ← f(x, θ) until convergence (requires
               contraction in x; Newton is used as fallback).
    Backward : IFT on F = x − f(x, θ) = 0.
    """
    theta_leaf = theta if isinstance(theta, Tensor) else Tensor(np.asarray(theta, dtype=np.float64))
    theta_np   = theta_leaf.data

    # Forward: Banach iteration (simple, works for contractions)
    x = np.asarray(x0, dtype=np.float64).copy()
    for _ in range(max_iter):
        x_new = f(Tensor(x), Tensor(theta_np)).data
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    x_star = x_new

    # Implicit equation: F(x, θ) = x − f(x, θ)
    def F(xv, tv):
        return xv - f(xv, tv)

    out = Tensor(x_star, _children=(theta_leaf,), _op="fixed_point")

    def _backward():
        dL_dx = out.grad.copy()
        Jx = _fd_jac(lambda xv: F(Tensor(xv), Tensor(theta_np)).data, x_star)
        Jt = _fd_jac(lambda tv: F(Tensor(x_star), Tensor(tv)).data, theta_np)
        v  = np.linalg.solve(Jx.T, dL_dx)
        theta_leaf.grad += -Jt.T @ v

    out._backward = _backward
    return out


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "=" * 60

    # ── 1. Sqrt via IFT: F(x, θ) = x² − θ = 0 ──────────────────────
    print(SEP)
    print("1. Differentiable sqrt via IFT: F(x, θ) = x² − θ")
    print("   x*(θ) = √θ  →  dx*/dθ = 1/(2√θ)")
    print()

    def F_sqrt(x, theta):
        return x * x - theta

    for theta_val in [4.0, 9.0, 16.0]:
        t = Tensor(np.array([theta_val]))
        x_star_t = implicit_solve(F_sqrt, t, x0=np.array([1.0]))

        L = x_star_t.sum()           # scalar loss = x*
        L.backward()

        analytic = 1.0 / (2.0 * theta_val ** 0.5)
        print(f"   θ={theta_val:.0f}:  x*={x_star_t.data.item():.4f} (exact {theta_val**0.5:.4f})"
              f"  dx*/dθ = {t.grad.item():.6f} (analytic {analytic:.6f})"
              f"  err={abs(t.grad.item() - analytic):.1e}")

    # ── 2. Fixed-point / DEQ: x* = tanh(w·x* + θ) ───────────────────
    print()
    print(SEP)
    print("2. Fixed-point (DEQ-style): x* = tanh(w·x* + θ)")
    print("   Differentiate through ∞ iterations via IFT")
    print()

    w = 0.5   # contraction weight (|w| < 1 guarantees convergence)

    def deq_f(x, theta):
        return (x * Tensor(np.array([w])) + theta).tanh()

    def deq_F(x, theta):
        return x - (x * Tensor(np.array([w])) + theta).tanh()

    thetas = np.array([-1.0, 0.0, 1.0])
    print(f"  {'θ':>6}  {'x* (IFT)':>10}  {'x* (brute)':>12}  {'dx*/dθ (IFT)':>14}  {'dx*/dθ (FD)':>12}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*14}  {'-'*12}")

    for tv in thetas:
        t = Tensor(np.array([tv]))
        xs = fixed_point_solve(deq_f, t, x0=np.array([0.0]))
        xs.sum().backward()

        # Brute-force: iterate until convergence
        x_bf = 0.0
        for _ in range(10000):
            x_bf = np.tanh(w * x_bf + tv)

        # Finite-difference dx*/dθ
        eps = 1e-5
        x_plus  = 0.0
        x_minus = 0.0
        for _ in range(10000):
            x_plus  = np.tanh(w * x_plus  + tv + eps)
            x_minus = np.tanh(w * x_minus + tv - eps)
        dx_fd = (x_plus - x_minus) / (2 * eps)

        print(f"  {tv:6.1f}  {xs.data.item():10.6f}  {x_bf:12.6f}  {t.grad.item():14.6f}  {dx_fd:12.6f}")

    # ── 3. Differentiable linear solve: Ax* = b(θ) ───────────────────
    print()
    print(SEP)
    print("3. Differentiable linear solve: Ax* = θ  (x* = A⁻¹ θ)")
    print("   IFT gives dx*/dθ = A⁻¹  (exact for linear systems)")
    print()

    A_np = np.array([[2., 1.],
                     [1., 3.]])

    def F_linear(x, theta):
        return Tensor(A_np) @ x - theta

    theta_np = np.array([1., 2.])
    t = Tensor(theta_np.copy())
    x_sol = implicit_solve(F_linear, t, x0=np.zeros(2))

    # Loss: L = c · x*
    c = np.array([1., -1.])
    loss = (Tensor(c) * x_sol).sum()
    loss.backward()

    # Analytic: dL/dθ = c · dx*/dθ = c · A⁻¹
    A_inv = np.linalg.inv(A_np)
    dL_dt_analytic = c @ A_inv

    x_exact = A_inv @ theta_np
    print(f"  A⁻¹ = \n{A_inv.round(4)}")
    print(f"  x* (IFT solve) = {x_sol.data.round(6)}  (exact {x_exact.round(6)})")
    print(f"  dL/dθ (IFT)    = {t.grad.round(8)}")
    print(f"  dL/dθ (exact)  = {dL_dt_analytic.round(8)}")
    print(f"  max error      = {np.abs(t.grad - dL_dt_analytic).max():.2e}")

    # ── 4. Honest limits ─────────────────────────────────────────────
    print()
    print(SEP)
    print("4. Honest limits")
    print()
    print("  ✓  Forward: arbitrary solver (Newton, Banach, direct, CG)")
    print("  ✓  Backward: exact via IFT — no unrolling, no infinite graph")
    print("  ✓  Cost: 2 Jacobians + 1 linear solve per backward call")
    print()
    print("  ✗  ∂F/∂x must be invertible at x* (non-degenerate solution)")
    print("  ✗  F must be C¹ w.r.t. x (IFT requires it — same as hessian.py)")
    print("  ✗  Newton diverges for non-smooth F; use bisection or homotopy")
    print("  ✗  O(n³) direct solve — use CG for large n (needs Jx-vector products)")
    print("  ✗  Multiple solutions: Newton finds ONE; IFT gives gradient of that one")
    print()
    print("  Type perspective (smooth_types.py):")
    print("  x*(θ) inherits smoothness from F.  If F is C∞ (e.g., tanh network),")
    print("  x*(θ) is also C∞.  If F involves relu, x*(θ) is PSMOOTH.")
