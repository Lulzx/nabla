"""
Adaptive ODE solver (Dormand-Prince RK45) with adjoint backward pass.

Why adaptive stepping matters
------------------------------
Fixed-step RK4 uses the same h everywhere. For stiff systems or problems
with localised features (fast transients, near-contact), this means either:
  - h too large → inaccurate
  - h too small everywhere → expensive

RK45 (Dormand-Prince) computes both a 4th and 5th order estimate per step,
uses the difference as a local error estimate, and adjusts h accordingly.

Backward pass (adjoint with stored steps)
------------------------------------------
The adaptive step sequence {(t_i, h_i)} is data-dependent — it depends on
the solution itself. A correct adjoint must account for this.

We use the "stored-step adjoint": save {t_i, h_i, y_i} during the forward
pass, then replay exactly those steps in the backward pass using
_rk4_vjp from odeint.py. This is exact for the stored trajectory.

The alternative (continuous adjoint with its own adaptive step control)
is more accurate but requires 2× the integration work. See torchdiffeq
for that approach.
"""

import numpy as np
from autograd import Tensor

# ------------------------------------------------------------------ #
# Dormand-Prince (DOPRI5) coefficients                                #
# ------------------------------------------------------------------ #

# Butcher tableau
_A = [
    [],
    [1/5],
    [3/40,      9/40],
    [44/45,    -56/15,     32/9],
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    [9017/3168,  -355/33,   46732/5247,   49/176,   -5103/18656],
    [35/384,     0,         500/1113,    125/192,  -2187/6784,   11/84],
]
_c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]

# 5th-order solution weights (same as last row of A above)
_b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]

# Error coefficients  e = b5 - b4  (b4 is the 4th-order weights)
_e = [71/57600, 0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40]


def _dopri5_vjp(f, y_np, t_i, h, p_np, v):
    """
    VJP of one DOPRI5 step w.r.t. y and params.
    Builds a small Nabla graph for the step, seeds backward with v.
    """
    y = Tensor(y_np)
    p = Tensor(p_np)

    k1 = f(y, t_i, p)
    k2 = f(y + k1 * (_A[1][0] * h),                                          t_i + _c[1]*h, p)
    k3 = f(y + k1*(_A[2][0]*h) + k2*(_A[2][1]*h),                            t_i + _c[2]*h, p)
    k4 = f(y + k1*(_A[3][0]*h) + k2*(_A[3][1]*h) + k3*(_A[3][2]*h),          t_i + _c[3]*h, p)
    k5 = f(y + k1*(_A[4][0]*h) + k2*(_A[4][1]*h) + k3*(_A[4][2]*h)
             + k4*(_A[4][3]*h),                                                t_i + _c[4]*h, p)
    k6 = f(y + k1*(_A[5][0]*h) + k2*(_A[5][1]*h) + k3*(_A[5][2]*h)
             + k4*(_A[5][3]*h) + k5*(_A[5][4]*h),                             t_i + _c[5]*h, p)

    y_next = y + (k1*_b5[0] + k3*_b5[2] + k4*_b5[3] + k5*_b5[4] + k6*_b5[5]) * h

    (Tensor(v) * y_next).sum().backward()
    return y.grad.copy(), p.grad.copy()


def _dopri5_step(f, y, t, h, p):
    """
    One Dormand-Prince step. Returns (y_next, error_estimate).
    Uses FSAL (first-same-as-last): k1 of next step = k7 of this step.
    """
    def call(y_, t_):
        return f(Tensor(y_), t_, Tensor(p)).data

    k1 = call(y, t)
    k2 = call(y + h * (_A[1][0]*k1),                                  t + _c[1]*h)
    k3 = call(y + h * (_A[2][0]*k1 + _A[2][1]*k2),                    t + _c[2]*h)
    k4 = call(y + h * (_A[3][0]*k1 + _A[3][1]*k2 + _A[3][2]*k3),     t + _c[3]*h)
    k5 = call(y + h * (_A[4][0]*k1 + _A[4][1]*k2 + _A[4][2]*k3
                      + _A[4][3]*k4),                                  t + _c[4]*h)
    k6 = call(y + h * (_A[5][0]*k1 + _A[5][1]*k2 + _A[5][2]*k3
                      + _A[5][3]*k4 + _A[5][4]*k5),                   t + _c[5]*h)

    y_next = y + h * (_b5[0]*k1 + _b5[2]*k3 + _b5[3]*k4
                     + _b5[4]*k5 + _b5[5]*k6)

    k7 = call(y_next, t + h)   # FSAL

    err = h * (_e[0]*k1 + _e[2]*k3 + _e[3]*k4 + _e[4]*k5 + _e[5]*k6 + _e[6]*k7)
    return y_next, err


# ------------------------------------------------------------------ #
# Adaptive integrator                                                  #
# ------------------------------------------------------------------ #

def odeint_adaptive(f, y0, t_span, params,
                    rtol=1e-4, atol=1e-6, max_steps=50_000,
                    h0=None, h_min=1e-10):
    """
    Integrate dy/dt = f(y, t, params) with adaptive RK45.

    Parameters
    ----------
    f        : callable(Tensor, float, Tensor) -> Tensor
    y0       : array_like (d,)
    t_span   : (t0, t1)
    params   : array_like or Tensor (p,)
    rtol, atol : relative and absolute tolerance for step control
    max_steps  : safety limit on the number of steps

    Returns
    -------
    t_out    : ndarray (N,)   — times of accepted steps
    y_out    : Tensor (N, d) — states at those times, with adjoint backward
    """
    y0_leaf = y0     if isinstance(y0,     Tensor) else Tensor(np.asarray(y0,     dtype=np.float64))
    p_leaf  = params if isinstance(params, Tensor) else Tensor(np.asarray(params, dtype=np.float64))
    y0_np   = y0_leaf.data
    p_np    = p_leaf.data
    t0, t1  = float(t_span[0]), float(t_span[1])

    # ---- Forward: adaptive RK45 ----
    h = h0 if h0 is not None else (t1 - t0) * 1e-2
    t  = t0
    y  = y0_np.copy()

    ts   = [t]
    ys   = [y.copy()]
    hs   = []          # accepted step sizes (for backward replay)

    n_steps = 0
    while t < t1 and n_steps < max_steps:
        h = min(h, t1 - t)

        y_next, err = _dopri5_step(f, y, t, h, p_np)

        # Error norm: mixed absolute/relative
        sc    = atol + rtol * np.maximum(np.abs(y), np.abs(y_next))
        err_norm = np.sqrt(np.mean((err / sc) ** 2))

        if err_norm <= 1.0:
            # Accept step
            t = t + h
            y = y_next
            ts.append(t)
            ys.append(y.copy())
            hs.append(h)
            n_steps += 1

        # PI step-size control (Hairer et al.)
        factor = 0.9 * err_norm ** (-0.2)
        factor = np.clip(factor, 0.1, 10.0)
        h = max(h * factor, h_min)

    t_out = np.array(ts)
    y_traj = np.stack(ys)                       # (N, d)

    out = Tensor(y_traj, _children=(y0_leaf, p_leaf), _op="odeint_adaptive")

    # ---- Backward: stored-step adjoint ----
    def _backward():
        a     = out.grad[-1].copy()
        dL_dp = np.zeros_like(p_np)

        for i in reversed(range(len(hs))):
            h_i = hs[i]
            # DOPRI5 VJP — matches the forward solver exactly, O(d) memory
            da, dp = _dopri5_vjp(f, ys[i], ts[i], h_i, p_np, a)
            a      = da
            dL_dp += dp
            a     += out.grad[i]

        y0_leaf.grad += a
        p_leaf.grad  += dL_dp

    out._backward = _backward
    return t_out, out


# ------------------------------------------------------------------ #
# Demo                                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    from autograd import stack, gradient_check
    from odeint import odeint

    SEP = "-" * 56

    # ---- 1. Step count comparison on a smooth problem ----
    print(SEP)
    print("1. Step counts: fixed-step RK4 vs adaptive RK45")

    def harmonic(y, t, p):
        # y'' + ω²y = 0  →  [y, y']' = [y', -ω²y]
        omega = p[0]
        return stack([y[1], -omega**2 * y[0]], axis=0)

    y0     = np.array([1.0, 0.0])
    params = np.array([5.0])      # ω = 5 (moderately fast oscillation)

    for n_fixed in [20, 50, 200, 1000]:
        t_fixed = np.linspace(0, 2*np.pi, n_fixed)
        y_fixed = odeint(harmonic, y0, t_fixed, params).data
        err = abs(y_fixed[-1, 0] - np.cos(5 * 2 * np.pi))
        print(f"  fixed  N={n_fixed:5d}  final_err={err:.2e}")

    t_out, y_adapt = odeint_adaptive(harmonic, y0, (0, 2*np.pi), params,
                                     rtol=1e-6, atol=1e-8)
    err_adapt = abs(y_adapt.data[-1, 0] - np.cos(5 * 2 * np.pi))
    print(f"  adaptive N={len(t_out):5d}  final_err={err_adapt:.2e}  "
          f"(rtol=1e-6, atol=1e-8)")

    # ---- 2. Stiff ODE: exponential decay ----
    print(SEP)
    print("2. Stiff ODE: dy/dt = -k*y  (k large)")

    def stiff_decay(y, t, p):
        k = p[0]
        return stack([-k * y[0]], axis=0)

    y0_s  = np.array([1.0])
    k_val = np.array([500.0])   # stiff! (RK4 stability limit: h < 2.79/k ≈ 0.0056)

    # RK4 stability limit: kh < 2.785  →  h < 2.785/500 = 0.00557
    # N=50  → h≈0.0102, kh≈5.1  → DIVERGES
    # N=200 → h≈0.0025, kh≈1.25 → stable
    true_val = float(np.exp(-500 * 0.5))
    for n_fixed in [50, 200]:
        t_s = np.linspace(0, 0.5, n_fixed)
        y_s = odeint(stiff_decay, y0_s, t_s, k_val).data
        err = abs(y_s[-1, 0] - true_val)
        stable = "OK" if err < 1.0 else "DIVERGED"
        print(f"  fixed  N={n_fixed:5d}  err={err:.2e}  {stable}")

    t_s_out, y_s_adapt = odeint_adaptive(stiff_decay, y0_s, (0, 0.5), k_val,
                                          rtol=1e-6, atol=1e-8)
    err_adapt = abs(y_s_adapt.data[-1, 0] - true_val)
    print(f"  adaptive N={len(t_s_out):5d}  err={err_adapt:.2e}  "
          f"(auto step-size keeps solution stable)")

    # ---- 3. Gradient check ----
    print(SEP)
    print("3. Gradient check: ∂y(t=0.5)/∂params  via adjoint vs FD")
    print("   (use final state at fixed time — avoids variable stored-time artefacts)")

    def simple(y, t, p):
        return stack([p[0] * y[0] + p[1]], axis=0)

    y0_g  = np.array([1.0])
    p_g   = Tensor(np.array([0.5, 0.1]))

    def run(p):
        _, y = odeint_adaptive(simple, y0_g, (0, 0.5), p, rtol=1e-7, atol=1e-9)
        return y[-1]   # final state at t=0.5 (fixed endpoint — FD-comparable)

    ok = gradient_check(run, p_g, atol=1e-3)
    print(f"  adaptive adjoint gradient check: {'PASS' if ok else 'FAIL'}")
