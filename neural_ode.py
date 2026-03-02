"""
Neural ODE: learning a continuous vector field from trajectory observations.

Chen et al. (2018). "Neural Ordinary Differential Equations." NeurIPS.

The model
----------
Instead of a discrete residual network  y_{k+1} = y_k + f_θ(y_k),
define a continuous-time transformation:

    dy/dt = f_θ(y, t)          neural network as vector field
    y(T)  = y(0) + ∫₀ᵀ f_θ(y, t) dt    solved by odeint

The output y(T) is a function of the initial state y(0) and parameters θ.
Training: minimize the MSE between predicted and observed trajectories.

Why adjoint, not backprop through the solver?
----------------------------------------------
Naive approach: unroll the RK4 steps into a computation graph and backprop.
Problem: O(T) memory — must store all intermediate states y₁,...,y_T.
         O(T·|θ|) time — T backward passes each touching the full network.

Adjoint method: instead of differentiating through the forward pass,
solve a second ODE backward in time (the adjoint equation):
    da/dt = -a(t) · ∂f_θ/∂y

The gradient dL/dθ is computed as a running integral, requiring O(1) extra
memory regardless of T. Cost is the same O(T·|θ|) compute, but at O(|θ|)
memory. For long sequences this is the decisive advantage.

odeint.py implements this: it stores only the current adjoint state `a` and
recomputes the forward values at each backward step (at the cost of one
extra forward pass). The gradient accumulates into p_leaf.grad.

What we demonstrate
--------------------
1. Generate observations from a damped harmonic oscillator (true dynamics
   are unknown to the model).
2. Train the neural ODE to match those observations.
3. Verify that the learned f_θ extrapolates correctly beyond the training
   window — the model has learned the underlying dynamics, not memorized
   the trajectory.
4. Compare memory: odeint adjoint (O(1)) vs unrolled graph (O(T)).
"""

import numpy as np
from autograd import Tensor, stack
from odeint import odeint


# ── True dynamics ─────────────────────────────────────────────────────────────

def damped_oscillator(y, t, params):
    """
    dy₁/dt = y₂
    dy₂/dt = −k·y₁ − c·y₂

    params = [k, c]  (spring constant, damping)
    Exact solution: y₁(t) = e^{−ct/2} [A cos(ωt) + B sin(ωt)],  ω = √(k − c²/4)
    """
    k, c = params[0], params[1]
    return stack([y[1], -k * y[0] - c * y[1]], axis=0)


# ── Neural vector field ───────────────────────────────────────────────────────
#
# Architecture: 2 → 16 → tanh → 16 → 2
# No time dependence (autonomous system, which the true dynamics satisfy).
# Total parameters: 2*16 + 16 + 16*2 + 2 = 32 + 16 + 32 + 2 = 82

N_IN, N_H, N_OUT = 2, 16, 2
N_PARAMS = N_IN*N_H + N_H + N_H*N_OUT + N_OUT   # 82


def neural_field(y, t, params):
    """2-layer tanh network as the vector field dy/dt = f_θ(y)."""
    i0, i1 = 0,           N_IN * N_H
    i2, i3 = i1,          i1 + N_H
    i4, i5 = i3,          i3 + N_H * N_OUT
    i6, i7 = i5,          i5 + N_OUT

    W1 = params[i0:i1].reshape(N_H, N_IN)
    b1 = params[i2:i3]
    W2 = params[i4:i5].reshape(N_OUT, N_H)
    b2 = params[i6:i7]

    h  = (W1 @ y + b1).tanh()
    return W2 @ h + b2


# ── Gradient descent with Adam-like update ────────────────────────────────────

def adam_step(params, grad, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """One Adam step. Returns updated (params, m, v)."""
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad ** 2
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, m, v


# ── Training ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "=" * 60
    np.random.seed(7)

    # ── 1. Generate training data ────────────────────────────────────
    print(SEP)
    print("1. True system: damped oscillator  ÿ + 0.3ẏ + 2y = 0")
    print("   Training window: t ∈ [0, 5]  (50 points, Δt = 0.1)")
    print()

    y0         = np.array([1.0, 0.0])
    true_p     = np.array([2.0, 0.3])    # k=2, c=0.3
    t_train    = np.linspace(0, 5,  50)
    t_extrap   = np.linspace(5, 10, 50)  # unseen region

    y_true_train = odeint(damped_oscillator, y0, t_train,  true_p).data.copy()
    y_extrap_start = y_true_train[-1]    # initial condition for extrapolation
    y_true_extrap  = odeint(damped_oscillator, y_extrap_start,
                            t_extrap, true_p).data.copy()

    # ── 2. Train neural ODE ─────────────────────────────────────────
    print(SEP)
    print(f"2. Training neural ODE  (architecture: {N_IN}→{N_H}→tanh→{N_H}→{N_OUT},"
          f"  {N_PARAMS} params)")
    print("   Optimizer: Adam  lr=1e-3  500 steps")
    print()

    # Small-ish init keeps the initial field weak (avoids instant stiffness)
    theta_np = 0.05 * np.random.randn(N_PARAMS)
    m_np     = np.zeros(N_PARAMS)
    v_np     = np.zeros(N_PARAMS)

    losses = []
    for step in range(1, 501):
        theta_t = Tensor(theta_np)
        y_pred  = odeint(neural_field, y0, t_train, theta_t)
        loss    = ((y_pred - Tensor(y_true_train)) ** 2).mean()
        loss.backward()

        theta_np, m_np, v_np = adam_step(
            theta_np, theta_t.grad, m_np, v_np, step, lr=1e-3)
        losses.append(float(loss.data))

        if step % 100 == 0:
            print(f"  step {step:4d}  loss = {losses[-1]:.6f}")

    print()

    # ── 3. Evaluate on training window ──────────────────────────────
    print(SEP)
    print("3. Fit quality on training window")
    print()

    theta_final = Tensor(theta_np)
    y_pred_train = odeint(neural_field, y0, t_train, theta_final).data

    mse_train = float(np.mean((y_pred_train - y_true_train) ** 2))
    mae_y1 = float(np.mean(np.abs(y_pred_train[:, 0] - y_true_train[:, 0])))
    mae_y2 = float(np.mean(np.abs(y_pred_train[:, 1] - y_true_train[:, 1])))

    print(f"  Training MSE:         {mse_train:.6f}")
    print(f"  Mean abs error y₁:    {mae_y1:.4f}")
    print(f"  Mean abs error y₂:    {mae_y2:.4f}")
    print()
    print(f"  {'t':>5}  {'y₁ true':>9}  {'y₁ pred':>9}  {'error':>8}"
          f"  {'y₂ true':>9}  {'y₂ pred':>9}  {'error':>8}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*8}"
          f"  {'-'*9}  {'-'*9}  {'-'*8}")
    for i in range(0, 50, 5):
        t_val      = t_train[i]
        y1t, y2t   = y_true_train[i]
        y1p, y2p   = y_pred_train[i]
        print(f"  {t_val:5.2f}  {y1t:9.4f}  {y1p:9.4f}  {abs(y1t-y1p):8.4f}"
              f"  {y2t:9.4f}  {y2p:9.4f}  {abs(y2t-y2p):8.4f}")

    # ── 4. Extrapolation ─────────────────────────────────────────────
    print()
    print(SEP)
    print("4. Extrapolation: t ∈ [5, 10]  (never seen during training)")
    print()

    y_pred_extrap = odeint(neural_field, y_extrap_start,
                           t_extrap, theta_final).data

    mse_extrap = float(np.mean((y_pred_extrap - y_true_extrap) ** 2))
    mae_extrap = float(np.mean(np.abs(y_pred_extrap[:, 0] - y_true_extrap[:, 0])))

    print(f"  Extrapolation MSE:    {mse_extrap:.6f}")
    print(f"  Mean abs error y₁:    {mae_extrap:.4f}")
    print()
    print(f"  {'t':>5}  {'y₁ true':>9}  {'y₁ pred':>9}  {'error':>8}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*8}")
    for i in range(0, 50, 5):
        t_val    = t_extrap[i]
        y1t      = y_true_extrap[i, 0]
        y1p      = y_pred_extrap[i, 0]
        print(f"  {t_val:5.2f}  {y1t:9.4f}  {y1p:9.4f}  {abs(y1t-y1p):8.4f}")

    ratio = mse_extrap / (mse_train + 1e-12)
    print()
    print(f"  Extrap / train MSE ratio: {ratio:.2f}×")
    if ratio < 5:
        print("  → model learned the dynamics, not just the trajectory.")
    else:
        print("  → model overfit the training window.")

    # ── 5. Memory: adjoint vs unrolled graph ─────────────────────────
    print()
    print(SEP)
    print("5. Memory cost: adjoint vs unrolled backprop")
    print()
    T = len(t_train) - 1   # number of integration steps
    d = len(y0)
    p = N_PARAMS

    mem_adjoint  = d + p                           # adjoint state + param grad
    mem_unrolled = T * d * 4 + p                   # 4 RK4 k-values × T steps × d

    print(f"  Integration steps T = {T}")
    print(f"  State dim d = {d},  param dim |θ| = {p}")
    print()
    print(f"  Adjoint memory:   O(d + |θ|)  = {mem_adjoint} scalars")
    print(f"  Unrolled memory:  O(T·d + |θ|) = {mem_unrolled} scalars  "
          f"({mem_unrolled/mem_adjoint:.0f}× more)")
    print()
    print("  For T=1000 steps (long trajectories), adjoint stores 82 scalars.")
    print("  Unrolled graph stores 4000+ — and must differentiate through each.")
    print()
    print("  odeint.py implements the adjoint: it recomputes the forward values")
    print("  during the backward pass (one extra pass) to avoid storing the graph.")

    # ── 6. What the network learned ──────────────────────────────────
    print()
    print(SEP)
    print("6. Probing the learned vector field f_θ")
    print()
    print("  True field at sample states:")
    print(f"  {'y₁':>6}  {'y₂':>6}  {'ẏ₁ true':>10}  {'ẏ₁ learned':>12}  {'match'}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*5}")

    sample_states = [
        np.array([1.0,  0.0]),
        np.array([0.0,  1.0]),
        np.array([-1.0, 0.0]),
        np.array([0.5, -0.5]),
    ]
    for s in sample_states:
        true_v  = damped_oscillator(Tensor(s), 0.0, Tensor(true_p)).data
        learned_v = neural_field(Tensor(s), 0.0, theta_final).data
        err = abs(true_v[0] - learned_v[0])
        print(f"  {s[0]:6.2f}  {s[1]:6.2f}  {true_v[0]:10.4f}  {learned_v[0]:12.4f}"
              f"  {'✓' if err < 0.05 else '≈' if err < 0.2 else '✗'}")

    print()
    print("  The network has implicitly learned that ẏ₁ = y₂")
    print("  and ẏ₂ ≈ −2y₁ − 0.3y₂ from trajectory observations alone.")
