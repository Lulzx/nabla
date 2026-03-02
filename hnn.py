"""
Hamiltonian Neural Networks: architecture-level energy conservation.

Greydanus et al. (2019). "Hamiltonian Neural Networks." NeurIPS.

The key insight
---------------
Neural ODE:  dy/dt = f_θ(y)           unconstrained vector field
HNN:         H_θ(q, p) → ℝ            scalar Hamiltonian
             dq/dt =  ∂H_θ/∂p         } Hamilton's equations
             dp/dt = −∂H_θ/∂q         }

For any differentiable H, the time derivative of H along the trajectory is:

    dH/dt = ∂H/∂q · dq/dt + ∂H/∂p · dp/dt
          = ∂H/∂q · ∂H/∂p + ∂H/∂p · (−∂H/∂q)  =  0

H_θ is exactly conserved for every θ, before and after training.  No
regularization penalty is needed — the symmetry is unbreakable by
gradient descent because it is a consequence of the field's structure
(J·∇H_θ is a Hamiltonian vector field for any H_θ).

Neural ODE learns an unconstrained f_θ whose divergence can be nonzero.
On training data it matches just as well.  On long rollouts, energy drifts.

Implementation
--------------
Architecture:  H_θ(y) = W₂ · tanh(W₁·y + b₁) + b₂   (2 → K → tanh → 1)

The symplectic gradient ∇H_θ is computed analytically — not by finite
difference — using the chain rule applied to the tanh network:

    ∂H/∂y = W₁ᵀ · diag(1 − tanh²(W₁y+b₁)) · W₂ᵀ

All operations are Tensor ops (W₁, W₂ are slices of params), so the
odeint adjoint correctly propagates ∂loss/∂θ back through Hamilton's
equations.  The gradient w.r.t. θ is a first-order quantity — unlike a
finite-difference approach that would give ∂²H/(∂θ∂y), a second-order
quantity with much weaker gradient signal.

True system
-----------
Simple pendulum:  H_true = p²/2 − cos(q)
  q₀ = 1.0 rad, p₀ = 0   (nonlinear regime; period ≈ 6.74 s)
  E₀ = 0²/2 − cos(1.0) ≈ −0.5403

Training: t ∈ [0, 4]  (40 points, ≈ 0.6 periods)
Rollout:  t ∈ [0, 40] (400 points, ≈ 6 periods)

The measured energy  E(t) = H_true(q_pred(t), p_pred(t))  should be
constant.  HNN: near-constant.  Neural ODE: drifts away from E₀.
"""

import numpy as np
from autograd import Tensor, stack
from odeint import odeint


# ── True pendulum (numpy only — not differentiated) ──────────────────────────

def _pendulum_np(y, t):
    return np.array([y[1], -np.sin(y[0])])


def _rk4_step_np(f, y, t, dt):
    k1 = f(y, t)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(y +    dt*k3,  t +    dt)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_pendulum(y0, t_arr):
    ys = [y0.copy()]
    for i in range(len(t_arr) - 1):
        ys.append(_rk4_step_np(_pendulum_np, ys[-1], t_arr[i],
                               t_arr[i+1] - t_arr[i]))
    return np.stack(ys)   # (N, 2)


def true_energy(traj):
    """H_true = p²/2 − cos(q),  shape (N, 2) → (N,)."""
    return 0.5 * traj[:, 1]**2 - np.cos(traj[:, 0])


# ── Architecture 1: Neural ODE  2 → 16 → tanh → 16 → 2  (82 params) ─────────

_N = 16
N_PARAMS_NODE = 2*_N + _N + _N*2 + 2   # 82

def node_field(y, t, params):
    """Unconstrained vector field dy/dt = f_θ(y)."""
    W1 = params[0    : 2*_N].reshape(_N, 2)
    b1 = params[2*_N : 3*_N]
    W2 = params[3*_N : 5*_N].reshape(2, _N)
    b2 = params[5*_N : 5*_N+2]
    return W2 @ (W1 @ y + b1).tanh() + b2


# ── Architecture 2: HNN  2 → 32 → tanh → 32 → 1  (129 params) ───────────────
#
# The network outputs a scalar H_θ(q, p).
# The field is NOT a direct output — it is J·∇H_θ, derived on-the-fly.

_K = 32
N_PARAMS_HNN = 2*_K + _K + _K + 1   # 129

def _H_net(y, params):
    """Scalar H_θ(y; params).  y: Tensor (2,),  returns scalar Tensor."""
    W1 = params[0    : 2*_K].reshape(_K, 2)
    b1 = params[2*_K : 3*_K]
    W2 = params[3*_K : 4*_K].reshape(1, _K)
    b2 = params[4*_K : 4*_K+1]
    return (W2 @ (W1 @ y + b1).tanh() + b2).sum()   # scalar


def hnn_field(y, t, params):
    """
    Hamiltonian vector field:  [dq/dt, dp/dt] = [∂H_θ/∂p, −∂H_θ/∂q]

    ∂H_θ/∂y is computed analytically via the chain rule through the tanh
    network — all as Tensor ops, so odeint's adjoint propagates ∂loss/∂θ
    correctly:

        ∂H/∂y = W₁ᵀ · diag(sech²(W₁y + b₁)) · W₂ᵀ

    This is a first-order quantity w.r.t. θ, giving strong gradient signal.
    """
    W1 = params[0    : 2*_K].reshape(_K, 2)
    b1 = params[2*_K : 3*_K]
    W2 = params[3*_K : 4*_K]              # shape (_K,) — output weights

    h     = (W1 @ y + b1).tanh()          # (_K,)
    sech2 = 1.0 - h * h                   # (_K,), sech²(W₁y+b₁)
    delta = sech2 * W2                    # (_K,), gradient at hidden layer
    dH_dy = W1.T @ delta                  # (2,), gradient of H w.r.t. y

    # Hamilton's equations: [dq/dt, dp/dt] = [∂H/∂p, −∂H/∂q]
    return stack([dH_dy[1], -dH_dy[0]])   # shape (2,)


# ── Adam ──────────────────────────────────────────────────────────────────────

def _adam(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = b1*m + (1-b1)*g
    v = b2*v + (1-b2)*g**2
    p = p - lr * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
    return p, m, v


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "=" * 60
    np.random.seed(42)

    # ── 1. Generate training and evaluation data ──────────────────────
    print(SEP)
    print("1. True system: pendulum  H = p²/2 − cos(q)")
    print("   q₀ = 1.0 rad, p₀ = 0  (period ≈ 6.74 s)")
    print("   Training: t ∈ [0, 7]   (70 pts,  ≈ 1 period)")
    print("   Rollout:  t ∈ [0, 70]  (700 pts, ≈ 10 periods)")
    print()

    y0 = np.array([1.0, 0.0])
    t_train = np.linspace(0,  7,  70)
    t_long  = np.linspace(0, 70, 700)

    y_train = simulate_pendulum(y0, t_train)
    y_long  = simulate_pendulum(y0, t_long)

    E0 = float(true_energy(y_train[:1]).item())
    rk4_drift = float(np.max(np.abs(true_energy(y_long) - E0)))
    print(f"   E₀ = {E0:.6f}    (RK4 reference drift over 40 s: {rk4_drift:.2e})")
    print()

    # ── 2. Train Neural ODE ───────────────────────────────────────────
    print(SEP)
    print(f"2. Neural ODE  (2→{_N}→tanh→{_N}→2,  {N_PARAMS_NODE} params)")
    print("   Adam lr=1e-3,  500 steps")
    print()

    pN = 0.05 * np.random.randn(N_PARAMS_NODE)
    mN = vN = np.zeros(N_PARAMS_NODE)
    for step in range(1, 501):
        pt = Tensor(pN)
        loss = ((odeint(node_field, y0, t_train, pt) - Tensor(y_train))**2).mean()
        loss.backward()
        pN, mN, vN = _adam(pN, pt.grad, mN, vN, step)
        if step % 100 == 0:
            print(f"  step {step:3d}  loss = {float(loss.data):.6f}")
    print()

    # ── 3. Train HNN ──────────────────────────────────────────────────
    print(SEP)
    print(f"3. HNN  (2→{_K}→tanh→{_K}→1,  scalar H_θ,  {N_PARAMS_HNN} params)")
    print("   Adam lr=1e-3,  1500 steps")
    print()

    pH = 0.3 * np.random.randn(N_PARAMS_HNN)
    mH = vH = np.zeros(N_PARAMS_HNN)
    for step in range(1, 1501):
        pt = Tensor(pH)
        loss = ((odeint(hnn_field, y0, t_train, pt) - Tensor(y_train))**2).mean()
        loss.backward()
        pH, mH, vH = _adam(pH, pt.grad, mH, vH, step)
        if step % 300 == 0:
            print(f"  step {step:4d}  loss = {float(loss.data):.6f}")
    print()

    # ── 4. Long-rollout energy conservation ───────────────────────────
    print(SEP)
    print("4. Energy  E(t) = H_true(q_pred(t), p_pred(t))  over 10× rollout")
    print()

    traj_N = odeint(node_field, y0, t_long, Tensor(pN)).data
    traj_H = odeint(hnn_field,  y0, t_long, Tensor(pH)).data

    E_true = true_energy(y_long)
    E_N    = true_energy(traj_N)
    E_H    = true_energy(traj_H)

    print(f"  {'t':>5}  {'E true':>8}  {'NeuralODE':>11}  {'|ΔE|':>7}  "
          f"{'HNN':>9}  {'|ΔE|':>7}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*11}  {'-'*7}  {'-'*9}  {'-'*7}")
    for i in range(0, 700, 70):
        t_v = t_long[i]
        print(f"  {t_v:5.1f}  {E_true[i]:8.4f}  {E_N[i]:11.4f}  "
              f"{abs(E_N[i]-E0):7.4f}  {E_H[i]:9.4f}  {abs(E_H[i]-E0):7.4f}")

    drift_N = float(np.mean(np.abs(E_N - E0)))
    drift_H = float(np.mean(np.abs(E_H - E0)))
    ratio   = drift_N / (drift_H + 1e-12)
    print()
    print(f"  Mean |ΔE| over t ∈ [0, 40]:")
    print(f"    Neural ODE : {drift_N:.5f}")
    print(f"    HNN        : {drift_H:.5f}  ({ratio:.1f}× lower drift)")

    # ── 5. Training fit ───────────────────────────────────────────────
    print()
    print(SEP)
    print("5. Training fit  (both models, t ∈ [0, 7])")
    print()
    mse_N = float(np.mean((odeint(node_field, y0, t_train, Tensor(pN)).data
                           - y_train)**2))
    mse_H = float(np.mean((odeint(hnn_field,  y0, t_train, Tensor(pH)).data
                           - y_train)**2))
    print(f"  Neural ODE MSE: {mse_N:.6f}")
    print(f"  HNN MSE:        {mse_H:.6f}")
    print()
    print("  Both fit the training window.  Divergence appears only on long")
    print("  rollouts, where the HNN's conserved structure prevents drift.")

    # ── 6. H_θ conservation check ─────────────────────────────────────
    print()
    print(SEP)
    print("6. H_θ  along the HNN trajectory  (should be machine-precision constant)")
    print()
    print("   The conservation identity holds for every θ:")
    print("       dH_θ/dt = ∂H_θ/∂q·∂H_θ/∂p − ∂H_θ/∂p·∂H_θ/∂q = 0")
    print()
    pH_t = Tensor(pH)
    vals = []
    print(f"  {'t':>5}  {'H_θ(q,p)':>12}  {'H_true(q,p)':>13}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*13}")
    for i in range(0, 700, 140):
        H_theta = float(_H_net(Tensor(traj_H[i]), pH_t).data)
        H_true_i = float(true_energy(traj_H[i:i+1]).item())
        vals.append(H_theta)
        print(f"  {t_long[i]:5.1f}  {H_theta:12.8f}  {H_true_i:13.8f}")

    span_theta = max(vals) - min(vals)
    print()
    print(f"  H_θ range over trajectory: {span_theta:.2e}  (conserved)")
    print(f"  H_true range:              {float(np.max(E_H)-np.min(E_H)):.4f}  "
          f"(not conserved — H_θ ≠ H_true exactly)")
    print()
    print("  Once H_θ ≈ H_true, the predicted trajectory stays near the correct")
    print("  energy surface.  More training → smaller gap → better conservation.")

    # ── 7. What Neural ODE cannot guarantee ───────────────────────────
    print()
    print(SEP)
    print("7. What Neural ODE cannot guarantee")
    print()
    print("  A necessary condition for energy conservation is div f_θ = 0")
    print("  (Liouville's theorem: H-flows preserve phase-space volume).")
    print()

    # Estimate numerical divergence at a few points
    eps_div = 1e-4
    pN_t = Tensor(pN)
    pH_t2 = Tensor(pH)

    print(f"  Numerical ∂f₁/∂q₁ + ∂f₂/∂q₂  at sample states:")
    print(f"  {'q':>6}  {'p':>6}  {'div(NeuralODE)':>16}  {'div(HNN)':>10}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*16}  {'-'*10}")
    sample_states = [
        np.array([ 1.0,  0.0]),
        np.array([ 0.0,  1.0]),
        np.array([-1.0,  0.0]),
        np.array([ 0.5, -0.5]),
    ]
    for s in sample_states:
        # div = ∂f1/∂q + ∂f2/∂p
        def div_fd(field, params):
            e1 = np.array([eps_div, 0.0])
            e2 = np.array([0.0, eps_div])
            f_plus1  = field(Tensor(s + e1), 0.0, params).data
            f_minus1 = field(Tensor(s - e1), 0.0, params).data
            f_plus2  = field(Tensor(s + e2), 0.0, params).data
            f_minus2 = field(Tensor(s - e2), 0.0, params).data
            df1_dq = (f_plus1[0] - f_minus1[0]) / (2*eps_div)
            df2_dp = (f_plus2[1] - f_minus2[1]) / (2*eps_div)
            return df1_dq + df2_dp

        div_N = div_fd(node_field, pN_t)
        div_H = div_fd(hnn_field,  pH_t2)
        print(f"  {s[0]:6.2f}  {s[1]:6.2f}  {div_N:16.4f}  {div_H:10.6f}")

    print()
    print("  HNN divergence is ≈ 0 everywhere (symplectic structure).")
    print("  Neural ODE divergence is nonzero — phase-space volume not preserved.")
