"""
Differentiating through contact and collision.

The problem: a simulation with contact has discontinuities where the
contact force switches on/off (y = 0 crossing). Naive autodiff gives
zero gradient through the impact — the event is invisible to the graph.

Three approaches, with honest notes on what each buys and costs:

1. Penalty method  — F_c = k * relu(-y)^2
   Smooth everywhere, but introduces stiffness (small k: inaccurate,
   large k: numerically stiff, requires tiny dt).

2. Log-barrier (IPC-style)  — F_c = -k * log(y/y0)  for y < y0
   Never allows penetration, but y=0 is a singularity.
   Equilibrium is at y = k/(mg), not y=0 exactly.

3. Event-based (non-smooth, exact)  — detect y=0 crossing, flip velocity
   Exact physics, but ∂L/∂restitution is NOT differentiable at the event.
   Smooth approximation: replace instantaneous flip with a short, stiff
   collision phase (viscoelastic contact).

We use approach 1 for gradient-based optimisation via odeint, and
approach 3 to show explicitly where the gradient fails.
"""

import numpy as np
from autograd import Tensor, stack, grad, value_and_grad
from odeint import odeint

G = 9.81   # m/s²


# ------------------------------------------------------------------ #
# Contact force models                                                 #
# ------------------------------------------------------------------ #

def penalty_contact(y, k_contact):
    """
    Penalty force: F = k * max(0, -y)^2.
    Always differentiable. Allows small penetration ∝ 1/k.
    """
    # relu(-y) = max(0, -y)
    penetration = (-y).relu()
    return k_contact * penetration ** 2


def logbarrier_contact(y, k_contact, y0=0.5):
    """
    Log-barrier: F = -k * log(y / y0)  for y < y0, else 0.
    Differentiable, never allows y ≤ 0. Adds a repulsive force near y=0.
    """
    # Clamp y to (ε, y0) for numerical safety, then zero above y0
    eps = Tensor(np.array(1e-6))
    y_safe = (y * Tensor(1.0)).relu() + eps   # always > 0
    # log-barrier active only for y < y0
    in_barrier = Tensor((y.data < y0).astype(float))
    return -k_contact * y_safe.log() * in_barrier


# ------------------------------------------------------------------ #
# Dynamics                                                             #
# ------------------------------------------------------------------ #

def bouncing_ball(state, t, params):
    """
    ODE for a ball under gravity with penalty contact.

    state   = [y, v]   (position, velocity)
    params  = [mass, k_contact]
    """
    y, v    = state[0], state[1]
    mass    = params[0]
    k_c     = params[1]

    F_gravity = -mass * Tensor(G)
    F_contact = penalty_contact(y, k_c)

    a = (F_gravity + F_contact) / mass
    return stack([v, a], axis=0)


# ------------------------------------------------------------------ #
# Demo                                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    SEP = "-" * 56

    # ---- 1. Forward simulation ----
    print(SEP)
    print("1. Bouncing ball: penalty contact (k=500)")

    t = np.linspace(0, 1.5, 300)
    y0 = np.array([2.0, 0.0])          # drop from y=2, at rest

    params_np = np.array([1.0, 500.0])  # mass=1, k_contact=500
    y_traj = odeint(bouncing_ball, y0, t, params_np).data

    y_pos = y_traj[:, 0]
    # dynamic max penetration from energy conservation: k*x^3/3 ≈ 0.5*m*v_impact^2
    # with v_impact = sqrt(2*g*h) = sqrt(2*9.81*2) ≈ 6.26 m/s → x ≈ 0.53 m
    print(f"  min position (penetration): {y_pos.min():.4f}  "
          f"(dynamic impact from 2m; static equilibrium would be ≈ -√(mg/k) = {-(G/500)**0.5:.4f})")
    print(f"  max bounce height after 1st contact: {y_pos[y_pos.argmin():].max():.4f}")

    # ---- 2. Gradient of bounce height w.r.t. contact stiffness ----
    print(SEP)
    print("2. ∂(bounce height)/∂k_contact  via adjoint")

    def bounce_height(params):
        y_traj = odeint(bouncing_ball, y0, t, params)
        return y_traj[:, 0].max()          # highest point in trajectory

    # Need max to be differentiable: use soft-max (log-sum-exp)
    def soft_bounce_height(params, tau=10.0):
        y_traj = odeint(bouncing_ball, y0, t, params)
        y_pos  = y_traj[:, 0]
        # soft-max: log(Σ exp(y * tau)) / tau  ≈  max(y)
        return (y_pos * tau).exp().sum().log() * Tensor(1.0 / tau)

    p = Tensor(params_np.copy())
    h = soft_bounce_height(p)
    h.backward()
    print(f"  ∂height/∂mass      = {p.grad[0]:.4f}")
    print(f"  ∂height/∂k_contact = {p.grad[1]:.6f}")

    # Verify with finite differences
    eps = 1.0
    h_plus  = soft_bounce_height(Tensor(params_np + np.array([0, eps]))).data
    h_minus = soft_bounce_height(Tensor(params_np - np.array([0, eps]))).data
    fd_dk = (h_plus - h_minus) / (2 * eps)
    print(f"  FD check ∂height/∂k_contact ≈ {fd_dk:.6f}  "
          f"match={'✓' if abs(p.grad[1] - fd_dk) / (abs(fd_dk)+1e-8) < 0.05 else '✗'}")

    # ---- 3. Optimise k_contact for max bounce height ----
    print(SEP)
    print("3. Optimise k_contact to maximise bounce height")
    print("   (higher k = stiffer contact = less energy absorbed in penetration)")

    k_opt = np.array([1.0, 100.0])   # start with soft contact
    lr    = 50.0

    for step in range(80):
        h_val, dk = value_and_grad(soft_bounce_height, argnums=0)(k_opt)
        k_opt[1] = np.clip(k_opt[1] + lr * dk[1], 10.0, 5000.0)
        if step % 20 == 0:
            print(f"  step {step:3d}  k={k_opt[1]:.1f}  soft_bounce_h={h_val:.4f}")

    # ---- 4. Honest note on discontinuities ----
    print(SEP)
    print("4. Where this breaks: the discontinuity problem")
    print()
    print("  Penalty method: gradient of bounce height w.r.t. k is smooth")
    print("  and correct as long as the trajectory has no zero-crossings")
    print("  within a single RK4 step.")
    print()
    print("  When k is very large, contact is nearly instantaneous —")
    print("  the contact phase compresses to < dt, the step crosses y=0,")
    print("  and the solver misses the contact event entirely.")
    print()
    print("  True fix: event detection (find t* where y(t*)=0, split step)")
    print("  + impulse model. This gives exact physics but ∂impulse/∂params")
    print("  is a Dirac delta — not useful for gradient descent.")
    print()
    print("  IPC (Incremental Potential Contact) sidesteps this by choosing")
    print("  k(h) that scales with step size, keeping contact energy finite")
    print("  regardless of h. Implementing that here would require coupling")
    print("  the contact model to the integrator time step — left as future work.")
