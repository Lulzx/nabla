"""
Differentiable ODE solver via the adjoint method.

    dy/dt = f(y, t, params),   y(t0) = y0

Forward pass  : fixed-step RK4, stores trajectory in O(N·d) memory.
Backward pass : backpropagates through ONE RK4 step at a time.
                Builds and discards a small Nabla graph per step →
                graph memory is O(d) instead of O(N·d) for unrolled autodiff.

Usage
-----
    from odeint import odeint

    def f(y, t, params):                 # must use Tensor ops
        a, b = params[0], params[1]
        return stack([a * y[1], -b * y[0]], axis=0)

    y_traj = odeint(f, y0, t, params)   # Tensor (N, d)
    loss   = (y_traj[-1] - target).pow(2).sum()
    loss.backward()
    # params.grad now holds ∂loss/∂params via the adjoint
"""

import numpy as np
from autograd import Tensor, stack


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _np(x):
    """Extract numpy array from Tensor or ndarray."""
    return x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float64)


def _rk4_np(f, y, t, dt, p):
    """Single RK4 step returning a numpy array (used in forward pass)."""
    def call(y_, t_):
        return f(Tensor(y_), t_, Tensor(p)).data
    k1 = call(y,           t)
    k2 = call(y + .5*dt*k1, t + .5*dt)
    k3 = call(y + .5*dt*k2, t + .5*dt)
    k4 = call(y +    dt*k3, t +    dt)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def _rk4_vjp(f, y_np, t_i, dt, p_np, v):
    """
    Vector-Jacobian product of one RK4 step.

    Builds a small Nabla graph for the step, seeds backward with v,
    then returns (∂L/∂y_i, ∂L/∂params).  The graph is immediately
    eligible for GC — this is the memory win vs unrolled autodiff.
    """
    y = Tensor(y_np)
    p = Tensor(p_np)

    k1 = f(y,                       t_i,          p)
    k2 = f(y + k1 * (dt * 0.5),     t_i + dt*0.5, p)
    k3 = f(y + k2 * (dt * 0.5),     t_i + dt*0.5, p)
    k4 = f(y + k3 * dt,             t_i + dt,     p)

    y_next = y + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0)

    (Tensor(v) * y_next).sum().backward()   # seed = v
    return y.grad.copy(), p.grad.copy()


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def odeint(f, y0, t, params):
    """
    Integrate dy/dt = f(y, t, params) with fixed-step RK4.

    Parameters
    ----------
    f      : callable(Tensor, float, Tensor) -> Tensor
             ODE right-hand side.  Must be written in Tensor ops.
    y0     : array_like or Tensor, shape (d,)
    t      : array_like, shape (N,)   monotonically increasing
    params : array_like or Tensor, shape (p,)

    Returns
    -------
    Tensor of shape (N, d).  Call .backward() on any scalar derived
    from it; gradients flow to y0 and params via the adjoint method.
    """
    # Accept both plain arrays and Tensors (so grad()/value_and_grad() work)
    y0_leaf = y0     if isinstance(y0,     Tensor) else Tensor(_np(y0))
    p_leaf  = params if isinstance(params, Tensor) else Tensor(_np(params))
    y0_np   = y0_leaf.data
    p_np    = p_leaf.data
    t_np    = np.asarray(t, dtype=np.float64)

    # ---- Forward: RK4, store full trajectory -------------------------
    ys = [y0_np]
    for i in range(len(t_np) - 1):
        dt = t_np[i+1] - t_np[i]
        ys.append(_rk4_np(f, ys[-1], t_np[i], dt, p_np))
    y_traj = np.stack(ys)                          # (N, d)

    out = Tensor(y_traj, _children=(y0_leaf, p_leaf), _op="odeint")

    # ---- Backward: adjoint, one step at a time -----------------------
    def _backward():
        # out.grad[i] = ∂L/∂y(t_i) flowing directly from the loss
        a     = out.grad[-1].copy()     # adjoint state  = dL/dy(T)
        dL_dp = np.zeros_like(p_np)

        for i in reversed(range(len(t_np) - 1)):
            dt = t_np[i+1] - t_np[i]

            # Backprop through this single RK4 step (tiny graph, O(d) memory)
            da, dp = _rk4_vjp(f, y_traj[i], t_np[i], dt, p_np, a)
            a      = da
            dL_dp += dp

            # Direct loss gradient at intermediate time steps
            a += out.grad[i]

        y0_leaf.grad += a
        p_leaf.grad  += dL_dp

    out._backward = _backward
    return out
