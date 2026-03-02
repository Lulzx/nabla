"""
Adjoint ODE demo: fit Lotka-Volterra parameters from noisy observations.

    dy1/dt =  α·y1 - β·y1·y2     (prey)
    dy2/dt =  δ·y1·y2 - γ·y2     (predator)

We generate noisy data from true parameters, start from a bad guess,
and recover the true parameters with gradient descent using odeint.
"""

import numpy as np
from autograd import Tensor, stack, gradient_check
from autograd import grad, value_and_grad
from odeint import odeint

SEP = "-" * 56


# ------------------------------------------------------------------ #
# ODE definition (must use Tensor ops)                                #
# ------------------------------------------------------------------ #

def lotka_volterra(y, t, params):
    α, β, γ, δ = params[0], params[1], params[2], params[3]
    prey, pred = y[0], y[1]
    return stack([
        α * prey - β * prey * pred,
        δ * prey * pred - γ * pred,
    ], axis=0)


# ------------------------------------------------------------------ #
# 1. Gradient check on odeint                                         #
# ------------------------------------------------------------------ #

def demo_gradient_check():
    print(SEP)
    print("1. Gradient check: ∂sum(y_traj)/∂params  via adjoint vs FD")

    np.random.seed(0)
    y0     = np.array([4.0, 2.0])
    t      = np.linspace(0, 1.0, 8)          # short, well-conditioned
    params = Tensor(np.array([1.2, 0.8, 2.5, 0.9]))

    def f(p):
        return odeint(lotka_volterra, y0, t, p)

    ok = gradient_check(f, params, atol=1e-3)
    print(f"   adjoint gradient check: {'PASS' if ok else 'FAIL'}")


# ------------------------------------------------------------------ #
# 2. Memory comparison (graph nodes)                                  #
# ------------------------------------------------------------------ #

def demo_memory_comparison():
    print(SEP)
    print("2. Memory: graph nodes — adjoint vs naive unrolled backprop")

    import sys

    y0     = np.array([4.0, 2.0])
    params = np.array([1.5, 1.0, 3.0, 1.0])

    for N in [10, 50, 200]:
        t = np.linspace(0, 2.0, N)

        # Adjoint: odeint builds ONE graph per backward step (O(d))
        # We measure the graph size of a single RK4 step VJP
        from odeint import _rk4_vjp
        y_dummy = np.array([4.0, 2.0])
        dt      = t[1] - t[0]

        # Count nodes in one RK4 step graph
        y_t = Tensor(y_dummy)
        p_t = Tensor(params)
        k1 = lotka_volterra(y_t, t[0], p_t)
        k2 = lotka_volterra(y_t + k1*(dt*0.5), t[0]+dt*0.5, p_t)
        k3 = lotka_volterra(y_t + k2*(dt*0.5), t[0]+dt*0.5, p_t)
        k4 = lotka_volterra(y_t + k3*dt,        t[0]+dt,     p_t)
        y_next = y_t + (k1 + k2*2.0 + k3*2.0 + k4) * (dt/6.0)

        # Count reachable nodes
        visited = set()
        def count(node):
            if id(node) not in visited:
                visited.add(id(node))
                for c in node._prev:
                    count(c)
        count(y_next)
        step_nodes = len(visited)

        print(f"   N={N:4d} steps │ adjoint graph/step: {step_nodes:4d} nodes  │ "
              f"naive would need: ~{step_nodes * N:6d} nodes")


# ------------------------------------------------------------------ #
# 3. Parameter estimation: fit α, β, γ, δ from noisy data            #
# ------------------------------------------------------------------ #

def demo_parameter_estimation():
    print(SEP)
    print("3. Lotka-Volterra parameter estimation")

    np.random.seed(42)

    TRUE_PARAMS = np.array([1.5, 1.0, 3.0, 1.0])
    y0 = np.array([10.0, 5.0])
    t  = np.linspace(0, 2.5, 40)

    # Ground truth + noise
    y_true = odeint(lotka_volterra, y0, t, TRUE_PARAMS).data
    y_obs  = y_true + 0.3 * np.random.randn(*y_true.shape)

    def loss(params):
        y_pred = odeint(lotka_volterra, y0, t, params)
        diff   = y_pred - Tensor(y_obs)
        return (diff * diff).mean()

    # Start from a bad initial guess
    params = np.array([1.0, 0.5, 2.0, 0.5])

    print(f"   true  : α={TRUE_PARAMS[0]} β={TRUE_PARAMS[1]} "
          f"γ={TRUE_PARAMS[2]} δ={TRUE_PARAMS[3]}")
    print(f"   init  : α={params[0]} β={params[1]} "
          f"γ={params[2]} δ={params[3]}")
    print()

    # Adam — handles the very different gradient scales across parameters
    lr, β1, β2, eps = 1e-2, 0.9, 0.999, 1e-8
    m = np.zeros_like(params)
    v = np.zeros_like(params)

    for step in range(1, 801):
        l, dp = value_and_grad(loss)(params)
        m = β1 * m + (1 - β1) * dp
        v = β2 * v + (1 - β2) * dp**2
        m_hat = m / (1 - β1**step)
        v_hat = v / (1 - β2**step)
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
        params = np.clip(params, 0.1, 10.0)

        if step % 200 == 0:
            print(f"   step {step:4d}  loss={l:.5f}  "
                  f"params=[{', '.join(f'{v_:.3f}' for v_ in params)}]")

    print()
    err = np.abs(params - TRUE_PARAMS) / TRUE_PARAMS * 100
    print(f"   fitted: α={params[0]:.3f} β={params[1]:.3f} "
          f"γ={params[2]:.3f} δ={params[3]:.3f}")
    print(f"   error : α={err[0]:.1f}% β={err[1]:.1f}% "
          f"γ={err[2]:.1f}% δ={err[3]:.1f}%")


if __name__ == "__main__":
    demo_gradient_check()
    demo_memory_comparison()
    demo_parameter_estimation()
