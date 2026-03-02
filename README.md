# nabla

Differentiable programming primitives built on NumPy. Reverse-mode AD via dynamic
computation graphs, forward-mode AD with perturbation tags, second-order methods,
ODE adjoints, and gradient estimators for stochastic and combinatorial nodes.
Zero dependencies beyond NumPy.

```python
from autograd import grad, value_and_grad, Tensor

df     = grad(lambda x: (x**3 - 2*x).sum())(np.array([1., 2., 3.]))
# → [1., 10., 25.]  (3x² − 2)

loss, (dW, db) = value_and_grad(mse, argnums=(0, 1))(W, b, X, y)
```

## Modules

| File | What it implements |
|------|--------------------|
| `autograd.py` | `Tensor` class. Dynamic graph, topological backprop, `grad()`, `value_and_grad()`, `gradient_check()`. |
| `examples.py` | Demos: scalar/broadcast/matmul/activations/softmax/MLP/gradient-check/Biot-Savart. |
| `forward_ad.py` | Forward-mode AD. Tagged dual numbers (`Dual`, `jvp`). Fixes perturbation confusion (Pearlmutter & Siskind 2008) via fresh per-call tags. Nested `jvp` for higher-order derivatives. |
| `smooth_types.py` | Differentiability type lattice: SMOOTH > C1 > PSMOOTH > LIPSCHITZ > CONTINUOUS. Composition degrades to min. `@custom_vjp` escape hatch. `TypedExpr` for static network type inference. |
| `hessian.py` | `hvp(f,x,v)`: Hessian-vector product via central-difference of gradient (2 reverse passes). `hessian(f,x)`: full matrix via n HVPs. `newton_cg`: Newton's method with CG linear solve, Armijo line search. |
| `conservative_grad.py` | Clarke subdifferential. AD output is always in ∂f(x) for definable programs (Bolte & Pauwels 2020). Subgradient descent on L1 regression. Chain-rule verification for conservative fields. |
| `implicit_diff.py` | `implicit_solve(F,θ,x0)`: solve F(x*,θ)=0 forward (Newton), backward via IFT. `fixed_point_solve(f,θ,x0)`: Banach iteration forward, IFT backward. Cost: 2 Jacobians + 1 linear solve per backward pass. |
| `stochastic_ad.py` | `reinforce`: score-function estimator ∂E[f(z)]/∂θ = E[f(z)·∂log p/∂θ]. `reparam`: reparameterization trick via reverse-mode AD through z=g(ε,θ). Variance comparison. Bernoulli REINFORCE. |
| `odeint.py` | Fixed-step RK4 integrator with adjoint backward pass. `odeint(f,y0,t,params)`. |
| `odeint_adaptive.py` | Adaptive RK45 (Dormand-Prince) with stored-step adjoint. Error-controlled step selection. |
| `diff_sort.py` | Differentiable sorting via annealing: smooth approximation to argsort. |
| `diff_combinatorial.py` | Blackbox differentiation for Dijkstra and Kruskal (Vlastelica et al. 2020). Gradient = (y*(w+λg)−y*(w))/λ. Learn-to-route demo. |
| `diff_contact.py` | Differentiable contact forces: penalty method (F=k·relu(−y)²) and log-barrier. Gradient of bounce height w.r.t. contact stiffness. |
| `symbolic_grad.py` | Expression tree with symbolic differentiation and simplification. |

## Core design

Each `Tensor` stores a `_backward` closure and a set of children. `.backward()` topologically sorts the graph and calls closures in reverse — the chain rule applied incrementally.

```
forward:   a → [*] → b → [exp] → c → [sum] → L
backward:  ∂L/∂a ← [∂*] ← ∂L/∂b ← [∂exp] ← ∂L/∂c ← 1
```

Broadcasting: `_unbroadcast` sums gradient axes that were broadcast in the forward pass.
Matmul: handles both matrix×matrix and matrix×vector (outer-product backward for 1D).

## Forward-mode AD

`jvp(f, x, v)` returns `(f(x), Df(x)·v)`. Each call allocates a fresh integer tag; the tangent of any `Dual` with a different tag is zero, preventing perturbation confusion when `f` internally differentiates. `Dual.val` may be a `Dual` (not stripped), enabling nested `jvp` for exact higher-order derivatives.

```python
from forward_ad import jvp
val, df   = jvp(lambda x: x.sin(), 1.0)
_,   d2f  = jvp(lambda x: jvp(lambda y: y.sin(), x)[1], 1.0)   # f''(x)
_,   d3f  = jvp(lambda x: jvp(lambda x: jvp(lambda y: y.sin(), x)[1], x)[1], 1.0)
```

## Second-order

`hessian.hvp(f, x, v)` uses central-difference of gradient: O(ε²) error, 2 reverse passes.
`hessian.hessian(f, x)` builds the full matrix: n HVPs, symmetrized.
`newton_cg` solves the Newton system via CG using only HVPs — never forms H.

Exact HVP without finite-difference: apply `jvp` from `forward_ad` to `grad(f)`.
Cost: one forward-over-reverse pass (the Pearlmutter 1994 R{} operator).

## Implicit differentiation

For programs with inner solvers (Newton, fixed-point iteration, argmin):

```python
from implicit_diff import implicit_solve, fixed_point_solve

# F(x*, θ) = 0 defines x*(θ).  Backward: IFT gives dL/dθ.
theta  = Tensor(np.array([9.0]))
x_star = implicit_solve(lambda x, t: x*x - t, theta, x0=np.array([1.0]))
# x_star ≈ 3.0;  x_star.sum().backward() puts 1/(2√θ) into theta.grad

# Fixed-point: x* = tanh(w·x* + θ), gradient through ∞ iterations
x_fp = fixed_point_solve(lambda x, t: (x * 0.5 + t).tanh(), theta, x0=np.zeros(1))
```

Backward cost: 2 finite-difference Jacobians + 1 linear solve. O(n³) for dense problems; use CG + Jx-vector products for large n.

## References

- Pearlmutter & Siskind (2008). Reverse-mode AD in a higher-order language. *PLDI*.
- Bolte & Pauwels (2020). A mathematical model for automatic differentiation in machine learning. *NeurIPS*.
- Vlastelica et al. (2020). Differentiation of blackbox combinatorial solvers. *ICLR*.
- Hairer, Nørsett & Wanner (1993). Solving ODEs I. (RK45 Butcher tableau)
- Williams (1992). Simple statistical gradient-following algorithms. *ML*.
- Kingma & Welling (2013). Auto-encoding variational Bayes. *ICLR*.

## Run

```sh
python3 examples.py           # core AD
python3 forward_ad.py         # perturbation confusion, nested JVP
python3 smooth_types.py       # differentiability types
python3 hessian.py            # Newton-CG, C² gap
python3 conservative_grad.py  # Clarke subdiff, L1 regression
python3 implicit_diff.py      # IFT: sqrt, DEQ, linear solve
python3 stochastic_ad.py      # REINFORCE vs reparameterization
python3 odeint.py             # ODE adjoint (pendulum)
python3 odeint_adaptive.py    # adaptive RK45
python3 diff_combinatorial.py # blackbox Dijkstra/MST
```

Python 3.8+, NumPy.
