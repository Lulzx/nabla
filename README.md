# nabla

Reverse-mode automatic differentiation built on NumPy. Computes exact gradients of arbitrary Python/NumPy functions by tracing a dynamic computation graph and backpropagating through it.

```python
from autograd import grad, value_and_grad, Tensor

# Differentiate any function
df = grad(lambda x: (x**3 - 2*x).sum())(np.array([1.0, 2.0, 3.0]))
# → [ 1. 10. 25.]  (3x² - 2)

# Multiple arguments
loss, (dW, db) = value_and_grad(mse_loss, argnums=(0, 1))(W, b)

# Physics: ∂E/∂I for coil energy E = ½LI²
dE_dI = grad(lambda I: (Tensor(0.5 * L) * I**2).sum())(I)
```

## How it works

Each `Tensor` records the operation that created it and a `_backward` closure. Calling `.backward()` performs a topological sort of the graph and propagates `∂L/∂node` in reverse — the chain rule applied automatically.

```
forward:   x → [op] → y → [op] → z = loss
backward:  ∂z/∂x ← [∂op] ← ∂z/∂y ← [∂op] ← ∂z/∂z = 1
```

## Operations

| Category | Ops |
|---|---|
| Arithmetic | `+` `-` `*` `/` `**` `@` (matmul) |
| Reductions | `sum`, `mean` (with `axis`, `keepdims`) |
| Activations | `relu`, `sigmoid`, `tanh`, `softmax` |
| Elementwise | `log`, `exp` |
| Shape | `reshape`, `transpose` / `.T` |
| Broadcasting | handled automatically |

## Functional API

```python
from autograd import grad, value_and_grad

# grad(fn, argnums=0) → returns gradient as ndarray
dL_dW = grad(loss_fn, argnums=0)(W, X, y)

# value_and_grad → returns (value, gradient)
loss, grads = value_and_grad(loss_fn, argnums=(0, 1))(W, b, X, y)
```

`argnums` selects which positional arguments to differentiate. Inputs are plain NumPy arrays; gradients are returned as NumPy arrays.

## Examples

```
python examples.py
```

Covers: scalar arithmetic, broadcasting, matmul, activation functions, softmax cross-entropy, a two-layer MLP solving XOR, a full numerical gradient check, and the functional API with a physics example (∂E/∂I_coil).

## Gradient check

Verify any function analytically vs finite differences:

```python
from autograd import gradient_check, Tensor

A = Tensor(np.random.randn(3, 4))
B = Tensor(np.random.randn(4, 2))
gradient_check(lambda a, b: (a @ b).tanh(), A, B)
# OK  max relative error = 1.63e-11
# OK  max relative error = 2.03e-11
```

## Requirements

- Python 3.8+
- NumPy
