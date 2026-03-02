"""
Differentiable gradient engine built on NumPy.

Implements reverse-mode automatic differentiation (backpropagation) via a
dynamic computation graph. Each Tensor records the operation that produced it
and how to propagate gradients back through that operation.
"""

import numpy as np


class Tensor:
    """A multi-dimensional array that tracks gradients through operations."""

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ------------------------------------------------------------------ #
    # Arithmetic                                                           #
    # ------------------------------------------------------------------ #

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other), _op="@")

        def _backward():
            # dL/dA = dL/dC @ B^T,  dL/dB = A^T @ dL/dC
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only scalar exponents"
        out = Tensor(self.data**exponent, _children=(self,), _op=f"**{exponent}")

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else Tensor(-other))

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other**-1

    # Reflected operators so that `scalar op Tensor` works
    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = lambda self, other: Tensor(other) - self
    __rtruediv__ = lambda self, other: Tensor(other) / self

    # ------------------------------------------------------------------ #
    # Reductions                                                           #
    # ------------------------------------------------------------------ #

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     _children=(self,), _op="sum")

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ------------------------------------------------------------------ #
    # Shape                                                                #
    # ------------------------------------------------------------------ #

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), _children=(self,), _op="reshape")

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, axes=None):
        out = Tensor(self.data.transpose(axes), _children=(self,), _op="T")

        def _backward():
            inv = None if axes is None else np.argsort(axes)
            self.grad += out.grad.transpose(inv)

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    # ------------------------------------------------------------------ #
    # Indexing                                                             #
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], _children=(self,), _op="[]")

        def _backward():
            # np.add.at handles repeated indices and advanced indexing correctly
            np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Activation functions                                                 #
    # ------------------------------------------------------------------ #

    def relu(self):
        out = Tensor(np.maximum(0, self.data), _children=(self,), _op="relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, _children=(self,), _op="sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, _children=(self,), _op="tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), _children=(self,), _op="log")

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def exp(self):
        ex = np.exp(self.data)
        out = Tensor(ex, _children=(self,), _op="exp")

        def _backward():
            self.grad += ex * out.grad

        out._backward = _backward
        return out

    def sqrt(self):
        return self ** 0.5

    def softmax(self, axis=-1):
        e = np.exp(self.data - self.data.max(axis=axis, keepdims=True))
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, _children=(self,), _op="softmax")

        def _backward():
            # Jacobian-vector product for softmax
            dot = (out.grad * s).sum(axis=axis, keepdims=True)
            self.grad += s * (out.grad - dot)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Backpropagation                                                      #
    # ------------------------------------------------------------------ #

    def backward(self):
        """Compute gradients for all leaf tensors via reverse-mode autodiff."""
        topo = []
        visited = set()

        def build(node):
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return (f"Tensor({self.data}, grad_fn=<{self._op}>)"
                if self._op else f"Tensor({self.data})")


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _unbroadcast(grad, shape):
    """Sum gradient axes that were broadcast during the forward pass."""
    if grad.shape == shape:
        return grad
    # Sum over leading dimensions added by broadcasting
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Sum over dimensions that were size-1 in the original
    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


# ------------------------------------------------------------------ #
# Array ops                                                            #
# ------------------------------------------------------------------ #

def stack(tensors, axis=0):
    """Stack a sequence of Tensors along a new axis (differentiable np.stack)."""
    out = Tensor(np.stack([t.data for t in tensors], axis=axis),
                 _children=tuple(tensors), _op="stack")

    def _backward():
        ax = axis % out.grad.ndim
        for i, t in enumerate(tensors):
            idx = [slice(None)] * out.grad.ndim
            idx[ax] = i
            t.grad += out.grad[tuple(idx)]

    out._backward = _backward
    return out


def cross(a, b):
    """Differentiable 3D cross product for (..., 3) Tensors."""
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return stack([
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    ], axis=-1)


# ------------------------------------------------------------------ #
# Functional API  –  grad(fn)(params)                                 #
# ------------------------------------------------------------------ #

def grad(fn, argnums=0):
    """
    Return a function that computes the gradient of *fn* w.r.t. the
    argument(s) given by *argnums*.

    Usage::

        def loss(W, b, X, y):
            return ((X @ W + b) - y).pow(2).mean()

        dW, db = mx.grad(loss, argnums=(0, 1))(W, b, X, y)

    Parameters
    ----------
    fn      : callable that accepts Tensor arguments and returns a scalar Tensor
    argnums : int or tuple[int]   which positional arguments to differentiate

    Returns
    -------
    grad_fn : callable with the same signature as *fn*;
              returns gradient(s) as numpy ndarray(s), same shapes as inputs.
    """
    single = isinstance(argnums, int)
    indices = (argnums,) if single else tuple(argnums)

    def grad_fn(*args, **kwargs):
        # Wrap the selected arguments in fresh Tensors that track grad
        tensors = list(args)
        tracked = []
        for i in indices:
            t = Tensor(np.asarray(args[i], dtype=np.float64))
            tensors[i] = t
            tracked.append(t)

        # Forward + backward
        out = fn(*tensors, **kwargs)
        if out.data.ndim != 0 and out.data.size != 1:
            raise ValueError(
                "grad() requires a scalar output; got shape "
                f"{out.data.shape}. Use .sum() or .mean() first."
            )
        out.backward()

        grads = [t.grad for t in tracked]
        return grads[0] if single else tuple(grads)

    return grad_fn


def value_and_grad(fn, argnums=0):
    """Like grad() but also returns the forward value alongside the gradient(s)."""
    single = isinstance(argnums, int)
    indices = (argnums,) if single else tuple(argnums)

    def vg_fn(*args, **kwargs):
        tensors = list(args)
        tracked = []
        for i in indices:
            t = Tensor(np.asarray(args[i], dtype=np.float64))
            tensors[i] = t
            tracked.append(t)

        out = fn(*tensors, **kwargs)
        out.backward()

        grads = [t.grad for t in tracked]
        g = grads[0] if single else tuple(grads)
        return out.data.copy(), g

    return vg_fn


# ------------------------------------------------------------------ #
# Numerical gradient check                                             #
# ------------------------------------------------------------------ #

def gradient_check(f, *tensors, eps=1e-5, atol=1e-4):
    """Check analytic vs numerical gradients; returns True if all tensors agree."""
    for t in tensors:
        t.grad = np.zeros_like(t.data)
    out = f(*tensors)
    out.backward()

    ok = True
    for t in tensors:
        analytic = t.grad.copy()
        t.grad = np.zeros_like(t.data)

        # Finite-difference estimate: perturb each element of t in-place
        num_grad = np.zeros_like(t.data)
        it = np.nditer(t.data, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = t.data[idx]

            t.data[idx] = orig + eps
            fp = f(*tensors).data.sum()

            t.data[idx] = orig - eps
            fm = f(*tensors).data.sum()

            t.data[idx] = orig
            num_grad[idx] = (fp - fm) / (2 * eps)
            it.iternext()

        rel_err = np.abs(num_grad - analytic) / (np.abs(num_grad) + np.abs(analytic) + 1e-8)
        if rel_err.max() > atol:
            print(f"FAIL  max relative error = {rel_err.max():.2e}")
            ok = False
        else:
            print(f"OK    max relative error = {rel_err.max():.2e}")
    return ok
