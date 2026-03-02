"""
Demonstrations of the autograd engine.
Covers scalar, vector, matrix ops, activations, and a small neural net.
"""

import numpy as np
from autograd import Tensor, gradient_check, grad, value_and_grad

SEP = "-" * 56


# ------------------------------------------------------------------ #
# 1. Scalar arithmetic                                                 #
# ------------------------------------------------------------------ #

def demo_scalar():
    print(SEP)
    print("1. Scalar arithmetic: f(x,y) = (x + y) * (x - y) = x² - y²")

    x = Tensor(3.0)
    y = Tensor(2.0)
    z = (x + y) * (x - y)
    z.backward()

    print(f"   z = {z.data}          (expected {3**2 - 2**2})")
    print(f"   dz/dx = {x.grad}      (expected {2*3.0})")
    print(f"   dz/dy = {y.grad}     (expected {-2*2.0})")


# ------------------------------------------------------------------ #
# 2. Broadcasting                                                      #
# ------------------------------------------------------------------ #

def demo_broadcast():
    print(SEP)
    print("2. Broadcasting: (3,) + (2,3) matrix")

    b = Tensor(np.array([1.0, 2.0, 3.0]))
    W = Tensor(np.ones((2, 3)))
    out = (W + b).sum()
    out.backward()

    print(f"   out = {out.data}  (expected {(np.ones((2,3)) + [1,2,3]).sum():.1f})")
    print(f"   db  = {b.grad}   (expected [2. 2. 2.])")
    print(f"   dW  =\n{W.grad}")


# ------------------------------------------------------------------ #
# 3. Matrix multiplication                                             #
# ------------------------------------------------------------------ #

def demo_matmul():
    print(SEP)
    print("3. Matrix multiplication gradient check")

    A = Tensor(np.random.randn(3, 4))
    B = Tensor(np.random.randn(4, 2))

    def f(a, b):
        return (a @ b).sum()

    out = f(A, B)
    out.backward()

    # Analytic: dL/dA = ones @ B^T, dL/dB = A^T @ ones
    ones_CB = np.ones((3, 2))
    print(f"   dA close? {np.allclose(A.grad, ones_CB @ B.data.T)}")
    print(f"   dB close? {np.allclose(B.grad, A.data.T @ ones_CB)}")


# ------------------------------------------------------------------ #
# 4. Activation functions                                              #
# ------------------------------------------------------------------ #

def demo_activations():
    print(SEP)
    print("4. Activation gradients vs finite differences")

    # Avoid x=0: ReLU is non-differentiable there (finite-diff gives 0.5)
    x = Tensor(np.array([-2.0, -1.0, 0.5, 1.0, 2.0]))

    for name, fn in [("relu", lambda t: t.relu()),
                     ("sigmoid", lambda t: t.sigmoid()),
                     ("tanh", lambda t: t.tanh())]:
        # Reset
        x.grad = np.zeros_like(x.data)
        out = fn(x)
        out.backward()
        analytic = x.grad.copy()
        x.grad = np.zeros_like(x.data)

        # Numerical
        eps = 1e-5
        num = np.array([
            (fn(Tensor(x.data + eps * np.eye(len(x.data))[i])).data.sum()
             - fn(Tensor(x.data - eps * np.eye(len(x.data))[i])).data.sum()) / (2 * eps)
            for i in range(len(x.data))
        ])
        print(f"   {name:8s}  max_err={np.abs(analytic - num).max():.2e}  "
              f"{'OK' if np.allclose(analytic, num, atol=1e-4) else 'FAIL'}")


# ------------------------------------------------------------------ #
# 5. Softmax + cross-entropy                                           #
# ------------------------------------------------------------------ #

def demo_softmax_crossentropy():
    print(SEP)
    print("5. Softmax cross-entropy loss")

    logits = Tensor(np.array([[2.0, 1.0, 0.1],
                               [0.5, 2.5, 0.3]]))
    targets = np.array([0, 1])   # class indices

    probs = logits.softmax(axis=1)
    N = logits.data.shape[0]
    log_probs = probs.log()

    # gather the log-probability of the correct class for each sample
    correct_log_probs = Tensor(log_probs.data[np.arange(N), targets])
    correct_log_probs._prev = {log_probs}

    def _bwd():
        g = np.zeros_like(log_probs.data)
        g[np.arange(N), targets] = correct_log_probs.grad
        log_probs.grad += g

    correct_log_probs._backward = _bwd

    loss = -correct_log_probs.mean()
    loss.backward()

    print(f"   loss = {loss.data:.4f}")
    print(f"   logits.grad =\n{logits.grad}")


# ------------------------------------------------------------------ #
# 6. Two-layer MLP on XOR                                             #
# ------------------------------------------------------------------ #

def demo_mlp_xor():
    print(SEP)
    print("6. Two-layer MLP on XOR (100 steps SGD)")

    np.random.seed(42)

    X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float))
    Y = np.array([[0], [1], [1], [0]], dtype=float)

    W1 = Tensor(np.random.randn(2, 8) * 0.5)
    b1 = Tensor(np.zeros((1, 8)))
    W2 = Tensor(np.random.randn(8, 1) * 0.5)
    b2 = Tensor(np.zeros((1, 1)))

    lr = 0.1

    for step in range(2000):
        # Forward
        h = (X @ W1 + b1).tanh()
        logits = h @ W2 + b2
        pred = logits.sigmoid()

        # MSE loss
        err = pred - Tensor(Y)
        loss = (err * err).mean()

        # Backward
        for t in [W1, b1, W2, b2]:
            t.grad = np.zeros_like(t.data)
        loss.backward()

        # SGD update
        for t in [W1, b1, W2, b2]:
            t.data -= lr * t.grad

        if step % 500 == 0:
            print(f"   step {step:4d}  loss={loss.data:.4f}")

    print(f"   final predictions: {pred.data.flatten().round(3)}")
    print(f"   expected:          [0, 1, 1, 0]")


# ------------------------------------------------------------------ #
# 7. Gradient check on a composite function                           #
# ------------------------------------------------------------------ #

def demo_gradient_check():
    print(SEP)
    print("7. Full gradient check on composite f(A,B) = tanh(A@B).sum()")

    A = Tensor(np.random.randn(3, 4) * 0.5)
    B = Tensor(np.random.randn(4, 2) * 0.5)

    def f(a, b):
        return (a @ b).tanh()

    print("   ", end="")
    gradient_check(f, A, B)


# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
# 8. Functional API: grad(fn)(params)                              #
# ------------------------------------------------------------------ #

def demo_functional_api():
    print(SEP)
    print("8. Functional API  grad(fn)(params)")

    # --- 8a. Simple scalar function ---
    def f(x):
        # x**3 - 2*x + 1  →  df/dx = 3x² - 2
        return (x**3 - 2*x + Tensor(1.0)).sum()

    x_val = np.array([1.0, 2.0, 3.0])
    df = grad(f)(x_val)
    expected = 3 * x_val**2 - 2
    print(f"   d/dx(x³-2x+1) at x={x_val}")
    print(f"   analytic : {df}")
    print(f"   expected : {expected}")
    print(f"   match    : {np.allclose(df, expected)}")

    # --- 8b. Multi-arg: value_and_grad on a linear regression loss ---
    print()
    np.random.seed(0)
    X_np = np.random.randn(10, 3)
    W_np = np.random.randn(3, 1)
    b_np = np.zeros((1, 1))
    y_np = X_np @ W_np + b_np + 0.1 * np.random.randn(10, 1)

    def mse_loss(W, b):
        X = Tensor(X_np)
        y = Tensor(y_np)
        pred = X @ W + b
        err = pred - y
        return (err * err).mean()

    loss_val, (dW, db) = value_and_grad(mse_loss, argnums=(0, 1))(W_np, b_np)
    print(f"   Linear regression MSE loss : {loss_val:.6f}")
    print(f"   dW shape={dW.shape},  db shape={db.shape}")

    # Verify dW matches closed-form: 2/N * X^T (X@W+b - y)
    N = X_np.shape[0]
    expected_dW = 2 / N * X_np.T @ (X_np @ W_np + b_np - y_np)
    print(f"   dW matches closed-form: {np.allclose(dW, expected_dW)}")

    # --- 8c. Physics example: ∂C_T/∂I_coil (coil energy) ---
    print()
    print("   Physics: E = ½ L I²  →  dE/dI = L·I")
    L = 1e-3     # 1 mH inductance
    I_val = np.array([5.0])  # 5 A

    def coil_energy(I_coil):
        return (Tensor(0.5 * L) * I_coil**2).sum()

    dE_dI = grad(coil_energy)(I_val)
    print(f"   dE/dI at I={I_val[0]} A  →  {dE_dI[0]:.6f} J/A  "
          f"(expected {L * I_val[0]:.6f})")


if __name__ == "__main__":
    demo_scalar()
    demo_broadcast()
    demo_matmul()
    demo_activations()
    demo_softmax_crossentropy()
    demo_mlp_xor()
    demo_gradient_check()
    demo_functional_api()
