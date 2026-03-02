"""
Forward-mode AD via tagged dual numbers — fixing perturbation confusion.

Perturbation confusion (Pearlmutter & Siskind, 2008)
----------------------------------------------------
Standard dual-number AD uses a single implicit perturbation everywhere.
When a function f(x) internally calls a differentiation primitive, and f's
argument x is itself an active dual (because we are differentiating f w.r.t.
x), the inner derivative picks up x's tangent and mixes it with the inner
perturbation — giving a wrong result.

The fix: each call to jvp() creates a *fresh integer tag*. A dual number
Dual(val, tang, tag) represents "value is `val`, derivative w.r.t.
perturbation #tag is `tang`." In arithmetic, the tangent of an operand with
a *different* tag is treated as zero. Foreign perturbations cannot bleed in.

Nested differentiation
----------------------
Allowing Dual.val to itself be a Dual (not stripped to float) enables
correct higher-order derivatives via nested jvp() calls — each level uses
a fresh tag so the two perturbation directions remain orthogonal.
Python's operator dispatch (via __rmul__ etc.) makes nested Dual arithmetic
work without any extra code.

This is the operational content of the "freshness" requirement that
differential linear logic (Ehrhard & Regnier 2003) enforces via typing.
"""

import math
import threading

# ── Perturbation tag counter ──────────────────────────────────────────────────

_lock    = threading.Lock()
_counter = 0

def _fresh():
    global _counter
    with _lock:
        _counter += 1
        return _counter


# ── NaiveDual: untagged, exhibits the bug ────────────────────────────────────

class NaiveDual:
    """Dual number without perturbation tags. Susceptible to perturbation confusion."""
    __slots__ = ('val', 'tang')

    def __init__(self, val, tang=0.0):
        self.val  = float(val)      # always a float — no nesting
        self.tang = float(tang)

    def _v(self, o): return o.val  if isinstance(o, NaiveDual) else float(o)
    def _t(self, o): return o.tang if isinstance(o, NaiveDual) else 0.0

    def __add__ (self, o): return NaiveDual(self.val + self._v(o), self.tang + self._t(o))
    def __radd__(self, o): return NaiveDual(float(o) + self.val,   self.tang)
    def __sub__ (self, o): return NaiveDual(self.val - self._v(o), self.tang - self._t(o))
    def __rsub__(self, o): return NaiveDual(float(o) - self.val,  -self.tang)
    def __neg__ (self):    return NaiveDual(-self.val, -self.tang)
    def __mul__ (self, o):
        v, t = self._v(o), self._t(o)
        return NaiveDual(self.val * v, self.tang * v + self.val * t)
    def __rmul__(self, o): return NaiveDual(float(o) * self.val, float(o) * self.tang)
    def __truediv__(self, o):
        v, t = self._v(o), self._t(o)
        return NaiveDual(self.val / v, (self.tang * v - self.val * t) / (v * v))
    def __pow__(self, n):
        n = float(n)
        return NaiveDual(self.val ** n, n * self.val ** (n - 1) * self.tang)
    def exp (self): e = math.exp(self.val);  return NaiveDual(e, e * self.tang)
    def log (self): return NaiveDual(math.log(self.val),  self.tang / self.val)
    def sin (self): return NaiveDual(math.sin(self.val),  math.cos(self.val)  * self.tang)
    def cos (self): return NaiveDual(math.cos(self.val), -math.sin(self.val)  * self.tang)
    def tanh(self): t = math.tanh(self.val); return NaiveDual(t, (1 - t*t) * self.tang)
    def relu(self): return NaiveDual(max(0., self.val), self.tang if self.val > 0 else 0.)
    def __float__(self): return self.val
    def __gt__(self, o): return self.val > (o.val if isinstance(o, NaiveDual) else o)
    def __lt__(self, o): return self.val < (o.val if isinstance(o, NaiveDual) else o)
    def __repr__(self): return f"NaiveDual({self.val:.4g}, tang={self.tang:.4g})"


def naive_deriv(f, x):
    """Derivative via untagged dual. BUGGY when f captures an outer dual."""
    xd = NaiveDual(x.val if isinstance(x, NaiveDual) else x, 1.0)
    r  = f(xd)
    return r.tang if isinstance(r, NaiveDual) else 0.0


# ── Dual: tagged, correct ────────────────────────────────────────────────────

class Dual:
    """
    Tagged dual number Dual(val, tang, tag).

    Key design decisions
    --------------------
    1. `val` may be a Dual (for nested differentiation).
       When jvp() is called with a Dual argument, the inner jvp wraps
       it as Dual(outer_dual, 1.0, inner_tag). The outer dual is preserved
       inside val, letting its tag-1 tangent propagate through arithmetic.
       Python's operator dispatch handles Dual-valued arithmetic automatically.

    2. In _t(o), the tangent of an operand with a different tag is 0.
       This prevents perturbations from leaking across nested jvp() calls
       (the perturbation confusion fix).

    3. NaiveDual arguments are always stripped to float (they carry no tag).
    """
    __slots__ = ('val', 'tang', 'tag')

    def __init__(self, val, tang=0.0, tag=0):
        # NaiveDual → extract float; Dual → preserve (enables nesting)
        if isinstance(val, NaiveDual):
            self.val = float(val.val)
        elif isinstance(val, Dual):
            self.val = val              # KEEP: allows Dual-valued arithmetic
        else:
            self.val = float(val)
        self.tang = tang                # may be float or Dual
        self.tag  = tag

    def _v(self, o): return o.val if isinstance(o, Dual) else float(o)
    def _t(self, o): return o.tang if (isinstance(o, Dual) and o.tag == self.tag) else 0.0

    def __add__ (self, o): return Dual(self.val + self._v(o), self.tang + self._t(o), self.tag)
    def __radd__(self, o): return Dual(o + self.val,           self.tang,             self.tag)
    def __sub__ (self, o): return Dual(self.val - self._v(o), self.tang - self._t(o), self.tag)
    def __rsub__(self, o): return Dual(o - self.val,          -self.tang,             self.tag)
    def __neg__ (self):    return Dual(-self.val,             -self.tang,             self.tag)
    def __mul__ (self, o):
        v, t = self._v(o), self._t(o)
        return Dual(self.val * v, self.tang * v + self.val * t, self.tag)
    def __rmul__(self, o): return Dual(o * self.val,  o * self.tang,  self.tag)
    def __truediv__(self, o):
        v, t = self._v(o), self._t(o)
        return Dual(self.val / v, (self.tang * v - self.val * t) / (v * v), self.tag)
    def __rtruediv__(self, o):
        return Dual(o / self.val, -o * self.tang / (self.val * self.val), self.tag)
    def __pow__(self, n):
        n = float(n)
        return Dual(self.val ** n, n * self.val ** (n - 1) * self.tang, self.tag)

    # Elementary functions — handle nested Dual val via method dispatch
    def _scalar_val(self):
        """Extract innermost float for scalar math functions."""
        v = self.val
        while isinstance(v, Dual): v = v.val
        return v

    def exp(self):
        if isinstance(self.val, Dual):
            e = self.val.exp()          # Dual.exp on nested
        else:
            e = math.exp(self.val)
        return Dual(e, e * self.tang, self.tag)

    def log(self):
        if isinstance(self.val, Dual):
            lv = self.val.log()
            return Dual(lv, self.tang / self.val, self.tag)
        return Dual(math.log(self.val), self.tang / self.val, self.tag)

    def sin(self):
        if isinstance(self.val, Dual):
            return Dual(self.val.sin(), self.val.cos() * self.tang, self.tag)
        return Dual(math.sin(self.val), math.cos(self.val) * self.tang, self.tag)

    def cos(self):
        if isinstance(self.val, Dual):
            return Dual(self.val.cos(), -self.val.sin() * self.tang, self.tag)
        return Dual(math.cos(self.val), -math.sin(self.val) * self.tang, self.tag)

    def tanh(self):
        if isinstance(self.val, Dual):
            t = self.val.tanh()
            return Dual(t, (1.0 - t * t) * self.tang, self.tag)
        t = math.tanh(self.val)
        return Dual(t, (1 - t * t) * self.tang, self.tag)

    def relu(self):
        # Subgradient convention: 0 at exactly 0
        sv = self._scalar_val()
        if isinstance(self.val, Dual):
            relu_val = self.val.relu()
        else:
            relu_val = max(0., self.val)
        tang = self.tang if sv > 0 else 0.
        return Dual(relu_val, tang, self.tag)

    def sqrt(self): return self ** 0.5

    def __float__(self): return float(self._scalar_val())
    def __gt__(self, o): return self._scalar_val() > (o._scalar_val() if isinstance(o, Dual) else float(o))
    def __lt__(self, o): return self._scalar_val() < (o._scalar_val() if isinstance(o, Dual) else float(o))
    def __ge__(self, o): return self._scalar_val() >= (o._scalar_val() if isinstance(o, Dual) else float(o))
    def __le__(self, o): return self._scalar_val() <= (o._scalar_val() if isinstance(o, Dual) else float(o))

    def __repr__(self):
        vs = repr(self.val) if isinstance(self.val, Dual) else f"{self.val:.4g}"
        ts = repr(self.tang) if isinstance(self.tang, Dual) else f"{self.tang:.4g}"
        return f"Dual({vs}, tang={ts}, tag={self.tag})"


# ── Forward-mode API ──────────────────────────────────────────────────────────

def jvp(f, x, v=1.0):
    """
    Jacobian-vector product: returns (f(x), f'(x) · v).

    Creates a fresh perturbation tag each call. If x is already a Dual
    (from an outer jvp call), it is preserved inside Dual.val — enabling
    correct nested differentiation without perturbation confusion.
    """
    tag = _fresh()
    xd  = Dual(x, v, tag)
    out = f(xd)
    if isinstance(out, Dual) and out.tag == tag:
        # Extract float from possibly-nested val
        val = out.val
        while isinstance(val, Dual): val = val.val
        return float(val), out.tang
    # out is a constant (no dependence on xd)
    val = out
    while isinstance(val, Dual): val = val.val
    return float(val), 0.0


def grad_fwd(f):
    """Return f'(x) via forward-mode AD."""
    return lambda x: jvp(f, x)[1]


def value_and_grad_fwd(f):
    """Return (f(x), f'(x)) via forward-mode AD."""
    return lambda x: jvp(f, x)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "=" * 60

    # ── 1. Perturbation confusion ────────────────────────────────────
    print(SEP)
    print("1. Perturbation confusion")
    print()
    print("   g(x) = x · (d/dy[y + x]  at y=0)")
    print("   d/dy[y + x] = 1  →  g(x) = x  →  g'(x) = 1  (for all x)")
    print()

    def g_naive(x):
        # Uses naive_deriv internally — shares the global tang=1 sentinel.
        dh_dy = naive_deriv(lambda y: y + x, 0.0)
        return x * dh_dy

    def g_tagged(x):
        # Uses jvp with a fresh tag — no leakage possible.
        _, dh_dy = jvp(lambda y: y + x, 0.0)
        return x * dh_dy

    x0 = 3.0
    naive_ans  = naive_deriv(g_naive,  x0)
    tagged_ans = jvp(g_tagged, x0)[1]

    print(f"   naive_deriv(g, {x0})  = {naive_ans:.1f}  "
          f"{'✓' if naive_ans  == 1.0 else '✗  ← perturbation confusion'}")
    print(f"   tagged jvp(g,  {x0})  = {tagged_ans:.1f}  "
          f"{'✓' if tagged_ans == 1.0 else '✗'}")
    print()
    print("   Naive: outer seeds x = NaiveDual(3, tang=1).")
    print("          Inner seeds y = NaiveDual(0, tang=1) — same sentinel.")
    print("          (y+x).tang = 1+1 = 2. Outer perturbation leaked in.")
    print()
    print("   Fix:   inner jvp creates y = Dual(0, 1, tag=FRESH).")
    print("          x.tag ≠ inner tag → x.tang ignored → (y+x).tang = 1. ✓")

    # ── 2. Nested jvp for higher-order derivatives ───────────────────
    print()
    print(SEP)
    print("2. Nested jvp — second derivative")
    print()
    print("   f(x) = x³   →  f'(x) = 3x²  →  f''(x) = 6x")
    print()
    print("   How: jvp(f, x) with x already a Dual preserves x inside")
    print("   Dual.val so the outer tangent propagates through arithmetic.")
    print("   Each nested jvp() uses a fresh tag; the two perturbation")
    print("   directions are orthogonal and never interfere.")
    print()

    def cube(x): return x * x * x

    x0   = 2.0
    val, f1 = jvp(cube, x0)
    _,   f2 = jvp(lambda x: jvp(cube, x)[1], x0)

    print(f"   f(2)   = {val:.1f}   (expected  8)")
    print(f"   f'(2)  = {f1:.1f}  (expected 12)")
    print(f"   f''(2) = {f2:.1f}  (expected 12)")

    # Verify on a few points
    import math as _math
    errors = [abs(jvp(lambda x: jvp(cube, x)[1], float(v))[1] - 6*v)
              for v in [1, 2, 3, 4]]
    print(f"   max error over [1,2,3,4]: {max(errors):.2e}")

    # ── 3. Third derivative ──────────────────────────────────────────
    print()
    print("   f(x) = sin(x)  →  f'''(x) = -cos(x)")
    print()

    def _sin(x): return x.sin() if isinstance(x, Dual) else _math.sin(x)

    d3_sin = lambda x: jvp(lambda x: jvp(lambda x: jvp(_sin, x)[1], x)[1], x)[1]
    x0 = _math.pi / 4
    got  = d3_sin(x0)
    want = -_math.cos(x0)
    print(f"   sin'''(π/4) via nested jvp = {got:.6f}")
    print(f"   exact -cos(π/4)            = {want:.6f}")
    print(f"   error: {abs(got - want):.2e}")

    # ── 4. Forward vs reverse cost ───────────────────────────────────
    print()
    print(SEP)
    print("3. Cost: forward (n JVPs) vs reverse (1 backward)")
    print()
    print("   f : ℝⁿ → ℝ  (ML loss):  reverse wins — 1 pass regardless of n")
    print("   f : ℝ  → ℝᵐ (Jacobian): forward wins — 1 JVP per output dim")
    print()

    import numpy as np
    from autograd import Tensor

    n = 8
    w_np = np.arange(1., n + 1.)
    x_np = np.arange(0.1, 0.1 * (n + 1), 0.1)

    # Forward: n scalar JVPs
    fwd_grad = np.array([jvp(lambda wi, i=i: wi * x_np[i], w_np[i])[1] for i in range(n)])

    # Reverse: 1 backward
    w_t = Tensor(w_np.copy())
    (w_t * Tensor(x_np)).sum().backward()

    print(f"   Forward gradient (n={n} JVPs): {fwd_grad}")
    print(f"   Reverse gradient (1 backward): {w_t.grad}")
    print(f"   Agree: {np.allclose(fwd_grad, w_t.grad)}")

    # ── 5. Float semantics gap ───────────────────────────────────────
    print()
    print(SEP)
    print("4. The float semantics gap (honest limitation)")
    print()
    print("   Tags guarantee: no perturbation confusion in the real-number model.")
    print("   They do not guarantee: the float computation approximates it.")
    print()
    print(f"   {'x':>12}  tagged ReLU'(x)")
    for xv in [1.0, 1e-5, 1e-15, -1e-15]:
        _, d = jvp(lambda x: x.relu(), xv)
        note = "  ← float noise determines sign" if abs(xv) < 1e-12 else ""
        print(f"   {xv:>12.2e}  {d}{note}")
    print()
    print("   The type/tag system guarantees correctness of the")
    print("   denotation f : ℝ → ℝ.  Float noise at x ≈ 0 is orthogonal.")
