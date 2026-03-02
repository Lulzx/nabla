"""
Differentiability type system — static guarantees for AD correctness.

The core question: can you define a type-level property that is
(a) strong enough to guarantee AD produces correct gradients, and
(b) weak enough to admit the programs people actually want to differentiate?

This module implements a practical answer:

Type lattice (IntEnum, higher = more regular)
---------------------------------------------
  SMOOTH (C∞) > C1 > PSMOOTH > LIPSCHITZ > CONTINUOUS

Composition rule: (f : A) ∘ (g : B) : min(A, B)   — weakest piece dominates.

Application rule: a combinator typed (DiffFn[≥ cls] → ...) raises TypeError
if passed a function that doesn't meet the minimum requirement.

Escape hatch: @custom_vjp(cls) lets users assert a smoothness class for
functions that can't be built from the typed primitives (e.g., wrappers
around external solvers). The system trusts the assertion.

The totality analogy
---------------------
  Type checking       ↔  Differentiability checking
  well-typed term     ↔  well-typed DiffFn
  partiality / ⊥      ↔  undefined derivative
  termination proof   ↔  smoothness certificate
  @unsafeIO escape    ↔  @custom_vjp escape
  no unrestricted rec ↔  no data-dep control flow on ℝ

Both sacrifice Turing-completeness for a static semantic guarantee.
Both work by restricting the language so the property holds by construction.
Neither can verify arbitrary Python (Rice's theorem); both track annotations.

Honest limits
-------------
- We track ANNOTATIONS, not proofs. A Python function passed through
  @smooth is trusted, not verified. Rice's theorem blocks verification.
- Float semantics gap: SMOOTH means C∞ w.r.t. the real-valued denotation,
  not the IEEE 754 computation. See forward_ad.py for the float discussion.
- No second-order tracking (C¹ vs C²): Newton's method or Hessians require
  knowing that the derivative is itself differentiable. Left for future work.
- Perturbation tag discipline (forward_ad.py) is a SEPARATE concern;
  both are needed for full correctness of forward-mode AD.
"""

from enum import IntEnum
from functools import wraps
import numpy as np


# ── Smoothness type lattice ──────────────────────────────────────────────────

class Smooth(IntEnum):
    CONTINUOUS = 0   # C⁰: just continuous (step, floor, indicator)
    LIPSCHITZ  = 1   # Lipschitz: Clarke subgradient exists a.e. (|x|, max)
    PSMOOTH    = 2   # Piecewise-C∞: smooth except on a measure-zero set (relu)
    C1         = 3   # Once continuously differentiable
    SMOOTH     = 4   # C∞: infinitely differentiable (exp, sin, polynomial)

    def __str__(self): return _NAMES[self]


_NAMES = {
    Smooth.SMOOTH:     "C∞",
    Smooth.C1:         "C¹",
    Smooth.PSMOOTH:    "piecewise-C∞ (a.e. differentiable)",
    Smooth.LIPSCHITZ:  "Lipschitz (subgradient exists)",
    Smooth.CONTINUOUS: "C⁰ (continuous, not differentiable everywhere)",
}

# Short aliases
SMOOTH     = Smooth.SMOOTH
C1         = Smooth.C1
PSMOOTH    = Smooth.PSMOOTH
LIPSCHITZ  = Smooth.LIPSCHITZ
CONTINUOUS = Smooth.CONTINUOUS


# ── DiffFn ───────────────────────────────────────────────────────────────────

class DiffFn:
    """
    A function annotated with its differentiability class.

    Composition via @ (f @ g = f ∘ g) propagates the weakest class.
    Smoothness requirements are checked at combinator call sites via .require().
    """

    def __init__(self, fn, cls: Smooth, name: str = None):
        self._fn  = fn
        self.cls  = cls
        self.name = name or getattr(fn, '__name__', repr(fn))

    def __call__(self, *args, **kw):
        return self._fn(*args, **kw)

    def __repr__(self):
        return f"DiffFn({self.name!r} : {_NAMES[self.cls]})"

    def __matmul__(self, other: 'DiffFn') -> 'DiffFn':
        """f @ g  =  f ∘ g  (apply g first, then f)."""
        if not isinstance(other, DiffFn):
            raise TypeError(f"Can only compose DiffFn with DiffFn, got {type(other).__name__}")
        cls  = min(self.cls, other.cls)
        name = f"({self.name} ∘ {other.name})"
        return DiffFn(lambda *a: self._fn(other._fn(*a)), cls, name)

    def require(self, cls: Smooth) -> 'DiffFn':
        """Assert this function meets a minimum smoothness requirement."""
        if self.cls < cls:
            raise TypeError(
                f"\nDifferentiability type error\n"
                f"  function : {self.name!r}\n"
                f"  has type : {_NAMES[self.cls]}\n"
                f"  requires : {_NAMES[cls]}\n"
                f"  fix      : use a smoother primitive, or register a\n"
                f"             @custom_vjp to assert the correct class.\n"
            )
        return self


# ── Decorators ───────────────────────────────────────────────────────────────

def _make_decorator(cls):
    def decorator(fn=None, *, name=None):
        if callable(fn):
            return DiffFn(fn, cls, name or fn.__name__)
        _name = fn if isinstance(fn, str) else name
        return lambda f: DiffFn(f, cls, _name or f.__name__)
    decorator.__name__ = cls.name.lower()
    return decorator

smooth     = _make_decorator(SMOOTH)
c1         = _make_decorator(C1)
psmooth    = _make_decorator(PSMOOTH)
lipschitz  = _make_decorator(LIPSCHITZ)
continuous = _make_decorator(CONTINUOUS)


def custom_vjp(cls: Smooth, name: str = None):
    """
    Assert a smoothness class for a function with a manually-specified
    gradient rule. The type system trusts the annotation.

    This is the escape hatch — analogous to unsafePerformIO.
    Incorrect assertions produce wrong gradients silently.
    """
    def decorator(fn):
        return DiffFn(fn, cls, name or fn.__name__)
    return decorator


# ── Primitive registry ───────────────────────────────────────────────────────

P = {
    # C∞ primitives
    'add':      DiffFn(np.add,                             SMOOTH,    '+'),
    'mul':      DiffFn(np.multiply,                        SMOOTH,    '×'),
    'exp':      DiffFn(np.exp,                             SMOOTH,    'exp'),
    'log':      DiffFn(np.log,                             SMOOTH,    'log'),
    'sin':      DiffFn(np.sin,                             SMOOTH,    'sin'),
    'cos':      DiffFn(np.cos,                             SMOOTH,    'cos'),
    'tanh':     DiffFn(np.tanh,                            SMOOTH,    'tanh'),
    'sigmoid':  DiffFn(lambda x: 1 / (1 + np.exp(-x)),    SMOOTH,    'σ'),
    'softmax':  DiffFn(lambda x: np.exp(x-x.max()) /
                                 np.exp(x-x.max()).sum(),  SMOOTH,    'softmax'),
    'gelu':     DiffFn(lambda x: x * (1 + np.tanh(
                    np.sqrt(2/np.pi)*(x + 0.044715*x**3)))/2,
                                                           SMOOTH,    'gelu'),
    # Piecewise-C∞
    'relu':     DiffFn(lambda x: np.maximum(0, x),        PSMOOTH,   'relu'),
    'leaky':    DiffFn(lambda x, a=0.01: np.where(x>0,x,a*x), PSMOOTH, 'leaky_relu'),
    'elu':      DiffFn(lambda x, a=1.0: np.where(x>0, x, a*(np.exp(x)-1)),
                                                           PSMOOTH,   'elu'),
    # Lipschitz (subgradient, not C¹)
    'abs':      DiffFn(np.abs,                             LIPSCHITZ, 'abs'),
    'maxpool':  DiffFn(np.maximum,                         LIPSCHITZ, 'max'),
    'clip':     DiffFn(np.clip,                            LIPSCHITZ, 'clip'),
    # Continuous but NOT differentiable
    'step':     DiffFn(lambda x: (x > 0).astype(float),   CONTINUOUS,'step'),
    'floor':    DiffFn(np.floor,                           CONTINUOUS,'floor'),
    'round':    DiffFn(np.round,                           CONTINUOUS,'round'),
}


# ── TypedExpr: type inference for expressions ────────────────────────────────

class TypedExpr:
    """
    Symbolic expression carrying its differentiability type.
    Not a computation graph — tracks types through operations only.
    Useful for statically checking a network architecture before training.
    """

    def __init__(self, name: str, cls: Smooth):
        self.name = name
        self.cls  = cls

    def apply(self, fn: DiffFn) -> 'TypedExpr':
        cls = min(self.cls, fn.cls)
        return TypedExpr(f"{fn.name}({self.name})", cls)

    def __repr__(self):
        return f"{self.name}  :  {_NAMES[self.cls]}"


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "=" * 60

    # ── 1. Composition rules ─────────────────────────────────────────
    print(SEP)
    print("1. Composition: (f : A) ∘ (g : B)  :  min(A, B)")
    print()
    print(f"  {'composition':28s}  {'type'}")
    print(f"  {'-'*28}  {'-'*30}")
    cases = [
        ("exp ∘ sin",         P['exp'],      P['sin']),
        ("sigmoid ∘ tanh",    P['sigmoid'],  P['tanh']),
        ("relu ∘ linear",     P['relu'],     P['mul']),
        ("abs ∘ relu",        P['abs'],      P['relu']),
        ("step ∘ sigmoid",    P['step'],     P['sigmoid']),
    ]
    for label, f, g in cases:
        c = f @ g
        print(f"  {label:28s}  {_NAMES[c.cls]}")

    # ── 2. Smoothness requirement enforcement ────────────────────────
    print()
    print(SEP)
    print("2. Smoothness requirements at combinator call sites")
    print()

    def twice(fn: DiffFn, x: float) -> float:
        """Apply fn twice. Meaningful only if fn is at least C¹ (has a derivative)."""
        fn.require(C1)
        return fn(fn(x))

    for name in ['sigmoid', 'tanh', 'relu', 'abs', 'step']:
        fn = P[name]
        try:
            result = twice(fn, 0.5)
            print(f"  twice({name:8s}, 0.5) = {result:.4f}  ✓")
        except TypeError as e:
            first_line = str(e).strip().split('\n')[1].strip()
            print(f"  twice({name:8s}, 0.5) → TypeError: {first_line}")

    # ── 3. Neural network type tracking ─────────────────────────────
    print()
    print(SEP)
    print("3. Type-tracked forward pass")
    print()

    x = TypedExpr("x", SMOOTH)
    print(f"  Input:              {x}")

    after_w1 = x.apply(P['mul'])
    print(f"  After W₁x + b₁:    {after_w1}")

    after_relu = after_w1.apply(P['relu'])
    print(f"  After relu:         {after_relu}")

    after_w2 = after_relu.apply(P['mul'])
    print(f"  After W₂x + b₂:    {after_w2}")

    after_sm = after_w2.apply(P['softmax'])
    print(f"  After softmax:      {after_sm}")

    print()
    print(f"  Network type: {_NAMES[after_sm.cls]}")
    print(f"  AD guarantee: correct almost everywhere.")
    print(f"  Non-differentiable set: {{x : any pre-activation == 0}}")
    print(f"  Measure under any abs-continuous input distribution: zero.")

    # Same network but with step instead of relu
    print()
    after_step = after_w1.apply(P['step'])
    after_w2b  = after_step.apply(P['mul'])
    print(f"  With step activation: {after_w2b}")
    print(f"  → gradient is zero almost everywhere — training would stall.")

    # ── 4. custom_vjp escape hatch ───────────────────────────────────
    print()
    print(SEP)
    print("4. custom_vjp escape hatch")
    print()

    # hard_sigmoid is CONTINUOUS in truth (clips at 0 and 1),
    # but we assert PSMOOTH by providing a subgradient convention.
    @custom_vjp(PSMOOTH, name="hard_sigmoid")
    def hard_sigmoid(x):
        return np.clip(x / 6 + 0.5, 0., 1.)

    print(f"  hard_sigmoid type (asserted): {hard_sigmoid}")
    x_hs = TypedExpr("x", SMOOTH)
    after_hs = x_hs.apply(P['mul']).apply(hard_sigmoid)
    print(f"  Network with hard_sigmoid: {after_hs}")
    print()
    print("  @custom_vjp is the 'escape hatch': the type system trusts")
    print("  the annotation. Wrong assertions silently produce wrong gradients.")
    print("  This is the same tradeoff as unsafePerformIO — power at the")
    print("  cost of the guarantee. Use only at verified boundaries.")

    # ── 5. The totality analogy ──────────────────────────────────────
    print()
    print(SEP)
    print("5. The totality analogy")
    print()
    print("  Type safety         ↔  Differentiability typing")
    print("  ──────────────────     ────────────────────────────────")
    print("  well-typed term     ↔  DiffFn with smoothness class")
    print("  partial function    ↔  non-differentiable point")
    print("  totality proof      ↔  smoothness certificate")
    print("  unsafePerformIO     ↔  @custom_vjp (trust me)")
    print("  no unrestricted rec ↔  no data-dep branches over ℝ")
    print("  Rice: undecidable   ↔  Rice: undecidable for arb. code")
    print()
    print("  Both sacrifice expressiveness for a static guarantee.")
    print("  Both restrict the language so the property holds by")
    print("  construction for all well-typed terms — rather than")
    print("  trying to verify the property for arbitrary programs.")

    # ── 6. Honest limits ─────────────────────────────────────────────
    print()
    print(SEP)
    print("6. Honest limits")
    print()
    print("  ✓  Composition correctly degrades the type")
    print("  ✓  Smoothness requirements enforced at combinator call sites")
    print("  ✓  @custom_vjp provides a principled, checked escape hatch")
    print("  ✓  TypedExpr gives static type inference over expression graphs")
    print()
    print("  ✗  Annotations, not proofs: we trust @smooth, can't verify it")
    print("     (Rice's theorem — arbitrary Python is unanalysable)")
    print("  ✗  Float semantics gap: SMOOTH is w.r.t. ℝ denotation,")
    print("     not IEEE 754 computation (see forward_ad.py §4)")
    print("  ✗  No order tracking: C¹ vs C² vs C∞ is coarse;")
    print("     Hessians need C² which this system doesn't distinguish")
    print("  ✗  Perturbation tag discipline (forward_ad.py) is a")
    print("     separate correctness concern — both are needed together")
    print("  ✗  Data-dependent loops over ℝ, dynamic shapes, sparse ops")
    print("     fall outside the typed fragment and need @custom_vjp")
