"""
Differentiable combinatorial solvers via blackbox differentiation.

Vlastelica et al., "Differentiation of Blackbox Combinatorial Solvers" (2020).

Core idea
---------
For a solver  y*(w) = argmin_{y ∈ Y} w·y  (shortest path, MST, etc.),
the true gradient ∂y*/∂w is zero almost everywhere (piecewise constant).
Instead, use a finite perturbation to estimate a useful gradient direction:

    ∂L/∂w  ≈  (y*(w + λ·∂L/∂y*) − y*(w)) / λ

Derivation
----------
Vlastelica defines the formula for argmax: y*(θ) = argmax y·θ.
Since our solver is argmin w·y = argmax y·(−w), we substitute θ = −w:

    ∇_θ L = (y*(θ) − y*(θ − λg)) / λ          [paper formula, argmax]
    ∇_w L = −∇_{−w} L = (y*(w + λg) − y*(w)) / λ   [sign flips once]

where g = ∂L/∂y* is the upstream gradient.

This gradient is NOT the true derivative (which doesn't exist).
It is a useful optimisation direction that:
  · is non-zero only when the perturbed solution differs from the original
  · correctly steers weights toward solutions preferred by the loss
  · degrades with large λ (path-changing noise) or small λ (near-zero → zero)

Honest limitations
------------------
  · Loss of the form L = w·y*(w) has gradient (y*(w+λw)−y*(w))/λ ≈ 0
    because scaling all weights uniformly doesn't change the argmin.
    Use a DIFFERENT loss signal than the solver weights — see demo.
  · Gradient is discontinuous in λ: for some losses it is 0 for all
    λ below a threshold, then jumps to ±1/λ. This makes tuning λ critical.
  · For dense graphs the perturbation may change many edges at once,
    producing a high-variance gradient estimate.
"""

import heapq
import numpy as np
from autograd import Tensor


# ------------------------------------------------------------------ #
# Dijkstra                                                             #
# ------------------------------------------------------------------ #

def _dijkstra(n_nodes, edges, weights, source, target):
    """Standard Dijkstra. Returns (edge_indicator, cost)."""
    adj = [[] for _ in range(n_nodes)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    INF = float('inf')
    dist = [INF] * n_nodes
    dist[source] = 0.0
    prev = [None] * n_nodes
    heap = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, ei in adj[u]:
            nd = d + weights[ei]
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = (ei, u)
                heapq.heappush(heap, (nd, v))

    indicator = np.zeros(len(edges))
    v = target
    while prev[v] is not None:
        ei, u = prev[v]
        indicator[ei] = 1.0
        v = u
    return indicator, dist[target]


def shortest_path(edge_weights, edges, n_nodes, source, target, lambda_=1.0):
    """
    Differentiable shortest path (Vlastelica et al. 2020).

    Forward  : exact Dijkstra
    Backward : ∂L/∂w = (y*(w + λ·out.grad) − y*(w)) / λ

    Parameters
    ----------
    edge_weights : Tensor or array (E,)
    edges        : list of (u, v)
    n_nodes      : int
    source, target : int
    lambda_      : perturbation magnitude (tune to be large enough that
                   perturbing in the gradient direction changes the path)
    """
    w_np   = edge_weights.data if isinstance(edge_weights, Tensor) else np.asarray(edge_weights)
    w_leaf = edge_weights if isinstance(edge_weights, Tensor) else Tensor(w_np)

    y_star, _ = _dijkstra(n_nodes, edges, w_np, source, target)
    out = Tensor(y_star, _children=(w_leaf,), _op="dijkstra")

    def _backward():
        # g = ∂L/∂y*  (upstream gradient from the loss)
        # Perturb routing weights in the direction of g, re-solve:
        #   ∂L/∂w ≈ (y*(w + λg) − y*(w)) / λ
        # This is Formula A from Vlastelica et al. for argmin solvers.
        # It is correct for gradient-descent (minimisation) losses.
        g = out.grad
        y_pert, _ = _dijkstra(n_nodes, edges, w_np + lambda_ * g, source, target)
        w_leaf.grad += (y_pert - y_star) / lambda_

    out._backward = _backward
    return out


# ------------------------------------------------------------------ #
# Kruskal MST                                                          #
# ------------------------------------------------------------------ #

def _kruskal(n_nodes, edges, weights):
    """Kruskal's MST. Returns edge indicator."""
    parent   = list(range(n_nodes))
    uf_rank  = [0] * n_nodes

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if uf_rank[px] < uf_rank[py]:
            px, py = py, px
        parent[py] = px
        if uf_rank[px] == uf_rank[py]:
            uf_rank[px] += 1
        return True

    indicator = np.zeros(len(edges))
    for i in np.argsort(weights):
        u, v = edges[i]
        if union(u, v):
            indicator[i] = 1.0
    return indicator


def minimum_spanning_tree(edge_weights, edges, n_nodes, lambda_=1.0):
    """
    Differentiable MST via Vlastelica blackbox gradient.
    Same gradient formula as shortest_path.
    """
    w_np   = edge_weights.data if isinstance(edge_weights, Tensor) else np.asarray(edge_weights)
    w_leaf = edge_weights if isinstance(edge_weights, Tensor) else Tensor(w_np)

    y_star = _kruskal(n_nodes, edges, w_np)
    out = Tensor(y_star, _children=(w_leaf,), _op="kruskal")

    def _backward():
        g = out.grad
        y_pert = _kruskal(n_nodes, edges, w_np + lambda_ * g)
        w_leaf.grad += (y_pert - y_star) / lambda_

    out._backward = _backward
    return out


# ------------------------------------------------------------------ #
# Demo                                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    SEP = "-" * 56
    np.random.seed(1)

    # ---- 1. Gradient direction check ----
    print(SEP)
    print("1. Gradient direction check: 3-node graph, two paths")
    print()
    print("   Graph: 0→1→2 (edges 0,1)  or  0→2 directly (edge 2)")
    print("   Loss: L = -y*[2]  (minimise: drive direct edge onto path)")

    n3    = 3
    e3    = [(0, 1), (1, 2), (0, 2)]
    w_np3 = np.array([1.0, 1.0, 3.0])   # 0→1→2 costs 2, direct costs 3

    # With L = -y*[2], g = [0, 0, -1], perturb w + λg = [1, 1, 3-λ].
    # Need 3 - λ < 2  →  λ > 1.  Use λ=1.5 → direct costs 1.5 < 2 → path flips.
    w3    = Tensor(w_np3.copy())
    path3 = shortest_path(w3, e3, n3, 0, 2, lambda_=1.5)
    print(f"   y*  = {path3.data}  (path: {'0→1→2' if path3.data[2]<0.5 else '0→2'})")

    loss3 = -path3[2]          # minimise: want direct edge on path
    loss3.backward()
    print(f"   ∂L/∂w = {w3.grad.round(3)}")
    print(f"   Interpretation: w[2]>0 → decrease w[2] to put direct edge on path,")
    print(f"                   w[0,1]<0 → increase w[0,1] to make indirect more costly.")
    print(f"   Sign correct: w[2]>0 ✓  w[0,1]<0 ✓"
          if w3.grad[2] > 0 and w3.grad[0] < 0 else "   Sign WRONG")

    # ---- 2. Learn-to-route: optimise routing weights for true costs ----
    print(SEP)
    print("2. Learn-to-route on 3×3 grid")
    print("   Routing weights w  ≠  true costs c")
    print("   Goal: learn w such that the shortest-w path has low true cost c")

    n_grid = 9
    edges_grid = []
    for r in range(3):
        for c in range(3):
            node = r * 3 + c
            if c < 2: edges_grid.append((node, node + 1))
            if r < 2: edges_grid.append((node, node + 3))
    E = len(edges_grid)

    # True costs: penalise right-side edges (columns 1-2)
    true_costs = np.array([
        3.0 if (edges_grid[e][0] % 3 >= 1 or edges_grid[e][1] % 3 >= 1) else 1.0
        for e in range(E)
    ])

    w_route = np.ones(E)   # start with uniform routing weights

    print(f"   Initial  true cost: ", end="")
    y0_route = _dijkstra(n_grid, edges_grid, w_route, 0, 8)[0]
    print(f"{(true_costs * y0_route).sum():.2f}  path: {np.where(y0_route>0.5)[0].tolist()}")

    lr = 0.3
    for step in range(150):
        w_t = Tensor(w_route.copy())
        y_t = shortest_path(w_t, edges_grid, n_grid, 0, 8, lambda_=0.5)
        # Loss: true cost of the path (different from routing weights!)
        loss = (Tensor(true_costs) * y_t).sum()
        loss.backward()
        w_route = np.clip(w_route - lr * w_t.grad, 0.1, None)

    y_final = _dijkstra(n_grid, edges_grid, w_route, 0, 8)[0]
    print(f"   Optimised true cost: {(true_costs * y_final).sum():.2f}  "
          f"path: {np.where(y_final>0.5)[0].tolist()}")

    # ---- 3. MST: gradient of total cost w.r.t. edge weights ----
    print(SEP)
    print("3. MST blackbox gradient on 5-node complete graph")

    n5     = 5
    edges5 = [(i, j) for i in range(n5) for j in range(i+1, n5)]
    w5     = np.array([4., 2., 6., 1., 3., 5., 8., 7., 2., 9.])
    # True costs differ from routing weights
    costs5 = np.array([1., 5., 1., 10., 1., 5., 1., 5., 10., 1.])

    w5_t  = Tensor(w5.copy())
    mst_t = minimum_spanning_tree(w5_t, edges5, n5, lambda_=0.5)
    loss5 = (Tensor(costs5) * mst_t).sum()
    loss5.backward()

    print(f"   MST edges (by w5): {np.where(_kruskal(n5, edges5, w5)>0)[0].tolist()}")
    print(f"   true cost        : {(costs5 * _kruskal(n5, edges5, w5)).sum():.1f}")
    print(f"   ∂(true cost)/∂w  : {w5_t.grad.round(2)}")
    print(f"   Non-zero entries indicate edges where changing w5 would swap")
    print(f"   the MST to a tree with lower true cost.")

    # Gradient descent step
    w5_opt = w5.copy()
    for _ in range(50):
        w5_t2 = Tensor(w5_opt.copy())
        mst_t2 = minimum_spanning_tree(w5_t2, edges5, n5, lambda_=0.5)
        (Tensor(costs5) * mst_t2).sum().backward()
        w5_opt = np.clip(w5_opt - 0.5 * w5_t2.grad, 0.1, None)

    mst_opt = _kruskal(n5, edges5, w5_opt)
    print(f"   After optimising w: MST true cost = {(costs5 * mst_opt).sum():.1f}  "
          f"edges: {np.where(mst_opt>0)[0].tolist()}")
