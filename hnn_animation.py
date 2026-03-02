"""
hnn_animation.py — animated comparison of HNN vs Neural ODE vs True pendulum.

Three panels update simultaneously:
  Left   : Phase portrait (q, p)   — energy contours + live trajectory trails
  Middle : Energy E(t) = H_true(q(t), p(t)) — conservation or drift
  Right  : Angle q(t) vs time      — trajectory accuracy

Run:
    python hnn_animation.py          # trains then shows animation
    python hnn_animation.py --save   # also writes hnn_animation.gif
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# ── import the engine ─────────────────────────────────────────────────────────
from autograd import Tensor, stack
from odeint import odeint


# ── True pendulum  H = p²/2 − cos(q) ─────────────────────────────────────────

def _pendulum_np(y, t):
    return np.array([y[1], -np.sin(y[0])])

def _rk4_step_np(f, y, t, dt):
    k1 = f(y, t)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(y +    dt*k3,  t +    dt)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_pendulum(y0, t_arr):
    ys = [y0.copy()]
    for i in range(len(t_arr) - 1):
        ys.append(_rk4_step_np(_pendulum_np, ys[-1], t_arr[i],
                               t_arr[i+1] - t_arr[i]))
    return np.stack(ys)

def true_energy(traj):
    return 0.5 * traj[:, 1]**2 - np.cos(traj[:, 0])


# ── Neural ODE  2 → 16 → tanh → 16 → 2 ──────────────────────────────────────

_N = 16
N_PARAMS_NODE = 2*_N + _N + _N*2 + 2

def node_field(y, t, params):
    W1 = params[0    : 2*_N].reshape(_N, 2)
    b1 = params[2*_N : 3*_N]
    W2 = params[3*_N : 5*_N].reshape(2, _N)
    b2 = params[5*_N : 5*_N+2]
    return W2 @ (W1 @ y + b1).tanh() + b2


# ── HNN  2 → 32 → tanh → 32 → 1  (scalar H_θ) ───────────────────────────────

_K = 32
N_PARAMS_HNN = 2*_K + _K + _K + 1

def hnn_field(y, t, params):
    W1 = params[0    : 2*_K].reshape(_K, 2)
    b1 = params[2*_K : 3*_K]
    W2 = params[3*_K : 4*_K]
    h     = (W1 @ y + b1).tanh()
    sech2 = 1.0 - h * h
    delta = sech2 * W2
    dH_dy = W1.T @ delta
    return stack([dH_dy[1], -dH_dy[0]])


# ── Adam ──────────────────────────────────────────────────────────────────────

def _adam(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = b1*m + (1-b1)*g
    v = b2*v + (1-b2)*g**2
    p = p - lr * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
    return p, m, v


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    np.random.seed(42)

    y0      = np.array([1.0, 0.0])
    t_train = np.linspace(0,  7,  70)
    t_long  = np.linspace(0, 70, 700)

    y_train = simulate_pendulum(y0, t_train)
    y_long  = simulate_pendulum(y0, t_long)
    E0      = float(true_energy(y_train[:1]).item())

    # Neural ODE — 500 Adam steps
    print("Training Neural ODE  (500 steps) …")
    pN = 0.05 * np.random.randn(N_PARAMS_NODE)
    mN = vN = np.zeros(N_PARAMS_NODE)
    for step in range(1, 501):
        pt   = Tensor(pN)
        loss = ((odeint(node_field, y0, t_train, pt) - Tensor(y_train))**2).mean()
        loss.backward()
        pN, mN, vN = _adam(pN, pt.grad, mN, vN, step)
        if step % 100 == 0:
            print(f"  NODE step {step:3d}  loss={float(loss.data):.6f}")

    # HNN — 1500 Adam steps
    print("Training HNN  (1500 steps) …")
    pH = 0.3 * np.random.randn(N_PARAMS_HNN)
    mH = vH = np.zeros(N_PARAMS_HNN)
    for step in range(1, 1501):
        pt   = Tensor(pH)
        loss = ((odeint(hnn_field, y0, t_train, pt) - Tensor(y_train))**2).mean()
        loss.backward()
        pH, mH, vH = _adam(pH, pt.grad, mH, vH, step)
        if step % 300 == 0:
            print(f"  HNN  step {step:4d}  loss={float(loss.data):.6f}")

    # Long rollouts
    print("Generating rollouts …")
    traj_true = y_long
    traj_node = odeint(node_field, y0, t_long, Tensor(pN)).data
    traj_hnn  = odeint(hnn_field,  y0, t_long, Tensor(pH)).data

    E_true = true_energy(traj_true)
    E_node = true_energy(traj_node)
    E_hnn  = true_energy(traj_hnn)

    return dict(
        t_train=t_train, t_long=t_long, E0=E0,
        traj_true=traj_true, traj_node=traj_node, traj_hnn=traj_hnn,
        E_true=E_true, E_node=E_node, E_hnn=E_hnn,
    )


# ── Animation ─────────────────────────────────────────────────────────────────

TRAIL   = 80      # number of past points shown as trail
STRIDE  = 2       # subsample for animation frames
INTERVAL= 30      # ms per frame

C_TRUE = "#2196F3"   # blue
C_HNN  = "#4CAF50"   # green
C_NODE = "#FF5722"   # deep orange


def make_animation(data, save_path=None):
    t       = data["t_long"]
    T_TRUE  = data["traj_true"]
    T_HNN   = data["traj_hnn"]
    T_NODE  = data["traj_node"]
    E_TRUE  = data["E_true"]
    E_HNN   = data["E_hnn"]
    E_NODE  = data["E_node"]
    E0      = data["E0"]
    t_train_end = data["t_train"][-1]

    # Subsample indices
    idx = np.arange(0, len(t), STRIDE)
    N   = len(idx)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 5), facecolor="#0d1117")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.13,
                        wspace=0.35)

    ax_phase  = fig.add_subplot(1, 3, 1)
    ax_energy = fig.add_subplot(1, 3, 2)
    ax_angle  = fig.add_subplot(1, 3, 3)

    for ax in (ax_phase, ax_energy, ax_angle):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.title.set_color("#e6edf3")

    # ── Panel 1: Phase portrait ────────────────────────────────────────────────
    # Energy contours of H_true = p²/2 − cos(q)
    qq = np.linspace(-np.pi, np.pi, 300)
    pp = np.linspace(-2.5, 2.5, 300)
    QQ, PP = np.meshgrid(qq, pp)
    HH = 0.5 * PP**2 - np.cos(QQ)
    levels = np.linspace(-1.2, 1.5, 20)
    ax_phase.contour(QQ, PP, HH, levels=levels, colors="#ffffff",
                     alpha=0.10, linewidths=0.5)
    # Highlight the true energy level
    ax_phase.contour(QQ, PP, HH, levels=[E0], colors=[C_TRUE],
                     alpha=0.35, linewidths=1.0)

    ax_phase.set_xlim(-np.pi, np.pi)
    ax_phase.set_ylim(-2.5, 2.5)
    ax_phase.set_xlabel("q  (angle, rad)")
    ax_phase.set_ylabel("p  (momentum)")
    ax_phase.set_title("Phase portrait")

    # Static full-length faint trails
    kw = dict(linewidth=0.4, alpha=0.15)
    ax_phase.plot(T_TRUE[:, 0], T_TRUE[:, 1], color=C_TRUE,  **kw)
    ax_phase.plot(T_HNN [:, 0], T_HNN [:, 1], color=C_HNN,   **kw)
    ax_phase.plot(T_NODE[:, 0], T_NODE[:, 1], color=C_NODE,  **kw)

    # Animated trails (drawn as thick lines)
    trail_true, = ax_phase.plot([], [], color=C_TRUE,  lw=1.4, alpha=0.85,
                                 label="True (RK4)")
    trail_hnn,  = ax_phase.plot([], [], color=C_HNN,   lw=1.4, alpha=0.85,
                                 label="HNN")
    trail_node, = ax_phase.plot([], [], color=C_NODE,  lw=1.4, alpha=0.85,
                                 label="Neural ODE")
    dot_true,   = ax_phase.plot([], [], "o", color=C_TRUE,  ms=6, zorder=5)
    dot_hnn,    = ax_phase.plot([], [], "o", color=C_HNN,   ms=6, zorder=5)
    dot_node,   = ax_phase.plot([], [], "o", color=C_NODE,  ms=6, zorder=5)

    ax_phase.legend(loc="upper right", fontsize=8, framealpha=0.3,
                    facecolor="#161b22", edgecolor="#30363d",
                    labelcolor="white")

    # ── Panel 2: Energy vs time ────────────────────────────────────────────────
    ax_energy.axhline(E0, color=C_TRUE, lw=0.8, alpha=0.5, linestyle="--",
                      label=f"E₀ = {E0:.3f}")
    # Training window shading
    ax_energy.axvspan(0, t_train_end, color="#ffffff", alpha=0.04)
    ax_energy.text(t_train_end * 0.45, E0 + 0.22, "train", color="#8b949e",
                   fontsize=8, ha="center")

    e_line_true, = ax_energy.plot([], [], color=C_TRUE, lw=1.2, alpha=0.75)
    e_line_hnn,  = ax_energy.plot([], [], color=C_HNN,  lw=1.8)
    e_line_node, = ax_energy.plot([], [], color=C_NODE, lw=1.8)
    vline        = ax_energy.axvline(0, color="#8b949e", lw=0.8, alpha=0.6)

    e_all = np.concatenate([E_TRUE, E_HNN, E_NODE])
    e_lo  = e_all.min() - 0.05
    e_hi  = e_all.max() + 0.05
    ax_energy.set_xlim(t[0], t[-1])
    ax_energy.set_ylim(e_lo, e_hi)
    ax_energy.set_xlabel("time (s)")
    ax_energy.set_ylabel("E(t)  =  H_true(q, p)")
    ax_energy.set_title("Energy conservation")
    ax_energy.legend(loc="lower left", fontsize=8, framealpha=0.3,
                     facecolor="#161b22", edgecolor="#30363d",
                     labelcolor="white")

    # Drift annotations (shown from the start, updated)
    node_ann = ax_energy.text(0.97, 0.08, "", transform=ax_energy.transAxes,
                               color=C_NODE, fontsize=8, ha="right",
                               va="bottom")
    hnn_ann  = ax_energy.text(0.97, 0.20, "", transform=ax_energy.transAxes,
                               color=C_HNN,  fontsize=8, ha="right",
                               va="bottom")

    # ── Panel 3: Angle q(t) vs time ───────────────────────────────────────────
    ax_angle.axvspan(0, t_train_end, color="#ffffff", alpha=0.04)
    ax_angle.text(t_train_end * 0.45,
                  T_TRUE[:, 0].max() * 0.85, "train",
                  color="#8b949e", fontsize=8, ha="center")

    q_line_true, = ax_angle.plot([], [], color=C_TRUE,  lw=1.2, alpha=0.75,
                                  label="True (RK4)")
    q_line_hnn,  = ax_angle.plot([], [], color=C_HNN,   lw=1.6,
                                  label="HNN")
    q_line_node, = ax_angle.plot([], [], color=C_NODE,  lw=1.6,
                                  label="Neural ODE")
    vline2       = ax_angle.axvline(0, color="#8b949e", lw=0.8, alpha=0.6)

    q_all = np.concatenate([T_TRUE[:, 0], T_HNN[:, 0], T_NODE[:, 0]])
    ax_angle.set_xlim(t[0], t[-1])
    ax_angle.set_ylim(q_all.min() - 0.1, q_all.max() + 0.1)
    ax_angle.set_xlabel("time (s)")
    ax_angle.set_ylabel("q(t)  (angle, rad)")
    ax_angle.set_title("Trajectory: angle q(t)")
    ax_angle.legend(loc="upper right", fontsize=8, framealpha=0.3,
                    facecolor="#161b22", edgecolor="#30363d",
                    labelcolor="white")

    # ── Super-title ────────────────────────────────────────────────────────────
    time_text = fig.text(0.50, 0.96, "", ha="center", va="top",
                          color="#e6edf3", fontsize=11, fontweight="bold")

    # ── Init ──────────────────────────────────────────────────────────────────
    def init():
        for art in (trail_true, trail_hnn, trail_node,
                    dot_true, dot_hnn, dot_node,
                    e_line_true, e_line_hnn, e_line_node,
                    q_line_true, q_line_hnn, q_line_node):
            art.set_data([], [])
        vline.set_xdata([0])
        vline2.set_xdata([0])
        time_text.set_text("")
        node_ann.set_text("")
        hnn_ann.set_text("")
        return (trail_true, trail_hnn, trail_node,
                dot_true, dot_hnn, dot_node,
                e_line_true, e_line_hnn, e_line_node,
                q_line_true, q_line_hnn, q_line_node,
                vline, vline2, time_text, node_ann, hnn_ann)

    # ── Update ─────────────────────────────────────────────────────────────────
    def update(frame_num):
        i = idx[frame_num]              # index into full 700-pt arrays
        t_now = t[i]

        # ── Phase portrait trails ──────────────────────────────────────────────
        lo = max(0, i - TRAIL)
        trail_true.set_data(T_TRUE[lo:i+1, 0], T_TRUE[lo:i+1, 1])
        trail_hnn .set_data(T_HNN [lo:i+1, 0], T_HNN [lo:i+1, 1])
        trail_node.set_data(T_NODE[lo:i+1, 0], T_NODE[lo:i+1, 1])
        dot_true.set_data([T_TRUE[i, 0]], [T_TRUE[i, 1]])
        dot_hnn .set_data([T_HNN [i, 0]], [T_HNN [i, 1]])
        dot_node.set_data([T_NODE[i, 0]], [T_NODE[i, 1]])

        # ── Energy lines (grow over time) ─────────────────────────────────────
        e_line_true.set_data(t[:i+1], E_TRUE[:i+1])
        e_line_hnn .set_data(t[:i+1], E_HNN [:i+1])
        e_line_node.set_data(t[:i+1], E_NODE[:i+1])
        vline .set_xdata([t_now])

        # Drift annotations (updated live)
        if i > 0:
            d_node = np.mean(np.abs(E_NODE[:i+1] - E0))
            d_hnn  = np.mean(np.abs(E_HNN [:i+1] - E0))
            node_ann.set_text(f"Neural ODE  mean|ΔE| = {d_node:.4f}")
            hnn_ann .set_text(f"HNN         mean|ΔE| = {d_hnn :.4f}")

        # ── Angle lines ────────────────────────────────────────────────────────
        q_line_true.set_data(t[:i+1], T_TRUE[:i+1, 0])
        q_line_hnn .set_data(t[:i+1], T_HNN [:i+1, 0])
        q_line_node.set_data(t[:i+1], T_NODE[:i+1, 0])
        vline2.set_xdata([t_now])

        # ── Super-title ────────────────────────────────────────────────────────
        periods = t_now / 6.74
        time_text.set_text(
            f"HNN vs Neural ODE — Simple Pendulum    "
            f"t = {t_now:5.1f} s  ({periods:.1f} periods)"
        )

        return (trail_true, trail_hnn, trail_node,
                dot_true, dot_hnn, dot_node,
                e_line_true, e_line_hnn, e_line_node,
                q_line_true, q_line_hnn, q_line_node,
                vline, vline2, time_text, node_ann, hnn_ann)

    ani = animation.FuncAnimation(
        fig, update, frames=N,
        init_func=init, interval=INTERVAL, blit=False
    )

    if save_path:
        print(f"Saving animation to {save_path} …")
        writer = animation.PillowWriter(fps=int(1000 / INTERVAL))
        ani.save(save_path, writer=writer, dpi=120)
        print("Saved.")
    else:
        plt.show()

    return ani


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    save = "--save" in sys.argv
    data = train()
    print("\nLaunching animation …")
    ani = make_animation(data, save_path="hnn_animation.gif" if save else None)
