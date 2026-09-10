"""
Dense lookahead sweep for the PID wall follower, plus figure.

Runs 29 lookahead values at 0.05 m resolution and writes lookahead_sweep.png.
Supersedes the earlier 6-point sweep, whose coarse spacing made a fragmented
stability boundary look like a contiguous feasible window.

Run:
    python plot_sweep.py
"""

import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from f110_gym.envs.f110_env import F110Env  # noqa: E402
import f110_gym  # noqa: E402
from wall_follower import WallFollower  # noqa: E402

MAP = os.path.join(os.path.dirname(f110_gym.__file__), "envs", "maps", "levine")
LOOKAHEADS = [round(x, 2) for x in np.arange(0.60, 2.001, 0.05)]

BLUE, GREY, GREEN, RED = "#2166ac", "#888888", "#2e8b57", "#c0392b"


def trial(env, lookahead, max_steps=40_000):
    ctrl = WallFollower(desired_dist=0.9, kp=1.0, kd=0.6, ki=0.0, lookahead=lookahead)
    obs, _, _, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
    errs = []
    for _ in range(max_steps):
        steer, speed, _, err = ctrl(obs["scans"][0])
        obs, _, _, _ = env.step(np.array([[steer, speed]]))
        errs.append(abs(err))
        if obs["collisions"][0]:
            return "crash", obs["lap_times"][0], float(np.mean(errs)), float(np.max(errs))
        if obs["lap_counts"][0] >= 1:
            return "lap", obs["lap_times"][0], float(np.mean(errs)), float(np.max(errs))
    return "timeout", obs["lap_times"][0], float(np.mean(errs)), float(np.max(errs))


def main():
    env = F110Env(map=MAP, map_ext=".pgm", num_agents=1)
    res = []
    print(f"{'L':>5} {'result':>8} {'sim_t':>7} {'mean':>7} {'max':>7}")
    for L in LOOKAHEADS:
        r, t, me, mx = trial(env, L)
        res.append(dict(L=L, r=r, t=t, mean=me, max=mx))
        print(f"{L:>5.2f} {r:>8} {t:>7.2f} {me:>7.3f} {mx:>7.3f}")

    laps = [d for d in res if d["r"] == "lap"]
    print(f"\n{len(laps)} of {len(res)} completed a lap")
    if laps:
        b = min(laps, key=lambda d: d["max"])
        print(f"best worst-case : L={b['L']:.2f}  mean {b['mean']:.3f}  max {b['max']:.3f}")
        b = min(laps, key=lambda d: d["mean"])
        print(f"best mean       : L={b['L']:.2f}  mean {b['mean']:.3f}  max {b['max']:.3f}")
        b = min(laps, key=lambda d: d["t"])
        print(f"fastest lap     : L={b['L']:.2f}  {b['t']:.2f} s")

    L = np.array([d["L"] for d in res])
    mean = np.array([d["mean"] for d in res])
    mx = np.array([d["max"] for d in res])
    lap = np.array([d["r"] == "lap" for d in res])
    tend = np.array([d["t"] for d in res])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1], "hspace": 0.12})

    for l in L[lap]:
        for a in (ax1, ax2):
            a.axvspan(l - 0.025, l + 0.025, color=GREEN, alpha=0.13, lw=0, zorder=0)

    ax1.plot(L, mx, "-", color=GREY, lw=1.6, alpha=0.85, zorder=2, label="Max |error|")
    ax1.plot(L, mean, "-", color=BLUE, lw=2.2, zorder=3, label="Mean |error|")
    for arr, c in ((mean, BLUE), (mx, GREY)):
        ax1.scatter(L[lap], arr[lap], s=42, color=c, edgecolors="white", lw=1.1, zorder=4)
        ax1.scatter(L[~lap], arr[~lap], s=30, facecolors="white", edgecolors=c, lw=1.3, zorder=4)
    ax1.scatter([], [], s=42, color="#555", label="Completed lap")
    ax1.scatter([], [], s=30, facecolors="white", edgecolors="#555", lw=1.3, label="Crashed")
    ax1.set_ylabel("Wall-distance error (m)")
    ax1.set_title("PID wall follower: error and survival vs lookahead distance\n"
                  "Levine map · $k_p$=1.0, $k_d$=0.6, $v$=2.5 m/s · 0.05 m resolution",
                  fontsize=11.5, pad=10)
    ax1.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    ax1.grid(alpha=0.25, zorder=0)
    ax1.set_ylim(0, max(mx.max() * 1.15, 1.0))

    ax2.bar(L[lap], tend[lap], width=0.038, color=GREEN, label="Lap time", zorder=3)
    ax2.bar(L[~lap], tend[~lap], width=0.038, color=RED, alpha=0.75,
            label="Time to crash", zorder=3)
    ax2.set_xlabel("Lookahead distance $L$ (m)")
    ax2.set_ylabel("Sim time (s)")
    ax2.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    ax2.grid(alpha=0.25, axis="y", zorder=0)
    ax2.set_xlim(L.min() - 0.05, L.max() + 0.05)

    fig.savefig("lookahead_sweep.png", dpi=170, bbox_inches="tight")
    print("\nwrote lookahead_sweep.png")


if __name__ == "__main__":
    main()
