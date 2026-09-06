"""Plot the lookahead sweep. Run after `python wall_follower.py sweep`."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOOKAHEAD = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
MEAN_CTE  = [0.391, 0.408, 0.140, 0.156, 0.204, 0.399]
MAX_CTE   = [2.931, 2.938, 0.939, 0.724, 2.078, 2.028]
LAPPED    = [False, False, False, True, True, False]

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(LOOKAHEAD, MEAN_CTE, "o-", lw=2, ms=7, color="#1f77b4", label="Mean |CTE|", zorder=3)
ax.plot(LOOKAHEAD, MAX_CTE, "s--", lw=2, ms=7, color="#7f7f7f", label="Max |CTE|", zorder=3)

crash_x = [l for l, ok in zip(LOOKAHEAD, LAPPED) if not ok]
ax.scatter(crash_x, [MEAN_CTE[LOOKAHEAD.index(x)] for x in crash_x],
           s=190, facecolors="none", edgecolors="#d62728", lw=2.2,
           zorder=4, label="Crashed before lap")

ax.axvspan(1.0, 1.5, color="#2ca02c", alpha=0.11, zorder=0)
ax.text(1.25, ax.get_ylim()[1] * 0.93, "feasible", ha="center",
        color="#2ca02c", fontsize=9, fontweight="bold")

ax.set_xlabel("Lookahead distance $L$ (m)")
ax.set_ylabel("Cross-track error (m)")
ax.set_title("Wall follower: tracking error vs lookahead\n"
             "Levine map, $k_p$=1.0, $k_d$=0.6, $v$=2.5 m/s", fontsize=11)
ax.grid(alpha=0.3, zorder=0)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("lookahead_sweep.png", dpi=160)
print("wrote lookahead_sweep.png")
