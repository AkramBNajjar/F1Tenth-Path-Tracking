"""Figures for the path-tracking study. Run after path_tracking.py works."""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from path_tracking import PurePursuit, Stanley, run, PATH

PP, ST, GREY = "#2166ac", "#c0392b", "#999999"

# ---------------- collect data ----------------
log_pp, s_pp = run(PurePursuit(Ld=1.0, vgain=0.6), verbose=False)
log_st, s_st = run(Stanley(k=3.5,  vgain=0.6), verbose=False)

pp_sweep, st_sweep = [], []
for Ld in (0.6, 0.8, 1.0, 1.5, 2.0):
    _, s = run(PurePursuit(Ld, 0.6), verbose=False); s["p"] = Ld; pp_sweep.append(s)
for k in (0.5, 1.0, 2.0, 3.5, 5.0, 8.0):
    _, s = run(Stanley(k, vgain=0.6), verbose=False); s["p"] = k; st_sweep.append(s)

# ================= FIGURE 1 : trajectory + error =================
fig = plt.figure(figsize=(12, 5.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], hspace=0.42, wspace=0.22)

ax = fig.add_subplot(gs[:, 0])
ax.plot(PATH[:, 0], PATH[:, 1], "--", color=GREY, lw=1.4, label="Reference raceline", zorder=1)
ax.plot(log_pp[:, 1], log_pp[:, 2], color=PP, lw=1.7, label="Pure pursuit", zorder=2)
ax.plot(log_st[:, 1], log_st[:, 2], color=ST, lw=1.7, alpha=0.85, label="Stanley", zorder=3)
ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Trajectory vs reference raceline", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="best"); ax.grid(alpha=0.25)

for i, (log, s, c, nm) in enumerate([(log_pp, s_pp, PP, "Pure pursuit ($L_d$=1.0)"),
                                     (log_st, s_st, ST, "Stanley ($k$=3.5)")]):
    a = fig.add_subplot(gs[i, 1])
    a.plot(log[:, 0], log[:, 7] * 100, color=c, lw=1.2)
    a.axhline(s["mean_cte"] * 100, color="k", ls=":", lw=1,
              label=f"mean {s['mean_cte']*100:.2f} cm")
    a.set_ylabel("CTE (cm)"); a.set_ylim(0, 12)
    a.set_title(nm, fontsize=10, loc="left")
    a.legend(frameon=False, fontsize=8.5); a.grid(alpha=0.25)
    if i == 1: a.set_xlabel("Lap time (s)")

fig.suptitle("Path tracking on the F1TENTH raceline · matched speed (vgain 0.6)",
             fontsize=12.5, y=0.98)
fig.savefig("tracking_comparison.png", dpi=165, bbox_inches="tight")
print("wrote tracking_comparison.png")

# ================= FIGURE 2 : the tradeoff =================
fig2, (b1, b2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

for sw, c, nm, sym in ((pp_sweep, PP, "Pure pursuit", "$L_d$ (m)"),
                       (st_sweep, ST, "Stanley", "$k$")):
    p    = [s["p"] for s in sw]
    mean = [s["mean_cte"] * 100 for s in sw]
    b1.plot(range(len(p)), mean, "o-", color=c, lw=2, ms=7, label=nm)
    for xi, (pi, mi) in enumerate(zip(p, mean)):
        b1.annotate(f"{pi:g}", (xi, mi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5, color=c)
b1.set_xticks([]); b1.set_ylabel("Mean cross-track error (cm)")
b1.set_xlabel("Tuning parameter (labelled: $L_d$ in m / Stanley $k$)")
b1.set_title("Both controllers have a U-shaped optimum", fontsize=11)
b1.legend(frameon=False, fontsize=9); b1.grid(alpha=0.25)

for sw, c, nm in ((pp_sweep, PP, "Pure pursuit"), (st_sweep, ST, "Stanley")):
    b2.plot([s["steer_rate"] for s in sw], [s["mean_cte"] * 100 for s in sw],
            "o-", color=c, lw=2, ms=7, label=nm)
b2.set_xlabel("Mean steering rate (rad/s)  —  control effort")
b2.set_ylabel("Mean cross-track error (cm)")
b2.set_title("Accuracy is bought with control effort", fontsize=11)
b2.legend(frameon=False, fontsize=9); b2.grid(alpha=0.25)

fig2.tight_layout()
fig2.savefig("tracking_tradeoff.png", dpi=165, bbox_inches="tight")
print("wrote tracking_tradeoff.png")

print("\n%-14s %8s %8s %10s" % ("controller", "mean_cm", "max_cm", "steer_r"))
for s, nm in ((s_pp, "pure_pursuit"), (s_st, "stanley")):
    print("%-14s %8.2f %8.2f %10.2f" % (nm, s["mean_cte"]*100, s["max_cte"]*100, s["steer_rate"]))
