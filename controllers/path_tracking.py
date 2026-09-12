"""
F1TENTH — path-tracking controllers: pure pursuit and Stanley.

Both track an optimized raceline (783 waypoints with position, heading,
curvature, and a target velocity profile) on `example_map`.

Unlike wall_follower.py, this measures TRUE cross-track error: the
perpendicular distance from the vehicle to the reference path, not the
distance to a wall.

Run:
    python path_tracking.py              # single run, both controllers
    python path_tracking.py sweep        # lookahead / gain sweep
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from f110_gym.envs.f110_env import F110Env  # noqa: E402
import f110_gym  # noqa: E402

# --------------------------------------------------------------------------
# Reference path
# --------------------------------------------------------------------------
# Point this at the examples/ folder of your f1tenth_gym clone.
EXAMPLES = os.path.expanduser("~/dev/f1tenth_gym/examples")

_WPT = np.loadtxt(os.path.join(EXAMPLES, "example_waypoints.csv"),
                  delimiter=";", skiprows=3)
PATH = _WPT[:, 1:3]   # x, y  (m)
# NOTE: the raceline's psi_rad column uses a heading convention rotated
# +pi/2 from standard atan2(dy, dx). Verified against path geometry:
# mean offset 1.5728 rad, std 0.058. Stanley reads this heading directly,
# so without the correction it saturates steering on the first step.
# Pure pursuit is purely geometric and never touches it, which is why the
# bug only showed up in one of the two controllers.
PSI = _WPT[:, 3] + np.pi / 2.0   # path heading (rad), convention-corrected
VREF = _WPT[:, 5]     # target speed (m/s)

WHEELBASE = 0.15875 + 0.17145   # lf + lr for the default F1TENTH params
STEER_LIMIT = 0.4               # rad

# Skip the opening transient before scoring. The spawn pose sits slightly off
# the raceline, so the first fraction of a second is a settling artifact, not
# tracking performance. Reporting it would flatter or penalise every run
# equally and hide the real differences between controllers.
WARMUP_STEPS = 100  # 1.0 s at 100 Hz


def cross_track_error(p):
    """Perpendicular distance from point p to the reference polyline.

    Projects p onto every segment of the path, clamps to the segment, and
    takes the minimum distance. This is the real definition of cross-track
    error -- distance to the *path*, not to the nearest waypoint, which
    would quantise the error to the waypoint spacing.
    """
    a = PATH
    b = np.roll(PATH, -1, axis=0)
    ab = b - a
    ap = p - a
    t = np.clip((ap * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.linalg.norm(proj - p, axis=1).min())


def nearest_index(p):
    return int(np.argmin(np.linalg.norm(PATH - p, axis=1)))


class PurePursuit:
    """Geometric path tracking.

    Picks a goal point one lookahead distance Ld ahead on the path, then
    solves for the steering angle of the circular arc from the rear axle to
    that goal point:

        delta = atan( 2 * L * sin(alpha) / Ld )

    where alpha is the angle to the goal point in the vehicle frame and L is
    the wheelbase. Written below as atan2(2*L*y_r, Ld^2), which is the same
    expression since sin(alpha) = y_r / Ld.

    Ld is the only tuning knob and it dominates behaviour:
      small Ld -> tight tracking, oscillation, instability at speed
      large Ld -> smooth, but cuts corners (mean error grows with Ld)
    """

    name = "pure_pursuit"

    def __init__(self, Ld=1.0, vgain=0.6):
        self.Ld = Ld
        self.vgain = vgain

    def __call__(self, x, y, theta, v):
        p = np.array([x, y])
        i = nearest_index(p)

        # walk forward along the path to the first point >= Ld away
        j = i
        for _ in range(len(PATH)):
            if np.linalg.norm(PATH[j] - p) >= self.Ld:
                break
            j = (j + 1) % len(PATH)
        gx, gy = PATH[j]

        # goal point into the vehicle frame
        dx, dy = gx - x, gy - y
        x_r = np.cos(-theta) * dx - np.sin(-theta) * dy
        y_r = np.sin(-theta) * dx + np.cos(-theta) * dy
        Ld_actual = np.hypot(x_r, y_r)

        steer = np.arctan2(2.0 * WHEELBASE * y_r, Ld_actual ** 2)
        steer = float(np.clip(steer, -STEER_LIMIT, STEER_LIMIT))
        speed = float(VREF[i] * self.vgain)
        return steer, speed


class Stanley:
    """Front-axle path tracking (Stanford's DARPA Grand Challenge controller).

    Corrects two errors at once, which pure pursuit conflates:

        delta = heading_error + atan( k * e / (v + k_soft) )

    The first term aligns the vehicle with the path direction. The second
    steers into the path proportionally to cross-track error e, with the
    correction shrinking as speed rises so it stays stable. k_soft keeps the
    denominator finite at low speed.

    Measured at the FRONT axle, not the rear -- that is the key difference
    from pure pursuit and why it tracks corner entry more tightly.
    """

    name = "stanley"

    def __init__(self, k=2.5, k_soft=1.0, vgain=0.6):
        self.k = k
        self.k_soft = k_soft
        self.vgain = vgain

    def __call__(self, x, y, theta, v):
        # project to the front axle
        fx = x + WHEELBASE * np.cos(theta)
        fy = y + WHEELBASE * np.sin(theta)
        p = np.array([fx, fy])
        i = nearest_index(p)

        # signed cross-track error: which side of the path are we on?
        dx, dy = fx - PATH[i, 0], fy - PATH[i, 1]
        e = np.hypot(dx, dy)
        path_normal = np.array([-np.sin(PSI[i]), np.cos(PSI[i])])
        if np.dot([dx, dy], path_normal) > 0:
            e = -e

        heading_error = np.arctan2(np.sin(PSI[i] - theta), np.cos(PSI[i] - theta))
        steer = heading_error + np.arctan2(self.k * e, v + self.k_soft)
        steer = float(np.clip(steer, -STEER_LIMIT, STEER_LIMIT))
        speed = float(VREF[i] * self.vgain)
        return steer, speed


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
START_POSE = np.array([[0.7, 0.0, 1.37079632679]])
_env = None


def get_env():
    global _env
    if _env is None:
        _env = F110Env(map=os.path.join(EXAMPLES, "example_map"),
                       map_ext=".png", num_agents=1)
    return _env


def run(controller, max_steps=30_000, verbose=True):
    env = get_env()
    obs, _, _, _ = env.reset(START_POSE)

    rows = []
    outcome = "timeout"
    for step in range(max_steps):
        x = obs["poses_x"][0]
        y = obs["poses_y"][0]
        th = obs["poses_theta"][0]
        v = obs["linear_vels_x"][0]

        steer, speed = controller(x, y, th, v)
        obs, _, _, _ = env.step(np.array([[steer, speed]]))

        rows.append((obs["lap_times"][0], obs["poses_x"][0], obs["poses_y"][0],
                     obs["poses_theta"][0], obs["linear_vels_x"][0],
                     steer, speed,
                     cross_track_error(np.array([obs["poses_x"][0],
                                                 obs["poses_y"][0]]))))

        if obs["collisions"][0]:
            outcome = "crash"
            break
        if obs["lap_counts"][0] >= 1:
            outcome = "lap"
            break

    log = np.array(rows)
    scored = log[WARMUP_STEPS:] if len(log) > WARMUP_STEPS else log
    cte = np.abs(scored[:, 7])
    stats = dict(outcome=outcome, lap_time=float(log[-1, 0]),
                 mean_cte=float(cte.mean()), max_cte=float(cte.max()),
                 rms_cte=float(np.sqrt((cte ** 2).mean())),
                 steer_rate=float(np.abs(np.diff(scored[:, 5])).mean() / 0.01))

    if verbose:
        print(f"  {controller.name:<14} {outcome:>6}  "
              f"lap {stats['lap_time']:6.2f}s  "
              f"mean {stats['mean_cte']:.4f}  rms {stats['rms_cte']:.4f}  "
              f"max {stats['max_cte']:.4f}  "
              f"steer_rate {stats['steer_rate']:.2f} rad/s")

    return log, stats


COLUMNS = ["t", "x", "y", "theta", "v", "steer", "speed", "cte"]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        print("\nPURE PURSUIT — lookahead sweep")
        print(f"  {'Ld':>5} {'vgain':>6} {'result':>7} {'lap_t':>7} "
              f"{'mean':>8} {'rms':>8} {'max':>8}")
        for vg in (0.5, 0.6, 0.7):
            for Ld in (0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5):
                _, s = run(PurePursuit(Ld, vg), verbose=False)
                print(f"  {Ld:>5.1f} {vg:>6.1f} {s['outcome']:>7} "
                      f"{s['lap_time']:>7.2f} {s['mean_cte']:>8.4f} "
                      f"{s['rms_cte']:>8.4f} {s['max_cte']:>8.4f}")

        print("\nSTANLEY — gain sweep")
        print(f"  {'k':>5} {'vgain':>6} {'result':>7} {'lap_t':>7} "
              f"{'mean':>8} {'rms':>8} {'max':>8}")
        for vg in (0.5, 0.6, 0.7):
            for k in (0.5, 1.0, 2.0, 3.5, 5.0, 8.0):
                _, s = run(Stanley(k, vgain=vg), verbose=False)
                print(f"  {k:>5.1f} {vg:>6.1f} {s['outcome']:>7} "
                      f"{s['lap_time']:>7.2f} {s['mean_cte']:>8.4f} "
                      f"{s['rms_cte']:>8.4f} {s['max_cte']:>8.4f}")
    else:
        print("\nSingle run, vgain=0.6:")
        for ctrl in (PurePursuit(Ld=1.0, vgain=0.6), Stanley(k=2.5, vgain=0.6)):
            log, _ = run(ctrl)
            np.savetxt(f"logs_{ctrl.name}.csv", log, delimiter=",",
                       header=",".join(COLUMNS), comments="")
        print("\nwrote logs_pure_pursuit.csv, logs_stanley.csv")
