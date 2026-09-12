"""
F1TENTH — Model Predictive Control for path tracking.

Formulated in PATH-FRAME ERROR COORDINATES rather than global x/y/theta.
That choice is the whole reason this works; see "Why error coordinates" below.

Run:
    python mpc.py              # single run
    python mpc.py sweep        # cost-weight and speed sweep

Requires cvxpy. Install carefully -- see the note at the bottom of this file.
"""

import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import cvxpy as cp  # noqa: E402
from f110_gym.envs.f110_env import F110Env  # noqa: E402
import f110_gym  # noqa: E402

EXAMPLES = os.path.expanduser("~/dev/f1tenth_gym/examples")

_W = np.loadtxt(os.path.join(EXAMPLES, "example_waypoints.csv"),
                delimiter=";", skiprows=3)
PATH = _W[:, 1:3]
PSI = _W[:, 3] + np.pi / 2.0     # convention-corrected path heading
KAPPA = _W[:, 4]                 # path curvature, 1/m
VREF = _W[:, 5]
SEG = np.linalg.norm(np.roll(PATH, -1, axis=0) - PATH, axis=1)

WHEELBASE = 0.15875 + 0.17145
STEER_LIMIT = 0.4
N_PTS = len(PATH)
WARMUP_STEPS = 100

START_POSE = np.array([[0.7, 0.0, 1.37079632679]])


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def cross_track_error(p):
    a = PATH
    b = np.roll(PATH, -1, axis=0)
    ab = b - a
    ap = p - a
    t = np.clip((ap * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None] * ab - p, axis=1).min())


class MPC:
    """Linear time-varying MPC on the path-frame error dynamics.

    STATE   e = [e_y, e_psi]
              e_y   = signed lateral offset from the path (+ = left)
              e_psi = heading error relative to the path tangent
    INPUT   delta  = steering angle

    MODEL (kinematic bicycle, linearised about small errors):

        e_y[k+1]   = e_y[k]   + dt * v[k] * e_psi[k]
        e_psi[k+1] = e_psi[k] + dt * v[k]/L * delta[k] - dt * v[k] * kappa[k]

    Path curvature kappa enters as a KNOWN DISTURBANCE sampled from the
    raceline over the horizon. That gives the optimiser feedforward for free:
    it knows a corner is coming and starts turning before any error appears.
    Pure pursuit approximates this with a lookahead heuristic; Stanley does
    not do it at all.

    COST    sum over horizon of  q_y*e_y^2 + q_psi*e_psi^2 + r*delta^2
            plus rd*(delta[k+1]-delta[k])^2  -- explicit control-effort and
            smoothness penalties, which is what lets MPC trade accuracy
            against actuator activity as a *tunable* rather than a property
            of the algorithm.

    CONSTRAINT  |delta| <= STEER_LIMIT, enforced inside the optimisation
                rather than clipped afterwards. This is the structural
                advantage over the geometric controllers: they compute a
                command and then saturate it, which silently invalidates
                their own geometry. MPC plans a trajectory that respects
                the limit.

    WHY ERROR COORDINATES
    ---------------------
    A first version of this tracked absolute x, y, theta against absolute
    waypoints. It failed at the same corner every lap regardless of cost
    weights -- weight-independence being the tell that the problem was
    structural, not tuning. Heading accumulates without bound over a lap
    (theta > 4 rad and climbing), so the affine linearisation terms scale
    with theta and the QP conditions badly. In error coordinates every state
    sits near zero, the linearisation stays valid, and the weights have
    physical meaning: q_y is "metres of lateral error I care about" rather
    than an abstract position penalty.
    """

    name = "mpc"

    def __init__(self, N=10, DT=0.05, vgain=0.6,
                 q_y=60.0, q_psi=5.0, r=1.0, rd=20.0, solve_every=5):
        self.N, self.DT, self.vgain = N, DT, vgain
        self.solve_every = solve_every
        self.prev_delta = 0.0
        self.hold = (0.0, 1.0)
        self.step_count = 0
        self.solves = 0
        self.solve_time = 0.0

        # Build the QP ONCE with parameters, so each timestep only updates
        # numbers and re-solves. Rebuilding the problem every step costs
        # ~20x more.
        E = cp.Variable((2, N + 1))
        U = cp.Variable(N)
        self.e0 = cp.Parameter(2)
        self.v = cp.Parameter(N, nonneg=True)
        self.kappa = cp.Parameter(N)
        self.dprev = cp.Parameter()

        cost = 0
        cons = [E[:, 0] == self.e0]
        for k in range(N):
            cons += [
                E[0, k + 1] == E[0, k] + DT * cp.multiply(self.v[k], E[1, k]),
                E[1, k + 1] == E[1, k]
                               + DT / WHEELBASE * cp.multiply(self.v[k], U[k])
                               - DT * cp.multiply(self.v[k], self.kappa[k]),
                cp.abs(U[k]) <= STEER_LIMIT,
            ]
            cost += (q_y * cp.square(E[0, k + 1])
                     + q_psi * cp.square(E[1, k + 1])
                     + r * cp.square(U[k]))
        for k in range(N - 1):
            cost += rd * cp.square(U[k + 1] - U[k])
        cost += rd * cp.square(U[0] - self.dprev)

        self.E, self.U = E, U
        self.prob = cp.Problem(cp.Minimize(cost), cons)

    def __call__(self, x, y, th, v):
        # Re-solve at 20 Hz, hold between solves. Realistic for real hardware
        # and 5x cheaper than solving every 100 Hz timestep.
        if self.step_count % self.solve_every:
            self.step_count += 1
            return self.hold
        self.step_count += 1

        p = np.array([x, y])
        i0 = int(np.argmin(np.linalg.norm(PATH - p, axis=1)))

        # signed lateral error: + means left of the path
        normal = np.array([-np.sin(PSI[i0]), np.cos(PSI[i0])])
        e_y = float(np.dot(p - PATH[i0], normal))
        e_psi = float(wrap(th - PSI[i0]))

        # sample curvature and reference speed forward by arc length
        idx = []
        i = i0
        for _ in range(self.N):
            idx.append(i)
            step = VREF[i] * self.vgain * self.DT
            d = 0.0
            while d < step:
                d += SEG[i]
                i = (i + 1) % N_PTS
        idx = np.array(idx)

        self.e0.value = np.array([e_y, e_psi])
        self.v.value = np.maximum(VREF[idx] * self.vgain, 0.5)
        self.kappa.value = KAPPA[idx]
        self.dprev.value = self.prev_delta

        t0 = time.time()
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True,
                            eps_abs=1e-4, eps_rel=1e-4, max_iter=8000)
            delta = float(self.U[0].value)
        except Exception:
            delta = self.prev_delta
        if not np.isfinite(delta):
            delta = self.prev_delta
        self.solve_time += time.time() - t0
        self.solves += 1

        delta = float(np.clip(delta, -STEER_LIMIT, STEER_LIMIT))
        self.prev_delta = delta
        self.hold = (delta, float(VREF[i0] * self.vgain))
        return self.hold


_env = None


def get_env():
    global _env
    if _env is None:
        _env = F110Env(map=os.path.join(EXAMPLES, "example_map"),
                       map_ext=".png", num_agents=1)
    return _env


def run(controller, max_steps=9000, verbose=True):
    env = get_env()
    obs, _, _, _ = env.reset(START_POSE)
    rows = []
    outcome = "timeout"
    for _ in range(max_steps):
        steer, speed = controller(obs["poses_x"][0], obs["poses_y"][0],
                                  obs["poses_theta"][0], obs["linear_vels_x"][0])
        obs, _, _, _ = env.step(np.array([[steer, speed]]))
        rows.append((obs["lap_times"][0], obs["poses_x"][0], obs["poses_y"][0],
                     obs["poses_theta"][0], obs["linear_vels_x"][0], steer, speed,
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
                 steer_rate=float(np.abs(np.diff(scored[:, 5])).mean() / 0.01),
                 ms_per_solve=1000 * controller.solve_time / max(controller.solves, 1))
    if verbose:
        print(f"  mpc  {outcome:>6}  lap {stats['lap_time']:6.2f}s  "
              f"mean {stats['mean_cte']*100:5.2f} cm  "
              f"max {stats['max_cte']*100:5.2f} cm  "
              f"steer_rate {stats['steer_rate']:.2f} rad/s  "
              f"{stats['ms_per_solve']:.1f} ms/solve")
    return log, stats


COLUMNS = ["t", "x", "y", "theta", "v", "steer", "speed", "cte"]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        print("\nMPC — horizon and speed sweep")
        print(f"  {'N':>3} {'vgain':>6} {'rd':>4} {'result':>7} {'lap_t':>7} "
              f"{'mean_cm':>8} {'max_cm':>7} {'srate':>6} {'ms':>6}")
        for N in (6, 10, 20):
            for vg in (0.6, 0.7, 0.8):
                _, s = run(MPC(N=N, vgain=vg), verbose=False)
                print(f"  {N:>3} {vg:>6.1f} {20:>4} {s['outcome']:>7} "
                      f"{s['lap_time']:>7.2f} {s['mean_cte']*100:>8.2f} "
                      f"{s['max_cte']*100:>7.2f} {s['steer_rate']:>6.2f} "
                      f"{s['ms_per_solve']:>6.1f}")
    else:
        log, _ = run(MPC())
        np.savetxt("logs_mpc.csv", log, delimiter=",",
                   header=",".join(COLUMNS), comments="")
        print("\nwrote logs_mpc.csv")

# ---------------------------------------------------------------------------
# INSTALLING cvxpy WITHOUT BREAKING THE SIM
#
#   pip install cvxpy
#
# will pull numpy 2.x, which breaks f110_gym (gym 0.26 does not support
# numpy 2). Downgrading numpy afterwards then breaks cvxpy, which was
# compiled against 2.x. Install both together instead:
#
#   pip install --force-reinstall --no-cache-dir "numpy<2" cvxpy
#
# Verify with:
#   python -c "import numpy, cvxpy; from f110_gym.envs.f110_env import F110Env; print('ok')"
# ---------------------------------------------------------------------------
