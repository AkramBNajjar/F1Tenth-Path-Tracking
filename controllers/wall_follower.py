"""F1TENTH PID wall follower — Saturday baseline."""
import os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from f110_gym.envs.f110_env import F110Env
import f110_gym

FOV = 4.7
NUM_BEAMS = 1080

def beam_at(scan, angle_deg):
    """Range in metres at an angle in the car frame. 0 = forward, + = left."""
    idx = int((np.radians(angle_deg) + FOV / 2.0) / FOV * (NUM_BEAMS - 1))
    return float(scan[np.clip(idx, 0, NUM_BEAMS - 1)])

class WallFollower:
    def __init__(self, desired_dist=0.9, kp=1.2, kd=0.6, ki=0.0, lookahead=0.5):
        self.desired_dist = desired_dist
        self.kp, self.kd, self.ki = kp, kd, ki
        self.lookahead = lookahead
        self._prev_err = 0.0
        self._integral = 0.0

    def __call__(self, scan, dt=0.01):
        a = beam_at(scan, 40.0)
        b = beam_at(scan, 90.0)
        theta = np.radians(50.0)
        alpha = np.arctan2(a * np.cos(theta) - b, a * np.sin(theta))
        dist_now = b * np.cos(alpha)
        dist_future = dist_now + self.lookahead * np.sin(alpha)
        err = self.desired_dist - dist_future
        self._integral += err * dt
        deriv = (err - self._prev_err) / dt
        self._prev_err = err
        steer = -(self.kp * err + self.kd * deriv + self.ki * self._integral)
        steer = float(np.clip(steer, -0.4, 0.4))
        speed = 2.5
        return steer, speed, dist_now, err

def run(controller, max_steps=30000, verbose=True):
    map_path = os.path.join(os.path.dirname(f110_gym.__file__), "envs", "maps", "levine")
    env = F110Env(map=map_path, map_ext=".pgm", num_agents=1)
    obs, _, _, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
    rows = []
    wall_start = time.time()
    outcome = "timeout"
    for step in range(max_steps):
        steer, speed, dist, err = controller(obs["scans"][0])
        obs, _, _, _ = env.step(np.array([[steer, speed]]))
        rows.append((obs["lap_times"][0], obs["poses_x"][0], obs["poses_y"][0],
                     obs["poses_theta"][0], obs["linear_vels_x"][0],
                     steer, speed, dist, err))
        if obs["collisions"][0]:
            outcome = "crash"
            break
        if obs["lap_counts"][0] >= 1:
            outcome = "lap_complete"
            break
    elapsed = time.time() - wall_start
    log = np.array(rows)
    if verbose:
        sim_t = log[-1, 0]
        print(f"outcome      : {outcome}")
        print(f"steps        : {len(log)}")
        print(f"sim time     : {sim_t:.2f} s")
        print(f"wall time    : {elapsed:.2f} s  ({sim_t/elapsed:.0f}x realtime)")
        print(f"mean |err|   : {np.abs(log[:, 8]).mean():.3f} m")
        print(f"max  |err|   : {np.abs(log[:, 8]).max():.3f} m")
    return log, outcome

COLUMNS = ["t", "x", "y", "theta", "v", "steer", "speed", "wall_dist", "err"]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        print(f"{'lookahead':>10} {'outcome':>13} {'lap_t':>7} {'mean':>7} {'max':>7}")
        for la in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
            log, outcome = run(WallFollower(kp=1.0, kd=0.6, lookahead=la), verbose=False)
            print(f"{la:>10.1f} {outcome:>13} {log[-1,0]:>7.2f} "
                  f"{np.abs(log[:,8]).mean():>7.3f} {np.abs(log[:,8]).max():>7.3f}")
    else:
        ctrl = WallFollower(desired_dist=0.9, kp=1.0, kd=0.6, ki=0.0, lookahead=1.0)
        log, outcome = run(ctrl)
        np.savetxt("logs_wall_follower.csv", log, delimiter=",",
                   header=",".join(COLUMNS), comments="")
        print("\nwrote logs_wall_follower.csv")
