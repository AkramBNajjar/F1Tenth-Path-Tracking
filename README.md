# F1TENTH Path-Tracking Study

Implementation and quantitative comparison of path-tracking controllers on the
[F1TENTH](https://f1tenth.org/) 1/10-scale autonomous vehicle platform.

The goal is not to build a car that drives. It is to **measure** how different
controllers trade tracking accuracy against speed and stability, and to
understand where each one breaks.

**Status:** wall-following complete in simulation. Pure pursuit in progress.

---

## Motivation

A path-tracking controller answers one question: given where I am, where I want
to be, and how fast I'm going, what steering angle do I command?

Every autonomous vehicle has one, and the tradeoffs are surprisingly sharp for
how simple the algorithms look. This repository implements several of them
against the same vehicle model, on the same track, with the same metrics, so
they can be compared on evidence rather than intuition.

---

## Setup

Verified on macOS (Apple Silicon), Python 3.11. See
[`SETUP_MACOS.md`](SETUP_MACOS.md) — the upstream `f1tenth_gym` package carries
stale dependency pins that need working around.

```bash
conda activate f1tenth
python wall_follower.py          # single run
python wall_follower.py sweep    # lookahead parameter sweep
python plot_sweep.py             # regenerate the figure
```

---

## Simulation environment

`f1tenth_gym` provides a single-track (bicycle) vehicle model with tire slip and
steering rate limits, plus a 2D LiDAR raycast against an occupancy map.

| | |
|---|---|
| Map | Levine (indoor loop, tight corners) |
| LiDAR | 1080 beams, 270° FOV |
| Timestep | 0.01 s (100 Hz) |
| Action | `[steering_angle (rad), speed (m/s)]` |
| Steering limit | ±0.4 rad |
| Throughput | ~115× realtime headless on an M-series Mac |

Running far faster than realtime is what makes parameter sweeps practical —
the six-point sweep below takes about two seconds.

---

## Controller 1 — PID wall follower

### Geometry

Two LiDAR beams are taken against the left wall: one perpendicular to the car
(`b`, at 90°) and one angled forward (`a`, at 40°), separated by θ = 50°.

The car's angle relative to the wall is

$$\alpha = \arctan\left(\frac{a\cos\theta - b}{a\sin\theta}\right)$$

which gives the true perpendicular distance to the wall

$$D_t = b\cos\alpha$$

Using the instantaneous distance alone produces steady oscillation, because the
controller only reacts to error it has already accumulated. Projecting a
lookahead distance $L$ forward along the car's heading gives the distance the
car *will* have if it holds its current course:

$$D_{t+1} = D_t + L\sin\alpha$$

PID acts on $e = D_\text{desired} - D_{t+1}$.

$L$ is the dominant tuning parameter. It is also the same idea that makes pure
pursuit work, which is why it's worth characterising carefully here first.

### Result: lookahead sweep

Constant 2.5 m/s, $k_p$ = 1.0, $k_d$ = 0.6, $k_i$ = 0.

| Lookahead $L$ (m) | Outcome | Lap time (s) | Mean \|CTE\| (m) | Max \|CTE\| (m) |
|---|---|---|---|---|
| 0.3 | crash | 5.36 | 0.391 | 2.931 |
| 0.5 | crash | 5.34 | 0.408 | 2.938 |
| 0.8 | crash | 18.90 | **0.140** | 0.939 |
| **1.0** | **lap complete** | 25.40 | 0.156 | **0.724** |
| 1.5 | lap complete | **24.87** | 0.204 | 2.078 |
| 2.0 | crash | 7.88 | 0.399 | 2.028 |

![Lookahead sweep](lookahead_sweep.png)

### Findings

**1. The feasible window is narrow and bounded on both sides.**
Only $L \in [1.0, 1.5]$ completes a lap. Below that, the controller reacts to
corners too late and clips the inside wall. Above it, the projected point is far
enough ahead that the controller effectively stops responding to the wall it is
currently beside. Performance is not monotonic in either direction — there is no
"more lookahead is smoother" rule to lean on.

**2. The best mean tracking error belongs to a controller that crashes.**
$L$ = 0.8 records the lowest mean CTE in the sweep (0.140 m) and hits a wall at
18.9 s. Tuning on average error alone selects a configuration that fails. Worst
case and average case disagree about which controller is better, which is the
practical argument for writing requirements against bounds rather than means.

**3. Lap time is bought with tracking margin.**
$L$ = 1.5 laps 0.53 s faster than $L$ = 1.0 but with 2.9× the worst-case error
(2.078 m vs 0.724 m). Both complete the lap; only one has margin left.

**4. Derivative gain is fragile at 100 Hz.**
Raising $k_d$ from 0.6 to 1.2, holding everything else fixed, took max CTE from
0.724 m to 25.9 m. Differentiating a fast, noisy signal amplifies the noise until
the controller fights itself.

### Caveat on the metrics

Mean CTE for crashed runs is averaged only over the seconds before impact, so
those values are not directly comparable to the completed laps — the $L$ = 0.3
and $L$ = 0.5 runs each survived roughly 5.3 s. Outcome is the primary metric;
mean CTE is only meaningful between runs that finished.

---

## Roadmap

- [x] Simulation environment, headless, with CSV logging
- [x] PID wall follower + lookahead characterisation
- [ ] Pure pursuit against a reference racing line
- [ ] Stanley controller (cross-track + heading error)
- [ ] MPC over the bicycle model
- [ ] Controller comparison at multiple speeds
- [ ] Port to ROS 2 nodes
- [ ] Hardware build (Raspberry Pi 5, RPLIDAR C1, 1/10 chassis)
- [ ] Sim-to-real: same controllers, measured gap

---

## References

- O'Kelly et al., *F1TENTH: An Open-source Evaluation Environment for Continuous
  Control and Reinforcement Learning*, NeurIPS 2019 Competition Track
- Coulter, *Implementation of the Pure Pursuit Path Tracking Algorithm*,
  CMU-RI-TR-92-01
- Hoffmann et al., *Autonomous Automobile Trajectory Tracking for Off-Road
  Driving*, ACC 2007 (Stanley controller)
