# F1TENTH Path-Tracking Study

Implementation and quantitative comparison of path-tracking controllers on the
[F1TENTH](https://f1tenth.org/) 1/10-scale autonomous vehicle platform.

The goal is not to build a car that drives. It is to **measure** how different
controllers trade tracking accuracy against speed, stability, and control
effort — and to find where each one breaks.

**Status:** three controllers working in simulation. MPC and ROS 2 next.
Hardware in Spring 2027.

---

## Motivation

A path-tracking controller answers one question: given where I am, where I want
to be, and how fast I'm going, what steering angle do I command?

Every autonomous vehicle has one, and the tradeoffs are sharper than the
algorithms look. This repository implements several against the same vehicle
model, on the same track, with the same metrics, so they can be compared on
evidence rather than intuition.

---

## Setup

Verified on macOS (Apple Silicon), Python 3.11. See
[`SETUP_MACOS.md`](SETUP_MACOS.md) — the upstream `f1tenth_gym` package ships
stale dependency pins that need working around.

```bash
conda activate f1tenth
python wall_follower.py            # reactive controller, single run
python wall_follower.py sweep      # lookahead sweep
python path_tracking.py            # pure pursuit + Stanley
python path_tracking.py sweep      # parameter sweeps for both
python plot_tracking.py            # regenerate figures
```

---

## Simulation environment

`f1tenth_gym` provides a single-track (bicycle) vehicle model with tire slip,
load transfer, and steering rate limits, plus a 2D LiDAR raycast against an
occupancy map.

| | |
|---|---|
| Vehicle model | Single-track with cornering stiffness, friction limit, CG height |
| LiDAR | 1080 beams, 270 deg FOV |
| Timestep | 0.01 s (100 Hz) |
| Action | `[steering_angle (rad), speed (m/s)]` |
| Steering limit | +/- 0.4 rad |
| Throughput | ~115x realtime headless on an M-series Mac |

Running far faster than realtime is what makes parameter sweeps practical — a
30-point sweep takes seconds.

---

## Two problems, two error metrics

The controllers here solve related but distinct problems, and it matters which
error is being reported.

**Reactive control** (wall follower) has no map and no reference path. It reads
LiDAR and holds a set distance from a wall. Error is *wall-distance error*.

**Path tracking** (pure pursuit, Stanley) is given a reference path and must
follow it. Error is **cross-track error** — the perpendicular distance from the
vehicle to the path. Computed here by projecting onto every path segment and
taking the minimum, not by distance to the nearest waypoint, which would
quantise error to the waypoint spacing.

These numbers are not comparable across the two sections, which is why they are
reported separately.

---

## Part 1 — PID wall follower (reactive)

### Geometry

Two LiDAR beams against the left wall: one perpendicular (`b`, 90 deg) and one
angled forward (`a`, 40 deg), separated by theta = 50 deg.

    alpha = atan( (a*cos(theta) - b) / (a*sin(theta)) )
    D_t   = b * cos(alpha)

Using instantaneous distance alone oscillates, because the controller only
reacts to error already accumulated. Projecting a lookahead `L` forward gives
the distance the car *will* have on its current course:

    D_t+1 = D_t + L*sin(alpha)

PID acts on `e = D_desired - D_t+1`.

### Result: lookahead sweep

29 values from 0.60 to 2.00 m at 0.05 m resolution. Levine map, 2.5 m/s,
kp = 1.0, kd = 0.6.

![Lookahead sweep](lookahead_sweep.png)

**Only 9 of 29 runs completed a lap, and the survivors are scattered** — 0.75
laps, 0.80–0.95 all crash, 1.00 laps, 1.05–1.40 all crash, 1.45–1.50 lap.
A 0.05 m change flips the outcome.

An earlier six-point sweep at 0.25 m spacing happened to sample 1.0 and 1.5 and
suggested a contiguous "feasible window" between them. **That was a sampling
artifact.** Finer resolution shows a fragmented stability boundary instead.

### Findings

**1. Outcome is discontinuous; error is not.** Mean error creeps smoothly from
0.12 m to 0.40 m across the whole range with no cliffs, while survival flips
repeatedly. The metric that can be measured continuously says almost nothing
about the binary outcome that actually matters.

**2. Two repeatable failure modes.** Crashes cluster at ~7.5 s and ~18.5 s of
sim time — the same two corners, not random.

**3. The optimum depends entirely on the objective.** Best worst-case error is
L = 1.00 (max 0.724 m); best mean is L = 0.75 (0.123 m); fastest lap is
L = 1.65 (24.86 s). Three objectives, three different controllers.

**4. Derivative gain is fragile at 100 Hz.** Raising kd from 0.6 to 1.2,
everything else fixed, took max error from 0.724 m to 25.9 m. Differentiating a
fast noisy signal amplifies the noise until the controller fights itself.

**Caveat:** mean error for crashed runs is averaged only over the seconds before
impact, so it is not comparable to completed laps. Outcome is the primary
metric.

---

## Part 2 — Pure pursuit and Stanley (path tracking)

Both track an optimized raceline of 783 waypoints carrying position, heading,
curvature, and a target velocity profile.

### Pure pursuit

Picks a goal point one lookahead distance `Ld` ahead on the path and solves for
the steering angle of the circular arc from the rear axle to it:

    delta = atan( 2*L*sin(alpha) / Ld )

where `L` is the wheelbase and `alpha` the angle to the goal point in the
vehicle frame. Purely geometric — no notion of heading error.

### Stanley

Corrects cross-track and heading error separately, measured at the **front**
axle:

    delta = psi_error + atan( k*e / (v + k_soft) )

The correction term shrinks with speed to stay stable; `k_soft` keeps the
denominator finite at rest.

### Results

Matched speed profile (vgain 0.6), same raceline, same map.

| Controller | Mean CTE | Max CTE | Lap time | Mean steering rate |
|---|---|---|---|---|
| Pure pursuit (Ld = 1.0) | 1.98 cm | 5.51 cm | 37.19 s | 0.12 rad/s |
| Stanley (k = 3.5) | **0.64 cm** | **3.05 cm** | 37.14 s | 2.51 rad/s |

![Trajectory and error](tracking_comparison.png)

![Tradeoff](tracking_tradeoff.png)

### Findings

**1. Stanley tracks ~3x tighter and pays ~20x the control effort.** At matched
speed it holds 0.64 cm against pure pursuit's 1.98 cm, but its mean steering
rate is 2.51 rad/s against 0.12. On a real vehicle that is actuator duty cycle
and ride quality, not a free win.

**2. The error signatures differ in kind, not just magnitude.** Pure pursuit
shows clean periodic humps — error rises to 4–5 cm through every corner and
falls on the straights. That is corner-cutting, exactly as the geometry
predicts: the controller aims at a point already partway around the bend.
Stanley shows low-amplitude high-frequency chatter instead. Predictable bias
versus continuous correction.

**3. Both have a U-shaped optimum.** Pure pursuit fails at Ld = 0.4 at every
speed tested (oscillates into the wall) and at 2.5 above vgain 0.5
(corner-cutting). Stanley is sluggish at k = 0.5 and unstable at k = 8. Neither
parameter is monotonic.

**4. Corner-cutting scales with lookahead as predicted.** At vgain 0.6, mean CTE
climbs monotonically 1.05 -> 6.78 cm as Ld goes 0.6 -> 2.0 m.

**Scope:** one track, one raceline, one speed profile. These are not general
claims about the controllers.

---

## A bug worth documenting

Stanley initially crashed 0.47 s into every run, at every gain value — including
gains an order of magnitude apart, which produced *identical* logs.

Gain-independence was the tell. If the gain does not matter, the steering
command is saturating, which means the error term is enormous from the first
step.

The raceline's `psi_rad` column uses a heading convention rotated pi/2 from
standard `atan2(dy, dx)`. Verified against path geometry: mean offset 1.5728 rad
across all 783 waypoints, standard deviation 0.058.

**Pure pursuit never caught it.** It is purely geometric and never reads the
stored heading — it derives everything from waypoint positions. Stanley consumes
path heading directly and so was fully exposed.

A silent data-convention mismatch that one consumer is immune to and another is
not is a real class of integration failure, and worth having found once.

---

## Roadmap

- [x] Simulation environment, headless, CSV logging
- [x] PID wall follower + stability characterisation
- [x] Pure pursuit against an optimized raceline
- [x] Stanley controller
- [x] True cross-track error and control-effort metrics
- [ ] MPC over the bicycle model
- [ ] Vehicle dynamics studies: kinematic vs dynamic model, friction sweep,
      understeer gradient, load transfer
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
  Driving*, ACC 2007
- Rajamani, *Vehicle Dynamics and Control*, 2nd ed.
