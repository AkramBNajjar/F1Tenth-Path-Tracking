"""Control math only. No ROS imports anywhere in this file.

The same code runs in the plain-Python study scripts and inside the ROS 2
nodes, so the ROS layer is genuinely only message plumbing. Keeping the
algorithm free of the middleware is what makes it testable without a robot.
"""

import os
import numpy as np

WHEELBASE = 0.15875 + 0.17145
STEER_LIMIT = 0.4


def load_raceline(examples_dir=None):
    """Return (path_xy, heading, curvature, speed) from the F1TENTH raceline."""
    examples_dir = examples_dir or os.environ.get(
        "F1TENTH_EXAMPLES", "/opt/f1tenth_gym/examples")
    w = np.loadtxt(os.path.join(examples_dir, "example_waypoints.csv"),
                   delimiter=";", skiprows=3)
    # NOTE: psi_rad uses a heading convention rotated +pi/2 from atan2(dy, dx).
    return w[:, 1:3], w[:, 3] + np.pi / 2.0, w[:, 4], w[:, 5]


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


class PurePursuit:
    """Geometric tracking: steer along the arc to a point Ld ahead."""

    def __init__(self, path, vref, lookahead=1.0, vgain=0.6):
        self.path, self.vref = path, vref
        self.Ld, self.vgain = lookahead, vgain

    def __call__(self, x, y, theta, v):
        p = np.array([x, y])
        i = int(np.argmin(np.linalg.norm(self.path - p, axis=1)))
        j = i
        for _ in range(len(self.path)):
            if np.linalg.norm(self.path[j] - p) >= self.Ld:
                break
            j = (j + 1) % len(self.path)
        dx, dy = self.path[j, 0] - x, self.path[j, 1] - y
        xr = np.cos(-theta) * dx - np.sin(-theta) * dy
        yr = np.sin(-theta) * dx + np.cos(-theta) * dy
        steer = np.arctan2(2.0 * WHEELBASE * yr, xr * xr + yr * yr)
        return (float(np.clip(steer, -STEER_LIMIT, STEER_LIMIT)),
                float(self.vref[i] * self.vgain))


class Stanley:
    """Front-axle tracking: correct cross-track and heading error together."""

    def __init__(self, path, psi, vref, k=3.5, k_soft=1.0, vgain=0.6):
        self.path, self.psi, self.vref = path, psi, vref
        self.k, self.k_soft, self.vgain = k, k_soft, vgain

    def __call__(self, x, y, theta, v):
        fx = x + WHEELBASE * np.cos(theta)
        fy = y + WHEELBASE * np.sin(theta)
        p = np.array([fx, fy])
        i = int(np.argmin(np.linalg.norm(self.path - p, axis=1)))
        dx, dy = fx - self.path[i, 0], fy - self.path[i, 1]
        e = np.hypot(dx, dy)
        if np.dot([dx, dy], [-np.sin(self.psi[i]), np.cos(self.psi[i])]) > 0:
            e = -e
        steer = wrap(self.psi[i] - theta) + np.arctan2(self.k * e, v + self.k_soft)
        return (float(np.clip(steer, -STEER_LIMIT, STEER_LIMIT)),
                float(self.vref[i] * self.vgain))


def cross_track_error(p, path):
    a = path
    b = np.roll(path, -1, axis=0)
    ab = b - a
    ap = p - a
    t = np.clip((ap * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None] * ab - p, axis=1).min())
