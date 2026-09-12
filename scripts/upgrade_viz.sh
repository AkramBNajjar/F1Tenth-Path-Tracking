#!/usr/bin/env bash
# Upgrades the visualization node: speed-coloured raceline, heading arrow,
# live telemetry text. Run from the repo root:  bash upgrade_viz.sh
# Then rebuild in the container: colcon build --symlink-install
set -e

PKG="ros2_ws/src/f1tenth_control"
[ -d "$PKG" ] || { echo "run from the repo root (ros2_ws/ not found)"; exit 1; }

cat > "$PKG/f1tenth_control/viz_node.py" << 'VIZEOF'
"""Visualization for the F1TENTH path-tracking study.

    /map            OccupancyGrid   the track
    /raceline       Marker          reference path, COLOURED BY TARGET SPEED
    /car            Marker          vehicle body
    /heading        Marker          which way the car is pointing
    /goal_point     Marker          pure pursuit target
    /lookahead      Marker          line from car to target
    /trail          Marker          driven path
    /telemetry      Marker          live speed and cross-track error as text

The speed-coloured raceline is the useful one. Red is slow, green is fast, so
the braking zones are visible as red patches before the car reaches them. That
makes the corner-entry behaviour from the study legible at a glance.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import rclpy
import yaml
from PIL import Image
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from f1tenth_control.controllers import (
    load_raceline, cross_track_error, WHEELBASE)


def quat_to_yaw(q):
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


def rgba(r, g, b, a=1.0):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def speed_color(t):
    """t in [0,1]: 0 = slowest (red), 0.5 = amber, 1 = fastest (green)."""
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        u = t / 0.5
        return rgba(0.85, 0.15 + 0.70 * u, 0.10)
    u = (t - 0.5) / 0.5
    return rgba(0.85 - 0.70 * u, 0.85, 0.10 + 0.20 * u)


def pt(x, y, z=0.0):
    p = Point()
    p.x, p.y, p.z = float(x), float(y), float(z)
    return p


class Viz(Node):
    def __init__(self):
        super().__init__("viz")
        self.declare_parameter("map_name", "example_map")
        self.declare_parameter("lookahead", 1.0)

        self.examples = os.environ.get("F1TENTH_EXAMPLES",
                                       "/opt/f1tenth_gym/examples")
        self.path, self.psi, self.kappa, self.vref = load_raceline(self.examples)
        self.Ld = self.get_parameter("lookahead").value
        self.trail = []

        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             history=QoSHistoryPolicy.KEEP_LAST)
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", latched)
        self.rl_pub = self.create_publisher(Marker, "/raceline", latched)
        self.car_pub = self.create_publisher(Marker, "/car", 10)
        self.head_pub = self.create_publisher(Marker, "/heading", 10)
        self.goal_pub = self.create_publisher(Marker, "/goal_point", 10)
        self.look_pub = self.create_publisher(Marker, "/lookahead", 10)
        self.trail_pub = self.create_publisher(Marker, "/trail", 10)
        self.tel_pub = self.create_publisher(Marker, "/telemetry", 10)

        self.create_subscription(Odometry, "/ego_racecar/odom", self.on_odom, 10)
        self.publish_map()
        self.publish_raceline()
        self.create_timer(5.0, self.publish_map)
        self.create_timer(5.0, self.publish_raceline)
        self.get_logger().info(
            "viz up: /map /raceline /car /heading /goal_point /lookahead "
            "/trail /telemetry")

    def publish_map(self):
        name = self.get_parameter("map_name").value
        with open(os.path.join(self.examples, name + ".yaml")) as f:
            meta = yaml.safe_load(f)
        a = np.flipud(np.array(
            Image.open(os.path.join(self.examples, name + ".png")).convert("L")))
        occ = np.full(a.shape, -1, dtype=np.int8)
        occ[a > meta["free_thresh"] * 255] = 0
        occ[a < meta["occupied_thresh"] * 255] = 100

        g = OccupancyGrid()
        g.header.frame_id = "map"
        g.header.stamp = self.get_clock().now().to_msg()
        g.info.resolution = float(meta["resolution"])
        g.info.width = int(a.shape[1])
        g.info.height = int(a.shape[0])
        g.info.origin.position.x = float(meta["origin"][0])
        g.info.origin.position.y = float(meta["origin"][1])
        g.info.origin.orientation.w = 1.0
        g.data = occ.flatten().tolist()
        self.map_pub.publish(g)

    def publish_raceline(self):
        """LINE_LIST with a per-segment colour from the target speed profile."""
        vmin, vmax = float(self.vref.min()), float(self.vref.max())
        span = max(vmax - vmin, 1e-6)

        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "raceline"
        m.id = 10
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.16
        m.pose.orientation.w = 1.0
        m.points, m.colors = [], []
        n = len(self.path)
        for i in range(n):
            j = (i + 1) % n
            c = speed_color((self.vref[i] - vmin) / span)
            m.points.append(pt(self.path[i, 0], self.path[i, 1], 0.02))
            m.points.append(pt(self.path[j, 0], self.path[j, 1], 0.02))
            m.colors.append(c)
            m.colors.append(c)
        self.rl_pub.publish(m)

    def on_odom(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        v = msg.twist.twist.linear.x
        now = self.get_clock().now().to_msg()
        cte = cross_track_error(np.array([x, y]), self.path)

        car = Marker()
        car.header.frame_id = "map"
        car.header.stamp = now
        car.ns, car.id = "car", 0
        car.type, car.action = Marker.CUBE, Marker.ADD
        car.pose.position.x = x + 0.5 * WHEELBASE * np.cos(yaw)
        car.pose.position.y = y + 0.5 * WHEELBASE * np.sin(yaw)
        car.pose.position.z = 0.06
        car.pose.orientation.z = float(np.sin(yaw / 2))
        car.pose.orientation.w = float(np.cos(yaw / 2))
        car.scale.x, car.scale.y, car.scale.z = 0.58, 0.31, 0.12
        car.color = rgba(0.10, 0.35, 0.75)
        self.car_pub.publish(car)

        head = Marker()
        head.header.frame_id = "map"
        head.header.stamp = now
        head.ns, head.id = "heading", 4
        head.type, head.action = Marker.ARROW, Marker.ADD
        head.scale.x, head.scale.y, head.scale.z = 0.07, 0.15, 0.15
        head.color = rgba(0.95, 0.95, 1.0, 0.95)
        head.pose.orientation.w = 1.0
        head.points = [pt(x, y, 0.14),
                       pt(x + 0.9 * np.cos(yaw), y + 0.9 * np.sin(yaw), 0.14)]
        self.head_pub.publish(head)

        p = np.array([x, y])
        i = int(np.argmin(np.linalg.norm(self.path - p, axis=1)))
        j = i
        for _ in range(len(self.path)):
            if np.linalg.norm(self.path[j] - p) >= self.Ld:
                break
            j = (j + 1) % len(self.path)
        gx, gy = self.path[j]

        goal = Marker()
        goal.header.frame_id = "map"
        goal.header.stamp = now
        goal.ns, goal.id = "goal", 1
        goal.type, goal.action = Marker.SPHERE, Marker.ADD
        goal.pose.position.x, goal.pose.position.y = float(gx), float(gy)
        goal.pose.position.z = 0.18
        goal.pose.orientation.w = 1.0
        goal.scale.x = goal.scale.y = goal.scale.z = 0.34
        goal.color = rgba(1.0, 0.45, 0.0)
        self.goal_pub.publish(goal)

        line = Marker()
        line.header.frame_id = "map"
        line.header.stamp = now
        line.ns, line.id = "lookahead", 2
        line.type, line.action = Marker.LINE_STRIP, Marker.ADD
        line.scale.x = 0.05
        line.color = rgba(1.0, 0.45, 0.0, 0.85)
        line.pose.orientation.w = 1.0
        line.points = [pt(x, y, 0.12), pt(gx, gy, 0.12)]
        self.look_pub.publish(line)

        self.trail.append((x, y))
        if len(self.trail) > 6000:
            self.trail = self.trail[-6000:]
        tr = Marker()
        tr.header.frame_id = "map"
        tr.header.stamp = now
        tr.ns, tr.id = "trail", 3
        tr.type, tr.action = Marker.LINE_STRIP, Marker.ADD
        tr.scale.x = 0.07
        tr.color = rgba(0.25, 0.85, 1.0, 0.95)
        tr.pose.orientation.w = 1.0
        tr.points = [pt(tx, ty, 0.05) for tx, ty in self.trail[::3]]
        self.trail_pub.publish(tr)

        tel = Marker()
        tel.header.frame_id = "map"
        tel.header.stamp = now
        tel.ns, tel.id = "telemetry", 5
        tel.type = Marker.TEXT_VIEW_FACING
        tel.action = Marker.ADD
        tel.pose.position.x, tel.pose.position.y = float(x), float(y)
        tel.pose.position.z = 1.4
        tel.pose.orientation.w = 1.0
        tel.scale.z = 0.85
        tel.color = rgba(1.0, 1.0, 1.0, 0.95)
        tel.text = "%.1f m/s   CTE %.1f cm" % (v, cte * 100)
        self.tel_pub.publish(tel)


def main(args=None):
    rclpy.init(args=args)
    node = Viz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
VIZEOF

echo "viz_node.py upgraded. In the container:"
echo "  cd /ws && colcon build --symlink-install && source install/setup.bash"
echo "  ros2 launch f1tenth_control pure_pursuit_viz.launch.py"
