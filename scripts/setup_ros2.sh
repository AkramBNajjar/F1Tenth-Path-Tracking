#!/usr/bin/env bash
# Scaffolds a ROS 2 workspace for the F1TENTH path-tracking controllers.
# Run once from the repo root:   bash setup_ros2.sh
set -e

ROOT="ros2_ws"
PKG="$ROOT/src/f1tenth_control"
mkdir -p "$PKG/f1tenth_control" "$PKG/launch" "$PKG/resource"
touch "$PKG/resource/f1tenth_control"

# ---------------------------------------------------------------- Dockerfile
cat > "$ROOT/Dockerfile" << 'DOCKEREOF'
# ROS 2 Jazzy on Ubuntu 24.04. Official images publish arm64, so this builds
# natively on Apple Silicon with no emulation.
FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip git ros-jazzy-ackermann-msgs \
    && rm -rf /var/lib/apt/lists/*

# f1tenth_gym pins are stale (see SETUP_MACOS.md). Install without them.
RUN pip3 install --break-system-packages --no-cache-dir \
      "numpy<2" scipy numba pillow pyyaml "pyglet==2.0.15" "gym==0.26.2"

RUN git clone --depth 1 https://github.com/f1tenth/f1tenth_gym.git /opt/f1tenth_gym \
    && pip3 install --break-system-packages --no-deps -e /opt/f1tenth_gym

ENV F1TENTH_EXAMPLES=/opt/f1tenth_gym/examples
WORKDIR /ws
CMD ["bash"]
DOCKEREOF

# --------------------------------------------------------------- package.xml
cat > "$PKG/package.xml" << 'PKGEOF'
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>f1tenth_control</name>
  <version>0.1.0</version>
  <description>Path-tracking controllers for the F1TENTH platform as ROS 2 nodes.</description>
  <maintainer email="akrambn939@gmail.com">Akram Badaoui-Najjar</maintainer>
  <license>MIT</license>

  <exec_depend>rclpy</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>nav_msgs</exec_depend>
  <exec_depend>ackermann_msgs</exec_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
PKGEOF

# ------------------------------------------------------------------ setup.py
cat > "$PKG/setup.py" << 'SETUPEOF'
from setuptools import setup
import os
from glob import glob

package_name = 'f1tenth_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Akram Badaoui-Najjar',
    maintainer_email='akrambn939@gmail.com',
    description='Path-tracking controllers for the F1TENTH platform as ROS 2 nodes.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'sim_bridge = f1tenth_control.sim_bridge_node:main',
            'pure_pursuit = f1tenth_control.pure_pursuit_node:main',
            'stanley = f1tenth_control.stanley_node:main',
        ],
    },
)
SETUPEOF

cat > "$PKG/setup.cfg" << 'CFGEOF'
[develop]
script_dir=$base/lib/f1tenth_control
[install]
install_scripts=$base/lib/f1tenth_control
CFGEOF

touch "$PKG/f1tenth_control/__init__.py"

# -------------------------------------------------------------- controllers.py
cat > "$PKG/f1tenth_control/controllers.py" << 'CTRLEOF'
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
CTRLEOF

# ------------------------------------------------------------ sim_bridge_node
cat > "$PKG/f1tenth_control/sim_bridge_node.py" << 'BRIDGEEOF'
"""Wraps f1tenth_gym as a ROS 2 node.

Publishes  /scan              sensor_msgs/LaserScan
           /ego_racecar/odom  nav_msgs/Odometry
Subscribes /drive             ackermann_msgs/AckermannDriveStamped

This is the piece that gets replaced by real hardware later. The controller
nodes never know whether they are talking to a simulator or a car, which is
the entire point of putting a message boundary here.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from f110_gym.envs.f110_env import F110Env


def yaw_to_quat(yaw):
    return 0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0))


class SimBridge(Node):
    def __init__(self):
        super().__init__("sim_bridge")
        self.declare_parameter("map_name", "example_map")
        self.declare_parameter("map_ext", ".png")
        self.declare_parameter("rate_hz", 100.0)

        examples = os.environ.get("F1TENTH_EXAMPLES", "/opt/f1tenth_gym/examples")
        map_path = os.path.join(
            examples, self.get_parameter("map_name").value)
        self.env = F110Env(map=map_path,
                           map_ext=self.get_parameter("map_ext").value,
                           num_agents=1)
        self.obs, _, _, _ = self.env.reset(
            np.array([[0.7, 0.0, 1.37079632679]]))

        self.cmd = (0.0, 0.0)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.odom_pub = self.create_publisher(Odometry, "/ego_racecar/odom", 10)
        self.create_subscription(AckermannDriveStamped, "/drive", self.on_drive, 10)

        rate = self.get_parameter("rate_hz").value
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info("sim bridge up, stepping at %.0f Hz" % rate)

    def on_drive(self, msg):
        self.cmd = (float(msg.drive.steering_angle), float(msg.drive.speed))

    def tick(self):
        steer, speed = self.cmd
        self.obs, _, _, _ = self.env.step(np.array([[steer, speed]]))
        if self.obs["collisions"][0]:
            self.get_logger().warn("collision, resetting")
            self.obs, _, _, _ = self.env.reset(
                np.array([[0.7, 0.0, 1.37079632679]]))
        now = self.get_clock().now().to_msg()

        scan = LaserScan()
        scan.header.stamp = now
        scan.header.frame_id = "ego_racecar/laser"
        scan.angle_min = -2.35
        scan.angle_max = 2.35
        scan.angle_increment = 4.7 / 1079.0
        scan.range_min = 0.0
        scan.range_max = 30.0
        scan.ranges = [float(r) for r in self.obs["scans"][0]]
        self.scan_pub.publish(scan)

        od = Odometry()
        od.header.stamp = now
        od.header.frame_id = "map"
        od.child_frame_id = "ego_racecar/base_link"
        od.pose.pose.position.x = float(self.obs["poses_x"][0])
        od.pose.pose.position.y = float(self.obs["poses_y"][0])
        qx, qy, qz, qw = yaw_to_quat(self.obs["poses_theta"][0])
        od.pose.pose.orientation.x = qx
        od.pose.pose.orientation.y = qy
        od.pose.pose.orientation.z = qz
        od.pose.pose.orientation.w = qw
        od.twist.twist.linear.x = float(self.obs["linear_vels_x"][0])
        od.twist.twist.angular.z = float(self.obs["ang_vels_z"][0])
        self.odom_pub.publish(od)


def main(args=None):
    rclpy.init(args=args)
    node = SimBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
BRIDGEEOF

# ------------------------------------------------------- controller node base
cat > "$PKG/f1tenth_control/pure_pursuit_node.py" << 'PPEOF'
"""Pure pursuit as a ROS 2 node.

Subscribes /ego_racecar/odom, publishes /drive. Logs cross-track error so a
run can be scored from the terminal without touching the simulator.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from f1tenth_control.controllers import (
    PurePursuit, load_raceline, cross_track_error)


def quat_to_yaw(q):
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__("pure_pursuit")
        self.declare_parameter("lookahead", 1.0)
        self.declare_parameter("vgain", 0.6)

        path, psi, kappa, vref = load_raceline()
        self.path = path
        self.ctrl = PurePursuit(
            path, vref,
            lookahead=self.get_parameter("lookahead").value,
            vgain=self.get_parameter("vgain").value)

        self.pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.create_subscription(Odometry, "/ego_racecar/odom", self.on_odom, 10)
        self.errs = []
        self.create_timer(2.0, self.report)
        self.get_logger().info("pure pursuit up: Ld=%.2f vgain=%.2f" % (
            self.get_parameter("lookahead").value,
            self.get_parameter("vgain").value))

    def on_odom(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        v = msg.twist.twist.linear.x

        steer, speed = self.ctrl(x, y, yaw, v)
        self.errs.append(cross_track_error(np.array([x, y]), self.path))

        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.drive.steering_angle = steer
        out.drive.speed = speed
        self.pub.publish(out)

    def report(self):
        if self.errs:
            e = np.abs(self.errs[-200:])
            self.get_logger().info("CTE mean %.2f cm  max %.2f cm" % (
                e.mean() * 100, e.max() * 100))


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
PPEOF

cat > "$PKG/f1tenth_control/stanley_node.py" << 'STEOF'
"""Stanley controller as a ROS 2 node. Same interface as pure_pursuit_node."""

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from f1tenth_control.controllers import (
    Stanley, load_raceline, cross_track_error)


def quat_to_yaw(q):
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


class StanleyNode(Node):
    def __init__(self):
        super().__init__("stanley")
        self.declare_parameter("k", 3.5)
        self.declare_parameter("vgain", 0.6)

        path, psi, kappa, vref = load_raceline()
        self.path = path
        self.ctrl = Stanley(path, psi, vref,
                            k=self.get_parameter("k").value,
                            vgain=self.get_parameter("vgain").value)

        self.pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.create_subscription(Odometry, "/ego_racecar/odom", self.on_odom, 10)
        self.errs = []
        self.create_timer(2.0, self.report)
        self.get_logger().info("stanley up: k=%.2f vgain=%.2f" % (
            self.get_parameter("k").value,
            self.get_parameter("vgain").value))

    def on_odom(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        v = msg.twist.twist.linear.x

        steer, speed = self.ctrl(x, y, yaw, v)
        self.errs.append(cross_track_error(np.array([x, y]), self.path))

        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.drive.steering_angle = steer
        out.drive.speed = speed
        self.pub.publish(out)

    def report(self):
        if self.errs:
            e = np.abs(self.errs[-200:])
            self.get_logger().info("CTE mean %.2f cm  max %.2f cm" % (
                e.mean() * 100, e.max() * 100))


def main(args=None):
    rclpy.init(args=args)
    node = StanleyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
STEOF

# ------------------------------------------------------------------- launch
cat > "$PKG/launch/pure_pursuit.launch.py" << 'L1EOF'
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='f1tenth_control', executable='sim_bridge',
             name='sim_bridge', output='screen'),
        Node(package='f1tenth_control', executable='pure_pursuit',
             name='pure_pursuit', output='screen',
             parameters=[{'lookahead': 1.0, 'vgain': 0.6}]),
    ])
L1EOF

cat > "$PKG/launch/stanley.launch.py" << 'L2EOF'
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='f1tenth_control', executable='sim_bridge',
             name='sim_bridge', output='screen'),
        Node(package='f1tenth_control', executable='stanley',
             name='stanley', output='screen',
             parameters=[{'k': 3.5, 'vgain': 0.6}]),
    ])
L2EOF

# --------------------------------------------------------------------- run.sh
cat > "$ROOT/run.sh" << 'RUNEOF'
#!/usr/bin/env bash
# Build the image once, then drop into the container with the workspace mounted.
set -e
cd "$(dirname "$0")"
docker build -t f1tenth-ros2 .
docker run -it --rm -v "$(pwd)":/ws f1tenth-ros2 bash
RUNEOF
chmod +x "$ROOT/run.sh"

echo "created:"
find "$ROOT" -type f | sort
