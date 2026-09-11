#!/usr/bin/env bash
# Adds a visualization node to f1tenth_control.
# Run from the repo root:   bash add_viz.sh
# Then rebuild inside the container:  colcon build --symlink-install
set -e

PKG="ros2_ws/src/f1tenth_control"
[ -d "$PKG" ] || { echo "run this from the repo root (ros2_ws/ not found)"; exit 1; }

cat > "$PKG/f1tenth_control/viz_node.py" << 'VIZEOF'
"""Publishes everything needed to see what the controller is doing.

    /map           nav_msgs/OccupancyGrid        the track, from the .png
    /raceline      nav_msgs/Path                 the reference path
    /car           visualization_msgs/Marker     vehicle body
    /goal_point    visualization_msgs/Marker     pure pursuit target
    /lookahead     visualization_msgs/Marker     line from car to target
    /trail         visualization_msgs/Marker     where the car has driven

The goal point marker is the useful one. Pure pursuit steers at a point a
fixed distance ahead on the path, and watching that point slide around
corners ahead of the car is the clearest possible explanation of why the
controller cuts corners.
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
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from f1tenth_control.controllers import load_raceline, WHEELBASE


def quat_to_yaw(q):
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


def rgba(r, g, b, a=1.0):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


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

        # latched QoS so late subscribers still receive the map
        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             history=QoSHistoryPolicy.KEEP_LAST)
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", latched)
        self.path_pub = self.create_publisher(Path, "/raceline", latched)
        self.car_pub = self.create_publisher(Marker, "/car", 10)
        self.goal_pub = self.create_publisher(Marker, "/goal_point", 10)
        self.look_pub = self.create_publisher(Marker, "/lookahead", 10)
        self.trail_pub = self.create_publisher(Marker, "/trail", 10)

        from nav_msgs.msg import Odometry
        self.create_subscription(Odometry, "/ego_racecar/odom", self.on_odom, 10)

        self.publish_map()
        self.publish_raceline()
        self.create_timer(5.0, self.publish_map)
        self.create_timer(5.0, self.publish_raceline)
        self.get_logger().info("viz up: /map /raceline /car /goal_point /lookahead /trail")

    # ------------------------------------------------------------------ map
    def publish_map(self):
        name = self.get_parameter("map_name").value
        with open(os.path.join(self.examples, name + ".yaml")) as f:
            meta = yaml.safe_load(f)
        img = Image.open(os.path.join(self.examples, name + ".png")).convert("L")
        a = np.array(img)
        # ROS expects row 0 at the bottom; images store row 0 at the top
        a = np.flipud(a)
        occ = np.full(a.shape, -1, dtype=np.int8)
        occ[a > meta["free_thresh"] * 255] = 0          # free
        occ[a < meta["occupied_thresh"] * 255] = 100    # wall

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

    # ------------------------------------------------------------- raceline
    def publish_raceline(self):
        p = Path()
        p.header.frame_id = "map"
        p.header.stamp = self.get_clock().now().to_msg()
        for (x, y), yaw in zip(self.path, self.psi):
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.z = float(np.sin(yaw / 2))
            ps.pose.orientation.w = float(np.cos(yaw / 2))
            p.poses.append(ps)
        self.path_pub.publish(p)

    # ----------------------------------------------------------------- live
    def on_odom(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        now = self.get_clock().now().to_msg()

        car = Marker()
        car.header.frame_id = "map"
        car.header.stamp = now
        car.ns = "car"
        car.id = 0
        car.type = Marker.CUBE
        car.action = Marker.ADD
        car.pose.position.x = x + 0.5 * WHEELBASE * np.cos(yaw)
        car.pose.position.y = y + 0.5 * WHEELBASE * np.sin(yaw)
        car.pose.position.z = 0.05
        car.pose.orientation.z = float(np.sin(yaw / 2))
        car.pose.orientation.w = float(np.cos(yaw / 2))
        car.scale.x, car.scale.y, car.scale.z = 0.58, 0.31, 0.12
        car.color = rgba(0.13, 0.40, 0.65)
        self.car_pub.publish(car)

        # pure pursuit goal point, recomputed here so viz stays decoupled
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
        goal.ns = "goal"
        goal.id = 1
        goal.type = Marker.SPHERE
        goal.action = Marker.ADD
        goal.pose.position.x = float(gx)
        goal.pose.position.y = float(gy)
        goal.pose.position.z = 0.1
        goal.pose.orientation.w = 1.0
        goal.scale.x = goal.scale.y = goal.scale.z = 0.28
        goal.color = rgba(0.95, 0.55, 0.10)
        self.goal_pub.publish(goal)

        line = Marker()
        line.header.frame_id = "map"
        line.header.stamp = now
        line.ns = "lookahead"
        line.id = 2
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.04
        line.color = rgba(0.95, 0.55, 0.10, 0.8)
        line.pose.orientation.w = 1.0
        a_, b_ = Point(), Point()
        a_.x, a_.y, a_.z = float(x), float(y), 0.08
        b_.x, b_.y, b_.z = float(gx), float(gy), 0.08
        line.points = [a_, b_]
        self.look_pub.publish(line)

        self.trail.append((x, y))
        if len(self.trail) > 4000:
            self.trail = self.trail[-4000:]
        tr = Marker()
        tr.header.frame_id = "map"
        tr.header.stamp = now
        tr.ns = "trail"
        tr.id = 3
        tr.type = Marker.LINE_STRIP
        tr.action = Marker.ADD
        tr.scale.x = 0.05
        tr.color = rgba(0.85, 0.20, 0.15, 0.9)
        tr.pose.orientation.w = 1.0
        tr.points = []
        for tx, ty in self.trail[::4]:
            pt = Point()
            pt.x, pt.y, pt.z = float(tx), float(ty), 0.03
            tr.points.append(pt)
        self.trail_pub.publish(tr)


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

# register the entry point
python3 - << 'PYEOF'
p = "ros2_ws/src/f1tenth_control/setup.py"
s = open(p).read()
if "viz = " not in s:
    s = s.replace("'sim_bridge = f1tenth_control.sim_bridge_node:main',",
                  "'sim_bridge = f1tenth_control.sim_bridge_node:main',\n            'viz = f1tenth_control.viz_node:main',")
    open(p, "w").write(s)
    print("setup.py: added viz entry point")
else:
    print("setup.py: viz already registered")
PYEOF

# dependencies
python3 - << 'PYEOF'
p = "ros2_ws/src/f1tenth_control/package.xml"
s = open(p).read()
for dep in ("visualization_msgs", "geometry_msgs", "std_msgs"):
    if dep not in s:
        s = s.replace("  <exec_depend>ackermann_msgs</exec_depend>",
                      "  <exec_depend>ackermann_msgs</exec_depend>\n  <exec_depend>%s</exec_depend>" % dep)
open(p, "w").write(s)
print("package.xml: deps ensured")
PYEOF

# launch file with viz included
cat > "ros2_ws/src/f1tenth_control/launch/pure_pursuit_viz.launch.py" << 'LEOF'
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='f1tenth_control', executable='sim_bridge',
             name='sim_bridge', output='screen'),
        Node(package='f1tenth_control', executable='pure_pursuit',
             name='pure_pursuit', output='screen',
             parameters=[{'lookahead': 1.0, 'vgain': 0.6}]),
        Node(package='f1tenth_control', executable='viz',
             name='viz', output='screen',
             parameters=[{'lookahead': 1.0}]),
    ])
LEOF

echo "done. rebuild inside the container:"
echo "  cd /ws && colcon build --symlink-install && source install/setup.bash"
echo "  ros2 launch f1tenth_control pure_pursuit_viz.launch.py"
