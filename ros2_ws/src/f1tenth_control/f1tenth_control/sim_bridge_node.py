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
