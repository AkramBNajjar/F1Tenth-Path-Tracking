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
