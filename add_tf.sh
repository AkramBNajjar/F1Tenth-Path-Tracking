#!/usr/bin/env bash
# Adds TF broadcasting to the sim bridge.
#
# Without this the bridge publishes odometry but no coordinate transforms, so
# Foxglove has no frame tree: follow mode has nothing to follow, and /scan
# (published in ego_racecar/laser) cannot be drawn in the map frame.
#
# Run from the repo root:  bash add_tf.sh
# Then rebuild in the container: colcon build --symlink-install
set -e

F="ros2_ws/src/f1tenth_control/f1tenth_control/sim_bridge_node.py"
[ -f "$F" ] || { echo "run from the repo root (sim_bridge_node.py not found)"; exit 1; }

python3 - << 'PYEOF'
p = "ros2_ws/src/f1tenth_control/f1tenth_control/sim_bridge_node.py"
s = open(p).read()

if "TransformBroadcaster" in s:
    print("TF already present, nothing to do")
    raise SystemExit(0)

# imports
s = s.replace(
"from ackermann_msgs.msg import AckermannDriveStamped",
"from ackermann_msgs.msg import AckermannDriveStamped\n"
"from geometry_msgs.msg import TransformStamped\n"
"from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster")

# set up broadcasters in __init__
s = s.replace(
'        rate = self.get_parameter("rate_hz").value',
'''        # TF tree: map -> ego_racecar/base_link -> ego_racecar/laser
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        st = TransformStamped()
        st.header.stamp = self.get_clock().now().to_msg()
        st.header.frame_id = "ego_racecar/base_link"
        st.child_frame_id = "ego_racecar/laser"
        st.transform.translation.x = 0.27
        st.transform.rotation.w = 1.0
        self.static_tf.sendTransform(st)

        rate = self.get_parameter("rate_hz").value''')

# broadcast the moving transform every tick
s = s.replace(
'        self.odom_pub.publish(od)',
'''        self.odom_pub.publish(od)

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "map"
        t.child_frame_id = "ego_racecar/base_link"
        t.transform.translation.x = float(self.obs["poses_x"][0])
        t.transform.translation.y = float(self.obs["poses_y"][0])
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf.sendTransform(t)''')

open(p, "w").write(s)
print("sim_bridge_node.py: TF broadcasting added")
PYEOF

python3 - << 'PYEOF'
p = "ros2_ws/src/f1tenth_control/package.xml"
s = open(p).read()
if "tf2_ros" not in s:
    s = s.replace("  <exec_depend>ackermann_msgs</exec_depend>",
                  "  <exec_depend>ackermann_msgs</exec_depend>\n  <exec_depend>tf2_ros</exec_depend>")
    open(p, "w").write(s)
    print("package.xml: tf2_ros added")
else:
    print("package.xml: tf2_ros already there")
PYEOF

python3 -m py_compile "$F" && echo "compiles OK"
echo
echo "rebuild in the container:"
echo "  cd /ws && colcon build --symlink-install && source install/setup.bash"
echo "  ros2 launch f1tenth_control pure_pursuit_viz.launch.py"
