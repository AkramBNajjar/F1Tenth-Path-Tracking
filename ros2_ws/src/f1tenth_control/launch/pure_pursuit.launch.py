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
