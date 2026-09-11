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
