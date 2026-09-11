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
            'viz = f1tenth_control.viz_node:main',
            'pure_pursuit = f1tenth_control.pure_pursuit_node:main',
            'stanley = f1tenth_control.stanley_node:main',
        ],
    },
)
