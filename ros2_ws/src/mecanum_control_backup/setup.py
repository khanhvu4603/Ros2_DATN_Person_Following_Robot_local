from setuptools import setup
import os
from glob import glob

package_name = 'mecanum_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    package_data={
        package_name: [
            'models/*',
            'sounds/*',
            'data/*',
        ],
    },
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orangepi',
    maintainer_email='orangepi@example.com',
    description='Mecanum control package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'imu_raw_publisher = mecanum_control.imu_raw_publisher:main',
            'lidar_processor   = mecanum_control.lidar_processor:main',
            'mecanum_controller= mecanum_control.mecanum_controller:main',
            'opencv_camera_node= mecanum_control.opencv_camera_node:main',
            'opencv_view_node  = mecanum_control.opencv_view_node:main',
            'person_detector   = mecanum_control.person_detector:main',
            'stm32_communicator= mecanum_control.stm32_communicator:main',
            'velocity_arbiter  = mecanum_control.velocity_arbiter:main',
            'dynamic_obstacle_tracker = mecanum_control.dynamic_obstacle_tracker:main',
            'manual_bridge     = mecanum_control.manual_bridge:main',
        ],
    },
)