from setuptools import setup

package_name = 'traffic_light_robot'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='raspb',
    maintainer_email='raspb@todo.todo',
    description='Traffic light detection robot',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_node_v2 = traffic_light_robot.detector_node_v2:main',
            'controller_node = traffic_light_robot.controller_node:main',
            'visualizer_node = traffic_light_robot.visualizer_node:main',
            'debug_traffic = traffic_light_robot.debug_traffic:main',
        ],
    },
)