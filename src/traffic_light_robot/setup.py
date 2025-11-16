from setuptools import setup

package_name = 'traffic_light_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='raspb',
    maintainer_email='bettaieb.walid1@gmail.com',
    description='Traffic light detection',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_node = traffic_light_robot.detector_node:main',
            'controller_node = traffic_light_robot.controller_node:main',
        ],
    },
)