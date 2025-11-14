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
    maintainer='user',
    maintainer_email='bettaieb88walid@gmail.com',
    description='Traffic Light Detection Node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'traffic_detector = traffic_light_robot.traffic_detector:main',
        ],
    },
)
