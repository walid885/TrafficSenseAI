from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('robot_description'),
                        'launch', 'gazebo.launch.py')
        ])
    )
    
    detector = Node(
        package='traffic_light_robot',
        executable='detector_node',
        output='screen'
    )
    
    controller = Node(
        package='traffic_light_robot',
        executable='controller_node',
        output='screen'
    )
    
    return LaunchDescription([
        gazebo_launch,
        detector,
        controller
    ])