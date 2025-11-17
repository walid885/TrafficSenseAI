import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    
    # 1. Get Package Paths
    pkg_robot_description = get_package_share_directory('robot_description')
    pkg_traffic_light_robot = get_package_share_directory('traffic_light_robot')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # 2. Configuration Files
    # Path to the World File
    world_file_path = os.path.join(pkg_robot_description, 'worlds', 'traffic_world.world')
    
    # Path to the URDF (Xacro)
    xacro_file = os.path.join(pkg_robot_description, 'urdf', 'robot.urdf.xacro')
    
    # Process the Xacro file to get the URDF XML
    robot_description_config = xacro.process_file(xacro_file)
    robot_desc = robot_description_config.toxml()

    # 3. Gazebo Server & Client (With World)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_file_path}.items()
    )

    # 4. Robot State Publisher (Required for TF)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[{'robot_description': robot_desc}]
    )

    # 5. Spawn Entity (Puts the robot in Gazebo)
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', 
                   '-entity', 'traffic_robot',
                   '-x', '0.0', '-y', '0.0', '-z', '0.2'], # Spawn slightly in the air
        output='screen'
    )

    # 6. Custom Nodes
    detector_node = Node(
        package='traffic_light_robot',
        executable='detector_node',
        name='traffic_light_detector',
        output='screen'
    )
    
    controller_node = Node(
        package='traffic_light_robot',
        executable='controller_node',
        name='autonomous_controller',
        output='screen'
    )

    # 7. Return Launch Description
    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        detector_node,
        controller_node
    ])