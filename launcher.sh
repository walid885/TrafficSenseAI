#!/bin/bash

# ROS2 Traffic Light Robot Launch Script

PROJECT_DIR="$HOME/Desktop/TrafficSenseAI"

# Build
cd "$PROJECT_DIR"
colcon build --packages-select traffic_light_robot
source install/setup.bash

# Launch main node in new terminal
gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 launch robot_description autonomous.launch.py; exec bash"

# Wait for launch to initialize
sleep 3

# Launch visualizer in new terminal
gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot visualizer_node; exec bash"

echo "Launch complete"