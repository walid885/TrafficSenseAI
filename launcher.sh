#!/bin/bash

PROJECT_DIR="$HOME/Desktop/TrafficSenseAI"

cd "$PROJECT_DIR"
colcon build --packages-select traffic_light_robot
source install/setup.bash

# Launch main with visible output
ros2 launch robot_description autonomous.launch.py &
LAUNCH_PID=$!

sleep 5

# Check if topics exist
if ! ros2 topic list | grep -q "/front_camera"; then
    echo "ERROR: Main launch failed - no camera topic"
    kill $LAUNCH_PID 2>/dev/null
    exit 1
fi

gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot visualizer_node; exec bash"

sleep 2

gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && rviz2; exec bash"

sleep 2

gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot rviz_visu; exec bash"

echo "Launch complete"