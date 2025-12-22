#!/bin/bash

PROJECT_DIR="$HOME/Desktop/TrafficSenseAI"
export QT_QPA_PLATFORM=xcb

trap 'pkill -f "ros2"; exit 0' INT TERM

cd "$PROJECT_DIR"
colcon build --packages-select traffic_light_robot
source install/setup.bash

# Launch simulation
gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 launch robot_description autonomous.launch.py; exec bash" &

sleep 10

# Launch debug node ONLY
gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot camera_debug; exec bash" &

echo "Camera Debug Running - Check terminal output and window"
echo "If percentages stay at 0.000%, robot cannot see lights"
echo "Press Ctrl+C to exit"

wait