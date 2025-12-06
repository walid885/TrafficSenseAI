#!/bin/bash

PROJECT_DIR="$HOME/Desktop/TrafficSenseAI"
export QT_QPA_PLATFORM=xcb

# Trap Ctrl+C
trap 'cleanup' INT TERM

cleanup() {
    echo "Shutting down..."
    pkill -f "ros2 launch robot_description"
    pkill -f "ros2 run traffic_light_robot"
    pkill -f "rviz2"
    pkill -f "gzserver"
    pkill -f "gzclient"
    sleep 2
    echo "Shutdown complete"
    exit 0
}

cd "$PROJECT_DIR"
colcon build --packages-select traffic_light_robot
source install/setup.bash

gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 launch robot_description autonomous.launch.py; exec bash" 2>/dev/null &
LAUNCH_PID=$!

sleep 3

gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot visualizer_node; exec bash" 2>/dev/null &

sleep 2

gnome-terminal -- bash -c "export QT_QPA_PLATFORM=xcb && cd $PROJECT_DIR && source install/setup.bash && rviz2; exec bash" 2>/dev/null &

sleep 1

#gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot rviz_visu; exec bash" 2>/dev/null &

#echo "Launch complete - Press Ctrl+C to shutdown all nodes"

# Keep script running
wait