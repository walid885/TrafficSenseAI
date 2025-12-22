#!/bin/bash

PROJECT_DIR="$HOME/Desktop/TrafficSenseAI"
export QT_QPA_PLATFORM=xcb

trap 'cleanup' INT TERM

cleanup() {
    echo "Shutting down Vision Tuner..."
    pkill -f "ros2 launch robot_description"
    pkill -f "ros2 run traffic_light_robot"
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

sleep 5

gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot hsv_tuner_node; exec bash" 2>/dev/null &

echo "Vision Tuner launched - Adjust trackbars in OpenCV window"
echo "Press Ctrl+C to shutdown"

wait