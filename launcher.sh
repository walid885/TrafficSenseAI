#!/bin/bash

PROJECT_DIR="$HOME/Desktop/TrafficSenseAI"
export QT_QPA_PLATFORM=xcb

trap 'cleanup' INT TERM

cleanup() {
    echo "=== SHUTDOWN ==="
    pkill -f "ros2"
    pkill -f "gzserver"
    pkill -f "gzclient"
    sleep 2
    exit 0
}

cd "$PROJECT_DIR"

echo "=== BUILDING ==="
colcon build --packages-select traffic_light_robot
source install/setup.bash

echo "=== LAUNCHING GAZEBO ==="
gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 launch robot_description autonomous.launch.py; exec bash" &
sleep 8

echo "=== LAUNCHING DETECTOR ==="
gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot detector_node_v2; exec bash" &
sleep 2

echo "=== LAUNCHING CONTROLLER ==="
gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot controller_node; exec bash" &
sleep 2

echo "=== LAUNCHING DEBUG ==="
gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot debug_traffic; exec bash" &
sleep 2

echo "=== LAUNCHING VISUALIZER ==="
gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot visualizer_node; exec bash" &

echo ""
echo "=== ALL NODES LAUNCHED ==="
echo "Watch debug terminal for: LIGHT: state | CMD_VEL: speed"
echo "Press Ctrl+C to shutdown"
echo ""

wait