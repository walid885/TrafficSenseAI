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
    
    if [ -f "$PROJECT_DIR/hsv_optimized_params.json" ]; then
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "TUNED HSV PARAMETERS:"
        echo "═══════════════════════════════════════════════════════════"
        cat "$PROJECT_DIR/hsv_optimized_params.json"
        echo ""
        echo "═══════════════════════════════════════════════════════════"
    fi
    
    exit 0
}

cd "$PROJECT_DIR"

echo "=== BUILDING ==="
colcon build --packages-select traffic_light_robot
source install/setup.bash

echo "=== LAUNCHING GAZEBO ==="
gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 launch robot_description autonomous.launch.py; exec bash" &
sleep 8

echo "=== LAUNCHING AUTO CALIBRATOR ==="
gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot interactive_hsv_tuner; exec bash" &

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "AUTO HSV CALIBRATOR LAUNCHED"
echo "═══════════════════════════════════════════════════════════"
echo "Collecting 100 calibration frames..."
echo "Press Q to quit after calibration"
echo "═══════════════════════════════════════════════════════════"

wait