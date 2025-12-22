#!/bin/bash

PROJECT_DIR="$HOME/Desktop/TrafficSenseAI"
export QT_QPA_PLATFORM=xcb

trap 'cleanup' INT TERM

cleanup() {
    echo "Shutting down Auto-Tuner..."
    pkill -f "ros2 launch robot_description"
    pkill -f "ros2 run traffic_light_robot"
    pkill -f "gzserver"
    pkill -f "gzclient"
    sleep 2
    
    # Display results if they exist
    if [ -f "$PROJECT_DIR/hsv_optimized_params.json" ]; then
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "TUNING RESULTS:"
        echo "═══════════════════════════════════════════════════════════"
        cat "$PROJECT_DIR/hsv_optimized_params.json"
        echo ""
        echo "═══════════════════════════════════════════════════════════"
    fi
    
    echo "Shutdown complete"
    exit 0
}

cd "$PROJECT_DIR"
colcon build --packages-select traffic_light_robot
source install/setup.bash

gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 launch robot_description autonomous.launch.py; exec bash" 2>/dev/null &

sleep 8

gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot detector_node; exec bash" 2>/dev/null &

sleep 3

gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot hsv_auto_tuner 2>&1 | tee hsv_tuning_output.log; exec bash" 2>/dev/null &
TUNER_PID=$!

echo "═══════════════════════════════════════════════════════════"
echo "HSV AUTO-TUNER ACTIVE"
echo "═══════════════════════════════════════════════════════════"
echo "Collecting 50 frames per state (RED, YELLOW, GREEN)"
echo "Process will auto-complete after RED detection and optimization"
echo "Output logged to: hsv_tuning_output.log"
echo "Press Ctrl+C to shutdown manually"
echo "═══════════════════════════════════════════════════════════"

wait