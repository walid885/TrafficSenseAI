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

# Launch simulation
gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 launch robot_description autonomous.launch.py; exec bash" 2>/dev/null &

sleep 8

# Launch detector_node_v2 (publishes /traffic_light_state properly)
gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot traffic_light_detector_v2; exec bash" 2>/dev/null &

sleep 3

# Launch auto-tuner
gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot hsv_auto_tuner 2>&1 | tee hsv_tuning_output.log; exec bash" 2>/dev/null &

echo "═══════════════════════════════════════════════════════════"
echo "HSV AUTO-TUNER ACTIVE"
echo "═══════════════════════════════════════════════════════════"
echo "Using detector_node_v2 for proper state publishing"
echo "Collecting 50 frames per state (RED, YELLOW, GREEN)"
echo "Check diagnostics output every 2 seconds"
echo "Press Ctrl+C to shutdown"
echo "═══════════════════════════════════════════════════════════"

wait