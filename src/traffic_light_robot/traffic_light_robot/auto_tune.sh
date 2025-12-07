#!/bin/bash

PROJECT_DIR="$HOME/Desktop/TrafficSenseAI"
export QT_QPA_PLATFORM=xcb

# Create results directory
mkdir -p tuning_results

# PID parameter ranges
KP_VALUES=(0.3 0.5 0.7 0.9 1.1 1.3)
KI_VALUES=(0.05 0.1 0.15 0.2 0.25 0.3)
KD_VALUES=(0.01 0.02 0.03 0.05 0.07)

# Or use grid search (smaller for testing)
# KP_VALUES=(0.5 0.8 1.1)
# KI_VALUES=(0.1 0.15 0.2)
# KD_VALUES=(0.02 0.03)

ITERATION=0
TOTAL_TESTS=$((${#KP_VALUES[@]} * ${#KI_VALUES[@]} * ${#KD_VALUES[@]}))

echo "=== PID Auto-Tuning Script ==="
echo "Testing $TOTAL_TESTS combinations"
echo "Kp values: ${KP_VALUES[@]}"
echo "Ki values: ${KI_VALUES[@]}"
echo "Kd values: ${KD_VALUES[@]}"
echo "=============================="

cleanup() {
    echo "Cleaning up processes..."
    pkill -f "ros2 launch robot_description"
    pkill -f "ros2 run traffic_light_robot"
    pkill -f "gzserver"
    pkill -f "gzclient"
    sleep 3
}

start_simulation() {
    echo "Starting Gazebo simulation..."
    cd "$PROJECT_DIR"
    source install/setup.bash
    
    ros2 launch robot_description autonomous.launch.py > /dev/null 2>&1 &
    LAUNCH_PID=$!
    
    sleep 8  # Wait for Gazebo to fully load
    echo "Simulation ready"
}

run_tuning_test() {
    local kp=$1
    local ki=$2
    local kd=$3
    local iter=$4
    
    echo "[$iter/$TOTAL_TESTS] Testing: Kp=$kp, Ki=$ki, Kd=$kd"
    
    cd "$PROJECT_DIR"
    source install/setup.bash
    
    timeout 35s ros2 run traffic_light_robot pid_tuner $kp $ki $kd $iter
    
    sleep 2
}

# Main loop
cleanup
start_simulation

for kp in "${KP_VALUES[@]}"; do
    for ki in "${KI_VALUES[@]}"; do
        for kd in "${KD_VALUES[@]}"; do
            ITERATION=$((ITERATION + 1))
            run_tuning_test $kp $ki $kd $ITERATION
        done
    done
done

cleanup

echo ""
echo "=== Tuning Complete ==="
echo "Analyzing results..."

# Generate summary report
python3 << 'EOF'
import json
import glob
import numpy as np

results = []
for file in glob.glob('tuning_results/iteration_*.json'):
    with open(file) as f:
        results.append(json.load(f))

results.sort(key=lambda x: x['score'])

print("\n=== TOP 10 PID CONFIGURATIONS ===\n")
print(f"{'Rank':<5} {'Kp':<7} {'Ki':<7} {'Kd':<7} {'Score':<10} {'MAE':<10} {'RMSE':<10} {'Variance':<12}")
print("-" * 80)

for i, r in enumerate(results[:10], 1):
    print(f"{i:<5} {r['kp']:<7.3f} {r['ki']:<7.3f} {r['kd']:<7.3f} "
          f"{r['score']:<10.2f} {r['mae']:<10.4f} {r['rmse']:<10.4f} {r['speed_variance']:<12.6f}")

print("\n=== BEST CONFIGURATION ===")
best = results[0]
print(f"Kp = {best['kp']:.3f}")
print(f"Ki = {best['ki']:.3f}")
print(f"Kd = {best['kd']:.3f}")
print(f"\nMetrics:")
print(f"  Score:         {best['score']:.2f}")
print(f"  MAE:           {best['mae']:.4f} m/s")
print(f"  RMSE:          {best['rmse']:.4f} m/s")
print(f"  Speed Var:     {best['speed_variance']:.6f}")
print(f"  Overshoot:     {best['overshoot']:.4f} m/s")
print(f"  Settling Time: {best['settling_time']:.2f}s" if best['settling_time'] else "  Settling Time: N/A")

# Save best config
with open('tuning_results/BEST_CONFIG.json', 'w') as f:
    json.dump(best, f, indent=2)

print("\nResults saved in tuning_results/")
print("Best config saved: tuning_results/BEST_CONFIG.json")
EOF

echo ""
echo "All plots saved in tuning_results/"
echo "Review plots and use best configuration in autonomous_controller.py"