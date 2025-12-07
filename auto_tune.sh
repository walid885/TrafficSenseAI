#!/bin/bash

PROJECT_DIR="$HOME/Desktop/TrafficSenseAI"
export QT_QPA_PLATFORM=xcb

mkdir -p "$PROJECT_DIR/tuning_results"

# PID parameter ranges
KP_VALUES=(0.3 0.5 0.7 0.9 1.1 1.3)
KI_VALUES=(0.05 0.1 0.15 0.2 0.25 0.3)
KD_VALUES=(0.01 0.02 0.03 0.05 0.07)

ITERATION=0
TOTAL_TESTS=$((${#KP_VALUES[@]} * ${#KI_VALUES[@]} * ${#KD_VALUES[@]}))
PARALLEL_SIMS=2

echo "=== PID Auto-Tuning Script ==="
echo "Testing $TOTAL_TESTS combinations (2 parallel simulations)"
echo "Kp values: ${KP_VALUES[@]}"
echo "Ki values: ${KI_VALUES[@]}"
echo "Kd values: ${KD_VALUES[@]}"
echo "=============================="

cleanup_sim() {
    local master_uri=$1
    pkill -f "ROS_MASTER_URI=$master_uri"
    pkill -f "GAZEBO_MASTER_URI=$master_uri"
    sleep 1
}

cleanup_all() {
    echo "Cleaning up all processes..."
    pkill -f "ros2 launch robot_description"
    pkill -f "ros2 run traffic_light_robot"
    pkill -f "gzserver"
    pkill -f "gzclient"
    sleep 3
    # Kill any zombie processes
    pkill -9 -f "gzserver" 2>/dev/null
    pkill -9 -f "pid_tuner" 2>/dev/null
}

trap cleanup_all INT TERM EXIT

echo "Building project..."
cd "$PROJECT_DIR"
colcon build --packages-select traffic_light_robot
source install/setup.bash

start_simulation() {
    local port=$1
    local sim_id=$2
    
    export ROS_DOMAIN_ID=$((port))
    export GAZEBO_MASTER_URI=http://localhost:$((11345 + port))
    
    cd "$PROJECT_DIR"
    source install/setup.bash
    
    gnome-terminal --title="Sim-$sim_id" -- bash -c "
        export ROS_DOMAIN_ID=$port
        export GAZEBO_MASTER_URI=http://localhost:$((11345 + port))
        cd $PROJECT_DIR
        source install/setup.bash
        ros2 launch robot_description autonomous.launch.py
    " 2>/dev/null &
    
    echo "Started simulation $sim_id (domain $port)"
    sleep 12
}

run_tuning_test() {
    local kp=$1
    local ki=$2
    local kd=$3
    local iter=$4
    local domain=$5
    
    export ROS_DOMAIN_ID=$domain
    
    cd "$PROJECT_DIR"
    source install/setup.bash
    
    timeout 40s ros2 run traffic_light_robot pid_tuner $kp $ki $kd $iter &
    local pid=$!
    
    wait $pid
    local status=$?
    
    if [ $status -eq 124 ]; then
        echo "  [$iter] Timed out"
    fi
    
    # Kill any lingering processes
    pkill -P $pid 2>/dev/null
    
    return $status
}

# Parallel execution function
run_parallel_tests() {
    local test_queue=("$@")
    local active_jobs=0
    local max_jobs=$PARALLEL_SIMS
    local domain_base=10
    
    # Start initial simulations
    for ((i=0; i<$max_jobs; i++)); do
        start_simulation $((domain_base + i)) $((i+1))
    done
    
    local queue_idx=0
    
    while [ $queue_idx -lt ${#test_queue[@]} ]; do
        if [ $active_jobs -lt $max_jobs ]; then
            local test_params=${test_queue[$queue_idx]}
            IFS=',' read -r kp ki kd iter <<< "$test_params"
            
            local domain=$((domain_base + active_jobs))
            
            echo "[$iter/$TOTAL_TESTS] Testing: Kp=$kp, Ki=$ki, Kd=$kd (Domain $domain)"
            
            (
                run_tuning_test $kp $ki $kd $iter $domain
                echo "DONE:$iter" >> /tmp/tuning_done_$$
            ) &
            
            ((active_jobs++))
            ((queue_idx++))
            sleep 1
        else
            # Wait for any job to complete
            wait -n
            ((active_jobs--))
        fi
    done
    
    # Wait for remaining jobs
    wait
    
    rm -f /tmp/tuning_done_$$
}

# Build test queue
test_queue=()
for kp in "${KP_VALUES[@]}"; do
    for ki in "${KI_VALUES[@]}"; do
        for kd in "${KD_VALUES[@]}"; do
            ITERATION=$((ITERATION + 1))
            test_queue+=("$kp,$ki,$kd,$ITERATION")
        done
    done
done

cleanup_all
run_parallel_tests "${test_queue[@]}"
cleanup_all

echo ""
echo "=== Tuning Complete ==="
echo "Analyzing results..."

cd "$PROJECT_DIR"
python3 << 'EOF'
import json
import glob
import numpy as np
import os

os.chdir('tuning_results')

results = []
for file in glob.glob('iteration_*.json'):
    try:
        with open(file) as f:
            results.append(json.load(f))
    except:
        pass

if not results:
    print("No results found!")
    exit(1)

results.sort(key=lambda x: x['score'])

print("\n=== TOP 10 PID CONFIGURATIONS ===\n")
print(f"{'Rank':<5} {'Kp':<7} {'Ki':<7} {'Kd':<7} {'Score':<10} {'MAE':<10} {'RMSE':<10} {'Variance':<12}")
print("-" * 80)

for i, r in enumerate(results[:min(10, len(results))], 1):
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
print(f"  Settling Time: {best['settling_time']:.2f}s" if best.get('settling_time') else "  Settling Time: N/A")

with open('BEST_CONFIG.json', 'w') as f:
    json.dump(best, f, indent=2)

print("\nResults: tuning_results/")
print("Best config: tuning_results/BEST_CONFIG.json")
print(f"Total successful tests: {len(results)}/{len(glob.glob('iteration_*.json'))}")
EOF

echo "Complete. Review tuning_results/ for plots and data."