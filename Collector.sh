#!/bin/bash
# Integrated data collection for TrafficSenseAI

PROJECT_DIR="$HOME/Desktop/TrafficSenseAI"
DATASET_DIR="${PROJECT_DIR}/yolo_dataset"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"

NUM_SAMPLES=600
SAMPLE_RATE=3
DRIVE_DURATION=240  # 4 minutes

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

cleanup() {
    log "Cleaning up..."
    pkill -f "dataset_collector_fixed"
    pkill -f "auto_robot_driver"
    pkill -f "ros2 launch robot_description"
    pkill -f "ros2 run traffic_light_robot"
    pkill -f "gzserver"
    pkill -f "gzclient"
    sleep 2
}

trap cleanup EXIT INT TERM

check_scripts() {
    log "Checking required scripts..."
    
    if [ ! -f "${SCRIPTS_DIR}/auto_dataset_collector.py" ]; then
        error "dataset_collector_fixed.py not found in ${SCRIPTS_DIR}"
    fi
    
    if [ ! -f "${SCRIPTS_DIR}/auto_robot_driver.py" ]; then
        error "auto_robot_driver.py not found in ${SCRIPTS_DIR}"
    fi
    
    chmod +x "${SCRIPTS_DIR}"/*.py
    log "Scripts OK ✓"
}

launch_gazebo() {
    log "Launching Gazebo simulation..."
    
    cd "${PROJECT_DIR}"
    colcon build --packages-select traffic_light_robot 2>&1 | grep -i "error" && error "Build failed"
    source install/setup.bash
    
    gnome-terminal -- bash -c "cd ${PROJECT_DIR} && source install/setup.bash && ros2 launch robot_description autonomous.launch.py; exec bash" 2>/dev/null &
    GAZEBO_PID=$!
    
    log "Waiting for Gazebo (20s)..."
    sleep 20
    
    if ! pgrep -x "gzserver" > /dev/null; then
        error "Gazebo failed to start"
    fi
    
    log "Gazebo running ✓"
}

verify_topics() {
    log "Verifying ROS2 topics..."
    
    local timeout=30
    local elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        if ros2 topic list 2>/dev/null | grep -q "/front_camera/image_raw"; then
            if ros2 topic list 2>/dev/null | grep -q "/odom"; then
                log "Topics available ✓"
                return 0
            fi
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    
    error "Required topics not found after ${timeout}s"
}

setup_dirs() {
    log "Setting up directories..."
    mkdir -p "${DATASET_DIR}"/{images,labels}/{train,val}
    mkdir -p "${SCRIPTS_DIR}"
    log "Directories ready ✓"
}

collect_data() {
    log "Starting collection..."
    log "Target: ${NUM_SAMPLES} samples"
    
    cd "${PROJECT_DIR}"
    source install/setup.bash
    
    # Start collector
    python3 "${SCRIPTS_DIR}/auto_dataset_collector.py" \
        "${DATASET_DIR}" "${NUM_SAMPLES}" "${SAMPLE_RATE}" &
    COLLECTOR_PID=$!
    
    sleep 3
    
    if ! ps -p $COLLECTOR_PID > /dev/null; then
        error "Collector failed to start"
    fi
    
    log "Collector running (PID: ${COLLECTOR_PID})"
    
    # Start driver
    python3 "${SCRIPTS_DIR}/auto_robot_driver.py" "${DRIVE_DURATION}" &
    DRIVER_PID=$!
    
    log "Robot driver started (PID: ${DRIVER_PID})"
    log "Collection in progress... Press Ctrl+C to stop"
    
    # Monitor progress
    while ps -p $COLLECTOR_PID > /dev/null 2>&1 && ps -p $DRIVER_PID > /dev/null 2>&1; do
        sleep 5
    done
    
    wait $COLLECTOR_PID 2>/dev/null || true
    wait $DRIVER_PID 2>/dev/null || true
    
    local collected=$(find "${DATASET_DIR}/images/train" -name "*.jpg" 2>/dev/null | wc -l)
    
    if [ $collected -eq 0 ]; then
        error "No images collected! Check logs above."
    fi
    
    log "Collection complete: ${collected} images ✓"
}

split_dataset() {
    local total=$(find "${DATASET_DIR}/images/train" -name "*.jpg" 2>/dev/null | wc -l)
    
    if [ $total -lt 20 ]; then
        warn "Only ${total} images. Skipping split."
        return
    fi
    
    log "Splitting dataset (80/20)..."
    
    cd "${DATASET_DIR}/images/train"
    
    # Shuffle and split
    local files=($(ls *.jpg | shuf))
    local val_count=$((total * 20 / 100))
    
    for ((i=0; i<val_count; i++)); do
        local img="${files[$i]}"
        local lbl="${img%.jpg}.txt"
        
        mv "$img" "../val/" 2>/dev/null
        mv "../../labels/train/$lbl" "../../labels/val/" 2>/dev/null
    done
    
    local train_final=$(ls "${DATASET_DIR}/images/train"/*.jpg 2>/dev/null | wc -l)
    local val_final=$(ls "${DATASET_DIR}/images/val"/*.jpg 2>/dev/null | wc -l)
    
    log "Split: ${train_final} train, ${val_final} val ✓"
}

create_data_yaml() {
    log "Creating data.yaml..."
    
    cat > "${DATASET_DIR}/data.yaml" << EOF
path: ${DATASET_DIR}
train: images/train
val: images/val

nc: 3
names: ['red', 'yellow', 'green']
EOF
    
    log "data.yaml created ✓"
}

show_summary() {
    echo
    echo "========================================"
    echo "  Collection Summary"
    echo "========================================"
    
    local train_imgs=$(ls "${DATASET_DIR}/images/train"/*.jpg 2>/dev/null | wc -l)
    local val_imgs=$(ls "${DATASET_DIR}/images/val"/*.jpg 2>/dev/null | wc -l)
    local total=$((train_imgs + val_imgs))
    
    echo "Total images: ${total}"
    echo "  - Training: ${train_imgs}"
    echo "  - Validation: ${val_imgs}"
    echo
    echo "Dataset location: ${DATASET_DIR}"
    echo "Config file: ${DATASET_DIR}/data.yaml"
    echo
    echo "Next steps:"
    echo "1. Inspect samples: ls ${DATASET_DIR}/images/train/*.jpg"
    echo "2. Train with YOLOv8: yolo train data=${DATASET_DIR}/data.yaml model=yolov8n.pt"
    echo "3. Or upload to Google Colab for faster training"
    echo "========================================"
}

main() {
    echo "======================================"
    echo "  Traffic Light Dataset Collection"
    echo "======================================"
    echo
    
    check_scripts
    setup_dirs
    launch_gazebo
    verify_topics
    collect_data
    split_dataset
    create_data_yaml
    show_summary
    
    cleanup
    log "Done!"
}

main