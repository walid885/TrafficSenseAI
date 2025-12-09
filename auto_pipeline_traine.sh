#!/bin/bash
# auto_yolo_pipeline.sh - Automated YOLO Training Pipeline for Traffic Light Detection
# Usage: ./auto_yolo_pipeline.sh [collect|train|deploy|full]

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ROOT="${HOME}/Desktop/TrafficSenseAI"
DATASET_DIR="${PROJECT_ROOT}/yolo_dataset"
MODELS_DIR="${PROJECT_ROOT}/models"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
WORKSPACE="${PROJECT_ROOT}"

# Collection parameters
NUM_SAMPLES=1200
SAMPLE_RATE=3  # Collect every Nth frame
COLLECTION_DURATION=300  # 5 minutes max

# Training parameters
BATCH_SIZE=64
SUBDIVISIONS=16
MAX_BATCHES=6000
TRAIN_SPLIT=0.8

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

check_dependencies() {
    log "Checking dependencies..."
    
    local deps=("ros2" "python3" "colcon" "wget" "tar")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            error "$dep not found. Please install it first."
        fi
    done
    
    python3 -c "import cv2" 2>/dev/null || error "OpenCV not installed: pip3 install opencv-python"
    python3 -c "import numpy" 2>/dev/null || error "NumPy not installed: pip3 install numpy"
    
    log "All dependencies satisfied"
}

setup_directories() {
    log "Setting up directory structure..."
    
    mkdir -p "${DATASET_DIR}"/{images,labels}/{train,val}
    mkdir -p "${MODELS_DIR}"
    mkdir -p "${SCRIPTS_DIR}"
    
    log "Directories created"
}

# ============================================================================
# DATASET COLLECTION
# ============================================================================
create_collector_script() {
    log "Creating dataset collector script..."
    
    cat > "${SCRIPTS_DIR}/auto_dataset_collector.py" << 'EOF'
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import sys
import signal

class AutoDatasetCollector(Node):
    def __init__(self, output_dir, max_samples, sample_rate):
        super().__init__('auto_dataset_collector')
        self.bridge = CvBridge()
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        
        self.counter = 0
        self.saved_count = 0
        
        # Robot pose (updated from odometry)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        
        # Camera intrinsics (adjust for your camera)
        self.fx = 554.25
        self.fy = 554.25
        self.cx = 320.5
        self.cy = 240.5
        self.img_width = 640
        self.img_height = 480
        
        # Traffic light world positions from SDF
        self.traffic_lights = [
            # GREEN light at x=8m
            {'x': 8, 'y': 2, 'z': 2.6, 'class_id': 2, 'radius': 0.15, 'active': True},
            {'x': 8, 'y': 2, 'z': 2.8, 'class_id': 1, 'radius': 0.25, 'active': False},
            {'x': 8, 'y': 2, 'z': 3.0, 'class_id': 0, 'radius': 0.25, 'active': False},
            
            # YELLOW light at x=28m
            {'x': 28, 'y': 2, 'z': 2.6, 'class_id': 2, 'radius': 0.25, 'active': False},
            {'x': 28, 'y': 2, 'z': 2.8, 'class_id': 1, 'radius': 0.15, 'active': True},
            {'x': 28, 'y': 2, 'z': 3.0, 'class_id': 0, 'radius': 0.12, 'active': False},
            
            # RED light at x=48m
            {'x': 48, 'y': 2, 'z': 2.6, 'class_id': 2, 'radius': 0.12, 'active': False},
            {'x': 48, 'y': 2, 'z': 2.8, 'class_id': 1, 'radius': 0.12, 'active': False},
            {'x': 48, 'y': 2, 'z': 3.0, 'class_id': 0, 'radius': 0.15, 'active': True},
        ]
        
        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        self.get_logger().info(f'Collector initialized. Target: {max_samples} samples')
        
        # Shutdown handler
        signal.signal(signal.SIGINT, self.shutdown_handler)
    
    def shutdown_handler(self, sig, frame):
        self.get_logger().info(f'Shutting down. Collected {self.saved_count} samples.')
        sys.exit(0)
    
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    def project_3d_to_2d(self, x_world, y_world, z_world):
        """Project 3D world point to 2D image"""
        # Transform to robot frame
        dx = x_world - self.robot_x
        dy = y_world - self.robot_y
        
        # Rotate to robot frame
        cos_yaw = np.cos(self.robot_yaw)
        sin_yaw = np.sin(self.robot_yaw)
        x_robot = dx * cos_yaw + dy * sin_yaw
        y_robot = -dx * sin_yaw + dy * cos_yaw
        
        # Camera frame (x=forward, y=left, z=up, camera height=1.0m)
        x_cam = x_robot
        y_cam = y_robot
        z_cam = z_world - 1.0
        
        if x_cam <= 0.1:  # Behind or too close
            return None
        
        # Project to image
        u = self.fx * (-y_cam / x_cam) + self.cx
        v = self.fy * (-z_cam / x_cam) + self.cy
        
        return int(u), int(v), x_cam
    
    def generate_bbox(self, u, v, radius_3d, distance):
        """Generate YOLO bounding box"""
        # Apparent size decreases with distance
        apparent_radius = max(5, int((radius_3d / distance) * self.fx * 1.5))
        
        x1 = max(0, u - apparent_radius)
        y1 = max(0, v - apparent_radius)
        x2 = min(self.img_width - 1, u + apparent_radius)
        y2 = min(self.img_height - 1, v + apparent_radius)
        
        # YOLO format (normalized)
        x_center = ((x1 + x2) / 2) / self.img_width
        y_center = ((y1 + y2) / 2) / self.img_height
        width = (x2 - x1) / self.img_width
        height = (y2 - y1) / self.img_height
        
        return x_center, y_center, width, height
    
    def image_callback(self, msg):
        if self.saved_count >= self.max_samples:
            self.get_logger().info('Target samples reached. Exiting...')
            raise SystemExit
        
        self.counter += 1
        if self.counter % self.sample_rate != 0:
            return
        
        # Convert image
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Generate annotations
        labels = []
        for tl in self.traffic_lights:
            # Only annotate active lights (emissive in SDF)
            if not tl['active']:
                continue
            
            result = self.project_3d_to_2d(tl['x'], tl['y'], tl['z'])
            if result is None:
                continue
            
            u, v, distance = result
            
            # Check if in frame
            if not (20 < u < self.img_width - 20 and 20 < v < self.img_height - 20):
                continue
            
            # Skip if too far (low resolution)
            if distance > 60:
                continue
            
            x_c, y_c, w, h = self.generate_bbox(u, v, tl['radius'], distance)
            
            # Validate bbox
            if w < 0.01 or h < 0.01 or w > 0.5 or h > 0.5:
                continue
            
            labels.append(f"{tl['class_id']} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        
        # Save only if labels exist
        if not labels:
            return
        
        # Save image
        img_filename = f"frame_{self.saved_count:06d}.jpg"
        img_path = os.path.join(self.output_dir, 'images', 'train', img_filename)
        cv2.imwrite(img_path, cv_image)
        
        # Save labels
        label_path = os.path.join(self.output_dir, 'labels', 'train', 
                                  img_filename.replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
        
        self.saved_count += 1
        
        if self.saved_count % 50 == 0:
            self.get_logger().info(f'Progress: {self.saved_count}/{self.max_samples}')

def main():
    if len(sys.argv) < 4:
        print("Usage: auto_dataset_collector.py <output_dir> <max_samples> <sample_rate>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    max_samples = int(sys.argv[2])
    sample_rate = int(sys.argv[3])
    
    rclpy.init()
    node = AutoDatasetCollector(output_dir, max_samples, sample_rate)
    
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
EOF

    chmod +x "${SCRIPTS_DIR}/auto_dataset_collector.py"
    log "Collector script created"
}

create_robot_driver() {
    log "Creating autonomous robot driver..."
    
    cat > "${SCRIPTS_DIR}/auto_robot_driver.py" << 'EOF'
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class AutoRobotDriver(Node):
    def __init__(self, duration):
        super().__init__('auto_robot_driver')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.duration = duration
        self.start_time = time.time()
        
        self.timer = self.create_timer(0.1, self.drive_callback)
        self.get_logger().info(f'Autonomous driver started for {duration}s')
    
    def drive_callback(self):
        elapsed = time.time() - self.start_time
        
        if elapsed > self.duration:
            self.get_logger().info('Drive complete')
            raise SystemExit
        
        msg = Twist()
        
        # Simple forward motion with slight variations
        if elapsed < 10:
            msg.linear.x = 0.3
        elif elapsed < 15:
            msg.linear.x = 0.5
        elif elapsed < 20:
            msg.linear.x = 0.2
        else:
            # Repeat pattern
            phase = (elapsed - 20) % 15
            if phase < 5:
                msg.linear.x = 0.4
            elif phase < 10:
                msg.linear.x = 0.3
            else:
                msg.linear.x = 0.5
        
        self.pub.publish(msg)

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: auto_robot_driver.py <duration_seconds>")
        sys.exit(1)
    
    duration = int(sys.argv[1])
    
    rclpy.init()
    node = AutoRobotDriver(duration)
    
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # Stop robot
        msg = Twist()
        node.pub.publish(msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
EOF

    chmod +x "${SCRIPTS_DIR}/auto_robot_driver.py"
    log "Driver script created"
}

collect_dataset() {
    log "Starting dataset collection..."
    
    # Check if Gazebo is running
    if ! pgrep -x "gzserver" > /dev/null; then
        warn "Gazebo not running. Launching world..."
        
        # Launch Gazebo in background
        cd "${WORKSPACE}"
        source install/setup.bash
        ros2 launch traffic_light_robot traffic_world.launch.py &
        GAZEBO_PID=$!
        
        log "Waiting for Gazebo to initialize (15s)..."
        sleep 15
    else
        log "Gazebo already running"
        GAZEBO_PID=""
    fi
    
    # Start collector in background
    cd "${WORKSPACE}"
    source install/setup.bash
    python3 "${SCRIPTS_DIR}/auto_dataset_collector.py" \
        "${DATASET_DIR}" "${NUM_SAMPLES}" "${SAMPLE_RATE}" &
    COLLECTOR_PID=$!
    
    sleep 2
    
    # Start autonomous driver
    python3 "${SCRIPTS_DIR}/auto_robot_driver.py" "${COLLECTION_DURATION}" &
    DRIVER_PID=$!
    
    log "Collection in progress..."
    log "Collector PID: ${COLLECTOR_PID}"
    log "Driver PID: ${DRIVER_PID}"
    
    # Wait for collection to complete
    wait $COLLECTOR_PID 2>/dev/null || true
    wait $DRIVER_PID 2>/dev/null || true
    
    # Cleanup
    if [ -n "$GAZEBO_PID" ]; then
        log "Stopping Gazebo..."
        kill $GAZEBO_PID 2>/dev/null || true
    fi
    
    # Count collected samples
    local collected=$(ls "${DATASET_DIR}/images/train"/*.jpg 2>/dev/null | wc -l)
    log "Collection complete: ${collected} images"
    
    if [ $collected -lt 100 ]; then
        error "Insufficient samples collected (${collected}). Need at least 100."
    fi
}

split_dataset() {
    log "Splitting dataset (${TRAIN_SPLIT} train / $((1-${TRAIN_SPLIT})) val)..."
    
    python3 << EOF
import os
import shutil
import random

dataset_dir = "${DATASET_DIR}"
train_split = ${TRAIN_SPLIT}

train_img_dir = os.path.join(dataset_dir, 'images', 'train')
val_img_dir = os.path.join(dataset_dir, 'images', 'val')
train_lbl_dir = os.path.join(dataset_dir, 'labels', 'train')
val_lbl_dir = os.path.join(dataset_dir, 'labels', 'val')

# Get all images
images = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
random.shuffle(images)

split_idx = int(len(images) * train_split)
val_images = images[split_idx:]

print(f"Moving {len(val_images)} samples to validation set...")

for img in val_images:
    shutil.move(os.path.join(train_img_dir, img), os.path.join(val_img_dir, img))
    
    lbl = img.replace('.jpg', '.txt')
    lbl_src = os.path.join(train_lbl_dir, lbl)
    if os.path.exists(lbl_src):
        shutil.move(lbl_src, os.path.join(val_lbl_dir, lbl))

print(f"Train: {split_idx}, Val: {len(val_images)}")
EOF

    log "Dataset split complete"
}

create_file_lists() {
    log "Creating file lists..."
    
    find "${DATASET_DIR}/images/train" -name "*.jpg" -type f > "${DATASET_DIR}/train.txt"
    find "${DATASET_DIR}/images/val" -name "*.jpg" -type f > "${DATASET_DIR}/val.txt"
    
    local train_count=$(wc -l < "${DATASET_DIR}/train.txt")
    local val_count=$(wc -l < "${DATASET_DIR}/val.txt")
    
    log "Train samples: ${train_count}"
    log "Val samples: ${val_count}"
}

# ============================================================================
# TRAINING
# ============================================================================
setup_darknet() {
    log "Setting up Darknet..."
    
    if [ ! -d "${MODELS_DIR}/darknet" ]; then
        cd "${MODELS_DIR}"
        git clone https://github.com/AlexeyAB/darknet
        cd darknet
        
        # Modify Makefile for CPU training (no GPU in VM)
        sed -i 's/GPU=1/GPU=0/' Makefile
        sed -i 's/CUDNN=1/CUDNN=0/' Makefile
        sed -i 's/OPENCV=0/OPENCV=1/' Makefile
        sed -i 's/OPENMP=0/OPENMP=1/' Makefile
        
        make -j$(nproc)
        
        log "Darknet compiled"
    else
        log "Darknet already exists"
    fi
    
    # Download pretrained weights
    if [ ! -f "${MODELS_DIR}/darknet/yolov4-tiny.conv.29" ]; then
        log "Downloading pretrained weights..."
        wget -q https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29 \
            -O "${MODELS_DIR}/darknet/yolov4-tiny.conv.29"
    fi
}

create_training_config() {
    log "Creating training configuration..."
    
    # Class names
    cat > "${MODELS_DIR}/darknet/data/traffic_light.names" << EOF
red
yellow
green
EOF

    # Data file
    cat > "${MODELS_DIR}/darknet/data/traffic_light.data" << EOF
classes = 3
train = ${DATASET_DIR}/train.txt
valid = ${DATASET_DIR}/val.txt
names = ${MODELS_DIR}/darknet/data/traffic_light.names
backup = ${MODELS_DIR}/weights/
EOF

    mkdir -p "${MODELS_DIR}/weights"
    
    # Download base config
    if [ ! -f "${MODELS_DIR}/darknet/cfg/yolov4-tiny-base.cfg" ]; then
        wget -q https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg \
            -O "${MODELS_DIR}/darknet/cfg/yolov4-tiny-base.cfg"
    fi
    
    # Modify config
    python3 << EOF
import re

with open("${MODELS_DIR}/darknet/cfg/yolov4-tiny-base.cfg", 'r') as f:
    content = f.read()

# Modify hyperparameters
content = re.sub(r'batch=\d+', 'batch=${BATCH_SIZE}', content)
content = re.sub(r'subdivisions=\d+', 'subdivisions=${SUBDIVISIONS}', content)
content = re.sub(r'max_batches\s*=\s*\d+', 'max_batches=${MAX_BATCHES}', content)
content = re.sub(r'steps=[\d,]+', 'steps=4800,5400', content)

# Modify classes and filters
lines = content.split('\n')
modified_lines = []

for i, line in enumerate(lines):
    if line.strip().startswith('classes='):
        modified_lines.append('classes=3')
    elif line.strip().startswith('filters='):
        # Check if this is before a yolo layer
        is_before_yolo = False
        for j in range(i+1, min(i+10, len(lines))):
            if '[yolo]' in lines[j]:
                is_before_yolo = True
                break
        
        if is_before_yolo:
            modified_lines.append('filters=24')  # (3+5)*3
        else:
            modified_lines.append(line)
    else:
        modified_lines.append(line)

with open("${MODELS_DIR}/darknet/cfg/yolov4-tiny-traffic.cfg", 'w') as f:
    f.write('\n'.join(modified_lines))

print("Config created")
EOF

    log "Training config created"
}

train_model() {
    log "Starting training..."
    warn "Training on CPU will take 15-25 hours. Consider using Google Colab for GPU training."
    
    read -p "Continue with CPU training? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Training skipped. Upload dataset to Colab manually."
        log "Dataset location: ${DATASET_DIR}"
        return
    fi
    
    cd "${MODELS_DIR}/darknet"
    
    ./darknet detector train \
        data/traffic_light.data \
        cfg/yolov4-tiny-traffic.cfg \
        yolov4-tiny.conv.29 \
        -dont_show -map \
        2>&1 | tee "${MODELS_DIR}/training.log"
    
    log "Training complete!"
    log "Best weights: ${MODELS_DIR}/weights/yolov4-tiny-traffic_best.weights"
}

# ============================================================================
# DEPLOYMENT
# ============================================================================
deploy_model() {
    log "Deploying model to ROS2 package..."
    
    local best_weights="${MODELS_DIR}/weights/yolov4-tiny-traffic_best.weights"
    local cfg_file="${MODELS_DIR}/darknet/cfg/yolov4-tiny-traffic.cfg"
    
    if [ ! -f "$best_weights" ]; then
        error "Trained weights not found at: $best_weights"
    fi
    
    # Copy to package
    local pkg_models="${PROJECT_ROOT}/src/traffic_light_robot/models"
    mkdir -p "$pkg_models"
    
    cp "$best_weights" "$pkg_models/"
    cp "$cfg_file" "$pkg_models/"
    
    # Update detector node
    local detector_node="${PROJECT_ROOT}/src/traffic_light_robot/traffic_light_robot/detector_node.py"
    
    if [ -f "$detector_node" ]; then
        log "Updating detector_node.py paths..."
        
        python3 << EOF
import re

with open("$detector_node", 'r') as f:
    content = f.read()

# Update paths
content = re.sub(
    r"readNetFromDarknet\([^)]+\)",
    "readNetFromDarknet(\n            '${pkg_models}/yolov4-tiny-traffic.cfg',\n            '${pkg_models}/yolov4-tiny-traffic_best.weights'\n        )",
    content
)

with open("$detector_node", 'w') as f:
    f.write(content)

print("Detector node updated")
EOF
    fi
    
    log "Model deployed successfully"
    log "Model location: ${pkg_models}"
}

test_deployment() {
    log "Testing deployment..."
    
    cd "${WORKSPACE}"
    colcon build --packages-select traffic_light_robot
    source install/setup.bash
    
    log "Launch Gazebo and run: ros2 run traffic_light_robot detector_node"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================
print_banner() {
    echo -e "${BLUE}"
    echo "=============================================="
    echo "  YOLO Traffic Light Training Pipeline"
    echo "  Automated Dataset Collection + Training"
    echo "=============================================="
    echo -e "${NC}"
}

show_usage() {
    echo "Usage: $0 [collect|train|deploy|full]"
    echo ""
    echo "Commands:"
    echo "  collect  - Collect and annotate dataset only"
    echo "  train    - Train YOLO model (requires dataset)"
    echo "  deploy   - Deploy trained model to ROS2 package"
    echo "  full     - Run complete pipeline (collect → train → deploy)"
    echo ""
}

main() {
    print_banner
    
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi
    
    local command=$1
    
    check_dependencies
    setup_directories
    
    case $command in
        collect)
            log "Running dataset collection..."
            create_collector_script
            create_robot_driver
            collect_dataset
            split_dataset
            create_file_lists
            log "Dataset ready at: ${DATASET_DIR}"
            log "Total samples: $(wc -l < ${DATASET_DIR}/train.txt) train, $(wc -l < ${DATASET_DIR}/val.txt) val"
            ;;
        
        train)
            log "Running training..."
            setup_darknet
            create_training_config
            train_model
            ;;
        
        deploy)
            log "Deploying model..."
            deploy_model
            test_deployment
            ;;
        
        full)
            log "Running complete pipeline..."
            
            # Collection
            create_collector_script
            create_robot_driver
            collect_dataset
            split_dataset
            create_file_lists
            
            # Training
            setup_darknet
            create_training_config
            train_model
            
            # Deployment
            deploy_model
            test_deployment
            
            log "Pipeline complete!"
            ;;
        
        *)
            error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
    
    log "Done!"
}

# Run main
main "$@"