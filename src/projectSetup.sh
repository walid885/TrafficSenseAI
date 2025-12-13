#!/bin/bash

# =================================================================
# FULL ROS 2 PROJECT SETUP AND LAUNCH SCRIPT
# Project: TrafficSenseAI
# ROS 2 Distro: Humble Hawksbill (Recommended for Ubuntu 22.04)
# =================================================================

# --- 1. Configuration ---
# The desired location for your project workspace
PROJECT_DIR="$HOME/Desktop/TrafficSenseAI"

# ROS 2 Distribution - Set your specific version
ROS_DISTRO="humble" 

# --- 2. Prerequisites & Installation Function ---

install_ros2_humble() {
    echo "--- 🛠️  Phase 1: Installing ROS 2 $ROS_DISTRO and Dependencies ---"

    # Set locale (Required for ROS setup)
    sudo apt update
    sudo apt install -y locales
    sudo locale-gen en_US en_US.UTF-8
    sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
    export LANG=en_US.UTF-8
    echo "✅ System locale set to UTF-8."

    # Setup Sources
    sudo apt install -y software-properties-common
    sudo add-apt-repository universe
    sudo apt update && sudo apt install -y curl gnupg lsb-release

    # Add ROS 2 GPG key
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

    # Add ROS 2 repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

    # Install ROS 2 Desktop (Includes ROS, RViz, Demos, Gazebo integration packages, etc.)
    sudo apt update
    echo "⏳ Installing ROS 2 Desktop. This may take some time..."
    sudo apt install -y ros-$ROS_DISTRO-desktop

    # Install ROS 2 build tools and rosdep
    sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-vcstool git
    
    # Initialize rosdep (needed only once per system)
    sudo rosdep init 2> /dev/null || true # Ignore 'already initialized' error
    rosdep update
    
    echo "✅ ROS 2 $ROS_DISTRO Installation Complete."
}

# --- 3. Project Setup and Cloning Function ---

setup_project() {
    echo "--- 📂 Phase 2: Setting up Project Workspace ---"

    if [ -d "$PROJECT_DIR" ]; then
        echo "⚠️ Project directory already exists at $PROJECT_DIR. Skipping cloning."
    else
        echo "Creating project directory: $PROJECT_DIR"
        mkdir -p "$PROJECT_DIR/src"
        cd "$PROJECT_DIR"

        # IMPORTANT: You must replace 'YOUR_REPO_URL' with the actual URL of your TrafficSenseAI GitHub/Git repository
        echo "🚨 Cloning project source (Please update the URL below if needed!):"
        # Example of how you would clone a repository:
        # git clone YOUR_REPO_URL src/TrafficSenseAI_Source
        
        echo "> Assuming project source files are copied into $PROJECT_DIR/src based on previous context."
    fi

    cd "$PROJECT_DIR"
}

# --- 4. Main Script Execution ---

# Check if ROS is installed. If not, install it.
if [ ! -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
    echo "ROS 2 $ROS_DISTRO not found. Starting full installation..."
    install_ros2_humble
else
    echo "✅ ROS 2 $ROS_DISTRO found. Skipping installation."
fi

# Set up the project directory
setup_project

# --- 5. Project Dependencies and Build ---
echo "--- ⚙️  Phase 3: Installing Workspace Dependencies and Building ---"

# Source the base environment for rosdep
source "/opt/ros/$ROS_DISTRO/setup.bash"

# Install all dependencies required by the packages in the 'src' directory
echo "🛠️ Installing project dependencies using rosdep..."
if rosdep install --from-paths src --ignore-src -y --rosdistro $ROS_DISTRO; then
    echo "✅ All dependencies installed successfully."
else
    echo "❌ Error: Rosdep failed to install all dependencies. Check the output for missing packages."
    exit 1
fi

# Build the entire workspace
echo "⏳ Building the entire project with colcon..."
if colcon build --symlink-install; then
    echo "✅ Colcon build successful!"
else
    echo "❌ Error: Colcon build failed. Check the errors above."
    exit 1
fi

# Source the local workspace setup file to access built packages
source install/setup.bash

# --- 6. Launch Simulation (Using your existing launch logic) ---

echo "--- ▶️ Phase 4: Launching Simulation ---"

# Trap Ctrl+C for clean shutdown (copied from your original launch script)
export QT_QPA_PLATFORM=xcb
cleanup() {
    echo ""
    echo "🚨 Shutting down ROS 2 nodes and processes..."
    pkill -f "ros2 launch robot_description"
    pkill -f "ros2 run traffic_light_robot"
    pkill -f "rviz2"
    pkill -f "gzserver"
    pkill -f "gzclient"
    sleep 2
    echo "✅ Shutdown complete."
    exit 0
}
trap 'cleanup' INT TERM

# 4.1. Launch Simulation and Robot Control (in a new terminal)
echo "1/2: Starting 'robot_description autonomous.launch.py' in a new terminal..."
gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 launch robot_description autonomous.launch.py; exec bash" 2>/dev/null &
LAUNCH_PID=$!

sleep 5

# 4.2. Launch Visualizer Node (in a new terminal)
echo "2/2: Starting 'traffic_light_robot visualizer_node' in a new terminal..."
gnome-terminal -- bash -c "cd $PROJECT_DIR && source install/setup.bash && ros2 run traffic_light_robot visualizer_node; exec bash" 2>/dev/null &

echo "---"
echo "✅ All components launched."
echo "Press **Ctrl+C** in this terminal to trigger the graceful shutdown."

# Keep the script running until the user hits Ctrl+C
wait "$LAUNCH_PID"