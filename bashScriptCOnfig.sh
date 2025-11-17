#!/bin/bash

# ROS & Gazebo Robotics Environment Setup
# For Ubuntu 20.04/22.04

set -e

echo "=========================================="
echo "ROS & Gazebo Robotics Setup"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "This script needs to be run with root privileges."
    echo "Please run: sudo bash $0"
    exit 1
fi

# Get the actual user (not root)
ACTUAL_USER=${SUDO_USER:-$USER}
ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)

# Get the directory where script is run from
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Running as root, configuring for user: $ACTUAL_USER"
echo "Workspace directory: $SCRIPT_DIR"

# Detect Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
echo "Detected Ubuntu version: $UBUNTU_VERSION"

# Update system
echo "Updating system packages..."
apt update && apt upgrade -y

# Install essential build tools
echo "Installing essential build tools..."
apt install -y build-essential cmake git curl wget gnupg2 lsb-release \
    software-properties-common apt-transport-https ca-certificates

# Install ROS (ROS Noetic for 20.04, ROS Humble for 22.04)
if [ "$UBUNTU_VERSION" == "20.04" ]; then
    echo "Installing ROS Noetic..."
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
    apt update
    apt install -y ros-noetic-desktop-full
    echo "source /opt/ros/noetic/setup.bash" >> $ACTUAL_HOME/.bashrc
    apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool
elif [ "$UBUNTU_VERSION" == "22.04" ]; then
    echo "Installing ROS 2 Humble..."
    rm -f /usr/share/keyrings/ros-archive-keyring.gpg
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list
    apt update
    apt install -y ros-humble-desktop-full
    echo "source /opt/ros/humble/setup.bash" >> $ACTUAL_HOME/.bashrc
    apt install -y python3-rosdep python3-colcon-common-extensions
else
    echo "Unsupported Ubuntu version. Please use 20.04 or 22.04"
    exit 1
fi

# Initialize rosdep
echo "Initializing rosdep..."
rosdep init 2>/dev/null || true
su - $ACTUAL_USER -c "rosdep update"

# Install Gazebo
echo "Installing Gazebo..."
if [ "$UBUNTU_VERSION" == "20.04" ]; then
    apt install -y ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control
elif [ "$UBUNTU_VERSION" == "22.04" ]; then
    apt install -y ros-humble-gazebo-ros-pkgs
fi

# Install Python development tools
echo "Installing Python tools..."
apt install -y python3-pip python3-venv
su - $ACTUAL_USER -c "pip3 install --user --upgrade pip"
su - $ACTUAL_USER -c "pip3 install --user numpy scipy matplotlib pandas"

# Setup workspace in current directory
echo "Setting up ROS workspace in current directory..."
su - $ACTUAL_USER -c "mkdir -p $SCRIPT_DIR/src"
cd $SCRIPT_DIR

if [ "$UBUNTU_VERSION" == "20.04" ]; then
    su - $ACTUAL_USER -c "cd $SCRIPT_DIR && catkin_make"
    echo "source $SCRIPT_DIR/devel/setup.bash" >> $ACTUAL_HOME/.bashrc
elif [ "$UBUNTU_VERSION" == "22.04" ]; then
    su - $ACTUAL_USER -c "cd $SCRIPT_DIR && colcon build"
    echo "source $SCRIPT_DIR/install/setup.bash" >> $ACTUAL_HOME/.bashrc
fi

chown -R $ACTUAL_USER:$ACTUAL_USER $SCRIPT_DIR

# Install robotics tools
echo "Installing robotics tools..."
if [ "$UBUNTU_VERSION" == "20.04" ]; then
    apt install -y \
        ros-noetic-robot-state-publisher \
        ros-noetic-joint-state-publisher \
        ros-noetic-xacro \
        ros-noetic-urdf \
        meshlab \
        libeigen3-dev
elif [ "$UBUNTU_VERSION" == "22.04" ]; then
    apt install -y \
        ros-humble-robot-state-publisher \
        ros-humble-joint-state-publisher \
        ros-humble-xacro \
        ros-humble-urdf \
        meshlab \
        libeigen3-dev
fi

# Clean up
echo "Cleaning up..."
apt autoremove -y
apt autoclean -y

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "Run: source ~/.bashrc"
echo "Your ROS workspace: $SCRIPT_DIR"
echo "=========================================="