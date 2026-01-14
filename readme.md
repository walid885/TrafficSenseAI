# TrafficSenseAI - ROS2 Traffic Light Detection System

## Prerequisites

- Ubuntu 22.04
- Internet connection
- Root/sudo access

## Setup Instructions

### 1. Add User to Sudoers (if needed)

```bash
su -
usermod -aG sudo <your-username>
exit
```

**Note:** Log out and log back in for changes to take effect.

### 2. Clean Existing Conflicts (if any)

```bash
sudo rm -f /etc/apt/sources.list.d/vscode.list
sudo rm -f /etc/apt/keyrings/packages.microsoft.gpg
sudo rm -f /usr/share/keyrings/microsoft.gpg
sudo apt update
```

### 3. Run Setup Script

Navigate to the project directory and execute the configuration script:

```bash
cd ~/Desktop/TrafficSenseAI
sudo bash ./bashScriptCOnfig.sh
```

The script performs the following operations:
- Installs ROS2 Humble
- Installs Gazebo simulator
- Installs Python tools (numpy, scipy, matplotlib, pandas)
- Installs robotics packages (URDF, XACRO, robot state publisher)
- Builds the workspace
- Configures the environment

### 4. Source Environment

```bash
source ~/.bashrc
```

Alternatively, source manually:

```bash
source install/setup.bash
```

### 5. Build Workspace (if needed)

```bash
cd ~/Desktop/TrafficSenseAI
colcon build
source install/setup.bash
```

## Running the Simulation

### Launch Gazebo with Robot

```bash
cd ~/Desktop/TrafficSenseAI
source install/setup.bash
ros2 launch robot_description gazebo.launch.py
```

### Run Traffic Detection (in new terminal)

```bash
cd ~/Desktop/TrafficSenseAI
source install/setup.bash
python3 src/traffic_light_robot/traffic_detector.py
```

## Project Structure

```
TrafficSenseAI/
├── src/
│   ├── robot_description/      # Robot URDF, meshes, worlds
│   │   ├── launch/
│   │   ├── urdf/
│   │   ├── meshes/
│   │   └── worlds/
│   └── traffic_light_robot/    # Python detection scripts
├── build/                      # Build files
├── install/                    # Installation files
├── log/                        # Build logs
└── bashScriptCOnfig.sh        # Setup script
```

## Operation Modes

### Building the Packages

```bash
cd ~/Desktop/TrafficSenseAI
colcon build --symlink-install
source install/setup.bash
```

### Autonomous Mode

```bash
ros2 launch robot_description autonomous.launch.py
```

### Teleop Mode (AZERTY)

**Terminal 1 - Launch Gazebo:**
```bash
ros2 launch robot_description gazebo.launch.py
```

**Terminal 2 - Run Teleop:**
```bash
source ~/Desktop/TrafficSenseAI/install/setup.bash
ros2 run robot_description azerty_teleop.py
```

If `azerty_teleop` is not executable:
```bash
chmod +x ~/Desktop/TrafficSenseAI/src/robot_description/scripts/azerty_teleop.py
```

### Keyboard Control Installation

```bash
sudo apt install ros-humble-teleop-twist-keyboard xterm -y
```

## Visualization

### Rebuild Traffic Light Robot Package

```bash
cd ~/Desktop/TrafficSenseAI
colcon build --packages-select traffic_light_robot
source install/setup.bash
```

### Run Visualizer

```bash
ros2 launch robot_description autonomous.launch.py
```

In a new terminal:
```bash
ros2 run traffic_light_robot visualizer_node
```

## Controller Tuning

```bash
cd ~/Desktop/TrafficSenseAI
colcon build --packages-select traffic_light_robot
./auto_tune.sh
```

## Troubleshooting

### Permission Denied
Ensure the user is added to the sudo group and has logged out and back in after the change.

### Package Not Found
```bash
sudo apt update
source /opt/ros/humble/setup.bash
```

### Gazebo Launch Issues
Verify that the workspace is properly sourced:
```bash
source install/setup.bash
```

### Build Errors
Clean and rebuild the workspace:
```bash
rm -rf build install log
colcon build
```

## Common Commands

**Rebuild workspace:**
```bash
colcon build
```

**Clean build:**
```bash
colcon build --cmake-clean-cache
```

**Source workspace:**
```bash
source install/setup.bash
```

**List available launch files:**
```bash
ros2 launch robot_description <TAB><TAB>
```

**Check ROS topics:**
```bash
ros2 topic list
```

## Dependencies

- ROS2 Humble
- Gazebo
- Python 3.10+
- NumPy, SciPy, Matplotlib, Pandas
- robot_state_publisher, joint_state_publisher
- XACRO, URDF tools

## Development Roadmap

1. Implementation of YOLO small version
2. Testing of YOLO system
3. Implementation of hybrid version (YOLO + HSV)
4. Addition of pedestrian entities to world implementation
5. Refactoring of world into improved version
6. System documentation and iteration reporting