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
Log out and log back in for changes to take effect.

### 2. Clean Existing Conflicts (if any)
```bash
sudo rm -f /etc/apt/sources.list.d/vscode.list
sudo rm -f /etc/apt/keyrings/packages.microsoft.gpg
sudo rm -f /usr/share/keyrings/microsoft.gpg
sudo apt update
```

### 3. Run Setup Script
Navigate to project directory and run:
```bash
cd ~/Desktop/TrafficSenseAI
sudo bash ./bashScriptCOnfig.sh
```

The script will:
- Install ROS2 Humble
- Install Gazebo simulator
- Install Python tools (numpy, scipy, matplotlib, pandas)
- Install robotics packages (URDF, XACRO, robot state publisher)
- Build the workspace
- Configure environment

### 4. Source Environment
```bash
source ~/.bashrc
```
Or manually:
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

## Troubleshooting

### Permission Denied
Ensure user is in sudo group and logged out/in after adding.

### Package Not Found
```bash
sudo apt update
source /opt/ros/humble/setup.bash
```

### Gazebo Won't Launch
Check if workspace is sourced:
```bash
source install/setup.bash
```

### Build Errors
Clean and rebuild:
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

// command to delete from the readme , but to have easier access right now raspb@Raspberry:~/Desktop/TrafficSenseAI$ ros2 launch robot_description gazebo.launch.py 