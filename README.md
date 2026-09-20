# APOLLO-NAV: Predictive Pursuit Algorithm
Autonomous Catcher node submitted for the TurtleBot Pursuit & Evasion Challenge (Tech Zephyr 4.0, IIT Bhubaneswar).

## Overview
- **Algorithm:** Apollonius dynamic lead-point trajectory solver.
- **State Estimation:** Extended Kalman Filter (CTRV model) tracking target velocity and heading.
- **Obstacle Handling:** 2D planar LiDAR sector-binned repulsive field weighting.
- **Robot Model:** TurtleBot 4 Lite (Gazebo Sim).

## Package Installation

### 1. Prerequisites
- ROS 2 (Humble / Iron)
- Python 3 with `numpy`

### 2. Build Instructions
Clone this repository into your ROS 2 workspace source directory:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone <YOUR-GITHUB-REPO-URL>
cd ~/ros2_ws
colcon build --packages-select apollo_pursuit
source install/setup.bash
