#!/bin/bash
cd ros2_ws/
colcon build && source install/setup.bash && ros2 run mujoco_image_viewer image_viewer 
