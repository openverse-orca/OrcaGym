# Configuration for Dexforce W1 gripper (position control for sliding fingers)
# Position control: each finger has its own actuator, but equality constraint ensures synchronization
# Control range: 0 (open) to 0.05 (closed) in meters - increased for full closure
dexforce_w1_gripper_config = {
    "left_hand": {
        "actuator_names": ["P_left_finger1", "P_left_finger2"],
        "body_names": ["left_finger1", "left_finger2"],
        "joint_names": ["LEFT_FINGER1", "LEFT_FINGER2"],
    },
    "right_hand": {
        "actuator_names": ["P_right_finger1", "P_right_finger2"],
        "body_names": ["right_finger1", "right_finger2"],
        "joint_names": ["RIGHT_FINGER1", "RIGHT_FINGER2"],
    }
}

