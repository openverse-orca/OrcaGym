from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Node(
        #     package='mujoco_image_viewer',
        #     executable='image_viewer',
        #     name='view1'
        # ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher_node',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher_node',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_footprint']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher_node',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'base_footprint', 'base_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher_node',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'lidar/sensor_frame']
        ),

    ])

