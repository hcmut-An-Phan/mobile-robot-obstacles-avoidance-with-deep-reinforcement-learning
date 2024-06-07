from launch import LaunchDescription
from launch_ros.actions import Node

from launch.actions import DeclareLaunchArgument
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    model_arg = DeclareLaunchArgument(
        name="model",
        default_value=os.path.join(get_package_share_directory("turtlebot3_description"), "urdf", "turtlebot3.urdf.xacro"),
        description="Absolute path to URDF file"
    )

    time_arg =DeclareLaunchArgument(
        name="use_sim_time",
        default_value="false",
        description="Use sim time if true"
    )

    robot_description = ParameterValue(Command(["xacro ", LaunchConfiguration("model")]), value_type=str)
    use_sim_time = LaunchConfiguration("use_sim_time")

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{'robot_description': robot_description, "use_sim_time": use_sim_time}]
    )

    joint_state_pubisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz",
        output="screen",
        arguments=["-d", os.path.join(get_package_share_directory("turtlebot3_description"), "rviz", "display.rviz")]
    )

    return LaunchDescription([
        model_arg,
        time_arg,
        robot_state_publisher,
        joint_state_pubisher_gui,
        rviz2
    ])

