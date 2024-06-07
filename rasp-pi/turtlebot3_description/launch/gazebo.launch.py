import os
from os import pathsep
from ament_index_python.packages import get_package_share_directory, get_package_prefix

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import Command, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    turtlebot3_description = get_package_share_directory("turtlebot3_description")
    turtlebot3_description_prefix = get_package_prefix("turtlebot3_description")
    gazebo_ros_dir = get_package_share_directory("gazebo_ros")

    model_arg = DeclareLaunchArgument(
        name="model",
        default_value=os.path.join(turtlebot3_description, "urdf", "turtlebot3.urdf.xacro"),
        description="Absolute path to URDF file"
    )

    model_path = os.path.join(turtlebot3_description, "models")
    model_path += pathsep + os.path.join(turtlebot3_description_prefix, "share")
    env_var = SetEnvironmentVariable("GAZEBO_MODEL_PATH", model_path)


    time_arg =DeclareLaunchArgument(
        name="use_sim_time",
        default_value="true",
        description="Use sim time if true"
    )

    world_arg = DeclareLaunchArgument(
        name="world",
        default_value=os.path.join(turtlebot3_description, "gazebo_world", "real_corridor"), 
        description="Absolute path to Gazebo world file"
    )

    robot_description = ParameterValue(Command(["xacro ", LaunchConfiguration("model")]), value_type=str)
    use_sim_time = LaunchConfiguration("use_sim_time")
    world = LaunchConfiguration("world")

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description" : robot_description, "use_sim_time": use_sim_time}]
    )


    start_gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_description, "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": world}.items()
    )

    start_gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, "launch", "gzclient.launch.py")
        )
    )


    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-entity", "turtlebot3_waffle", "-topic", "robot_description"],
        output="screen"
    )

    return LaunchDescription([
        model_arg,
        time_arg,
        world_arg,
        robot_state_publisher,
        env_var,
        start_gazebo_server,
        start_gazebo_client,
        spawn_robot
    ])
