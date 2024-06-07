import os

from launch import LaunchDescription
from launch.actions import TimerAction
from launch.actions import RegisterEventHandler, IncludeLaunchDescription
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from ament_index_python.packages import get_package_share_directory




def generate_launch_description():
    
    pkg_path = get_package_share_directory("turtlebot3_controller")
    pkg_lidar = get_package_share_directory("rtf_lds_driver")
    ekf_params_file = os.path.join(pkg_path, "config", "ekf.yaml")
    
    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("turtlebot3_description"), "urdf", "turtlebot3.urdf.xacro"]
            ),
        ]
    )
    robot_description = {"robot_description": robot_description_content}


    controller_params_file = PathJoinSubstitution(
        [
            FindPackageShare("turtlebot3_controller"),
            "config",
            "controller.yaml",
        ]
    )  
    

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, controller_params_file],
        output="both",
    )


    # Maybe need this because robot state publisstart_controller_manager_cmdher must finish starting up before control manager
    delayed_controller_manager = TimerAction(period=3.0, actions=[controller_manager])


    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
        remappings=[
            ("/diff_drive_controller/cmd_vel_unstamped", "/cmd_vel"),
        ],
    )


    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager"
        ]
    )


    diff_drive_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "diff_drive_controller",
            "--controller-manager",
            "/controller_manager"
        ]
    )

   
    # Delay start of robot_controller after `joint_state_broadcaster`
    delay_robot_controller_spawner_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[diff_drive_controller_spawner],
        )
    )
    
    
    # Spawn imu_sensor_broadcaser
    start_imu_broadcaster_cmd = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["imu_broadcaster"],
    )
    

    # Start odom_publisher
    start_odom_publisher = Node(
        package="turtlebot3_controller",
        executable="odom_publisher"
    )


    # Start robot localization using an Extended Kalman Filter
    start_robot_localization_cmd = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_params_file],
    )
    
    
    # Start lidar node
    start_lidar_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(pkg_lidar, "launch", "hlds_laser.launch.py")]
        )
    )

    return LaunchDescription([
        delayed_controller_manager,
        robot_state_pub_node,
        joint_state_broadcaster_spawner,
        delay_robot_controller_spawner_after_joint_state_broadcaster_spawner,
        start_imu_broadcaster_cmd,
        start_odom_publisher,
        start_robot_localization_cmd,
        start_lidar_cmd
    ])

