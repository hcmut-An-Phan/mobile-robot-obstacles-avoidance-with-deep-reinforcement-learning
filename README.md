launch Rviz:
$ ros2 launch turtlebot3_description display.launch.py

launch Gazebo:
(the part in () is option)
$ ros2 launch turtlebot3_description gazebo.launch.py (model:=path/to/model/file world:=path/to/world/file)

control robot in gazebo with libgazebo_ros_diff_drive:
$ ros2 run teleop_twist_keyboard teleop_twist_keyboard

vew odometry:
$ rviz2 -d ./src/turtlebot3_description/rviz/control.rviz

*** ROS2 control ***
launch gazebo (this using turtlebor3_controllers.yaml):
(the part in () is option)
$ ros2 launch turtlebot3_description gazebo.launch.py (model:=path/to/model/file world:=path/to/file.world)

launch controller:
$ ros2 launch turtlebot3_controller controller.launch.py

check controller:
$ ros2 control list_controllers

check hardware:
$ ros2 control list_hardware_components
$ ros2 control list_hardware_interfaces

publish to /simple_velocity_controller/commands
$ ros2 topic pub /simple_velocity_controller/commands std_msgs/msg/Float64MultiArray "layout: 
    dim: []
    data_offset: 0
data: []"

friendlier use, control simulated robot with keyboard using diff_drive_controllers.yaml and diff_drive_controller.launch.py:
$ ros2 run teleop_twist_keyboard  teleop_twist_keyboard --ros-args -r /cmd_vel:=/diff_drive_controller/cmd_vel_unstamped

sudo killall -9 gazebo gzserver gzclient