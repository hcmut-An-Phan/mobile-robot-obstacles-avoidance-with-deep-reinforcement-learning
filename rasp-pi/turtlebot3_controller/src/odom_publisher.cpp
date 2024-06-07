#include "turtlebot3_controller/odom_publisher.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <cmath>
using std::placeholders::_1;


SimpleOdomPublisher::SimpleOdomPublisher(const std::string& name)
                                  : Node(name)
                                  , left_wheel_prev_pos_(0.0)
                                  , right_wheel_prev_pos_(0.0)
                                  , x_(0.0)
                                  , y_(0.0)
                                  , theta_(0.0)
                                  , first_time_pub_(true)
{
    declare_parameter("wheel_radius", 0.033);
    declare_parameter("wheel_separation", 0.287);

    wheel_radius_ = get_parameter("wheel_radius").as_double();
    wheel_separation_ = get_parameter("wheel_separation").as_double();

    RCLCPP_INFO_STREAM(get_logger(), "Using wheel radius " << wheel_radius_);
    RCLCPP_INFO_STREAM(get_logger(), "Using wheel separation " << wheel_separation_);
     
    prev_time_ = get_clock()->now();

    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>("/joint_states", qos, std::bind(&SimpleOdomPublisher::jointCallback, this, _1));
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/turtlebot3/odom_noisy", qos);

   
    odom_msg_.header.frame_id = "odom";
    odom_msg_.child_frame_id = "base_footprint_ekf";
    odom_msg_.pose.pose.orientation.x = 0.0;
    odom_msg_.pose.pose.orientation.y = 0.0;
    odom_msg_.pose.pose.orientation.z = 0.0;
    odom_msg_.pose.pose.orientation.w = 1.0;

    transform_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    transform_stamped_.header.frame_id = "odom";
    transform_stamped_.child_frame_id = "base_footprint";

}


void SimpleOdomPublisher::jointCallback(const sensor_msgs::msg::JointState &msg)
{
  double dp_left;
  double dp_right;

  if (first_time_pub_)
  {
    first_time_pub_ = false;
    dp_left = 0.0;
    dp_right = 0.0;
  }
  else
  {
    dp_left = msg.position.at(0) - left_wheel_prev_pos_;
    dp_right = msg.position.at(1) - right_wheel_prev_pos_;
  }

  rclcpp::Time msg_time =  msg.header.stamp;
  rclcpp::Duration dt = msg_time - prev_time_;

  left_wheel_prev_pos_ = msg.position.at(0);
  right_wheel_prev_pos_ = msg.position.at(1);
  prev_time_ = msg_time;

  // double fi_left = dp_left / dt.seconds();
  // double fi_right = dp_right / dt.seconds();

  double fi_left = msg.velocity.at(0);
  double fi_right = msg.velocity.at(1);

  double linear = (wheel_radius_ * fi_right + wheel_radius_ * fi_left) / 2;
  double angular = (wheel_radius_ * fi_right - wheel_radius_ * fi_left) / wheel_separation_;

  double d_s = (wheel_radius_ * dp_right + wheel_radius_ * dp_left)/2;
  double d_theta = (wheel_radius_ * dp_right - wheel_radius_ * dp_left) / wheel_separation_;


  //RCLCPP_INFO_STREAM(get_logger(), "theta: " << theta_ / M_1_PI * 180);

  x_ += d_s * cos(theta_ + d_theta/2);
  y_ += d_s * sin(theta_ + d_theta/2);

  theta_ += d_theta;
  if (theta_ >= 2*M_PI)
  {
    theta_ -= 2*M_PI;
  }
  else if (theta_ <= -2*M_PI)
  {
    theta_ += 2*M_PI;
  }
  

  tf2::Quaternion q;
  q.setRPY(0, 0, theta_);
  odom_msg_.header.stamp = get_clock()->now();
  odom_msg_.pose.pose.position.x = x_;
  odom_msg_.pose.pose.position.y = y_;
  odom_msg_.pose.pose.orientation.x = q.getX();
  odom_msg_.pose.pose.orientation.y = q.getY();
  odom_msg_.pose.pose.orientation.z = q.getZ();
  odom_msg_.pose.pose.orientation.w = q.getW();
  odom_msg_.twist.twist.linear.x = linear;
  odom_msg_.twist.twist.angular.z = angular;
  odom_pub_->publish(odom_msg_);

  transform_stamped_.transform.translation.x = x_;
  transform_stamped_.transform.translation.y = y_;
  transform_stamped_.transform.rotation.x = q.getX();
  transform_stamped_.transform.rotation.y = q.getY();
  transform_stamped_.transform.rotation.z = q.getZ();
  transform_stamped_.transform.rotation.w = q.getW();
  transform_stamped_.header.stamp = get_clock()->now();
  transform_broadcaster_->sendTransform(transform_stamped_);
}

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SimpleOdomPublisher>("odom_publisher_node");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}