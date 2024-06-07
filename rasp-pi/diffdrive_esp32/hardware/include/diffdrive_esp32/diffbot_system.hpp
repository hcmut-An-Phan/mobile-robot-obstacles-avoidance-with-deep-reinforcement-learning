#ifndef DIFFDRIVE_ESP32__DIFFBOT_SYSTEM_HPP_
#define DIFFDRIVE_ESP32__DIFFBOT_SYSTEM_HPP_

#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/clock.hpp"
#include "rclcpp/duration.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"

#include "diffdrive_esp32/visibility_control.h"

#include "diffdrive_esp32/esp32_comms.hpp"
#include "diffdrive_esp32/wheel.hpp"

namespace diffdrive_esp32
{
class DiffDriveEsp32Hardware : public hardware_interface::SystemInterface
{

struct Config
{
  std::string left_wheel_name = "";
  std::string right_wheel_name = "";
  std::string device = "";
  int baud_rate = 0;
  int timeout_ms = 0;
  int enc_counts_per_rev = 0;
};

public:
  RCLCPP_SHARED_PTR_DEFINITIONS(DiffDriveEsp32Hardware);

  DIFFDRIVE_ESP32_PUBLIC
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;

  DIFFDRIVE_ESP32_PUBLIC
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  DIFFDRIVE_ESP32_PUBLIC
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  DIFFDRIVE_ESP32_PUBLIC
  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  DIFFDRIVE_ESP32_PUBLIC
  hardware_interface::CallbackReturn on_cleanup(
    const rclcpp_lifecycle::State & previous_state) override;

  DIFFDRIVE_ESP32_PUBLIC
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  DIFFDRIVE_ESP32_PUBLIC
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  DIFFDRIVE_ESP32_PUBLIC
  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  DIFFDRIVE_ESP32_PUBLIC
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  ArduinoComms comms_;
  Config cfg_;

  Wheel wheel_l_;
  Wheel wheel_r_;
};

}  // namespace diffdrive_esp32

#endif  // DIFFDRIVE_ESP32__DIFFBOT_SYSTEM_HPP_
