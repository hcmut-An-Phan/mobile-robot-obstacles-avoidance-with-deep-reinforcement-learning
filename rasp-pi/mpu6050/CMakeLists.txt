cmake_minimum_required(VERSION 3.16)
project(mpu6050 LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
  add_compile_options(-Wall -Wextra)
endif()

# find dependencies
set(THIS_PACKAGE_INCLUDE_DEPENDS
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
)

find_package(ament_cmake REQUIRED)
foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()

#COMPILE
add_library(
  mpu6050_hardware_interface 
  SHARED
  src/mpu6050_hardware_interface.cpp
  src/mpu6050_lib.cpp
)

target_compile_features(mpu6050_hardware_interface PUBLIC cxx_std_17)

target_include_directories(mpu6050_hardware_interface PUBLIC
$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
$<INSTALL_INTERFACE:include/mpu6050_hardware_interface>)

ament_target_dependencies(
  mpu6050_hardware_interface PUBLIC
  ${THIS_PACKAGE_INCLUDE_DEPENDS})

# Link i2c library to mpu6050_hardware_interface target
target_link_libraries(mpu6050_hardware_interface PUBLIC i2c)

# Export hardware plugin
pluginlib_export_plugin_description_file(hardware_interface mpu6050_hardware.xml)

# INSTALL
install(
  DIRECTORY include/
  DESTINATION include/mpu6050_hardware_interface
)

install(
  TARGETS mpu6050_hardware_interface
  EXPORT export_mpu6050_hardware_interface
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# EXPORT
ament_export_libraries(mpu6050_hardware_interface)
include_directories(include)

# Create Cpp executables
add_executable(mpu6050_offsets src/mpu6050_lib.cpp src/mpu6050_offsets.cpp)
add_executable(mpu6050_covariances src/mpu6050_lib.cpp src/mpu6050_covariances.cpp)
add_executable(test_mpu6050 src/mpu6050_lib.cpp src/test_mpu6050.cpp)

# Install Cpp executables
install(TARGETS
  mpu6050_offsets
  mpu6050_covariances
  test_mpu6050
  DESTINATION lib/${PROJECT_NAME}
)

# Link i2c to mpu6050_offsets and mpu6050_covariances targets
target_link_libraries(mpu6050_offsets i2c)
target_link_libraries(mpu6050_covariances i2c)
target_link_libraries(test_mpu6050 i2c)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_targets(export_mpu6050_hardware_interface HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})
ament_package()
