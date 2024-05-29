cmake_minimum_required(VERSION 3.0.2)
project(testing_kuka)


find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_geometry
  image_transport
  message_filters
  message_generation
  message_runtime
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf2_ros
  vision_msgs
  visualization_msgs
)

find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)


################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  YoloResult.msg
  GraspDetectionResult.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
  vision_msgs
  visualization_msgs
  geometry_msgs
)


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    cv_bridge
    geometry_msgs
    image_geometry
    image_transport
    message_filters
    message_runtime
    pcl_ros
    roscpp
    rospy
    sensor_msgs
    std_msgs
    tf2_ros
    vision_msgs
    visualization_msgs
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# add_library(${PROJECT_NAME}
#   src/tracker_with_cloud_node.cpp
# )
add_executable(${PROJECT_NAME} src/tracker_with_cloud_node.cpp)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)
add_dependencies(${PROJECT_NAME} testing_kuka_generate_messages_cpp)


add_executable(testing_pcl_node src/testing_pcl_node.cpp)
target_link_libraries(testing_pcl_node
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)


install(PROGRAMS
  script/tracker_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
