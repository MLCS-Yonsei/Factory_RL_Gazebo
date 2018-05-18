cmake_minimum_required(VERSION 2.8.3)
project(description)

find_package(catkin REQUIRED)

catkin_package(
  CATKIN_DEPENDS
)

install(DIRECTORY controller/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config)
install(DIRECTORY controller/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/meshes)
install(DIRECTORY controller/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf)
