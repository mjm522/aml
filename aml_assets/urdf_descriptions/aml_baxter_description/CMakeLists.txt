cmake_minimum_required(VERSION 2.8.3)
project(aml_baxter_description)


find_package(catkin
  REQUIRED
  COMPONENTS
  rospy
  xacro
  baxter_core_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  xacro
  baxter_core_msgs
)

catkin_package()

foreach(dir meshes urdf)
   install(DIRECTORY ${dir}/
      DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
