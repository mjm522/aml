/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2013, Open Source Robotics Foundation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Open Source Robotics Foundation
 *     nor the names of its contributors may be
 *     used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: Hariharasudan Malaichamee
 * Author: Dave Coleman
 Desc:   Customized the default gazebo_ros_control_plugin.cpp
*/

// Overload the default plugin
#include <gazebo_ros_control/gazebo_ros_control_plugin.h>

// Controller Manager
#include <controller_manager_msgs/SwitchController.h>

// Intera_Gripper
#include <intera_core_msgs/AssemblyState.h>
#include <intera_core_msgs/IOComponentCommand.h>

namespace intera_gripper_gazebo_plugin
{

class InteraGripperGazeboRosControlPlugin : public gazebo_ros_control::GazeboRosControlPlugin
{
private:
    ros::Subscriber robot_state_sub_;
    ros::Subscriber right_gripper_state_sub;

    // Rate to publish assembly state
    ros::Timer timer_;

    // Cache the message
    intera_core_msgs::AssemblyState assembly_state_;

    boost::mutex mtx_;  // mutex for re-entrent calls to modeCommandCallback

    // enabled tracks the current status of the robot that is being published & is_disabled keeps track of the action taken
    bool enable_cmd, is_disabled, right_gripper_is_started;

public:

    void Load(gazebo::physics::ModelPtr parent, sdf::ElementPtr sdf)
    {
        // Load parent class first
        GazeboRosControlPlugin::Load(parent, sdf);

        // Intera_Gripper customizations:
        ROS_INFO_STREAM_NAMED("intera_gripper_gazebo_ros_control_plugin",
                              "Loading Intera_Gripper specific simulation components");

        // Subscribe to a topic that switches' Intera_Gripper's msgs
        right_gripper_state_sub
            = model_nh_.subscribe < intera_core_msgs::IOComponentCommand> ("/io/end_effector/right_gripper/command",
                                                                           1, &InteraGripperGazeboRosControlPlugin::rightEndEffectorCommandCallback, this);

        // Subscribe to the topic that publishes the robot's state
        robot_state_sub_
            = model_nh_.subscribe < intera_core_msgs::AssemblyState> ("/robot/state",
                                                                      1, &InteraGripperGazeboRosControlPlugin::enableCommandCallback, this);

        enable_cmd = false;
        is_disabled = false;
        right_gripper_is_started = false;
    }

    void enableCommandCallback(const intera_core_msgs::AssemblyState msg)
    {
        enable_cmd = msg.enabled;
        std::vector < std::string > start_controllers;
        std::vector < std::string > stop_controllers;

        // Check if we got disable signal and if the controllers are not already disabled
        if (!enable_cmd && !is_disabled)
        {
            stop_controllers.push_back("right_gripper_controller");

            if (!controller_manager_->switchController(start_controllers, stop_controllers,
                                                       controller_manager_msgs::SwitchController::Request::BEST_EFFORT))
            {
                ROS_ERROR_STREAM_NAMED("intera_gripper_gazebo_ros_control_plugin",
                                       "Failed to switch controllers");
            }
            else
            {
                //Resetting the command modes to the initial configuration
                ROS_INFO("Gripper is disabled");
                right_gripper_is_started = false;
                is_disabled = true;
            }
        }
    }

    void rightEndEffectorCommandCallback(const intera_core_msgs::IOComponentCommand msg)
    {
        if (!right_gripper_is_started && enable_cmd)
        {
            std::vector < std::string > start_controllers;
            std::vector < std::string > stop_controllers;

            start_controllers.push_back("right_gripper_controller");
            if (!controller_manager_->switchController(start_controllers, stop_controllers,
                                                       controller_manager_msgs::SwitchController::Request::STRICT))
            {
                ROS_ERROR_STREAM_NAMED("intera_gripper_gazebo_ros_control_plugin",
                                       "Failed to switch controllers");
            }
            else {
                ROS_INFO("Gripper is enabled");
                ROS_INFO("Right Grippercontroller was successfully started");
                right_gripper_is_started=true;
                is_disabled=false;
            }
        }
        else
        {
            return;
        }
    }
};

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN (InteraGripperGazeboRosControlPlugin);
}  // namespace
