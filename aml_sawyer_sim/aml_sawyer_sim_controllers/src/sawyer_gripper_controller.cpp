/*********************************************************************
 # Copyright (c) 2014 Kei Okada
 # All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions are met:
 #
 # 1. Redistributions of source code must retain the above copyright notice,
 #    this list of conditions and the following disclaimer.
 # 2. Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 # 3. Neither the name of the Rethink Robotics nor the names of its
 #    contributors may be used to endorse or promote products derived from
 #    this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 # POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

#include <sawyer_sim_controllers/sawyer_gripper_controller.h>
#include <pluginlib/class_list_macros.h>
#include <yaml-cpp/yaml.h>

namespace sawyer_sim_controllers
{

SawyerGripperController::SawyerGripperController()
    : new_command(true),
      update_counter(0),
      mimic_idx_(0),
      main_idx_(0)
{
}

SawyerGripperController::~SawyerGripperController()
{
    gripper_command_sub.shutdown();
}

bool SawyerGripperController::init(hardware_interface::EffortJointInterface *robot,
                                   ros::NodeHandle &nh)
{
    // Store nodehandle
    nh_ = nh;

    // Get joint sub-controllers
    XmlRpc::XmlRpcValue xml_struct;
    if (!nh_.getParam("joints", xml_struct))
    {
        ROS_ERROR("No 'joints' parameter in controller (namespace '%s')",
                  nh_.getNamespace().c_str());
        return false;
    }

    // Make sure it's a struct
    if (xml_struct.getType() != XmlRpc::XmlRpcValue::TypeStruct)
    {
        ROS_ERROR("The 'joints' parameter is not a struct (namespace '%s')",
                  nh_.getNamespace().c_str());
        return false;
    }

    // Get number of joints
    n_joints = xml_struct.size();

    gripper_controllers.resize(n_joints);
    int i = 0;  // track the joint id
    for (XmlRpc::XmlRpcValue::iterator joint_it = xml_struct.begin();
         joint_it != xml_struct.end(); ++joint_it)
    {
        // Get joint controller
        if (joint_it->second.getType() != XmlRpc::XmlRpcValue::TypeStruct)
        {
            ROS_ERROR("The 'joints/joint_controller' parameter is not a struct (namespace '%s')",
                      nh_.getNamespace().c_str());
            return false;
        }

        // Get joint controller name
        std::string joint_controller_name = joint_it->first;

        // Get the joint-namespace nodehandle
        {
            ros::NodeHandle joint_nh(nh_, "joints/" + joint_controller_name);
            ROS_INFO_STREAM_NAMED("init",
                                  "Loading sub-controller '" << joint_controller_name
                                  << "', Namespace: " << joint_nh.getNamespace());

            gripper_controllers[i].reset(new effort_controllers::JointPositionController());
            gripper_controllers[i]->init(robot, joint_nh);

        }  // end of joint-namespaces

        // Set mimic indices
        if (gripper_controllers[i]->joint_urdf_->mimic)
        {
            mimic_idx_ = i;
        }
        else
        {
            main_idx_ = i;
        }

        // Add joint name to map (allows unordered list to quickly be mapped to the ordered index)
        joint_to_index_map.insert(std::pair<std::string, std::size_t>(gripper_controllers[i]->getJointName(), i));

        // increment joint i
        ++i;
    }

    // Get controller topic name that it will subscribe to
    if (nh_.getParam("topic", topic_name))
    { // They provided a custom topic to subscribe to

        // Get a node handle that is relative to the base path
        ros::NodeHandle nh_base("~");

        // Create command subscriber custom to sawyer
        gripper_command_sub = nh_base.subscribe<intera_core_msgs::IOComponentCommand>(topic_name, 1,
                                                                                      &SawyerGripperController::commandCB, this);
    }
    else  // default "command" topic
    {
        // Create command subscriber custom to sawyer
        gripper_command_sub = nh_.subscribe<intera_core_msgs::IOComponentCommand>("command", 1,
                                                                                  &SawyerGripperController::commandCB, this);
    }
    return true;
}

void SawyerGripperController::starting(const ros::Time& time)
{
}

void SawyerGripperController::stopping(const ros::Time& time)
{
}

void SawyerGripperController::update(const ros::Time& time,
                                     const ros::Duration& period)
{
    // Debug info
    update_counter++;
    //TODO: Change to ROS Param (20 Hz)
    if (update_counter % 5 == 0)
    {
        updateCommands();
    }

    // Apply joint commands
    for (size_t i = 0; i < n_joints; i++)
    {
        // Update the individual joint controllers
        gripper_controllers[i]->update(time, period);
    }
}

void SawyerGripperController::updateCommands()
{
    // Check if we have a new command to publish
    if (!new_command)
        return;

    // Go ahead and assume we have proccessed the current message
    new_command = false;

    // Get latest command
    const intera_core_msgs::IOComponentCommand &command = *(gripper_command_buffer.readFromRT());

    if (command.op != "set")
    {
        // ROS_INFO("Gripper update command op : %s", command.op.c_str());
        return;
    }

    // ROS_INFO("Gripper update command args : %s", command.args.c_str());

    //Asuume single [] in args: e.g. signals": {"position_m": {"data": [99.9958333], "format": {"type": "float"}}}}
    std::size_t start_pos = command.args.find('[');
    std::size_t end_pos = command.args.find(']');
    if (start_pos == std::string::npos || end_pos == std::string::npos)
    {
        ROS_INFO("Not found '[' or ']' in Gripper update command args: %s", command.args.c_str());
        return;
    }

    double cmd_position = gripper_controllers[main_idx_]->getPosition();
    cmd_position = atof(command.args.substr(start_pos+1, end_pos-start_pos-1).c_str());

    // ROS_INFO("Gripper command str: %s value %3.5f",
    //          command.args.substr(start_pos+1, end_pos-start_pos-1).c_str(), cmd_position);

    // Check Command Limits:
    if (cmd_position < gripper_controllers[main_idx_]->joint_urdf_->limits->lower)
    {
        cmd_position = gripper_controllers[main_idx_]->joint_urdf_->limits->lower;
    }
    else if (cmd_position > gripper_controllers[main_idx_]->joint_urdf_->limits->upper)
    {
        cmd_position = gripper_controllers[main_idx_]->joint_urdf_->limits->upper;
    }

    // cmd = ratio * range
    // cmd_position = (cmd_position/100.0) *
    //                (gripper_controllers[main_idx_]->joint_urdf_->limits->upper - gripper_controllers[main_idx_]->joint_urdf_->limits->lower);

    // Update the individual joint controllers
    ROS_DEBUG_STREAM(gripper_controllers[main_idx_]->joint_urdf_->name << "->setCommand(" << cmd_position << ")");
    gripper_controllers[main_idx_]->setCommand(cmd_position);
    gripper_controllers[mimic_idx_]->setCommand(gripper_controllers[mimic_idx_]->joint_urdf_->mimic->multiplier*cmd_position+gripper_controllers[mimic_idx_]->joint_urdf_->mimic->offset);
}

void SawyerGripperController::commandCB(const intera_core_msgs::IOComponentCommandConstPtr& msg)
{
    // the writeFromNonRT can be used in RT, if you have the guarantee that
    //  * no non-rt thread is calling the same function (we're not subscribing to ros callbacks)
    //  * there is only one single rt thread
    gripper_command_buffer.writeFromNonRT(*msg);

    new_command = true;
}

}  // namespace

PLUGINLIB_EXPORT_CLASS(sawyer_sim_controllers::SawyerGripperController,
                       controller_interface::ControllerBase)
