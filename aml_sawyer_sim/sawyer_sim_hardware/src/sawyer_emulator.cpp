/*********************************************************************
 # Copyright (c) 2015, Rethink Robotics
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

/**
 *  \author Hariharasudan Malaichamee
 *  \desc   Node emulating the Sawyer hardware interfaces for simulation
 *      commands
 */

#include <sawyer_sim_hardware/sawyer_emulator.h>

namespace sawyer_en {

// Topics to subscribe and publish
const std::string SAWYER_STATE_TOPIC = "robot/state";
const std::string SAWYER_ENABLE_TOPIC = "robot/set_super_enable";
const std::string SAWYER_STOP_TOPIC = "robot/set_super_stop";
const std::string SAWYER_RESET_TOPIC = "robot/set_super_reset";
const std::string SAWYER_DISPLAY_TOPIC = "robot/head_display";

const std::string SAWYER_RIGHT_GRIPPER_ST = "io/end_effector/state";
const std::string SAWYER_RIGHT_GRIPPER_PROP = "io/end_effector/config";

const std::string SAWYER_RIGHT_GRIPPER_ST_SUB = "io/end_effector/right_gripper/state";
const std::string SAWYER_RIGHT_GRIPPER_PROP_SUB = "io/end_effector/right_gripper/config";

const std::string SAWYER_JOINT_TOPIC = "robot/joint_states";

const std::string SAWYER_RIGHT_LASER_TOPIC = "sim/laserscan/right_hand_range/state";
const std::string SAWYER_RIGHT_IR_TOPIC = "robot/range/right_hand_range/state";
const std::string SAWYER_RIGHT_IR_STATE_TOPIC = "robot/analog_io/right_hand_range/state";
const std::string SAWYER_RIGHT_IR_INT_TOPIC = "robot/analog_io/right_hand_range/value_uint32";

const std::string SAWYER_NAV_LIGHT_TOPIC = "robot/digital_io/command";
const std::string SAWYER_RIGHTIL_TOPIC = "robot/digital_io/right_inner_light/state";
const std::string SAWYER_RIGHTOL_TOPIC = "robot/digital_io/right_outer_light/state";
const std::string SAWYER_TORSO_RIGHTIL_TOPIC = "robot/digital_io/torso_right_inner_light/state";
const std::string SAWYER_TORSO_RIGHTOL_TOPIC = "robot/digital_io/torso_right_outer_light/state";

const std::string SAWYER_HEAD_STATE_TOPIC = "robot/head/head_state";

const std::string SAWYER_RIGHT_GRAVITY_TOPIC = "robot/limb/right/gravity_compensation_torques";

const std::string SAWYER_SIM_STARTED = "robot/sim/started";

const int IMG_LOAD_ON_STARTUP_DELAY = 1;  // Timeout for publishing a single RSDK image on start up

enum nav_light_enum
{
    right_inner_light,
    torso_right_inner_light,
    right_outer_light,
    torso_right_outer_light
};

std::map<std::string, nav_light_enum> nav_light;
/**
 * Method to initialize the default values for all the variables, instantiate the publishers and subscribers
 */
bool sawyer_emulator::init()
{
    //Default values for the assembly state
    assembly_state.enabled = false;  // true if enabled
    assembly_state.stopped = false;  // true if stopped -- e-stop asserted
    assembly_state.error = false;    // true if a component of the assembly has an error
    assembly_state.estop_button = intera_core_msgs::AssemblyState::ESTOP_BUTTON_UNPRESSED;  // button status
    assembly_state.estop_source = intera_core_msgs::AssemblyState::ESTOP_SOURCE_NONE;  // If stopped is true, the source of the e-stop.

    // IONodeStatus (io/end_effector/state)
    right_grip_node_st.node.name = "End_effector";
    right_grip_node_st.node.status.tag = "readay";
    right_grip_node_st.node.status.key = "io/node/ready";
    right_grip_node_st.node.status.msg = "";

    right_grip_node_st.devices.push_back(intera_core_msgs::IOComponentStatus());
    right_grip_node_st.devices[0].name  = "right_gripper";
    right_grip_node_st.devices[0].status.tag = "readay";
    right_grip_node_st.devices[0].status.key = "io/gripper/ready";
    right_grip_node_st.devices[0].status.msg = "calibrated";

    right_grip_node_st.commands.push_back(ros::Time());

    //////////////////////////////////////////////////////////////////////////////////////
    // IODeviceStatus (io/end_effector/right_gripper/state)
    for (int tt =0; tt <50; tt++)
    {
        right_grip_st.commands.push_back(ros::Time());
    }

    // IODeviceStatus.device
    right_grip_st.device.name = "right_gripper";
    right_grip_st.device.status.tag = "readay";
    right_grip_st.device.status.key = "io/gripper/ready";
    right_grip_st.device.status.msg = "calibrated";

    // IODeviceStatus.ports
    int index = 0;
    right_grip_st.ports.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.ports[index].name = "current_limit_mA";
    right_grip_st.ports[index].format = "{\"data_type\":\"u16\",\"role\":\"sink\",\"type\":\"int\"}";
    right_grip_st.ports[index].data = "[1000]";
    index++;

    right_grip_st.ports.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.ports[index].name = "current_mA";
    right_grip_st.ports[index].format = "{\"data_type\":\"s16\",\"role\":\"source\",\"type\":\"int\"}";
    right_grip_st.ports[index].data = "[-44]";
    index++;

    right_grip_st.ports.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.ports[index].name = "error_flags";
    right_grip_st.ports[index].format = "{\"data_type\":\"u8\",\"role\":\"source\",\"type\":\"int\"}";
    right_grip_st.ports[index].data = "[0]";
    index++;

    right_grip_st.ports.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.ports[index].name = "holding_current_mA";
    right_grip_st.ports[index].format = "{\"data_type\":\"u16\",\"role\":\"sink\",\"type\":\"int\"}";
    right_grip_st.ports[index].data = "[349]";
    index++;

    right_grip_st.ports.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.ports[index].name = "ident";
    right_grip_st.ports[index].format = "{\"data_type\":\"u32\",\"role\":\"source\",\"type\":\"int\"}";
    right_grip_st.ports[index].data = "[65537]";
    index++;

    right_grip_st.ports.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.ports[index].name = "op_flag_calibrated";
    right_grip_st.ports[index].format = "{\"data_type\":\"u8\",\"role\":\"source\",\"type\":\"int\"}";
    right_grip_st.ports[index].data = "[0]";
    index++;

    right_grip_st.ports.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.ports[index].name = "op_mode";
    right_grip_st.ports[index].format = "{\"data_type\":\"u8\",\"role\":\"source\",\"type\":\"int\"}";
    right_grip_st.ports[index].data = "[4]";
    index++;

    right_grip_st.ports.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.ports[index].name = "pos_dead_zone";
    right_grip_st.ports[index].format = "{\"data_type\":\"u8\",\"role\":\"sink\",\"type\":\"int\"}";
    right_grip_st.ports[index].data = "[31]";
    index++;

    right_grip_st.ports.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.ports[index].name = "pos_target";
    right_grip_st.ports[index].format = "{\"data_type\":\"u16\",\"role\":\"sink\",\"type\":\"int\"}";
    right_grip_st.ports[index].data = "[610]";
    index++;

    right_grip_st.ports.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.ports[index].name = "position_steps";
    right_grip_st.ports[index].format = "{\"data_type\":\"u16\",\"role\":\"source\",\"type\":\"int\"}";
    right_grip_st.ports[index].data = "[0]";
    index++;

    right_grip_st.ports.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.ports[index].name = "pwm_out";
    right_grip_st.ports[index].format = "{\"data_type\":\"u16\",\"role\":\"source\",\"type\":\"int\"}";
    right_grip_st.ports[index].data = "[0]";
    index++;

    right_grip_st.ports.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.ports[index].name = "speed_steps";
    right_grip_st.ports[index].format = "{\"data_type\":\"u16\",\"role\":\"sink\",\"type\":\"int\"}";
    right_grip_st.ports[index].data = "[1024]";

    for (unsigned int i=0; i<right_grip_st.ports.size(); i++)
    {
        right_grip_st.ports[i].status = intera_core_msgs::IOStatus();
        right_grip_st.ports[i].status.tag = "readay";
        right_grip_st.ports[i].status.key = "io/Port/ready";
        right_grip_st.ports[i].status.msg = "";
    }

    // IODeviceStatus.signals
    index = 0;
    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[index].name = "calibrate";
    right_grip_st.signals[index].format = "{\"role\":\"output\",\"type\":\"bool\"}";
    right_grip_st.signals[index].data = "[false]";
    index++;

    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[index].name = "cmd_grip";
    right_grip_st.signals[index].format = "{\"role\":\"output\",\"type\":\"bool\"}";
    right_grip_st.signals[index].data = "[false]";
    index++;

    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[index].name = "dead_zone_m";
    right_grip_st.signals[index].format = "{\"role\":\"output\",\"type\":\"float\",\"units\":\"m\"}";
    right_grip_st.signals[index].data = "[0.002]";
    index++;

    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[index].name = "force_response_n";
    right_grip_st.signals[index].format = "{\"role\":\"input\",\"type\":\"float\"}";
    right_grip_st.signals[index].data = "[0.25]";
    index++;

    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[index].name = "go";
    right_grip_st.signals[index].format = "{\"role\":\"output\",\"type\":\"bool\"}";
    right_grip_st.signals[index].data = "[true]";
    index++;

    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[index].name = "has_error";
    right_grip_st.signals[index].format = "{\"role\":\"input\",\"type\":\"bool\"}";
    right_grip_st.signals[index].data = "[false]";
    index++;

    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[index].name = "is_calibrated";
    right_grip_st.signals[index].format = "{\"role\":\"input\",\"type\":\"bool\"}";
    right_grip_st.signals[index].data = "[true]";
    index++;

    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[index].name = "is_gripping";
    right_grip_st.signals[index].format = "{\"role\":\"input\",\"type\":\"bool\"}";
    right_grip_st.signals[index].data = "[false]";
    index++;

    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[index].name = "is_moving";
    right_grip_st.signals[index].format = "{\"role\":\"input\",\"type\":\"bool\"}";
    right_grip_st.signals[index].data = "[false]";
    index++;

    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[index].name = "object_kg";
    right_grip_st.signals[index].format = "{\"role\":\"output\",\"type\":\"float\",\"units\":\"kg\"}";
    right_grip_st.signals[index].data = "[0.0]";
    index++;

    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[index].name = "position_m";
    right_grip_st.signals[index].format = "{\"role\":\"output\",\"type\":\"float\",\"units\":\"m\"}";
    right_grip_st.signals[index].data = "[0.041667]";
    index++;

    gripper_signal_index = index; // srore for intera_interface.gripper.get_position()
    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[index].name = "position_response_m";
    right_grip_st.signals[index].format = "{\"role\":\"input\",\"type\":\"float\",\"units\":\"m\"}";
    right_grip_st.signals[index].data = "[0.0]";
    index++;

    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[index].name = "reboot";
    right_grip_st.signals[index].format = "{\"role\":\"output\",\"type\":\"bool\"}";
    right_grip_st.signals[index].data = "[false]";
    index++;

    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[index].name = "speed_mps";
    right_grip_st.signals[index].format = "{\"role\":\"output\",\"type\":\"float\",\"units\":\"m/s\"}";
    right_grip_st.signals[index].data = "[1.5]";

    for (unsigned int i=0; i<right_grip_st.signals.size(); i++)
    {
        right_grip_st.signals[i].status = intera_core_msgs::IOStatus();
        right_grip_st.signals[i].status.tag = "readay";
        right_grip_st.signals[i].status.key = "io/signal/ready";
        right_grip_st.signals[i].status.msg = "";
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // IONodeConfiguration (io/end_effector/config)
    // IONodeConfiguration.node
    right_grip_prop.node.name ="End_effector";
    right_grip_prop.node.config = "{\"id\":\"/io/end_effector\",\"name\":\"End_effector\",\"params\":{},\"plugins\":[\"package:/end_effector/config/rethink_end_effector_device_plugin.json\",\"package://end_effector/config/rethink_electric_gripper_plugin.json\",\"package://end_effector/config/rethink_smart_end_effector_plugin.json\",\"package://end_effector/config/rethink_vacuum_gripper_plugin.json\"]}";

    // IONodeConfiguration.devices
    right_grip_prop.devices.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_prop.devices[0].name = "right_gripper";
    right_grip_prop.devices[0].config = "{\"auto\":true,\"device_type\":513,\"engine_params\":{\"autoEnroll\":true,\"configurable\":true},\"id\":\"/io/end_effector/right_gripper\",\"label\":\"Electric Parallel Gripper\",\"manufacturer_name\":\"Rethink Robotics, Inc.\",\"params\":{},\"plugin\":\"RethinkElectricGripperPlugin\",\"product_name\":\"Electric Parallel Gripper\",\"type\":\"ElectricParallelGripper\"}";

    //////////////////////////////////////////////////////////////////////////////////////
    // IODeviceConfiguration (io/end_effector/right_gripper/config)
    right_grip_dev.device.name = "right_gripper";
    right_grip_dev.device.config = "{\"auto\":true,\"name\":\"right_gripper\"}";

    // IODeviceConfiguration.ports
    index = 0;
    right_grip_dev.ports.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.ports[index].name = "current_limit_mA";
    right_grip_dev.ports[index].config = "{\"default\":[1000],\"fixed\":true,\"format\":{\"data_type\":\"u16\",\"role\":\"sink\",\"type\":\"int\"},\"name\":\"current_limit_mA\",\"params\":{\"count\":2,\"index\":4},\"type\":\"Command\"}";
    index++;

    right_grip_dev.ports.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.ports[index].name = "current_mA";
    right_grip_dev.ports[index].config = "{\"fixed\":true,\"format\":{\"data_type\":\"s16\",\"role\":\"source\",\"type\":\"int\"},\"name\":\"current_mA\",\"params\":{\"count\":2,\"index\":4,\"update_rate_hz\":10.0},\"type\":\"Response\"}";
    index++;

    right_grip_dev.ports.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.ports[index].name = "error_flags";
    right_grip_dev.ports[index].config = "{\"fixed\":true,\"format\":{\"data_type\":\"u8\",\"role\":\"source\",\"type\":\"int\"},\"name\":\"error_flags\",\"params\":{\"count\":1,\"index\":3,\"response\":\"error_flags\"},\"type\":\"Response\"}";
    index++;

    right_grip_dev.ports.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.ports[index].name = "holding_current_mA";
    right_grip_dev.ports[index].config = "{\"default\":[349],\"fixed\":true,\"format\":{\"data_type\":\"u16\",\"role\":\"sink\",\"type\":\"int\"},\"name\":\"holding_current_mA\",\"params\":{\"count\":2,\"index\":8},\"type\":\"Command\"}";
    index++;

    right_grip_dev.ports.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.ports[index].name = "ident";
    right_grip_dev.ports[index].config = "{\"fixed\":true,\"format\":{\"data_type\":\"u32\",\"role\":\"source\",\"type\":\"int\"},\"name\":\"ident\",\"params\":{\"count\":3,\"index\":1},\"type\":\"Ident\"}";
    index++;

    right_grip_dev.ports.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.ports[index].name = "op_flag_calibrated";
    right_grip_dev.ports[index].config = "{\"fixed\":true,\"format\":{\"data_type\":\"u8\",\"role\":\"source\",\"type\":\"int\"},\"name\":\"op_flag_calibrated\",\"params\":{\"count\":1,\"index\":2,\"mask\":1,\"response\":\"is_calibrated\"},\"type\":\"Response\"}";
    index++;

    right_grip_dev.ports.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.ports[index].name = "op_mode";
    right_grip_dev.ports[index].config = "{\"fixed\":true,\"format\":{\"data_type\":\"u8\",\"role\":\"source\",\"type\":\"int\"},\"name\":\"op_mode\",\"params\":{\"count\":1,\"index\":1,\"response\":\"op_mode\"},\"type\":\"Response\"}";
    index++;

    right_grip_dev.ports.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.ports[index].name = "pos_dead_zone";
    right_grip_dev.ports[index].config = "{\"fixed\":true,\"format\":{\"data_type\":\"u8\",\"role\":\"sink\",\"type\":\"int\"},\"name\":\"pos_dead_zone\",\"params\":{\"count\":1,\"index\":1},\"type\":\"Command\"}";
    index++;

    right_grip_dev.ports.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.ports[index].name = "pos_target";
    right_grip_dev.ports[index].config = "{\"fixed\":true,\"format\":{\"data_type\":\"u16\",\"role\":\"sink\",\"type\":\"int\"},\"name\":\"pos_target\",\"params\":{\"count\":2,\"index\":2},\"type\":\"Command\"}";
    index++;

    right_grip_dev.ports.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.ports[index].name = "position_steps";
    right_grip_dev.ports[index].config = "{\"fixed\":true,\"format\":{\"data_type\":\"u16\",\"role\":\"source\",\"type\":\"int\"},\"name\":\"position_steps\",\"params\":{\"count\":2,\"index\":6,\"response\":\"position\",\"update_rate_hz\":10.0},\"type\":\"Response\"}";
    index++;

    right_grip_dev.ports.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.ports[index].name = "pwm_out";
    right_grip_dev.ports[index].config = "{\"fixed\":true,\"format\":{\"data_type\":\"u16\",\"role\":\"source\",\"type\":\"int\"},\"name\":\"pwm_out\",\"params\":{\"count\":2,\"index\":10},\"type\":\"Response\"}";
    index++;

    right_grip_dev.ports.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.ports[index].name = "speed_steps";
    right_grip_dev.ports[index].config = "{\"fixed\":true,\"format\":{\"data_type\":\"u16\",\"role\":\"sink\",\"type\":\"int\"},\"name\":\"speed_steps\",\"params\":{\"count\":2,\"index\":6},\"type\":\"Command\"}";


    // IODeviceConfiguration.signals
    index = 0;
    right_grip_dev.signals.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.signals[index].name = "calibrate";
    right_grip_dev.signals[index].config = "{\"default\":[false], \"fixed\":true,\"format\":{\"role\":\"output\",\"type\":\"bool\"},\"name\":\"calibrate\",\"params\":{\"value\":\"calibrate\"},\"type\":\"Gripper\"}";
    index++;

    right_grip_dev.signals.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.signals[index].name = "cmd_grip";
    right_grip_dev.signals[index].config = "{\"default\":[false],\"fixed\":true,\"format\":{\"role\":\"output\",\"type\":\"bool\"},\"name\":\"cmd_grip\",\"params\":{\"script\":\"if (cmd_grip) pos_target = 0; else pos_target = 610;\"},\"sink\":\"pos_target\",\"type\":\"Derived\"}";
    index++;

    right_grip_dev.signals.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.signals[index].name = "dead_zone_m";
    right_grip_dev.signals[index].config = "{\"default\":[0.002],\"fixed\":true,\"format\":{\"max\":0.020333,\"min\":0.0,\"role\":\"output\",\"type\":\"float\",\"units\":\"m\"},\"name\":\"dead_zone_m\",\"params\":{\"port_max\":305,\"port_min\":3,\"signal_max\":0.0203330000000000,\"signal_min\":0.000100800000000000},\"sink\":\"pos_dead_zone\",\"type\":\"Convert\"}";
    index++;

    right_grip_dev.signals.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.signals[index].name = "force_response_n";
    right_grip_dev.signals[index].config = "{\"default\":[0.0],\"fixed\":true,\"format\":{\"multiple\":0.250000000000000,\"role\":\"input\",\"type\":\"float\"},\"name\":\"force_response_n\",\"params\":{\"port_max\":1000,\"port_min\":-1000,\"signal_max\":10.0,\"signal_min\":-10.0},\"source\":\"current_mA\",\"type\":\"Convert\"}";
    index++;

    right_grip_dev.signals.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.signals[index].name = "go";
    right_grip_dev.signals[index].config = "{\"default\":[true],\"fixed\":true,\"format\":{\"role\":\"output\",\"type\":\"bool\"},\"name\":\"go\",\"params\":{\"value\":\"go\"},\"type\":\"Gripper\"}";
    index++;

    right_grip_dev.signals.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.signals[index].name = "has_error";
    right_grip_dev.signals[index].config = "{\"default\":[false],\"fixed\":true,\"format\":{\"role\":\"input\",\"type\":\"bool\"},\"name\":\"has_error\",\"params\":{},\"source\":\"error_flags\",\"type\":\"Data\"}";
    index++;

    right_grip_dev.signals.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.signals[index].name = "is_calibrated";
    right_grip_dev.signals[index].config = "{\"default\":[true],\"fixed\":true,\"format\":{\"role\":\"input\",\"type\":\"bool\"},\"name\":\"is_calibrated\",\"params\":{},\"source\":\"op_flag_calibrated\",\"type\":\"Data\"}";
    index++;

    right_grip_dev.signals.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.signals[index].name = "is_gripping";
    right_grip_dev.signals[index].config = "{\"default\":[false],\"fixed\":true,\"format\":{\"role\":\"input\",\"type\":\"bool\"},\"name\":\"is_gripping\",\"params\":{\"value\":\"gripping\"},\"type\":\"Gripper\"}";
    index++;

    right_grip_dev.signals.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.signals[index].name = "is_moving";
    right_grip_dev.signals[index].config = "{\"default\":[false],\"fixed\":true,\"format\":{\"role\":\"input\",\"type\":\"bool\"},\"name\":\"is_moving\",\"params\":{\"value\":\"moving\"},\"type\":\"Gripper\"}";
    index++;

    right_grip_dev.signals.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.signals[index].name = "object_kg";
    right_grip_dev.signals[index].config = "{\"default\":[0.0],\"fixed\":true,\"format\":{\"max\":4.0,\"min\":0.0,\"role\":\"output\",\"type\":\"float\",\"units\":\"kg\"},\"label\":\"Object Mass - Electric Parallel Gripper Tip\",\"name\":\"object_kg\",\"params\":{\"urdf_link_name\":\"right_gripper\"},\"type\":\"Mass\"}";
    index++;

    right_grip_dev.signals.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.signals[index].name = "position_m";
    right_grip_dev.signals[index].config = "{\"default\":[0.041667],\"fixed\":true,\"format\":{\"max\":0.041667,\"min\":0.0,\"role\":\"output\",\"type\":\"float\",\"units\":\"m\"},\"name\":\"position_m\",\"params\":{\"convert\":\"position\",\"port_max\":610,\"port_min\":0,\"signal_max\":0.0416670000000000,\"signal_min\":0.0},\"sink\":\"pos_target\",\"type\":\"Convert\"}";
    index++;

    right_grip_dev.signals.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.signals[index].name = "position_response_m";
    right_grip_dev.signals[index].config = "{\"default\":[0.0],\"fixed\":true,\"format\":{\"max\":0.0416670000000000,\"min\":0.0,\"role\":\"input\",\"type\":\"float\",\"units\":\"m\"},\"name\":\"position_response_m\",\"params\":{\"convert\":\"position_response\",\"port_max\":610,\"port_min\":50,\"signal_max\":0.0416670000000000,\"signal_min\":0.0},\"source\":\"position_steps\",\"type\":\"Convert\"}";
    index++;

    right_grip_dev.signals.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.signals[index].name = "reboot";
    right_grip_dev.signals[index].config = "{\"default\":[false],\"fixed\":true,\"format\":{\"role\":\"output\",\"type\":\"bool\"},\"name\":\"reboot\",\"params\":{\"value\":\"reboot\"},\"type\":\"Gripper\"}";
    index++;

    right_grip_dev.signals.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_dev.signals[index].name = "speed_mps";
    right_grip_dev.signals[index].config = "{\"default\":[1.5],\"fixed\":true,\"format\":{\"role\":\"output\",\"type\":\"float\",\"units\":\"m/s\"},\"name\":\"speed_mps\",\"params\":{\"port_max\":1024,\"port_min\":50,\"signal_max\":0.25,\"signal_min\":0.02},\"sink\":\"speed_steps\",\"type\":\"Convert\"}";


    //////////////////////////////////////////////////////////////////////////////////////
    rightIL_nav_light.isInputOnly = false;
    rightOL_nav_light.isInputOnly = false;
    torso_rightIL_nav_light.isInputOnly = false;
    torso_rightOL_nav_light.isInputOnly = false;

    rightIL_nav_light.state = intera_core_msgs::DigitalIOState::OFF;
    rightOL_nav_light.state = intera_core_msgs::DigitalIOState::OFF;
    torso_rightIL_nav_light.state = intera_core_msgs::DigitalIOState::OFF;
    torso_rightOL_nav_light.state = intera_core_msgs::DigitalIOState::OFF;

    head_msg.pan = 0;
    head_msg.isTurning = false;

    isStopped = false;

    right_gravity.header.frame_id="base";

    // Initialize the map that would be used in the nav_light_cb
    nav_light["right_inner_light"] = right_inner_light;
    nav_light["torso_right_inner_light"] = torso_right_inner_light;
    nav_light["right_outer_light"] = right_outer_light;
    nav_light["torso_right_outer_light"] = torso_right_outer_light;

    // Initialize the publishers
    assembly_state_pub = n.advertise<intera_core_msgs::AssemblyState>(SAWYER_STATE_TOPIC, 1);

    right_grip_st_pub = n.advertise<intera_core_msgs::IONodeStatus>(SAWYER_RIGHT_GRIPPER_ST, 1);
    right_grip_prop_pub = n.advertise<intera_core_msgs::IONodeConfiguration>(SAWYER_RIGHT_GRIPPER_PROP, 1);

    right_grip_st_sub_pub = n.advertise<intera_core_msgs::IODeviceStatus>(SAWYER_RIGHT_GRIPPER_ST_SUB, 1);
    right_grip_prop_sub_pub = n.advertise<intera_core_msgs::IODeviceConfiguration>(SAWYER_RIGHT_GRIPPER_PROP_SUB, 1);

    right_ir_pub = n.advertise<sensor_msgs::Range>(SAWYER_RIGHT_IR_TOPIC, 1);
    right_ir_state_pub = n.advertise<intera_core_msgs::AnalogIOState>(SAWYER_RIGHT_IR_STATE_TOPIC, 1);
    right_ir_int_pub = n.advertise<std_msgs::UInt32>(SAWYER_RIGHT_IR_INT_TOPIC, 1);

    right_inner_light_pub = n.advertise<intera_core_msgs::DigitalIOState>(SAWYER_RIGHTIL_TOPIC, 1);
    right_outer_light_pub = n.advertise<intera_core_msgs::DigitalIOState>(SAWYER_RIGHTOL_TOPIC, 1);
    torso_right_inner_light_pub = n.advertise<intera_core_msgs::DigitalIOState>(SAWYER_TORSO_RIGHTIL_TOPIC, 1);
    torso_right_outer_light_pub = n.advertise<intera_core_msgs::DigitalIOState>(SAWYER_TORSO_RIGHTOL_TOPIC, 1);

    right_grav_pub = n.advertise<intera_core_msgs::SEAJointState>(SAWYER_RIGHT_GRAVITY_TOPIC, 1);

    head_pub = n.advertise<intera_core_msgs::HeadState>(SAWYER_HEAD_STATE_TOPIC,1);

    // Latched Simulator Started Publisher
    sim_started_pub = n.advertise<std_msgs::Empty>(SAWYER_SIM_STARTED, 1, true);

    // Initialize the subscribers
    enable_sub = n.subscribe(SAWYER_ENABLE_TOPIC, 100, &sawyer_emulator::enable_cb, this);
    stop_sub = n.subscribe(SAWYER_STOP_TOPIC, 100, &sawyer_emulator::stop_cb, this);
    reset_sub = n.subscribe(SAWYER_RESET_TOPIC, 100, &sawyer_emulator::reset_cb, this);
    jnt_st = n.subscribe(SAWYER_JOINT_TOPIC, 100, &sawyer_emulator::update_jnt_st, this);

    // right_laser_sub = n.subscribe(SAWYER_RIGHT_LASER_TOPIC, 100, &sawyer_emulator::right_laser_cb, this);
    nav_light_sub = n.subscribe(SAWYER_NAV_LIGHT_TOPIC, 100, &sawyer_emulator::nav_light_cb, this);
}

/**
 * Method that publishes the emulated interfaces' states and data at 100 Hz
 * @param img_path that refers the path of the image that loads on start up
 */
void sawyer_emulator::publish(const std::string &img_path)
{
    ros::Rate loop_rate(100);

    arm_kinematics::Kinematics kin;
    kin.init_grav();

    image_transport::ImageTransport it(n);
    image_transport::Publisher display_pub = it.advertise(SAWYER_DISPLAY_TOPIC, 1);

    // Read OpenCV Mat image and convert it to ROS message
    cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
    try
    {
        cv_ptr->image = cv::imread(img_path, CV_LOAD_IMAGE_UNCHANGED);
        if (cv_ptr->image.data)
        {
            cv_ptr->encoding = sensor_msgs::image_encodings::BGR8;
            sleep(IMG_LOAD_ON_STARTUP_DELAY);  // Wait for the model to load
            display_pub.publish(cv_ptr->toImageMsg());
        }
    }
    catch (std::exception e)
    {
        ROS_WARN("Unable to load the Startup picture on Sawyer's display screen %s",e.what());
    }

    ROS_INFO("Simulator is loaded and started successfully");
    std_msgs::Empty started_msg;
    sim_started_pub.publish(started_msg);

    while (ros::ok())
    {
        assembly_state_pub.publish(assembly_state);
        ros::Time stamp = ros::Time::now();
        right_grip_node_st.time = stamp;
        right_grip_node_st.commands[0] = stamp;
        right_grip_st_pub.publish(right_grip_node_st);

        right_grip_prop.time = stamp;
        right_grip_prop_pub.publish(right_grip_prop);

        right_grip_st.time = stamp;
        // intera_io/io_interface.py should handle this, but it cannot.
        for (int tt=0; tt <50; tt++)
        {
            right_grip_st.commands[tt] = (stamp + ros::Duration(tt*0.001));
        }

        right_grip_st_sub_pub.publish(right_grip_st);

        right_grip_dev.time = stamp;
        right_grip_prop_sub_pub.publish(right_grip_dev);

        right_ir_pub.publish(right_ir);
        right_ir_state_pub.publish(right_ir_state);
        right_ir_int_pub.publish(right_ir_int);
        right_inner_light_pub.publish(rightIL_nav_light);
        right_outer_light_pub.publish(rightOL_nav_light);
        torso_right_inner_light_pub.publish(torso_rightIL_nav_light);
        torso_right_outer_light_pub.publish(torso_rightOL_nav_light);

        head_pub.publish(head_msg);
        kin.getGravityTorques(jstate_msg, right_gravity, assembly_state.enabled);
        right_gravity.header.stamp = ros::Time::now();
        right_grav_pub.publish(right_gravity);
        ros::spinOnce();
        loop_rate.sleep();
    }
}
/**
 * Method to enable the robot
 */
void sawyer_emulator::enable_cb(const std_msgs::Bool &msg)
{
    if (msg.data && !isStopped)
    {
        assembly_state.enabled = true;
    }
    else
    {
        assembly_state.enabled = false;
    }
    assembly_state.stopped = false;
    assembly_state.estop_button = intera_core_msgs::AssemblyState::ESTOP_BUTTON_UNPRESSED;
    assembly_state.estop_source = intera_core_msgs::AssemblyState::ESTOP_SOURCE_NONE;
    enable = assembly_state.enabled;
}

/**
 * Method to stop the robot and capture the source of the stop
 */
void sawyer_emulator::stop_cb(const std_msgs::Empty &msg)
{
    assembly_state.enabled = false;
    assembly_state.stopped = true;
    assembly_state.estop_button = intera_core_msgs::AssemblyState::ESTOP_BUTTON_UNPRESSED;
    assembly_state.estop_source = intera_core_msgs::AssemblyState::ESTOP_SOURCE_UNKNOWN;
    enable = false;
    isStopped = true;
}

/**
 * Method resets all the values to False and 0s
 */
void sawyer_emulator::reset_cb(const std_msgs::Empty &msg)
{
    assembly_state.enabled = false;
    assembly_state.stopped = false;
    assembly_state.estop_button = intera_core_msgs::AssemblyState::ESTOP_BUTTON_UNPRESSED;
    assembly_state.estop_source = intera_core_msgs::AssemblyState::ESTOP_SOURCE_NONE;
    assembly_state.error = false;
    enable = false;
    isStopped = false;
}

void sawyer_emulator::nav_light_cb(const intera_core_msgs::DigitalOutputCommand &msg)
{
    int res;
    if (msg.value)
    {
        res = intera_core_msgs::DigitalIOState::ON;
    }
    else
    {
        res = intera_core_msgs::DigitalIOState::OFF;
    }

    switch (nav_light.find(msg.name)->second)
    {
        case right_inner_light:
            rightIL_nav_light.state = res;
            break;
        case torso_right_inner_light:
            torso_rightIL_nav_light.state = res;
            break;
        case right_outer_light:
            rightOL_nav_light.state = res;
            break;
        case torso_right_outer_light:
            torso_rightOL_nav_light.state = res;
            break;
        default:
            ROS_ERROR("Not a valid component id");
            break;
    }
}

void sawyer_emulator::update_jnt_st(const sensor_msgs::JointState &msg)
{
    jstate_msg = msg;
    float threshold = 0.0009;
    right_gravity.actual_position.resize(right_gravity.name.size());
    right_gravity.actual_velocity.resize(right_gravity.name.size());
    right_gravity.actual_effort.resize(right_gravity.name.size());

    for (int i = 0; i < msg.name.size(); i++)
    {
        if (msg.name[i] == "head_pan")
        {
            if (fabs(float(head_msg.pan) - float(msg.position[i])) > threshold)
            {
                head_msg.isTurning = true;
            }
            else
            {
                head_msg.isTurning = false;
            }
            head_msg.pan = msg.position[i];
        }
        else if (msg.name[i] == "r_gripper_l_finger_joint")
        {
            std::ostringstream stm ;
            // stm << (msg.position[i]/0.020833)*100;
            stm << msg.position[i];

            // intera_interface.gripper.get_position() will return "position_response_m"
            right_grip_st.signals[gripper_signal_index].data = std::string("[") + stm.str() + std::string("]");
        }
        else
        {
            for (int j=0;j<right_gravity.name.size();j++)
            {
                if (msg.name[i] == right_gravity.name[j])
                {
                    right_gravity.actual_position[j] = msg.position[i];
                    right_gravity.actual_velocity[j] = msg.velocity[i];
                    right_gravity.actual_effort[j] = msg.effort[i];
                    break;
                }
            }
        }
    }
}

}  //namespace

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "sawyer_emulator");

    std::string img_path = argc > 1 ? argv[1] : "";
    sawyer_en::sawyer_emulator emulate;
    bool result = emulate.init();
    emulate.publish(img_path);

    return 0;
}
