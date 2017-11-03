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

    //Default values for the right gripper end effector states
    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[0].name = "calibrate";
    right_grip_st.signals[0].format =
"{\n"
"    \"type\": \"bool\",\n"
"    \"role\": \"input\"\n"
"}";
    right_grip_st.signals[0].data = "[true]";
    right_grip_st.signals[0].status = intera_core_msgs::IOStatus();
    right_grip_st.signals[0].status.tag = "readay";
    right_grip_st.signals[0].status.key = "io/ready";
    right_grip_st.signals[0].status.msg = "";

    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[1].name = "position_m";
    right_grip_st.signals[1].format =
"{\n"
"    \"type\": \"float\",\n"
"    \"role\": \"input\"\n"
"}";
    right_grip_st.signals[1].data = "[0.0]";
    right_grip_st.signals[1].status = intera_core_msgs::IOStatus();
    right_grip_st.signals[1].status.tag = "readay";
    right_grip_st.signals[1].status.key = "io/ready";
    right_grip_st.signals[1].status.msg = "";

    right_grip_st.signals.push_back(intera_core_msgs::IODataStatus());
    right_grip_st.signals[2].name = "position_response_m";
    right_grip_st.signals[2].format =
"{\n"
"    \"type\": \"float\",\n"
"    \"role\": \"output\"\n"
"}";
    right_grip_st.signals[2].data = "[0.0]";
    right_grip_st.signals[2].status = intera_core_msgs::IOStatus();
    right_grip_st.signals[2].status.tag = "readay";
    right_grip_st.signals[2].status.key = "io/ready";
    right_grip_st.signals[2].status.msg = "";

    right_grip_prop.devices.push_back(intera_core_msgs::IOComponentConfiguration());
    right_grip_prop.devices[0].name = "right_gripper";
    right_grip_prop.devices[0].config = "";

    right_grip_dev.device.name = "right_gripper";

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

    right_grip_st_pub = n.advertise<intera_core_msgs::IODeviceStatus>(SAWYER_RIGHT_GRIPPER_ST, 1);
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

        right_grip_st_pub.publish(right_grip_st);
        right_grip_prop_pub.publish(right_grip_prop);

        right_grip_st_sub_pub.publish(right_grip_st);
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
            stm << (msg.position[i]/0.020833)*100;
            // intera_interface.gripper.get_position() will return "position_response_m"
            right_grip_st.signals[2].data = std::string("[") + stm.str() + std::string("]");
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
