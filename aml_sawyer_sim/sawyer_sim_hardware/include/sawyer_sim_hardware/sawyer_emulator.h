/*********************************************************************
 # Copyright (c) 2013-2015, Rethink Robotics
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

#ifndef sawyer_emulator_H_
#define sawyer_emulator_H_

#include "ros/ros.h"
#include <std_msgs/Bool.h>
#include <std_msgs/Empty.h>
#include <std_msgs/UInt32.h>

//Sawyer Specific Messages
#include <intera_core_msgs/AssemblyState.h>
#include <intera_core_msgs/JointCommand.h>
#include <intera_core_msgs/AnalogIOState.h>
#include <intera_core_msgs/DigitalOutputCommand.h>
#include <intera_core_msgs/DigitalIOState.h>
#include <intera_core_msgs/HeadState.h>
#include <intera_core_msgs/SEAJointState.h>

#include <sensor_msgs/JointState.h>

//ROS-Opencv Headers
#include <image_transport/image_transport.h>
#include <opencv/cvwimage.h>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Range.h>
#include <sensor_msgs/LaserScan.h>

#include <sawyer_sim_kinematics/arm_kinematics.h>
#include <cmath>
#include <map>

namespace sawyer_en
{

    class sawyer_emulator
    {

    public:
        /**
         * Method to initialize the default values for all the variables, instantiate the publishers and    * subscribers
         * @param img_path that refers the path of the image that loads on start up
         */
        sawyer_emulator()
        {
        }

        bool init();

        /**
         * Method to start the publishers
         * @param Nodehandle to initialize the image transport
         * @param img_path that refers the path of the image that loads on start up
         */
        void publish(const std::string &img_path);

    private:
        bool enable;
        //Subscribers
        ros::Subscriber enable_sub, stop_sub, reset_sub,
            right_laser_sub, nav_light_sub, jnt_st;

        // Infrared publishers
        ros::Publisher right_ir_pub, right_ir_int_pub, right_ir_state_pub;

        // Navigator publishers
        ros::Publisher right_itb_innerL_pub,
            torso_right_innerL_pub, right_itb_outerL_pub, torso_right_outerL_pub;

        // General state publishers
        ros::Publisher assembly_state_pub, head_pub;

        // Gravity Publishers
        ros::Publisher right_grav_pub;

        ros::NodeHandle n;
        ros::Timer head_nod_timer;

        intera_core_msgs::HeadState head_msg;
        intera_core_msgs::AssemblyState assembly_state;

        intera_core_msgs::AnalogIOState right_ir_state;

        intera_core_msgs::DigitalIOState rightIL_nav_light,
            rightOL_nav_light, torso_rightIL_nav_light, torso_rightOL_nav_light;

        intera_core_msgs::SEAJointState right_gravity;
        sensor_msgs::JointState jstate_msg;
        sensor_msgs::Range right_ir;
        std_msgs::UInt32 right_ir_int;

        bool isStopped;

        /**
         * Callback function to enable the robot
         */
        void enable_cb(const std_msgs::Bool &msg);

        /**
         * Callback function to stop the robot and capture the source of the stop
         */
        void stop_cb(const std_msgs::Empty &msg);

        /**
         * Callback function to reset all the state values to False and 0s
         */
        void reset_cb(const std_msgs::Empty &msg);

        /**
         * Callback function to update the right laser values
         */
        void right_laser_cb(const sensor_msgs::LaserScan &msg);

        /**
         * Callback function to update the navigators' light values
         */
        void nav_light_cb(const intera_core_msgs::DigitalOutputCommand &msg);

        /**
         * Method that updates the gravity variable
         */
        void update_jnt_st(const sensor_msgs::JointState &msg);
    };
}  // namespace

#endif /* SAWYER_EMULATOR_H_ */
