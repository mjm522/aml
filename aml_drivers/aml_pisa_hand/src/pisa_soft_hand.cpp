#include <sys/mman.h>
#include <cmath>
#include <time.h>
#include <signal.h>
#include <stdexcept>

// ROS headers
#include <ros/ros.h>
#include <ros/console.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Duration.h>
#include <controller_manager/controller_manager.h>

// qb tools
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include "qbmove_communications.h"
#include "definitions.h"

///////////////////////////////////////////////////////////////
/*
The code might fail to open the port, this might due to the permission issues,
run,  *******sudo chmod 0666 /dev/ttyUSB0********to enable anyone to read and write the port when the
device is connected.
*/
///////////////////////////////////////////////////////////////

class PisaSoftHand
  {
  public:
    // from RobotHW
    PisaSoftHand(ros::NodeHandle nh);
    bool start();
    void stop();
    void reset();
    bool read(ros::Time time,  ros::Duration period);
    void write(ros::Time time, ros::Duration period);
    // from QBtools
    // port selection by id
    int port_selection(const int id, char* my_port);
    int open_port(char*);
    void set_input(short int);
    void pos_cmd_callback(const std_msgs::Float32::ConstPtr& msg);
    void read_status_callback(const std_msgs::Float32::ConstPtr& msg);

  private:

    // Node handle
    ros::NodeHandle nh_;

    // QB tools Parameters
    int device_id_;
    comm_settings comm_settings_t_;
    char port_[255];
    bool everything_ok;

  protected:

  };

  PisaSoftHand::PisaSoftHand(ros::NodeHandle nh) :
    nh_(nh)
  {
    everything_ok = false;
    if( ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug) )
    ros::console::notifyLoggerLevelsChanged();
    //subsciber for sending position cmd
  }

  bool PisaSoftHand::start()
  {
    nh_.param("device_id", device_id_, BROADCAST_ID);

    // Finally, do the qb tools thing
    // get the port by id
    while(true)
    {
      if( port_selection(device_id_, port_) )
      {
        // open the port
        assert(open_port(port_));
        // and activate the hand
        commActivate(&comm_settings_t_, device_id_, 1);
        ROS_INFO("Done initialization !");
        return true;
      }
      else
      {
        ROS_WARN("Hand not found in all available ports, trying again...");
        // randomize the waiting to avoid conflict in files (probabilistically speaking)
        sleep( 4*((double) rand() / (RAND_MAX)) );
      }
    }

  }

  // port selection by id
  int PisaSoftHand::port_selection(const int id, char* my_port)
  {
    int num_ports = 0;
    char ports[10][255];

    char system_cmd[80];
    strcpy(system_cmd,"sudo chmod 0666 ");

    num_ports = RS485listPorts(ports);

    ROS_DEBUG_STREAM("Search id in " << num_ports << " serial ports available...");

    if(num_ports)
    {
      for(int i = 0; i < num_ports; i++)
      {
        ROS_DEBUG_STREAM("Checking port: " << ports[i]);

        //TODO: how to run a sudo command from this file to open the port?
        // strcat(system_cmd, ports[i]);
        // std::system(system_cmd);

        int aux_int;
        comm_settings comm_settings_t;
        char list_of_devices[255];

        openRS485(&comm_settings_t, ports[i]);

        if(comm_settings_t.file_handle == INVALID_HANDLE_VALUE)
        {
          ROS_DEBUG_STREAM("Couldn't connect to the serial port. Continue with the next available.");
          continue;
        }

        aux_int = RS485ListDevices(&comm_settings_t, list_of_devices);

        ROS_DEBUG_STREAM( "Number of devices: " << aux_int );

        if(aux_int > 1 || aux_int <= 0)
        {
          ROS_WARN_STREAM("The current port has " << aux_int << " devices connected, but it must be only one... that is not a SoftHand");
        }
        else
        {
          ROS_DEBUG_STREAM("List of devices:");
          for(int d = 0; d < aux_int; ++d)
          {
            ROS_DEBUG_STREAM( static_cast<int>(list_of_devices[d]) );
            ROS_DEBUG_STREAM( "searching id" << id );
            if( static_cast<int>(list_of_devices[d]) == id )
            {
              ROS_DEBUG_STREAM("Hand found at port: " << ports[i] << " !");
              strcpy(my_port, ports[i]);
              closeRS485(&comm_settings_t);
              sleep(1);
              return 1;
            }
            sleep(1);
          }
        }
        closeRS485(&comm_settings_t);
      }
      return 0;
    }
    else
    {
        ROS_ERROR("No serial port available.");
        return 0;
    }
  }

  int PisaSoftHand::open_port(char* port) 
  {
    ROS_DEBUG_STREAM("Opening serial port: " << port << " for hand_id: " << device_id_);
    fflush(stdout);

    openRS485(&comm_settings_t_, port);

    if(comm_settings_t_.file_handle == INVALID_HANDLE_VALUE)
    {
        ROS_ERROR("Couldn't connect to the selected serial port.");
        return 0;
    }
    usleep(500000);
    printf("Done.\n");
    everything_ok = true;
    return 1;
  }

 bool PisaSoftHand::read(ros::Time time, ros::Duration period)
  {
      // read from hand
      static short int inputs[2];
      commGetMeasurements(&comm_settings_t_, device_id_, inputs);

      static short int currents[2];
      commGetCurrents(&comm_settings_t_, device_id_, currents);

      ROS_INFO("The current reading is %d",currents[0]*1.0);

      return true;
  }

  void PisaSoftHand::write(ros::Time time, ros::Duration period)
  {
      static int warning = 0;

      // write to the hand
      short int pos;
      pos = (short int)(17000.0*1);
      set_input(pos);

      return;
  }

  void PisaSoftHand::set_input(short int pos)
  {
    static short int inputs[2];

    inputs[0] = pos;
    inputs[1] = 0;

    commSetInputs(&comm_settings_t_, device_id_, inputs);
    return;
  }

  void PisaSoftHand::pos_cmd_callback(const std_msgs::Float32::ConstPtr& msg)
  {
    if (!everything_ok)
    {
      ROS_INFO("Something is wrong somewhere ...");
      return;
    }

    float cmd = msg->data;
    short int pos;
    ROS_INFO("Received command %d", cmd);

    if (cmd < 0.)
    {
      cmd = 0.;
    }
    else if(cmd > 1.)
    {
      cmd = 1.;
    }

    pos = (short int)(17000.0*cmd);

    ROS_DEBUG_STREAM("The commanded value: " << pos);

    set_input(pos);

  }

  void PisaSoftHand::read_status_callback(const std_msgs::Float32::ConstPtr& msg)
  {
    if (!everything_ok)
    {
      ROS_INFO("Something is wrong somewhere ...");
      return;
    }


    if (msg->data == 3)
    {
      struct timespec ts = {0, 0};
    
      ros::Time now(ts.tv_sec, ts.tv_nsec);
      ros::Duration period(1.0);

      read(now, period);
    }
    
  }

int main( int argc, char** argv )
{
  // initialize ROS
  ros::init(argc, argv, "pisa_soft_hand", ros::init_options::NoSigintHandler);

  // ros spinner
  ros::AsyncSpinner spinner(1);
  spinner.start();

  // construct the lwr
  ros::NodeHandle sh_nh("");
  PisaSoftHand sh_robot(sh_nh);

  // configuration routines
  sh_robot.start();

  // timer variables
  struct timespec ts = {0, 0};
  ros::Time last(ts.tv_sec, ts.tv_nsec), now(ts.tv_sec, ts.tv_nsec);
  ros::Duration period(1.0);

  //start the subscriber
  ros::Subscriber sh_pos_cmd = sh_nh.subscribe("soft_hand_pos_cmd", 1000, &PisaSoftHand::pos_cmd_callback, &sh_robot);
  ros::Subscriber sh_read_status = sh_nh.subscribe("soft_hand_read_current", 1000, &PisaSoftHand::read_status_callback, &sh_robot);

  while(ros::ok()) 
  {
    
    // get the time / period
    if (!clock_gettime(CLOCK_REALTIME, &ts)) 
    {
      now.sec = ts.tv_sec;
      now.nsec = ts.tv_nsec;
      period = now - last;
      last = now;
    } 
    else 
    {
      ROS_FATAL("Failed to poll realtime clock!");
      break;
    }

  }
  return 0;
}
