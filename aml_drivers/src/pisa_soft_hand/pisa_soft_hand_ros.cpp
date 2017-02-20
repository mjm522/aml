#include <sys/mman.h>
#include <cmath>
#include <time.h>
#include <signal.h>
#include <stdexcept>

// ROS headers
#include <ros/ros.h>
#include <controller_manager/controller_manager.h>
#include <std_msgs/Duration.h>


// qb tools
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <qbmoveAPI/qbmove_communications.h>


class PisaSoftHand
{
private:
	// Node handle
    ros::NodeHandle nh_;
	// QB tools Parameters
    int device_id_;
    comm_settings comm_settings_t_;
	char port_[255];

public:
	bool start();
	// from QBtools
    // port selection by id
    int port_selection(const int id, char* my_port);
    int open_port(char*);
    void set_input(short int);
protected:
};

 PisaSoftHand::PisaSoftHand(ros::NodeHandle nh) :
    nh_(nh)
  {}

   bool PisaSoftHand::start()
  {

    // construct a new lwr device (interface and state storage)
    this->device_.reset( new PisaSoftHand::SHRDevice() );

     nh_.param("device_id", device_id_, BROADCAST_ID);

    // initialize and set to zero the state and command values
    this->device_->init();
    this->device_->reset();

    ROS_INFO("Register state and position interfaces");

    // register ros-controls interfaces
    this->registerInterface(&state_interface_);
    this->registerInterface(&position_interface_);

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

    num_ports = RS485listPorts(ports);

    ROS_DEBUG_STREAM("Search id in " << num_ports << " serial ports available...");

    if(num_ports)
    {
      for(int i = 0; i < num_ports; i++)
      {
        ROS_DEBUG_STREAM("Checking port: " << ports[i]);

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
    return 1;
  }

 void PisaSoftHand::set_input(short int pos)
  {
    static short int inputs[2];

    inputs[0] = pos;
    inputs[1] = 0;

    commSetInputs(&comm_settings_t_, device_id_, inputs);
    return;
  }


 int main( int argc, char** argv )
{
  // initialize ROS
  ros::init(argc, argv, "lwr_hw_interface", ros::init_options::NoSigintHandler);

  // ros spinner
  ros::AsyncSpinner spinner(1);
  spinner.start();

  // custom signal handlers
  signal(SIGTERM, quitRequested);
  signal(SIGINT, quitRequested);
  signal(SIGHUP, quitRequested);

  // construct the lwr
  ros::NodeHandle sh_nh("");
  soft_hand_hw::PisaSoftHand sh_robot(sh_nh);

  // configuration routines
  sh_robot.start();

  // timer variables
  struct timespec ts = {0, 0};
  ros::Time last(ts.tv_sec, ts.tv_nsec), now(ts.tv_sec, ts.tv_nsec);
  ros::Duration period(1.0);

  //the controller manager
  // controller_manager::ControllerManager manager(&sh_robot, sh_nh);

  // run as fast as possible
  while( !g_quit ) 
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

    // read the state from the soft hand
    if(!sh_robot.read(now, period))
    {
      g_quit = true;
      break;
    }

    // update the controllers
    // manager.update(now, period);

    // write the command to the lwr
    // sh_robot.write(now, period);
  }

  std::cerr <<" Stopping spinner..." << std::endl;
  spinner.stop();

  std::cerr << "Stopping soft hand..." << std::endl;
  sh_robot.stop();

  std::cerr << "Done!" << std::endl;

  return 0;
}