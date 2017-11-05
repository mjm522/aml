# sawyer_simulator
---------------------------------

Gazebo simulation with emulated interfaces for the Sawyer robot
* based on [baxter_simulator](https://github.com/RethinkRobotics/baxter_simulator)
* using `intera_common/intera_core_msgs` not `baxter_common/baxter_core_msgs`

### Repository overview
---------------------------------------
     .
     |
     +-- sawyer_simulator/        Metapackage
     |
     +-- sawyer_sim_description/  Urdf for Gazebo 
     |   +-- urdf/
     |   +-- launch/
     |
     +-- sawyer_gazebo/           Gazebo interface for the sawyer that loads the models into simulation
     |   +-- src/
     |   +-- launch/
     |   +-- worlds/
     |
     +-- sawyer_sim_controllers/  Controller plugins for sawyer
     |   +-- src/
     |   +-- include/
     |   +-- config/
     |
     +-- sawyer_sim_hardware/     This emulates the hardware interfaces of sawyer 
     |   +-- src/
     |   +-- include/
     |   +-- config/
     |   +-- launch/
     |
     +-- sawyer_sim_kinematics/   Implementation of IK, FK and gravity compensation for sawyer 
     |   +-- src/
     |   +-- include/
     |   +-- launch/

### TODO
---------------------------------------
* Gravity compensation (bug? urdf?)
* Tuning pid gain in sawyer_sim_hardware/config/sawyer_sim_controllers.yaml
* Sensor reading
