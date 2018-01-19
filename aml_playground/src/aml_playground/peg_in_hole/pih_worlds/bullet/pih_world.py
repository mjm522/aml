import numpy as np
import pandas as pd
import pybullet as pb
from aml_robot.bullet.bullet_robot import BulletRobot
from aml_data_collec_utils.record_sample import RecordSample
from aml_playground.peg_in_hole.pih_worlds.bullet.config import pih_world_config


#global macros, how to access it from bullet directly?

MOUSE_MOVE_EVENT = 1
MOUSE_BUTTON_EVENT = 2
KEY_IS_DOWN = 3
KEY_WAS_RELEASED = 4


class BoxObject(BulletRobot):

    def __init__(self, box_id):

        super(BoxObject, self).__init__(robot_id=box_id, config=pih_world_config)

    def get_effect(self):
        p, q   = self.get_pos_ori()
        status = {}
        status['pos'] = p
        #all the files in package follows np.quaternion convention, that is
        # w,x,y,z while ros follows x,y,z,w convention
        status['ori'] = np.array([q[3],q[0],q[1],q[2]])

        return status

    def reset(self):
        #implement this
        pass

class PegHole():

    def __init__(self, hole_id, pos = [0. ,0. ,0.], ori = [0., 0., 0., 1]):

        self._pos = pos
        self._ori = ori
        self._id = hole_id
        self.set_default_pos(np.array(pos), np.array(ori))

    def set_default_pos(self, pos, ori):

        pb.resetBasePositionAndOrientation(self._id, pos, ori)
        self._default_pos = pos
        self._default_ori = ori


class PIHWorld():
    """
    Peg in the hole environment that makes use of the bullet physics engine
    """

    def __init__(self, config):

        """
        Constructor of the class
        Args: 
        config:
                dt : time step of the simulation
                world_path : path to the URDF file depicting the world
                peg_path: path to the URDF file depicting the peg to be inserted
                hole_path: path to the URDF of a static body that has a hole
                robot_path: path to the URDF file of the manipulator
                ctrl_type: specify what types of control scheme is to be use to control the manipulator
        """

        phys_id = pb.connect(pb.SHARED_MEMORY)
        
        if (phys_id<0):
            phys_id = pb.connect(pb.GUI)
        
        pb.resetSimulation()
        
        pb.setTimeStep(config['dt'])

        pb.setGravity(0., 0.,-10.)

        #load the world urdf file, returns an id
        self._world_id = pb.loadURDF(config['world_path'])
  
        manipulator = 

        pb.setRealTimeSimulation(0)

        self._peg = BoxObject(box_id=pb.loadURDF(config['peg_path']))

        self._hole = PegHole(hole_id=pb.loadURDF(config['hole_path'], 
                             useFixedBase=True), 
                             pos=[0,0,1], 
                             ori=[0, 0, -0.707, 0.707])

        
        self._manipulator = BulletRobot(robot_id=pb.loadURDF(config['robot_path'], useFixedBase=True, globalScaling=1.5), 
                                        ee_link_idx=2, 
                                        config=pih_world_config, 
                                        enable_force_torque_sensors = True)

        pb.resetBasePositionAndOrientation(self._world_id, np.array([0., 0., -0.5]), np.array([0.,0.,0.,1]))

        self._peg.configure_default_pos(np.array([0, -1, 1.5]), np.array([0., 0., 0., 1]))

        self._manipulator.configure_default_pos(np.array([0, -1.3, 3.]),  np.array([0., 1, 0., 0]))

        self._config   = config

        self._gains = [1, 1, 1]
        
        self._forces = []
        self._torques = []
        self._contact_points = []

        #there are three control modes implemented in this file
        #key word 'vel' : velocity control mode
        #key word 'torq' : torque control mode
        #key word 'pos' : position control mode
        self._ctrlr_type = 'vel'


        #demo collecting variables
        self._left_button_down = False
        self._demo_collection_start = False
        self._demo_point_count = 0
        #variable required for correct operation
        self._traj_point_1, _  = self._manipulator.get_ee_pose()


    def reset(self, noise=0.01):
        """
        This function is very necessary for the RL setup
        the first part resets the peg to the oiriginal part
        the secodn part reset the manipulators to the original part added by an optional noise
        Args:
        noise = coefficient of the gain command
        """

        self._peg.reset()
        self._manipulator.set_jnt_state([0.,0.,0.])


    def step(self):

        pb.stepSimulation()

    def update(self, ctrl_cmd):
        """
        This function updates the control commands for the PIH world
        the only entity that can be controlled is the manipulator
        Args: 
        ctrl_cmd = np.array, dimension: num_joint_states x 1
        """

        if self._ctrlr_type == 'vel':

            self._manipulator.set_joint_velocities(ctrl_cmd)

        elif self._ctrlr_type =='torq':

            self._manipulator.set_joint_torques(ctrl_cmd)

        elif self._ctrlr_type == 'pos':

            self._manipulator.set_jnt_state(ctrl_cmd)

        else:
            raise Exception("Unknown type control")


    def compute_os_ctrlr_cmd(self, os_set_point, Kp):
        """
        This function computes the control command for the manipulator
        in the operational space
        Args: 
        os_set_point = desired operational space set point
        Kp =  gains in the operational space
        """

        return np.multiply(Kp, np.array([0, -0.05, -0.05]))

    def compute_js_ctrlr_cmd(self, js_set_point, Kp):
        """
        This function computes the control command for the manipulator 
        in the joint space
        Args:
        js_set_point = desired joint setpoint
        Kp = gains of the system
        """

        return np.multiply(Kp, np.array([0, -0.05, -0.05]))

    def on_shutdown(self):

        d = {'contact point' : pd.Series(self._contact_points),
             'forces'        : pd.Series(self._forces),
             'torques'       : pd.Series(self._torques)}

        self._forces = []; self._torques = []; self._contact_points = []

        df = pd.DataFrame(d)
        df = df.rename_axis('Gains: '+str(self._gains), axis=1)
        file_name = self._config['data_folder_path']+'pih '+str(self._gains)+'.csv'
        df.to_csv(file_name)
        print df['contact point']

    def get_force_torque_details(self):

        ee_in_contact_with_box = False
        in_contact = False

        [fx,fy,fz,tx,ty,tz] = [0,0,0,0,0,0]
        contact_point = [0.,0.,0.]

        contact_details = self._manipulator.get_contact_points()

        if len(contact_details) > 4:

            in_contact = True

            if contact_details[2] == self._peg._id and contact_details[3] == self._manipulator._ee_link_idx:

                ee_in_contact_with_box = True

        if in_contact:

            [fx,fy,fz], [tx,ty,tz] = self._manipulator.get_joint_details(joint_idx = self._manipulator._ee_link_idx, flag = 'force_torque')

            if ee_in_contact_with_box:

                # print "\n\nForce: ", fx, fy, fz
                # print "\nTorque: ", tx, ty, tz
                contact_point = contact_details[5]

            else:
                print "Robot in contact with other object. Object ID:", "Square_Hole_Table" if contact_details[2] == self._hole._id else contact_details[2]
                print "Force: ", fx, fy, fz
                print "Torque: ", tx, ty, tz

        self._forces.append((fx,fy,fz))
        self._torques.append((tx,ty,tz))
        self._contact_points.append(contact_point)


    def draw_trajectory(self, point_1, point_2, colour=[1,0,0], line_width=1.5):
        """
        This function adds colour line between points point_1 and point_2 in the bullet
        Args:
        point_1: starting point => [x,y,z]
        point_2: ending_point => [x,y,z]
        """
        pb.addUserDebugLine(point_1, point_2, lifeTime=0, lineColorRGB=colour, lineWidth=line_width)

    def collect_demo(self, demo_draw_interwal=10):
        """
        This function is for collecting the demo trajectory
        Args: 
        demo_draw_interwal = this parameter decides at what interwal the plot needs to be updated
        """
        #get the tuple of mouse events
        mouse_events = pb.getMouseEvents()

        #check the tuple only if its length is more than zero
        if len(mouse_events) > 0:

            if mouse_events[0][0] == MOUSE_BUTTON_EVENT:
                
                #left button
                if mouse_events[0][3] == 0:

                    #button is down
                    if mouse_events[0][4] == KEY_IS_DOWN:

                        self._left_button_down = True

                    #button is released
                    if mouse_events[0][4] == KEY_WAS_RELEASED:

                        self._left_button_down = False

            #collect the demo only if the button was pressed
            if self._left_button_down:

                if mouse_events[0][0] == MOUSE_MOVE_EVENT:

                    self._demo_collection_start = True

                    print "Start collecting demo"

                    #start collecting demo point
                    self._demo_point_count += 1

                    traj_point_2, _ = self._manipulator.get_ee_pose()

                    print "traj_point", traj_point_2

                    #draw the lines in specific interwal
                    if self._demo_point_count % demo_draw_interwal == 0:

                        self.draw_trajectory(point_1=self._traj_point_1, point_2=traj_point_2)

                        #store the previous value to draw from this point
                        self._traj_point_1 = traj_point_2
            
            else:

                #stop the collection only if it was started
                if self._demo_collection_start:

                    print "Stop collecting demo"

                    self._demo_collection_start = False
                    self._demo_point_count = 0 
            