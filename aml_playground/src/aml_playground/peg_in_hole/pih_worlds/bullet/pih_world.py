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
  
        pb.setRealTimeSimulation(0)

        # self._peg = BoxObject(box_id=pb.loadURDF(config['peg_path']))

        # self._hole = PegHole(hole_id=pb.loadURDF(config['hole_path'], 
        #                      useFixedBase=True), 
        #                      pos=[0,0,1], 
        #                      ori=[0, 0, -0.707, 0.707])

        self._robot_id = pb.loadURDF(config['robot_path'], useFixedBase=True, globalScaling=1.5)

        self._manipulator = BulletRobot(robot_id=self._robot_id,
                                        ee_link_idx=3, 
                                        config=pih_world_config, 
                                        enable_force_torque_sensors = True)


        pb.resetBasePositionAndOrientation(self._world_id, np.array([0., 0., -0.5]), np.array([0.,0.,0.,1]))

        # self._peg.configure_default_pos(np.array([0, -1, 1.5]), np.array([0., 0., 0., 1]))

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

        #for saving data
        self._ee_pos_array = []
        self._ee_vel_array = []
        self._js_pos_array = []
        self._js_vel_array = []


    def reset(self, noise=0.01):
        """
        This function is very necessary for the RL setup
        the first part resets the peg to the oiriginal part
        the secodn part reset the manipulators to the original part added by an optional noise
        Args:
        noise = coefficient of the gain command
        """

        # self._peg.reset()
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


    def compute_os_imp_ctrlr_cmd(self, os_set_point, Kp):
        """
        This function computes the operation space impedance controller
        according to the slides of Morteza
        """

        ee_pos, ee_ori = self._manipulator.get_ee_pose()
        ee_vel, ee_omg = self._manipulator.get_ee_velocity_from_bullet()
        jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque = self._manipulator.get_jnt_state()

        delta_pos      = os_set_point - ee_pos
        delta_vel      = np.zeros(3) - ee_vel

        ee_force = np.zeros(3)


        #compute the jacobian
        #note: the objVeolocities and objAccelerations have to be list and not numpy array
        linearJacobian, angularJacobian = pb.calculateJacobian(bodyUniqueId=self._robot_id, 
                                                               linkIndex= self._manipulator._ee_link_idx,
                                                               localPosition=ee_pos,
                                                               objPositions=jnt_pos+[0], #last inactive joint
                                                               objVelocities=np.zeros(4).tolist(),
                                                               objAccelerations=np.zeros(4).tolist())
        #the return is of type tuple, so convert it to array
        linearJacobian  = np.asarray(linearJacobian)
        angularJacobian = np.asarray(angularJacobian)



        #compute mass matrix
        Mq = np.asarray(pb.calculateMassMatrix(bodyUniqueId=self._robot_id,
                                               objPositions=jnt_pos+[0])) #last inactive joint


        #Compute cartesian space inertia matrix
        Mq_inv    = np.linalg.inv(Mq)
        Mcart_inv = np.dot(np.dot(linearJacobian, Mq_inv), linearJacobian.transpose())
        Mcart     = np.linalg.pinv(Mcart_inv, rcond=1e-3)


        #inertia shaping, as same as eee inertia
        Md_inv  = Mcart_inv #(np.linalg.inv(self._Md))


        # #secondary pose torque
        # tau_pose = np.zeros_like(q) - np.dot(self._kd_q, dq)
        # tau_pose = np.dot(null_proj, tau_pose)


        #from morteza slide
        xdd = np.zeros(3)
        tmp = np.dot(Mcart, Md_inv)
        f = ee_force + np.dot(Mcart, xdd) + np.dot(tmp, (np.multiply(Kp, delta_pos) + np.multiply(np.sqrt(Kp), delta_vel)) ) - np.dot(tmp, ee_force)
        tau_task = np.dot( np.dot(linearJacobian.transpose(), Mcart),  f)

        cmd = tau_task 

        return cmd


    def compute_os_ctrlr_cmd(self, os_set_point, Kp):
        """
        This function computes the control command for the manipulator
        in the operational space
        Args: 
        os_set_point = desired operational space set point
        Kp =  gains in the operational space
        """

        ee_pos, ee_ori = self._manipulator.get_ee_pose()
        ee_vel, ee_omg = self._manipulator.get_ee_velocity_from_bullet()
        jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque = self._manipulator.get_jnt_state()


        #compute the jacobian
        #note: the objVeolocities and objAccelerations have to be list and not numpy array
        linearJacobian, angularJacobian = pb.calculateJacobian(bodyUniqueId=self._robot_id, 
                                                               linkIndex=self._manipulator._ee_link_idx,
                                                               localPosition=ee_pos,
                                                               objPositions=jnt_pos,
                                                               objVelocities=np.zeros(3).tolist(),
                                                               objAccelerations=np.zeros(3).tolist())

        linearJacobian  = np.asarray(linearJacobian)
        angularJacobian = np.asarray(angularJacobian)


        #proportional term * error in setpoint - derivative term * ee_velocity
        error_term = np.multiply(Kp, os_set_point-ee_pos) - np.multiply(np.sqrt(Kp), ee_vel)

        ctrl_cmd = np.dot(np.linalg.pinv(linearJacobian, rcond=1e-3), error_term)

        #converts nan to zero
        return np.nan_to_num(ctrl_cmd)

    def compute_js_ctrlr_cmd(self, js_set_point, Kp):
        """
        This function computes the control command for the manipulator 
        in the joint space
        Args:
        js_set_point = desired joint setpoint
        Kp = gains of the system


        for the inbuilt function:
            targetVelocities = optional
            linkIndices = to which force has to be applied = joint indices
            controlMode = pb.POSITION_CONTROL, pb.TORQUE_CONTROL, pb.VELOCITY_CONTROL
        """
        #torque = bullet.calculateInverseDynamics(id_robot, obj_pos, obj_vel, obj_acc)
        # actions = pi.act(obs)
    
        #print(" ".join(["%+0.2f"%x for x in obs]))
        #print("Motors")
        #print(motors)

        #for m in range(len(motors)):
            #print("motor_power")
            #print(motor_power[m])
            #print("actions[m]")
            #print(actions[m])
        #p.setJointMotorControl2(human, motors[m], controlMode=p.TORQUE_CONTROL, force=motor_power[m]*actions[m]*0.082)
        #p.setJointMotorControl2(human1, motors[m], controlMode=p.TORQUE_CONTROL, force=motor_power[m]*actions[m]*0.082)
            
        # forces = [0.] * len(motors)
        # for m in range(len(motors)):
        #     forces[m] = motor_power[m]*actions[m]*0.082
        # pb.setJointMotorControlArray(human, motors,controlMode=p.TORQUE_CONTROL, forces=forces)

        # pb.setJointMotorControlArray(bodyUniqueId=,
        #                              linkIndices=,
        #                              controlMode=,
        #                              targetPositions=,
        #                              targetVelocities=,
        #                              forces=,
        #                              positionGains=,
        #                              velocityGains=)

        jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque = self._manipulator.get_jnt_state()

        error_term = np.multiply(Kp, js_set_point-jnt_pos) - np.multiply(np.sqrt(Kp), jnt_vel)

        return error_term

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


    def draw_trajectory(self, point_1, point_2, colour=[1,0,0], line_width=4.5):
        """
        This function adds colour line between points point_1 and point_2 in the bullet
        Args:
        point_1: starting point => [x,y,z]
        point_2: ending_point => [x,y,z]
        """
        pb.addUserDebugLine(point_1, point_2, lifeTime=0, lineColorRGB=colour, lineWidth=line_width)


    def get_observation(self):

        jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque = self._manipulator.get_jnt_state()

        ee_pos, ee_ori = self._manipulator.get_ee_pose()

        ee_vel, ee_omg = self._manipulator.get_ee_velocity_from_bullet()

        # np.hstack([jnt_pos, jnt_vel, ee_pos, ee_ori, ee_vel, ee_omg])

        return np.hstack([jnt_pos, jnt_vel, ee_pos, ee_vel])

    def collect_demo(self, demo_draw_interwal=10):
        """
        This function is for collecting the demo trajectory. It works with the help of mouse events.
        To record a demo, click the left button on the mouse and drag the end effector of the manipulator.
        To stop and save the recorded demo, simply release the left button press.
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

                    ee_vel, ee_omg = self._manipulator.get_ee_velocity_from_bullet()

                    jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque = self._manipulator.get_jnt_state()

                    print "traj_point", traj_point_2

                    self._ee_pos_array.append(traj_point_2)
                    self._ee_vel_array.append(ee_vel)
                    self._js_pos_array.append(jnt_pos)
                    self._js_vel_array.append(jnt_vel)
 
                    #draw the lines in specific interwal
                    if self._demo_point_count % demo_draw_interwal == 0:

                        self.draw_trajectory(point_1=self._traj_point_1, point_2=traj_point_2)

                        #store the previous value to draw from this point
                        self._traj_point_1 = traj_point_2
            
            else:

                #stop the collection only if it was started
                if self._demo_collection_start:

                    print "Stop collecting demo"

                    data = np.hstack([np.round(np.asarray(self._js_pos_array).squeeze() ,3), 
                                      np.round(np.asarray(self._js_vel_array).squeeze() ,3), 
                                      np.round(np.asarray(self._ee_pos_array).squeeze(), 3), 
                                      np.round(np.asarray(self._ee_vel_array).squeeze(), 3),])

                    file_name = self._config['demo_folder_path']+'pih_js_ee_pos_data'+'.csv'

                    np.savetxt(file_name, data, delimiter=",")

                    self._demo_collection_start = False
                    self._demo_point_count = 0

                    self._ee_pos_array = []
                    self._ee_vel_array = []
            
