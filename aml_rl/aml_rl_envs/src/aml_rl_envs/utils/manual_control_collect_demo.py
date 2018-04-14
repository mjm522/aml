import os
import time
import numpy as np
import pybullet as p
from collect_demo import plot_demo


def min_jerk_step(x, xd, xdd, goal, tau, dt):
    # computes the update of x,xd,xdd for the next time step dt given that we are
    # currently at x,xd,xdd, and that we have tau until we want to reach
    # the goal
    if tau<dt:
        return goal,0,0
    dist = goal - x;
    a1   = 0
    a0   = xdd * tau**2
    v1   = 0
    v0   = xd * tau
    t1=dt;
    t2=dt**2;
    t3=dt**3;
    t4=dt**4;
    t5=dt**5;
    c1 = (6.*dist + (a1 - a0)/2. - 3.*(v0 + v1))/tau**5;
    c2 = (-15.*dist + (3.*a0 - 2.*a1)/2. + 8.*v0 + 7.*v1)/tau**4;
    c3 = (10.*dist+ (a1 - 3.*a0)/2. - 6.*v0 - 4.*v1)/tau**3;
    c4 = xdd/2.;
    c5 = xd;
    c6 = x;
    x   = c1*t5 + c2*t4 + c3*t3 + c4*t2 + c5*t1 + c6;
    xd  = 5.*c1*t4 + 4*c2*t3 + 3*c3*t2 + 2*c4*t1 + c5;
    xdd = 20.*c1*t3 + 12.*c2*t2 + 6.*c3*t1 + 2.*c4;
    return x,xd,xdd


def make_demonstrations(start, goal, only_pos=True):
    # generate the minimum jerk trajectory
    t   = start
    td  = 0
    tdd = 0
    T   = []
    
    #for the demonstration
    tau = 1.0
    dt  = 0.01
    
    #get demonstration
    for i in range(int(2*tau/dt)):
        t,td,tdd = min_jerk_step(t, td, tdd, goal, tau-i*dt, dt)
        T.append([t, td, tdd])
    T = np.asarray(T).squeeze()

    if only_pos:
        return T[:,0]
    else:
        return T 


def get_desired_path(start, goal, only_pos):
    des_path = {}
    des_path['info'] = 'each joint_idx in the dictionary has a 200x3 interpolated trajectory if not only_pos is set else contains a 200x1 position trajectory'
    
    for k in range(len(start)):
        traj = make_demonstrations(start[k], goal[k], only_pos=only_pos)

        des_path['joint_' + str(k)] = np.asarray(traj).squeeze()

    return des_path


class ManualDemoCollect():

    def __init__(self, manipulator, demo_path=os.environ["AML_DATA"]+'/data/new_data'):
        """
        constructor of the class
        Args:
        manipulator of type sawyer/threelink/kuka
        """

        self._robot = manipulator
        self._robot_id = manipulator._robot_id
        self._start = None
        self._end = None

        if not os.path.exists(demo_path):
            os.makedirs(demo_path)

        self._demo_folder_path = demo_path

        self._file_name = self._demo_folder_path + '/block_js_ee_pos_data_dual_rotate_2'+'.csv'

        self.setup_manual_control(default_joint_state=self._robot._jnt_postns)

    def setup_manual_control(self, default_joint_state=None):
        """
        setup manula control sliders on the bullet window
        """
        
        self.joint_ids=[]
        self.param_ids=[]

        # gravId = p.addUserDebugParameter("gravity",-10,10,-10)

        jnt_idx = -1
        for j in self._robot._movable_jnts:
            jnt_idx += 1
            # p.changeDynamics(self._robot_id,j,linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self._robot_id, j)
            #print(info)
            joint_name = info[1]
            joint_type = info[2]
            if (joint_type==p.JOINT_PRISMATIC or joint_type==p.JOINT_REVOLUTE):
                self.joint_ids.append(j)
                if default_joint_state is None:
                    self.param_ids.append(p.addUserDebugParameter(joint_name.decode("utf-8"),-4,4,0.))
                else:
                    self.param_ids.append(p.addUserDebugParameter(joint_name.decode("utf-8"),-4,4, default_joint_state[jnt_idx]))

        p.setRealTimeSimulation(1)

    def get_start_goal(self):
        """
        this function is to capture a start joint position
        and goal end position based on key board events
        """

        while self._start is None or self._end is None:

            keyboard_events = p.getKeyboardEvents()
            
            if 115L in keyboard_events.keys():

                jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque = self._robot.get_jnt_state()

                self._start = jnt_pos 

                # print "Demo start location saved: Start is now: \t", self._start

            elif 101L in keyboard_events.keys():

                jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque = self._robot.get_jnt_state()

                self._end = jnt_pos

                # print "Demo end location saved: Start is now: \t", self._end

            for i in range(len(self.param_ids)):
                c = self.param_ids[i]
                targetPos = p.readUserDebugParameter(c)
                p.setJointMotorControl2(self._robot_id, self.joint_ids[i], p.POSITION_CONTROL, targetPos, force=5*240.)


    def collect_demo(self):
        """
        this function extrapolates between two data point and 
        take the data
        """

        self.get_start_goal()

        num_joints = len(self._robot.movable_joints)

        if self._start is not None and self._end is not None:

            min_jerk_path = get_desired_path(self._start, self._end, only_pos=False)

        #data holder
        data = np.zeros([200, 2*num_joints+3*3+4])

        #arrange joint position and joint velocity
        for k in range(num_joints):

            #store the position data
            data[:, k] =  min_jerk_path['joint_'+str(k)][:,0]
            #store the velocity data
            data[:, k+num_joints] =  min_jerk_path['joint_'+str(k)][:,1]

        self._robot.set_joint_state(self._start)

        #apply the same data to obtain additional data
        for k in range(200):

            # apply the same position on to the robot
            for i in range(len(self._robot.movable_joints)):

                p.setJointMotorControl2(self._robot_id, self._robot.movable_joints[i], p.POSITION_CONTROL, data[k, i], force=5*240.)

            time.sleep(0.01)

            ee_pos, ee_ori = self._robot.ee_pose()
            ee_vel, ee_omg = self._robot.ee_velocity()

            data[k, 2*num_joints:2*num_joints+3]     = ee_pos
            data[k, 2*num_joints+3:2*num_joints+7]   = ee_ori
            data[k, 2*num_joints+7:2*num_joints+10]  = ee_vel
            data[k, 2*num_joints+10:2*num_joints+13] = ee_omg

        self._demo_data = np.round(data, 3)

        #save the data
        np.savetxt(self._file_name, self._demo_data, delimiter=",")

    
    def check_demo(self, start_idx=18):
        """
        This function replays a store set of data
        """

        if not os.path.isfile(self._file_name):

            raise Exception("The given path to demo does not exist, given path: \n" + self._file_name)

        demo_data  = np.genfromtxt(self._file_name, delimiter=',')

        plot_demo(demo_data, start_idx=start_idx)

        raw_input("Plotted end effector trajectory, press any key to play")

        self._robot.set_joint_state(demo_data[0, :len(self._robot.movable_joints)])

        for k in range(200):

            for i in range(len(self._robot.movable_joints)):

                p.setJointMotorControl2(self._robot_id, self._robot.movable_joints[i], p.POSITION_CONTROL, demo_data[k, i], force=5*240.)

            time.sleep(0.01)

        raw_input("Demo done!, press any key to exit")

        
