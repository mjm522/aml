import numpy as np
import quaternion
from baxter_robot import BaxterArm
import time
import baxter_interface
from baxter_interface import CHECK_VERSION
import rospy
import tf
from tf import TransformListener

import sys
sys.argv

curr_time_in_sec = lambda: int(round(time.time() * 1e6))

def compute_w(q,qdot):
	Q = np.array([[-q[1],-q[2],-q[3]],
	    		  [q[0],-q[3],q[2]],
	    		  [q[3],q[0],-q[1]],
	    		  [-q[2],q[1],q[0]]])
	w = 2*np.dot(Q.T,qdot)
	return w

def min_jerk_step(x,xd,xdd,goal, tau, dt):
	# function [x,xd,xdd] = min_jerk_step(x,xd,xdd,goal,tau, dt) computes
	# the update of x,xd,xdd for the next time step dt given that we are
	# currently at x,xd,xdd, and that we have tau until we want to reach
	# the goal

	if tau<dt:
		return np.nan, np.nan, np.nan

	dist = goal - x

	a1   = 0
	a0   = xdd * tau**2
	v1   = 0
	v0   = xd * tau

	t1=dt
	t2=dt**2
	t3=dt**3
	t4=dt**4
	t5=dt**5

	c1 = (6.*dist + (a1 - a0)/2. - 3.*(v0 + v1))/(tau**5)
	c2 = (-15.*dist + (3.*a0 - 2.*a1)/2. + 8.*v0 + 7.*v1)/(tau**4)
	c3 = (10.*dist+ (a1 - 3.*a0)/2. - 6.*v0 - 4.*v1)/(tau**3)
	c4 = xdd/2.
	c5 = xd
	c6 = x

	x   = c1*t5 + c2*t4 + c3*t3 + c4*t2 + c5*t1 + c6
	xd  = 5.*c1*t4 + 4.*c2*t3 + 3.*c3*t2 + 2.*c4*t1 + c5
	xdd = 20.*c1*t3 + 12.*c2*t2 + 6.*c3*t1 + 2.*c4
	return x, xd, xdd


def min_jerk_step_pos(start_pos, goal_pos, tau, dt):
	timesteps = np.arange(0, 2*tau, dt)
	final_p = np.zeros((len(timesteps),3))
	for j in range(3):
		# generate the minimum jerk trajectory between each component of position
		t=start_pos[j]
		td=0
		tdd=0
		goal = goal_pos[j]
		T = np.zeros((len(timesteps),3))
		for i in range(len(timesteps)):
			t,td,tdd = min_jerk_step(t,td,tdd,goal,tau-i*dt,dt)
			T[i,:]   = np.array([t, td, tdd])
		  	#print i, '\t', T[i,:]
		#print T[:,j]
		final_p[:,j] = T[:,0].copy()

	#differentiate
	final_v = np.diff(final_p, axis=0)/dt
	#add initial velocity
	final_v = np.vstack([np.zeros((1,3)),final_v])

	#compute angular acceleration
	final_a = np.diff(final_v, axis=0)/dt
	#add initial acceleration
	final_a = np.vstack([np.zeros((1,3)),final_a])

	return final_p, final_v, final_a


def  min_jerk_step_qt(start_qt, goal_qt, tau, dt):
    timesteps = np.arange(0, 2*tau, dt)
    final_q = np.zeros((len(timesteps),4))
    for j in range(4):
    	# generate the minimum jerk trajectory between each component of
    	# quarternions
    	t=start_qt[j]
    	td=0
    	tdd=0
    	goal = goal_qt[j]
    	T = np.zeros((len(timesteps),3))
    	for i in range(len(timesteps)):
    		t,td,tdd = min_jerk_step(t,td,tdd,goal,tau-i*dt,dt)
    		T[i,:]   = np.array([t, td, tdd])
    	  	#print i, '\t', T[i,:]
    	#print T[:,j]
    	final_q[:,j] = T[:,0].copy()

    #normalize the quarternions
    for i in range(len(timesteps)):
        tmp = final_q[i,:]
        final_q[i,:] = tmp/np.linalg.norm(tmp)

    #differentiate
    final_dot = np.diff(final_q, axis=0)/dt
    #add initial velocity
    final_dot = np.vstack([np.zeros((1,4)),final_dot])

    #compute angular velocity
    final_w = np.zeros((len(timesteps),3))
    for i in range(len(timesteps)):
        final_w[i,:] = compute_w(final_q[i,:], final_dot[i,:])

    #compute angular acceleration
    final_al = np.diff(final_w, axis=0)/dt
    #add initial acceleration
    final_al = np.vstack([np.zeros((1,3)),final_al])

    #append zero as the first component.
    #print final_a.shape
    final_w = np.hstack([np.zeros((len(timesteps),1)),final_w])
    final_al = np.hstack([np.zeros((len(timesteps),1)),final_al])

    #converting to a quaternion array
    final_q = quaternion.as_quat_array(final_q)

    return final_q, final_w, final_al

def quatdiff(quat_curr, quat_des):
    return quat_des[0]*quat_curr[1:4] - quat_curr[0]*quat_des[1:4] + np.cross(quat_des[1:4],quat_curr[1:4])

class MinJerkController():

    def __init__(self, extern_call=False, trial_arm=None, aux_arm=None, tau=5., dt=0.05):
        
        if extern_call:
            if trial_arm is None or aux_arm is None:
                print "Pass the arm handles, it can't be none"
                raise ValueError
            else:
                self.left_arm  = trial_arm
                self.right_arm = aux_arm
        else:
            self.left_arm 	= BaxterArm('left') #object of type Baxter from baxter_mechanism
            self.right_arm  = BaxterArm('right')
            self.start_pos 	= None
            self.goal_pos 	= None
            self.start_qt 	= None
            self.goal_qt 	= None
            self.tau 		= tau
            self.dt 		= dt
            self.timesteps  = int(self.tau/self.dt)

            baxter = baxter_interface.RobotEnable(CHECK_VERSION)
            baxter.enable()
        self.osc_pos_threshold = 0.01

    def configure(self, start_pos, start_ori, goal_pos, goal_ori):

        self.start_pos  = start_pos
        self.goal_pos   = goal_pos
        self.start_qt   = quaternion.as_float_array(start_ori)[0]
        self.goal_qt    = quaternion.as_float_array(goal_ori)[0]
  
    def get_arm_handle(self, limb_idx=0):
        if limb_idx == 0:
            return self.left_arm
        elif limb_idx == 1:
            return self.right_arm
        else:
            print "Unknown limb index"
            raise ValueError
    
    def set_neutral(self):
        self.left_arm.move_to_neutral()
        self.right_arm.move_to_neutral()

    def tuck_arms(self):
        tuck_l = np.array([-1.0, -2.07,  3.0, 2.55,  0.0, 0.01,  0.0])
        tuck_r = np.array([1.0, -2.07, -3.0, 2.55, -0.0, 0.01,  0.0])
        self.left_arm.move_to_joint_position(tuck_l)
        self.right_arm.move_to_joint_position(tuck_r)

    def untuck_arms(self):
        untuck_l = np.array([-0.08, -1.0, -1.19, 1.94,  0.67, 1.03, -0.50])
        untuck_r = np.array([0.08, -1.0,  1.19, 1.94, -0.67, 1.03,  0.50])
        self.left_arm.move_to_joint_position(untuck_l)
        self.right_arm.move_to_joint_position(untuck_r)

    def get_min_jerk_trajectory(self):

        final_q, final_w, final_al  = min_jerk_step_qt(self.start_qt, self.goal_qt,  self.tau, self.dt)

        final_p, final_v, final_a   = min_jerk_step_pos(self.start_pos, self.goal_pos, self.tau, self.dt)

        min_jerk_traj = {}
        #position trajectory
        min_jerk_traj['pos_traj'] = final_p
        #velocity trajectory
        min_jerk_traj['vel_traj'] = final_v
        #acceleration trajectory
        min_jerk_traj['acc_traj'] = final_al
        #orientation trajectory
        min_jerk_traj['ori_traj'] = final_q
        #angular velocity trajectory
        min_jerk_traj['omg_traj'] = final_w[:,1:4]
        #angular acceleration trajectory
        min_jerk_traj['alp_traj'] = final_a[:,1:4]

        return min_jerk_traj

    def osc_position_cmd(self, goal_pos, goal_ori=None, limb_idx=0, orientation_ctrl=False):
        
        arm = self.get_arm_handle(limb_idx)
        
        jnt_start = arm.angles()

        error                   = 100.
        alpha                   = 0.1
        t                       = 0

        curr_pos, curr_ori  = arm.get_ee_pose()
        jac                 = arm.get_jacobian_from_joints()

        arm_state      = arm._update_state()
        q              = arm_state['position']

        dq             = arm_state['velocity']
        delta_pos      = goal_pos-curr_pos


        if orientation_ctrl:
            if goal_ori is None:
                print "For orientation control, pass goal orientation!"
                raise ValueError

            delta_ori       = quatdiff(quaternion.as_float_array(goal_ori)[0], quaternion.as_float_array(curr_ori)[0])
            delta           = np.hstack([delta_pos, delta_ori])
        else:

            jac             = jac[0:3,:]
            delta           = delta_pos

        jac_star            = np.dot(jac.T, (np.linalg.inv(np.dot(jac, jac.T))))
        null_q              = alpha*np.dot(jac_star, delta) + np.dot((np.eye(len(jnt_start)) - np.dot(jac_star,jac)),(jnt_start - q))
        u                   = q + null_q

        if np.any(np.isnan(u)) or np.linalg.norm(delta_pos) < self.osc_pos_threshold:
            u               = q

        return u
    
    def osc_torque_cmd(self, goal_pos, goal_ori=None, limb_idx=0, orientation_ctrl=False):
        arm = self.get_arm_handle(limb_idx)

        #proportional gain
        kp              = 10.
        #derivative gain
        kd              = np.sqrt(kp);
        #null space control gain
        alpha           = 0*0.15;

        jnt_start     = arm.angles()

        #to fix the nan issues that happen
        u_old           = np.zeros_like(jnt_start)

        # calculate position of the end-effector
        ee_xyz, ee_ori  = arm.get_ee_pose()

        # calculate the Jacobian for the end effector
        jac_ee         = arm.get_jacobian_from_joints()
        arm_state      = arm._state

        q              = arm_state['position']

        dq             = arm_state['velocity']

        # calculate the inertia matrix in joint space
        Mq             = arm.get_arm_inertia() 

        # convert the mass compensation into end effector space
        Mx_inv         = np.dot(jac_ee, np.dot(np.linalg.inv(Mq), jac_ee.T))
        svd_u, svd_s, svd_v = np.linalg.svd(Mx_inv)

        # cut off any singular values that could cause control problems
        singularity_thresh  = .00025
        for i in range(len(svd_s)):
            svd_s[i] = 0 if svd_s[i] < singularity_thresh else \
                1./float(svd_s[i])

        # numpy returns U,S,V.T, so have to transpose both here
        # convert the mass compensation into end effector space
        Mx   = np.dot(svd_v.T, np.dot(np.diag(svd_s), svd_u.T))

        x_des   = goal_pos - ee_xyz
 
        if orientation_ctrl:
            if goal_ori is None:
                print "For orientation control, pass goal orientation!"
                raise ValueError
            else:
                if type(goal_ori) is np.quaternion:
                    omg_des  = quatdiff(quaternion.as_float_array(goal_ori)[0], quaternion.as_float_array(ee_ori)[0])
                elif len(goal_ori) == 3:
                    omg_des = goal_ori
                else:
                    print "Wrong dimension"
                    raise ValueError
        else:
            omg_des = np.zeros(3)

        # calculate desired force in (x,y,z) space
        Fx                  = np.dot(Mx, np.hstack([x_des, omg_des]))
        # transform into joint space, add vel and gravity compensation
        u                   = (kp * np.dot(jac_ee.T, Fx) - np.dot(Mq, kd * dq))

        # calculate our secondary control signal
        # calculated desired joint angle acceleration

        prop_val            = ((jnt_start - q) + np.pi) % (np.pi*2) - np.pi

        q_des               = (kp * prop_val - kd * dq).reshape(-1,)

        u_null              = np.dot(Mq, q_des)

        # calculate the null space filter
        Jdyn_inv            = np.dot(Mx, np.dot(jac_ee, np.linalg.inv(Mq)))

        null_filter         = np.eye(len(q)) - np.dot(jac_ee.T, Jdyn_inv)

        u_null_filtered     = np.dot(null_filter, u_null)

        #changing the rest q as the last updated q
        jnt_start           = q 

        u                   += alpha*u_null_filtered

        if np.any(np.isnan(u)):
            u               = u_old
        else:
            u_old           = u

        return u
    
    def relative_jac(self, rel_pos):

        #left_arm is the master arm
        
        jac_left = self.left_arm.get_jacobian_from_joints()
        
        jac_right = self.right_arm.get_jacobian_from_joints()

        def make_skew(v):
            return np.array([[0., -v[2], v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])

        tmp1 = np.vstack([np.hstack([np.eye(3),-make_skew(rel_pos)]),np.hstack([np.zeros((3,3)),np.eye(3)])])
        
        pos_ee_l, rot_ee_l = self.left_arm.get_cartesian_pos_from_joints()
        
        pos_ee_r, rot_ee_r = self.right_arm.get_cartesian_pos_from_joints()

        tmp2 = np.vstack([np.hstack([-rot_ee_l,np.zeros((3,3))]),np.hstack([np.zeros((3,3)),-rot_ee_l])])

        tmp3 = np.vstack([np.hstack([rot_ee_r, np.zeros((3,3))]), np.hstack([np.zeros((3,3)), rot_ee_r])])

        jac_rel = np.hstack([np.dot(np.dot(tmp1, tmp2), jac_left), np.dot(tmp3, jac_right)])

        jac_master_rel = np.vstack([np.hstack([jac_left, np.zeros_like(jac_left)]), jac_rel])

        return jac_master_rel

    def osc_torque_cmd_2(self, arm_data, goal_pos, goal_ori=None, orientation_ctrl=False):

        #proportional gain
        kp              = 10.
        #derivative gain
        kd              = np.sqrt(kp);
        #null space control gain
        alpha           = 0.15;

        jnt_start = arm_data['jnt_start']
        ee_xyz = arm_data['ee_point']
        jac_ee = arm_data['jacobian']
        q      = arm_data['position']
        dq     = arm_data['velocity']
        
        #to fix the nan issues that happen
        u_old  = np.zeros_like(jnt_start)

        # calculate the inertia matrix in joint space
        Mq     = arm_data['inertia'] 

        # convert the mass compensation into end effector space
        Mx_inv         = np.dot(jac_ee, np.dot(np.linalg.inv(Mq), jac_ee.T))
        svd_u, svd_s, svd_v = np.linalg.svd(Mx_inv)

        # cut off any singular values that could cause control problems
        singularity_thresh  = .00025
        for i in range(len(svd_s)):
            svd_s[i] = 0 if svd_s[i] < singularity_thresh else \
                1./float(svd_s[i])

        # numpy returns U,S,V.T, so have to transpose both here
        # convert the mass compensation into end effector space
        Mx   = np.dot(svd_v.T, np.dot(np.diag(svd_s), svd_u.T))

        x_des   = goal_pos #- ee_xyz
 
        if orientation_ctrl:
            if goal_ori is None:
                print "For orientation control, pass goal orientation!"
                raise ValueError
            else:
                if type(goal_ori) is np.quaternion:
                    omg_des  = quatdiff(quaternion.as_float_array(goal_ori)[0], quaternion.as_float_array(ee_ori)[0])
                elif len(goal_ori) == 3:
                    omg_des = goal_ori
                else:
                    print "Wrong dimension"
                    raise ValueError
        else:
            omg_des = np.zeros(3)

        # calculate desired force in (x,y,z) space
        Fx                  = np.dot(Mx, np.hstack([x_des, omg_des]))
        # transform into joint space, add vel and gravity compensation
        u                   = (kp * np.dot(jac_ee.T, Fx) - np.dot(Mq, kd * dq))

        # calculate our secondary control signal
        # calculated desired joint angle acceleration

        prop_val            = ((jnt_start - q) + np.pi) % (np.pi*2) - np.pi

        q_des               = (kp * prop_val - kd * dq).reshape(-1,)

        u_null              = np.dot(Mq, q_des)

        # calculate the null space filter
        Jdyn_inv            = np.dot(Mx, np.dot(jac_ee, np.linalg.inv(Mq)))

        null_filter         = np.eye(len(q)) - np.dot(jac_ee.T, Jdyn_inv)

        u_null_filtered     = np.dot(null_filter, u_null)

        #changing the rest q as the last updated q
        jnt_start           = q 

        u                   += alpha*u_null_filtered

        if np.any(np.isnan(u)):
            u               = u_old
        else:
            u_old           = u

        return u

    def plot_min_jerk_traj(self):
        min_jerk_traj = self.get_min_jerk_trajectory()

        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.subplot(311)
        plt.title('orientation')
        plt.plot(min_jerk_traj['ori_traj'][:,0]) 
        plt.plot(min_jerk_traj['ori_traj'][:,1]) 
        plt.plot(min_jerk_traj['ori_traj'][:,2]) 
        plt.plot(min_jerk_traj['ori_traj'][:,3])
        # 
        plt.subplot(312)
        plt.title('omega')
        plt.plot(min_jerk_traj['omg_traj'][:,0]) 
        plt.plot(min_jerk_traj['omg_traj'][:,1]) 
        plt.plot(min_jerk_traj['omg_traj'][:,2]) 
        # 
        plt.subplot(313)
        plt.title('alpha')
        plt.plot(min_jerk_traj['alp_traj'][:,0]) 
        plt.plot(min_jerk_traj['alp_traj'][:,1]) 
        plt.plot(min_jerk_traj['alp_traj'][:,2])

        plt.figure(2)
        plt.subplot(311)
        plt.title('position')
        plt.plot(min_jerk_traj['pos_traj'][:,0]) 
        plt.plot(min_jerk_traj['pos_traj'][:,1]) 
        plt.plot(min_jerk_traj['pos_traj'][:,2]) 
        # 
        plt.subplot(312)
        plt.title('velocity')
        plt.plot(min_jerk_traj['vel_traj'][:,0]) 
        plt.plot(min_jerk_traj['vel_traj'][:,1]) 
        plt.plot(min_jerk_traj['vel_traj'][:,2]) 
        # 
        plt.subplot(313)
        plt.title('acceleration')
        plt.plot(min_jerk_traj['acc_traj'][:,0]) 
        plt.plot(min_jerk_traj['acc_traj'][:,1]) 
        plt.plot(min_jerk_traj['acc_traj'][:,2])

        plt.show()

def test_position_control(limb_idx=0):
    #0 is left and 1 is right
    
    baxter_ctrlr = MinJerkController()
    arm = baxter_ctrlr.get_arm_handle(limb_idx)
    baxter_ctrlr.set_neutral()
    baxter_ctrlr.untuck_arms()

    start_pos, start_ori  =  arm.get_ee_pose()

    if limb_idx == 0:
        goal_pos = start_pos + np.array([0.,0.8, 0.])
    else:
        goal_pos = start_pos - np.array([0.,0.8, 0.])

    angle    = 90.0
    axis     = np.array([1.,0.,0.]); axis = np.sin(0.5*angle*np.pi/180.)*axis/np.linalg.norm(axis)
    goal_ori = np.quaternion(np.cos(0.5*angle*np.pi/180.), axis[0], axis[1], axis[2], axis[3])

    baxter_ctrlr.configure(start_pos, start_ori, goal_pos, goal_ori)

    min_jerk_traj   = baxter_ctrlr.get_min_jerk_trajectory()

    print "Starting position controller"

    for t in range(baxter_ctrlr.timesteps):
        cmd = baxter_ctrlr.osc_position_cmd(goal_pos=min_jerk_traj['pos_traj'][t,:],
         goal_ori=None, limb_idx=limb_idx, orientation_ctrl=False)
        arm.exec_position_cmd(cmd)
        time.sleep(0.1)

    final_pos, final_ori  =  arm.get_ee_pose()
    print "ERROR in position \t", np.linalg.norm(final_pos-goal_pos)
    print "ERROR in orientation \t", np.linalg.norm(final_ori-goal_ori)

def test_torque_control(limb_idx=0):
    
    baxter_ctrlr = MinJerkController()
    arm = baxter_ctrlr.get_arm_handle(limb_idx)
    #baxter_ctrlr.set_neutral()
    baxter_ctrlr.untuck_arms()

    start_pos, start_ori  =  arm.get_ee_pose()
    
    if limb_idx == 0:
        goal_pos = start_pos + np.array([0.,0.8, 0.])
    else:
        goal_pos = start_pos - np.array([0.,0.8, 0.])
    
    angle    = 90.0
    axis     = np.array([1.,0.,0.]); axis = np.sin(0.5*angle*np.pi/180.)*axis/np.linalg.norm(axis)
    goal_ori = np.quaternion(np.cos(0.5*angle*np.pi/180.), axis[0], axis[1], axis[2], axis[3])

    baxter_ctrlr.configure(start_pos, start_ori, goal_pos, goal_ori)

    min_jerk_traj   = baxter_ctrlr.get_min_jerk_trajectory()

    print "Starting torque controller"
    mags = []
    for t in range(baxter_ctrlr.timesteps):
        cmd = baxter_ctrlr.osc_torque_cmd(goal_pos=min_jerk_traj['pos_traj'][t,:],
         goal_ori=None, limb_idx=limb_idx, orientation_ctrl=False)
        arm.exec_torque_cmd(cmd)
        print "the command \n", cmd
        mags.append(np.linalg.norm(cmd))
        time.sleep(0.1)

    import matplotlib.pyplot as plt
    plt.plot(mags)
    plt.show()
    print max(mags)
    print min(mags)

    final_pos, final_ori  =  arm.get_ee_pose()
    print "ERROR in position \t", np.linalg.norm(final_pos-goal_pos)
    print "ERROR in orientation \t", np.linalg.norm(final_ori-goal_ori)

def test_coop_position_control():
    limb_idx_l = 0 #0 is left and 1 is right
    limb_idx_r = 1 #0 is left and 1 is right
    
    baxter_ctrlr = MinJerkController()
    arm_l = baxter_ctrlr.get_arm_handle(limb_idx_l)
    arm_r = baxter_ctrlr.get_arm_handle(limb_idx_r)
    baxter_ctrlr.set_neutral()
    baxter_ctrlr.untuck_arms()

    start_pos, start_ori  =  arm_l.get_ee_pose()
    goal_pos = start_pos + np.array([0.,0.8, 0.])
    angle    = 90.0
    axis     = np.array([1.,0.,0.]); axis = np.sin(0.5*angle*np.pi/180.)*axis/np.linalg.norm(axis)
    goal_ori = np.quaternion(np.cos(0.5*angle*np.pi/180.), axis[0], axis[1], axis[2], axis[3])

    baxter_ctrlr.configure(start_pos, start_ori, goal_pos, goal_ori)

    left_pos, left_ori   = arm_l.get_ee_pose()
    right_pos, right_ori = arm_r.get_ee_pose()

    pos_rel = right_pos - left_pos
        
    vel_rel = np.array([0.,0.,0.])
        
    omg_rel = np.array([0.,0.,0.])

    jnt_l = arm_l.angles()
    jnt_r = arm_r.angles()

    min_jerk_traj   = baxter_ctrlr.get_min_jerk_trajectory()

    print "Starting co-operative position controller"

    for t in range(baxter_ctrlr.timesteps):

        jac_master_rel = baxter_ctrlr.relative_jac(pos_rel)

        jac_tmp = np.dot(jac_master_rel, jac_master_rel.T)
        
        jac_star = np.dot(jac_master_rel.T, (np.linalg.inv(jac_tmp + np.eye(jac_tmp.shape[0])*0.01))) 

        pos_tmp = np.hstack([min_jerk_traj['vel_traj'][t,:], min_jerk_traj['omg_traj'][t,:], vel_rel, omg_rel])
        
        jnt_vel = np.dot(jac_star, pos_tmp)

        if np.any(np.isnan(jnt_vel)):
            jnt_vel_l = np.zeros(7)
            jnt_vel_r = np.zeros(7)
        else:
            jnt_vel_l = jnt_vel[:7]
            jnt_vel_r = jnt_vel[7:]
        
        jnt_l += jnt_vel_l*baxter_ctrlr.dt
        jnt_r += jnt_vel_r*baxter_ctrlr.dt
        
        arm_l.exec_position_cmd(jnt_l)

        arm_r.exec_position_cmd(jnt_r)

        time.sleep(0.1)

    final_pos, final_ori  =  arm_l.get_ee_pose()
    print "ERROR in position \t", np.linalg.norm(final_pos-goal_pos)
    print "ERROR in orientation \t", np.linalg.norm(final_ori-goal_ori)

def test_coop_torque_control():
    limb_idx_l = 0 #0 is left and 1 is right
    limb_idx_r = 1 #0 is left and 1 is right
    
    baxter_ctrlr = MinJerkController()
    arm_l = baxter_ctrlr.get_arm_handle(limb_idx_l)
    arm_r = baxter_ctrlr.get_arm_handle(limb_idx_r)
    #baxter_ctrlr.set_neutral()
    baxter_ctrlr.untuck_arms()

    start_pos, start_ori  =  arm_l.get_ee_pose()
    goal_pos = start_pos + np.array([0.,-0.0, 0.9])
    angle    = 45.0
    axis     = np.array([1.,0.,0.]); axis = np.sin(0.5*angle*np.pi/180.)*axis/np.linalg.norm(axis)
    goal_ori = np.quaternion(np.cos(0.5*angle*np.pi/180.), axis[0], axis[1], axis[2], axis[3])

    baxter_ctrlr.configure(start_pos, start_ori, goal_pos, goal_ori)

    left_pos, left_ori   = arm_l.get_ee_pose()
    right_pos, right_ori = arm_r.get_ee_pose()

    pos_rel = right_pos - left_pos
        
    vel_rel = np.array([0.,0.,0.])
        
    omg_rel = np.array([0.,0.,0.])

    jnt_l = arm_l.angles()
    jnt_r = arm_r.angles()

    min_jerk_traj   = baxter_ctrlr.get_min_jerk_trajectory()

    print "Starting co-operative torque controller"

    # rate = rospy.Rate(10)
    mags_r = []
    mags_l = []
    for t in range(baxter_ctrlr.timesteps):

        state_l  = arm_l._state
        state_r  = arm_r._state
        
        state_l['jnt_start'] = jnt_l
        state_r['jnt_start'] = jnt_r

        jac_ee_l = state_l['jacobian']
        jac_ee_r = state_r['jacobian']

        left_pos  = state_l['ee_point'] 
        left_ori  = state_l['ee_ori'] 
        right_pos = state_r['ee_point'] 
        right_ori = state_r['ee_ori']

        jac_master_rel = baxter_ctrlr.relative_jac(pos_rel)

        jac_tmp = np.dot(jac_master_rel, jac_master_rel.T)
        
        jac_star = np.dot(jac_master_rel.T, (np.linalg.inv(jac_tmp + np.eye(jac_tmp.shape[0])*0.001))) 

        pos_tmp = np.hstack([min_jerk_traj['vel_traj'][t,:], min_jerk_traj['omg_traj'][t,:], vel_rel, omg_rel])
        
        jnt_vel = np.dot(jac_star, pos_tmp)

        ee_req_vel_l        = np.dot(jac_ee_l, jnt_vel[:7]) 
        ee_req_vel_r        = np.dot(jac_ee_r, jnt_vel[7:]) 

        cmd_l = baxter_ctrlr.osc_torque_cmd_2(arm_data=state_l, 
                                              goal_pos=ee_req_vel_l[0:3]*baxter_ctrlr.dt, 
                                              goal_ori=None, 
                                              orientation_ctrl=False)

        cmd_r = baxter_ctrlr.osc_torque_cmd_2(arm_data=state_r, 
                                              goal_pos=ee_req_vel_r[0:3]*baxter_ctrlr.dt, 
                                              goal_ori=None, 
                                              orientation_ctrl=False)

        #cmd = [0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.10]
        arm_l.exec_torque_cmd(cmd_l)
        #print "cmd \n", cmd
        #print "magnitude \t", np.linalg.norm(cmd)
        arm_r.exec_torque_cmd(cmd_r)
        mags_r.append(np.linalg.norm(cmd_r))
        mags_l.append(np.linalg.norm(cmd_l))
        time.sleep(0.1)

        jnt_r = state_r['position']
        jnt_l = state_l['position']

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(mags_l)
    plt.figure()
    plt.plot(mags_r)
    #plt.show()

    arm_l.exec_position_cmd2(np.zeros(7))
    arm_r.exec_position_cmd2(np.zeros(7))
    
    final_pos, final_ori  =  arm_l.get_ee_pose()
    print "ERROR in position \t", np.linalg.norm(final_pos-goal_pos)
    print "ERROR in orientation \t", np.linalg.norm(final_ori-goal_ori)

def test_follow_gps_policy(vel_ee_master, arm_l, arm_r, dt):

    baxter_ctrlr = MinJerkController(extern_call=True, trial_arm=arm_l, aux_arm=arm_r)
    
    left_pos, left_ori   = arm_l.get_ee_pose()
    # vel_ee_master = arm_l.get_ee_velocity()[0:3]
    right_pos, right_ori = arm_r.get_ee_pose()

    pos_rel = right_pos - left_pos
        
    vel_rel = np.array([0.,0.,0.])
        
    omg_rel = np.array([0.,0.,0.])

    jnt_l = arm_l.angles()
    jnt_r = arm_r.angles()

    #print "Starting co-operative torque controller"

    # rate = rospy.Rate(10)
    mags_r = []
    mags_l = []


    state_l  = arm_l._state
    state_r  = arm_r._state
    
    state_l['jnt_start'] = jnt_l
    state_r['jnt_start'] = jnt_r

    jac_ee_l = state_l['jacobian']
    jac_ee_r = state_r['jacobian']

    left_pos  = state_l['ee_point'] 
    left_ori  = state_l['ee_ori'] 
    right_pos = state_r['ee_point'] 
    right_ori = state_r['ee_ori']

    rel_jac = baxter_ctrlr.relative_jac(pos_rel)
    #only for position
    jac_master_rel = np.vstack([rel_jac[0:3,:],rel_jac[6:9,:]])

    jac_tmp = np.dot(jac_master_rel, jac_master_rel.T)
    
    jac_star = np.dot(jac_master_rel.T, (np.linalg.inv(jac_tmp + np.eye(jac_tmp.shape[0])*0.001))) 

    pos_tmp = np.hstack([vel_ee_master, vel_rel])
    
    jnt_vel = np.dot(jac_star, pos_tmp)

    ee_req_vel_l        = np.dot(jac_ee_l, jnt_vel[:7])[0:3]
    ee_req_vel_r        = np.dot(jac_ee_r, jnt_vel[7:])[0:3]

    #print ee_req_vel_l
    #print ee_req_vel_r

    # cmd_l = baxter_ctrlr.osc_torque_cmd_2(arm_data=state_l, goal_pos=ee_req_vel_l[0:3]*baxter_ctrlr.dt, goal_ori=None, orientation_ctrl=False)
    # print "cmd_l \n", cmd_l

    #aux arm in position mode
    #cmd_r = jnt_r + jnt_vel[7:]*0.8

    #aux arm in vel mode
    cmd_r = jnt_vel[7:]

    #aux arm in torque mode
    # cmd_r = baxter_ctrlr.osc_torque_cmd_2(arm_data=state_r, goal_pos=ee_req_vel_r[0:3], goal_ori=None, orientation_ctrl=False)
    
    print "cmd_r \n", cmd_r

    return cmd_r


def test_follow_gps_policy2(arm_l, arm_r):

    #left_arm_start =[-0.8, 0.7, 0.1, 7.48607379e-01, 0.5, 1.25333658e+00, -1.81064349e-04]
    #rigt_arm_start = [0.8, 0.7, -0.1, 7.48607379e-01, -0.5, 1.25333658e+00, 1.81064349e-04]

    #arm_l.move_to_joint_position(left_arm_start)
    #arm_r.move_to_joint_position(rigt_arm_start)

    baxter_ctrlr = MinJerkController(extern_call=True, trial_arm=arm_l, aux_arm=arm_r)
    
    left_pos,  left_ori  = arm_l.get_ee_pose()
    
    right_pos, right_ori = arm_r.get_ee_pose() 

    state_r  = arm_r._state
    state_r['jnt_start'] = arm_r.angles()

    #print right_pos - left_pos
    #print quatdiff(quaternion.as_float_array(right_ori)[0], quaternion.as_float_array(left_ori)[0])
    #the imaginary coordinate translation w.rt left ee
    #following is the initiall difference when using left_arm_start and rigt_arm_start
    const_tran = np.array([-0.00507125, -0.85750604, -0.00270199])

    #the imaginary coordinate rotation w.rt left ee
    const_qt = np.quaternion(1.,0.,0.,0.)
    
    #find the rotation to coordinate frame
    req_tansf = left_ori*const_qt*left_ori.conjugate()

    #compute required change in position
    req_pos_diff = left_pos + const_tran - right_pos

    #compute required change in orientation
    #req_ori_diff = quatdiff(quaternion.as_float_array(right_ori)[0], quaternion.as_float_array(req_tansf)[0])
    
    #following is the initiall difference when using left_arm_start and rigt_arm_start
    req_ori_diff = np.array([-0., -0.0, 0. ])#np.array([-0.12093466, -0.00218386, 0.5374029 ])

    #cmd_r = np.dot(arm_r._state['jacobian'].T, np.hstack([req_pos_diff, req_ori_diff]))

    cmd_r = baxter_ctrlr.osc_torque_cmd_2(arm_data=state_r, 
                                          goal_pos=req_pos_diff, 
                                          goal_ori=req_ori_diff, 
                                          orientation_ctrl=True)

    return cmd_r

def test_reach_both_sides_box(torque_mode=False):
    flag_box = False
    box_tf = TransformListener()
    baxter_ctrlr = MinJerkController()
    left_task_complete = False
    right_task_complete = False
    box_length  = 0.410 #m
    box_breadth = 0.305
    box_height  = 0.107
    #for better show off ;)
    baxter_ctrlr.set_neutral()

    while not rospy.is_shutdown():

        try:
            tfmn_time = box_tf.getLatestCommonTime('base', 'box')
            flag_box = True
        except tf.Exception:
            print "Some exception occured while getting the transformation!!!"

        if flag_box:
            flag_box = False

            box_pos, box_ori     = box_tf.lookupTransform('base', 'box', tfmn_time)

            box_pos = np.array([box_pos[0],box_pos[1],box_pos[2]])
            box_ori = np.quaternion(box_ori[3],box_ori[0],box_ori[1],box_ori[2])

            left_pos,  left_ori  = baxter_ctrlr.left_arm.get_ee_pose()
    
            right_pos, right_ori = baxter_ctrlr.right_arm.get_ee_pose()

            left_goal_pos  = box_pos + np.dot(quaternion.as_rotation_matrix(box_ori), np.array([box_length/2., -box_height/2.,0.]))
            right_goal_pos = box_pos + np.dot(quaternion.as_rotation_matrix(box_ori), np.array([-box_length/2.,-box_height/2.,0.]))
        
            error_left = np.linalg.norm(left_pos-left_goal_pos)
            error_right = np.linalg.norm(right_pos-right_goal_pos)

            if torque_mode:
                
                #minimum jerk trajectory for left arm
                baxter_ctrlr.configure(left_pos, left_ori, left_goal_pos, box_ori)
                min_jerk_traj_left   = baxter_ctrlr.get_min_jerk_trajectory()
                #minimum jerk trajectory for right arm
                baxter_ctrlr.configure(right_pos, right_ori, right_goal_pos, box_ori)
                min_jerk_traj_right  = baxter_ctrlr.get_min_jerk_trajectory()
                
                for t in range(baxter_ctrlr.timesteps):
                    #compute the left joint torque command
                    left_cmd  = baxter_ctrlr.osc_torque_cmd(goal_pos=min_jerk_traj_left['pos_traj'][t,:], 
                                                            goal_ori=min_jerk_traj_left['ori_traj'][t],  
                                                            limb_idx=0, 
                                                            orientation_ctrl=True)
                    #compute the right joint torque command if the error is above some threshold
                    right_cmd = baxter_ctrlr.osc_torque_cmd(goal_pos=min_jerk_traj_right['pos_traj'][t,:], 
                                                            goal_ori=min_jerk_traj_right['ori_traj'][t], 
                                                            limb_idx=1, 
                                                            orientation_ctrl=True)
                    baxter_ctrlr.left_arm.exec_torque_cmd(left_cmd)
                    baxter_ctrlr.right_arm.exec_torque_cmd(right_cmd)
                    time.sleep(0.1)
                
                break
            
            else:
                
                #compute the left joint position command if the error is above some threshold
                left_cmd  = baxter_ctrlr.osc_position_cmd(goal_pos=left_goal_pos, 
                                                          goal_ori=box_ori, 
                                                          limb_idx=0, 
                                                          orientation_ctrl=False)
                #compute the right joint position command if the error is above some threshold
                right_cmd = baxter_ctrlr.osc_position_cmd(goal_pos=right_goal_pos, 
                                                          goal_ori=box_ori, 
                                                          limb_idx=1, 
                                                          orientation_ctrl=False)

                baxter_ctrlr.left_arm.move_to_joint_position(left_cmd)
                baxter_ctrlr.right_arm.move_to_joint_position(right_cmd)

                print "error_right \t", error_right
                print "error_left \t", error_left

                if (error_right < baxter_ctrlr.osc_pos_threshold) and (error_left < baxter_ctrlr.osc_pos_threshold):
                    #break the loop if within certain threshold
                    break

def test_lift_both_sides_box():

    baxter_ctrlr = MinJerkController()
    
    left_pos,  left_ori  = baxter_ctrlr.left_arm.get_ee_pose()
    right_pos, right_ori = baxter_ctrlr.right_arm.get_ee_pose()

    left_goal_pos = left_pos +  np.array([0.0, 0., 0.15])

    left_goal_ori = left_ori
    #minimum jerk trajectory for left arm
    baxter_ctrlr.configure(left_pos, left_ori, left_goal_pos, left_goal_ori)
    min_jerk_traj_left   = baxter_ctrlr.get_min_jerk_trajectory()

    #the imaginary coordinate rotation w.rt left ee
    const_qt = np.quaternion(1.,0.,0.,0.)
    #find the rotation to coordinate frame
    req_tansf = left_ori*const_qt*left_ori.conjugate()
    #compute required change in position

    req_pos_diff = right_pos - left_pos

    req_ori_diff = np.array([0., 0., 0. ])#np.array([-0.12093466, -0.00218386, 0.5374029 ])

    for t in range(baxter_ctrlr.timesteps):
        #compute the left joint torque command
        left_cmd  = baxter_ctrlr.osc_torque_cmd(goal_pos=min_jerk_traj_left['pos_traj'][t,:], 
                                                goal_ori=min_jerk_traj_left['ori_traj'][t],  
                                                limb_idx=0, 
                                                orientation_ctrl=True)
      
        state_r  = baxter_ctrlr.right_arm._state
        state_r['jnt_start'] = baxter_ctrlr.right_arm.angles()

        right_cmd = baxter_ctrlr.osc_torque_cmd_2(arm_data=state_r, 
                                              goal_pos=req_pos_diff, 
                                              goal_ori=req_ori_diff, 
                                              orientation_ctrl=True)
        
        baxter_ctrlr.left_arm.exec_torque_cmd(left_cmd)
        baxter_ctrlr.right_arm.exec_torque_cmd(right_cmd)
        time.sleep(0.1)

##==============================
# Test code
#===============================
if __name__ == "__main__":

    rospy.init_node('baxter_classical_controller')

    #arguments that can be passed in
    # python classical_controllers.py calib
    # will self calibrate the arm, default is not to do it
    # python classical_controllers.py torque
    # will play the arm in torque mode

    #get the arguments passed to the script
    cmdargs = str(sys.argv)

    #test_torque_control(0)
    #test_position_control()
    #test_coop_position_control()
    
    #test_coop_torque_control()

    #test_follow_gps_policy()


    #ctrlr = MinJerkController()

    #test_follow_gps_policy2(ctrlr.left_arm, ctrlr.right_arm)

    if 'calib' in cmdargs:
        from camera_calib import Baxter_Eye_Hand_Calib
        calib = Baxter_Eye_Hand_Calib()
        calib.self_caliberate()
    
    if 'torque' in cmdargs:
        torque_mode = True
    else:
        torque_mode = False

    test_reach_both_sides_box(torque_mode)
    #test_lift_both_sides_box()
