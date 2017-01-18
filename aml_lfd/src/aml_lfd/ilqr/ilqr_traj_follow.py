import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import cProfile
from aml_robot.baxter_robot import BaxterArm
import time
import rospy

from aml_lfd.utilities.utilities import quat_mult, compute_exp, load_demo_data

#[.95, .8, .5, .2, .1, .05, .02, .01, .005, .002, .001, 0]
class DDP_TrajFollowClass():
    def __init__(self, idx, H, model, target_traj, reward, model_bias):
        self.idx            = idx
        self.H              = H
        self.target_traj    = target_traj
        self.reward         = reward
        self.cost           = 0.
        self.model          = model
        self.model_bias     = model_bias
        self.magic_factors  = [.95, .8, .5, .2, .1, .05, .02, .01, .005, .002, .001, 0]
        self.num_states     = None
        self.num_ctrls      = None
        self.num_links      = None
        self.configure()

        self.right_arm  = BaxterArm('right')
    
    def configure(self):
        self.num_states = len(self.reward['state_multipliers'])
        self.num_ctrls  = len(self.reward['input_multipliers'])
        self.num_links  = len(self.reward['input_multipliers'])
        self.A  = np.zeros((self.num_states, self.num_states))
        self.B  = np.zeros((self.num_states, self.num_ctrls))
        self.Mq = np.zeros((self.num_links, self.num_links))
        self.Cq = np.zeros((self.num_links,1))
        self.Q  = np.diag(self.reward['state_multipliers'])
        self.Qf = np.diag(self.reward['state_multipliers'])
        self.R  = np.diag(self.reward['input_multipliers'])
        self.Rf = np.diag(self.reward['input_multipliers'])
        self.Alist = [self.A for _ in range(self.H)]
        self.Blist = [self.B for _ in range(self.H)]
        self.Qlist = [self.Q for _ in range(self.H)]
        self.Rlist = [self.R for _ in range(self.H)]
        #K has same size as A
        self.Klist = [np.random.random((self.num_states, self.num_states)) for _ in range(self.H-1)]
        self.target_xlist = [np.zeros((self.num_states,1)) for _ in range(self.H)]
        self.target_ulist = [np.zeros((self.num_ctrls,1))  for _ in range(self.H-1)]

        self.dxlist = [np.zeros(self.num_states) for _ in range(self.H)]
        self.xlist  = [np.zeros(self.num_states) for _ in range(self.H)]
        self.ulist  = [np.zeros(self.num_ctrls)  for _ in range(self.H)]

    def compute_dx(self, xtarget, x):
        dx  = x - xtarget
        dx[self.idx['q']] = self.quat_multiply(lq=np.multiply(np.array([1., -1., -1., -1.]),xtarget[self.idx['q']]), 
                                               rq=x[self.idx['q']])
        dx[self.idx['q'][0]] = 1
        return dx

    def euler_to_q(self, euler):
        c1 = np.cos(euler[2] * 0.5)
        c2 = np.cos(euler[1] * 0.5)
        c3 = np.cos(euler[0] * 0.5)
        s1 = np.sin(euler[2] * 0.5)
        s2 = np.sin(euler[1] * 0.5)
        s3 = np.sin(euler[0] * 0.5)

        q = np.zeros(4)

        q[0] = c1*c2*c3 + s1*s2*s3
        q[1] = c1*s2*c3 + s1*c2*s3
        q[2] = s1*c2*c3 - c1*s2*s3
        q[3] = c1*c2*s3 - s1*s2*c3
        return q
    
    def quat_multiply(self, lq, rq):
        #// quaternion entries in order: w, x, y, z
        quat    = np.zeros(4)
        quat[0] = lq[3]*rq[3] - lq[0]*rq[0] - lq[1]*rq[1] - lq[2]*rq[2]
        quat[1] = lq[3]*rq[1] - lq[0]*rq[2] + lq[1]*rq[3] + lq[2]*rq[0]
        quat[2] = lq[3]*rq[2] + lq[0]*rq[1] - lq[1]*rq[0] + lq[2]*rq[3]
        quat[3] = lq[3]*rq[0] + lq[0]*rq[3] + lq[1]*rq[2] - lq[2]*rq[1]
        return quat

    def err_simulate(self, index, dx0, u0, xnom0, xnom1, sim_time, model_bias, magic_factor=0, x1star=0):
        # dx0 is a state relative to some nominal trajectory
        x0  = self.compose_dx(xnom0, dx0)
        # x1 = run simulator on x0, u0
        x1  = self.simulate_f(index, x0, u0, sim_time, model_bias, magic_factor, x1star)
        # compute state relative to nominal trajectory at next step
        dx1 = self.compute_dx(xnom1, x1)
        return dx1

    def compose_dx(self,x0, dx):
        x1                = x0 + dx
        x0[self.idx['q']] = x0[self.idx['q']]/np.linalg.norm(x0[self.idx['q']])
        quat0             = x0[self.idx['q']]
        dq                = np.hstack([ dx[[self.idx['q'][1:4]]], np.sqrt(1-np.linalg.norm(dx[[self.idx['q'][1:4]]])**2) ])
        x1[self.idx['q']] = self.quat_multiply(quat0, dq)
        return x1

    def simulate_f(self, index, x0, delta_u0, sim_time, model_bias=None, magic_factor=0, x1_star=0):

        dt = sim_time   
        x1 = np.zeros_like(x0)
        ##
        genq = x0[self.idx['genq']]

        Mq   = self.right_arm.get_arm_inertia(joint_angles=genq[:7])

        ## for our feedback controllers it's important to have past inputs and
        ## change in inputs stored into the state:
        x1[self.idx['u_prev']] = x0[self.idx['u_prev']] + delta_u0 # control at previous time, namely at time 0
        x1[self.idx['u_delta_prev']] = delta_u0
        ## position and orientation merely require integration (we use Euler
        ## integration):
        
        ddgenq = np.dot(np.linalg.inv(Mq),(x0[self.idx['u_prev']]))
        
        dgenq  = np.hstack([genq[7:14], ddgenq])

        x1_genq = x0[self.idx['genq']] + dt*dgenq

        x1[self.idx['genq']] = x1_genq
       
        J = self.right_arm.get_jacobian_from_joints(joint_angles=x1_genq[:7])

        ee_pose_dot = np.dot(J, dgenq[:7])

        x1[self.idx['xyz_dot']] = ee_pose_dot[0:3]

        x1[self.idx['xyz']] = x0[self.idx['xyz']] + ee_pose_dot[0:3]*dt

        x1[self.idx['q']] = quat_mult(compute_exp(dt/2.*ee_pose_dot[3:6]), x0[self.idx['q']])
    
        x1 = magic_factor*x1_star + (1.-magic_factor)*x1

        return x1

    def quaternion_from_axis_rotation(self, axis_rotation):
        rotation_angle = np.linalg.norm(axis_rotation)
        quat = np.zeros(4)
        if(rotation_angle < 1e-4):  # avoid division by zero -- also: can use simpler computation in this case, since for small angles sin(x) = x is a good approximation
            quat[1:4] = axis_rotation/2
        else:
            normalized_axis = axis_rotation / rotation_angle
            quat[1:4] = normalized_axis * np.sin(rotation_angle/2)
        quat[0] = np.sqrt(1 - np.linalg.norm(quat[1:4],2)**2)
        return quat

    def express_vector_in_quat_frame(self, vin, q):
        vout = self.rotate_vector(vin, np.vstack([q[0],-q[1:4]]))
        return vout

    def rotate_vector(self, vin, q):
    #   // return   ( ( q * quaternion(vin) ) * q_conj ) .complex_part
        vout = self.quat_multiply( quat_multiply ( q, np.hstack( np.vstack([0, vin]), np.vstack([q[0],- q[1:4]]) ) ) )
        vout = vout[1:4]
        return vout

    #this function is to stack states while re-simulating the 
    #calculated trajectory
    def stack_state(self, u_old, u_new):
        arm_state      = self.right_arm.state
        ee_pos, ee_ori = self.right_arm.ee_pose()
        ee_ori         = np.array([ee_ori[0], ee_ori[1], ee_ori[2], ee_ori[3]])

        return np.hstack([u_old,   
                          unew-u_old,  
                          np.dot(arm_state['jacobian'], 
                          arm_state['velocity']),  
                          ee_pos, np.zeros(3), 
                          arm_state['position'], 
                          arm_state['velocity'] , 
                          ee_ori]).T

    def linearized_dynamics(self, x0, u0, xt0, xt1, dt_sim, model_bias, magic_factor, x1star):

        dx0     = self.compute_dx(xt0, x0)
        fx_t0   = self.err_simulate(0, dx0, u0, xt0, xt1, dt_sim, model_bias, magic_factor, x1star)
        epsilon = np.ones(dx0.shape[0])*.01
        #epsilon(end-4:end-1) = ones(1,4) %% seems better to linearize with large step for the inputs
        #%% recall last entry of state corresponds to intercept, do not
        #%% linearize!

        for i in range(0,dx0.shape[0]-1): #[0],epsilon.shape[1]
            delta       = np.zeros(dx0.shape[0])
            delta[i]    = epsilon[i]

            fx_t1m      = self.err_simulate(index=i, 
                                            dx0=dx0 - delta, 
                                            u0=u0, 
                                            xnom0=xt0, 
                                            xnom1=xt1, 
                                            sim_time=dt_sim, 
                                            model_bias=model_bias, 
                                            magic_factor=magic_factor, 
                                            x1star=x1star)

            fx_t1p      = self.err_simulate(index=i, 
                                            dx0=dx0 + delta, 
                                            u0=u0, 
                                            xnom0=xt0, 
                                            xnom1=xt1, 
                                            sim_time=dt_sim, 
                                            model_bias=model_bias, 
                                            magic_factor=magic_factor, 
                                            x1star=x1star)

            self.A[:,i] = ((fx_t1p - fx_t1m)/epsilon[i]/2).reshape(-1)
        
        epsilon = np.ones(u0.shape[0])

        for i in range(0,u0.shape[0]):
            delta       = np.zeros(u0.shape[0])
            delta[i]    = epsilon[i]

            fx_t1m      = self.err_simulate(index=i, 
                                            dx0=dx0, 
                                            u0=u0 - delta, 
                                            xnom0=xt0, 
                                            xnom1=xt1, 
                                            sim_time=dt_sim, 
                                            model_bias=model_bias, 
                                            magic_factor=magic_factor, 
                                            x1star=x1star)

            fx_t1p      = self.err_simulate(index=i, 
                                            dx0=dx0, 
                                            u0=u0 + delta, 
                                            xnom0=xt0, 
                                            xnom1=xt1, 
                                            sim_time=dt_sim, 
                                            model_bias=model_bias, 
                                            magic_factor=magic_factor, 
                                            x1star=x1star)

            self.B[:,i] = ((fx_t1p - fx_t1m)/epsilon[i]/2).reshape(-1)

        self.A[:,-1] = (fx_t0 - np.dot(self.A[:,:-1],dx0[:-1]) - np.dot(self.B, u0)).reshape(-1)

    def ddp_for_trajectory_following(self, start_state  ):
        
        open_loop_flag  = 0
        
        num_ddp_iters   = len(self.magic_factors)
        nom_traj        = self.target_traj

        ddp_iter        = 0
        #this is for allocating additional iterations with the last magic factor
        additional_iter = 0
        #pre-allocating space
        im_trajs = [nom_traj for _ in range(num_ddp_iters+additional_iter+2)]

        while (ddp_iter <= num_ddp_iters):
            print "Iteration \t",ddp_iter+1
            
            if(ddp_iter == len(self.magic_factors)):
                num_ddp_iters = num_ddp_iters + additional_iter
                magic_factor  = 0
            elif(ddp_iter < len(self.magic_factors)):
                magic_factor = self.magic_factors[ddp_iter]
            
            ddp_iter += 1
            #backward pass
            lqr_traj = self.lqr_backups_for_trajectory_following(nom_traj=nom_traj, 
                                                                 magic_factor=magic_factor)
            #forward pass
            nom_traj = self.lqr_run_controller_in_nonlinear_sim(start_state=start_state,
                                                                lqr_traj=lqr_traj,
                                                                magic_factor=magic_factor, 
                                                                open_loop_flag=open_loop_flag)

            qscore, rscore = self.score_lqr_trajectory(traj=nom_traj)
            
            newCost = qscore + rscore   # should decrease
            print 'Change in cost\t', self.cost-newCost
            self.cost = newCost

            #store the trajectory
            im_trajs[ddp_iter] = copy.deepcopy(nom_traj)
      
        result_traj = self.lqr_run_controller_in_nonlinear_sim(start_state=start_state,
                                                               lqr_traj=lqr_traj,
                                                               magic_factor=magic_factor,
                                                               open_loop_flag=open_loop_flag)
        return lqr_traj, result_traj, im_trajs

    def lqr_backups_for_trajectory_following(self, nom_traj, magic_factor):
        # nom_traj: we linearize around this traj
        # target_traj: this is our target
        # simulate_f: function that we can call as follows: next_state =
        #            simulate_f(current_state, inputs, simtime, params, model_bias)
        # model: parameters and features of the dynamics model (simulate_f)
        # idx: how we index into features and state using named indexes
        # model_bias: offsets we use for the model at each time slice
        # reward
        # magic_factor: how much the dynamics is altered to automatically reach the
        # target
        target_traj = self.target_traj

        #linear_terms_state = reward.linear_terms_state
        H = self.H

        # state is augmented with inputs from past 2 timesteps, so state vector
        # actually of length dim_x + 2*dim_u
        if self.num_states != nom_traj['x'][0].shape[0] or self.num_states != target_traj['x'][0].shape[0] \
                or self.num_ctrls != nom_traj['u'][0].shape[0] or self.num_ctrls != target_traj['u'][0].shape[0] \
                or H != len(nom_traj['x']) or H != len(target_traj['x']) \
                or H != len(nom_traj['u']) or H != len(target_traj['u']) \
                or H != len(nom_traj['t']) or H != len(target_traj['t']):
            print ('dimension mismatch for inputs to lqr_trajectory(...)\n "nom_traj\t" {} "\n target_traj \t" {} "\n state_multipliers \t" {} \
            "\n input_multipliers \t" {}').format(\
            nom_traj['x'].shape,\
            target_traj['x'].shape,\
            self.num_states, self.num_ctrls)

        Ps = self.Qf

        lqr_traj = {}
        lqr_traj['t'] = nom_traj['t']
        
        for i in range(H-2,-1,-1):
            dt = lqr_traj['t'][i+1]-lqr_traj['t'][i]
            
            #this function will populate self.A and self.B
            self.linearized_dynamics(x0=nom_traj['x'][i].T, 
                                     u0=nom_traj['u'][i].T, 
                                     xt0=target_traj['x'][i].T, 
                                     xt1=target_traj['x'][i+1].T, \
                                     dt_sim=dt, 
                                     model_bias=self.model_bias[i].T, 
                                     magic_factor=magic_factor, 
                                     x1star=target_traj['x'][i+1].T)
            
            Q = self.Q*dt
            R = self.R*dt
            
            #feedback gain of the ilqr controller
            K   = -np.dot(np.linalg.inv(R + np.dot(np.dot(self.B.T,Ps),self.B)) , np.dot(np.dot(self.B.T,Ps), self.A)) #
            tmp = self.A + np.dot(self.B, K)
            #value function of the next state, this is 
            #the dynamic programming pass
            Ps  = Q + np.dot(np.dot(K.T,R),K) + np.dot(np.dot(tmp.T,Ps),tmp)

            #% add nominal inputs to offset term in K

            # print "Ps \n", self.Qf
            # print "first \n", np.linalg.inv(R + np.dot(np.dot(self.B.T,Ps),self.B))
            # print "second \n", np.dot(np.dot(self.B.T,Ps), self.A)
            # print "A \n", self.A
            # print "B \n", self.B
            # print "K \n", K
            # raw_input("Enter")

            self.Klist[i]        = K.copy()
            self.target_xlist[i] = target_traj['x'][i].copy()
            self.target_ulist[i] = target_traj['u'][i].copy()

            self.Alist[i]        = self.A.copy()
            self.Blist[i]        = self.B.copy()
            self.Qlist[i]        = self.Q.copy()
            self.Rlist[i]        = self.R.copy()

        self.target_xlist[H-1] = target_traj['x'][-1].copy()
        #self.target_ulist[H-2] = target_traj['u'][-1].copy()

        self.Alist[H-1] = self.Alist[H-2].copy()
        self.Blist[H-1] = self.Blist[H-2].copy()
        self.Qlist[H-1] = self.Qf.copy()
        self.Rlist[H-1] = self.Rf.copy()

        lqr_traj['A'] = self.Alist
        lqr_traj['B'] = self.Blist
        lqr_traj['Q'] = self.Qlist
        lqr_traj['R'] = self.Rlist
        lqr_traj['K'] = self.Klist

        lqr_traj['target_x'] = self.target_xlist 
        lqr_traj['target_u'] = self.target_ulist
        lqr_traj['nom_x']    = nom_traj['x']
        lqr_traj['nom_u']    = nom_traj['u']

        return lqr_traj 

    def lqr_run_controller_in_nonlinear_sim(self, start_state, lqr_traj, magic_factor, open_loop_flag):     
        model_bias = self.model_bias
        
        # run lqr controllers on non-linear dynamics
        if(open_loop_flag == 1):
            print 'running in open loop ...'
        
        traj = {}
        traj['target_x'] = lqr_traj['target_x']
        traj['t']        = lqr_traj['t']

        H = self.H
        x = start_state.copy() #lqr_traj.nom_x(1,:)'

        for i in range (0, H-1):
            dx = self.compute_dx(lqr_traj['target_x'][i], x)
            #% add observation noise here ...
            if(open_loop_flag==1):
                u = lqr_traj['K'][:,-1]
            else:
                u = np.dot(lqr_traj['K'][i],dx)
            
            self.dxlist[i] = dx.copy()
            self.xlist[i]  = x.copy()
            self.ulist[i]  = u.copy()
            
            sim_time = lqr_traj['t'][i+1] - lqr_traj['t'][i]

            x = self.simulate_f(index=i, 
                                x0=x, 
                                delta_u0=u, 
                                sim_time=sim_time, 
                                model_bias=model_bias[i,:].T, 
                                magic_factor=magic_factor, 
                                x1_star=traj['target_x'][i+1].T)#state-noise: + randn(1,2)
            #x(1:end-4) = x(1:end-4)+randn(length(x)-4,1)*.01
            
        dx = self.compute_dx(lqr_traj['target_x'][-1].T, x)
        self.dxlist[H-1] = dx.copy()
        self.xlist[H-1]  = x.copy()
    
        traj['dx'] = self.dxlist
        traj['x']  = self.xlist
        traj['u']  = self.ulist
        return traj

    def score_lqr_trajectory(self, traj):
        reward = self.reward
        
        qscore = 0
        rscore = 0

        for i in range(0,len(traj['x'])):
            dx = traj['dx'][i]
            qscore = qscore + np.dot(np.dot(dx,self.Q),dx.T)
            u = traj['u'][i]
            rscore = rscore + np.dot(np.dot(u,self.R),u.T)
        return qscore, rscore


######     MAIN ########
def find_optimal_control_sequence(data, dt=0.01):
    cost = 0

    num_ctrls  = 7 
    num_states = 7
    num_ee_pos = 3
    num_ee_ori = 4
    num_ee_omg = 3
    num_data_pts   = len(data)

    k = 0
    idx = {}
    #state variables
    idx['u_prev']       = range(k, k+num_ctrls);   k += num_ctrls    # 0-6   => previous control input
    idx['u_delta_prev'] = range(k,k+num_ctrls);    k += num_ctrls    # 7-13  => change in control signals
    idx['xyz_dot']      = range(k,k+num_ee_pos);   k += num_ee_pos   # 14-16 => end effecter velocities
    idx['xyz']          = range(k,k+num_ee_pos);   k += num_ee_pos   # 17-19 => end effecter positions
    idx['abg_dot']      = range(k,k+num_ee_omg);   k += num_ee_omg   # 20-22 => end-effecter angular velocities
    idx['genq']         = range(k,k+num_states*2); k += num_states*2 # 23-37 => generalized coordinates
    idx['q']            = range(k,k+num_ee_ori);   k += num_ee_ori   # 38-41 => it's critical for compose_dx.m, compute_dx.m (and likely other functions) for the quaternion to be the last entry in the state

    # non-state variables used as features in the model:
    idx['inputs']       = range(k,k+num_ctrls);    k += num_ctrls   
    idx['uvw']          = range(k,k+num_ee_omg);   k += num_ee_omg

    #horizon length = length of the trajectory to be followed
    H = num_data_pts


    move_quaternion_target = data[-1]['ee_ori']

    target_move_state = np.hstack([np.zeros(num_ctrls), 
                                   np.zeros(num_ctrls),
                                   np.zeros(num_ee_pos),
                                   np.zeros(num_ee_pos),
                                   np.zeros(num_ee_omg),
                                   np.zeros(num_states*2),
                                   move_quaternion_target])                              

    target_traj = {}
    #target_traj_x = []

    start_state   = copy.deepcopy(target_move_state)
    #just pre allocating memory
    target_traj_x = [start_state for _ in range(H)]

    for i in range(H):
        target_move_state[idx['xyz']]  = data[i]['ee_pos']
        target_move_state[idx['genq']] = np.hstack([data[i]['position'], data[i]['velocity']])
        target_traj_x[i]               = target_move_state.copy()

    #state trajectory
    target_traj['x'] = target_traj_x
    #control trajectory
    target_traj['u'] = [np.zeros(num_ctrls)  for _ in range(H)]
    #time trajectory
    target_traj['t'] = np.asarray([i for i in range(H)])*dt


    model_bias_est = np.zeros((H,6))

    u_prev_mult       = 0.5 # weightage given for control signal
    u_delta_prev_mult = 10  # weightage given for change in control signal
    xyz_dot_mult      = 1.  # weightage given for ee velocity
    xyz_mult          = 1.  # weightage given for ee position
    pqr_mult          = 0.4 # weightage given for ee angular velocity
    q_mult            = 0.5 # weightage given for ee angular position
    genq_mult         = 0.5 # weightage given for joint position and joint velocity

    reward={}
    #running cost for state 
    reward['state_multipliers'] = np.hstack([u_prev_mult       * np.ones(num_ctrls),
                                             u_delta_prev_mult * np.ones(num_ctrls),
                                             xyz_dot_mult      * np.ones(num_ee_pos),  
                                             xyz_mult          * np.ones(num_ee_pos),
                                             pqr_mult          * np.ones(num_ee_omg),
                                             genq_mult         * np.ones(num_states*2), 
                                             q_mult            * np.ones(num_ee_ori)]).T
    #running cost for control signal
    reward['input_multipliers'] = np.ones(num_ctrls)*0.1

    
    ddp = DDP_TrajFollowClass(idx=idx, 
                              H=H, 
                              model=data,
                              target_traj=target_traj, 
                              reward=reward, 
                              model_bias=model_bias_est)

    [lqr_traj, result_traj, im_trajs] = ddp.ddp_for_trajectory_following(start_state)

    # let's check how well we perform:
    # note: for any given system, we could run all the code we have in matlab,
    # and then just use the lines below "on-board" or in a C-code control loop

    x = start_state
    #preallocating memory
    traj_xlist = [np.zeros_like(x) for _ in range(H)]
    
    rate = 40
    rate = rospy.timer.Rate(rate)
    print "Starting to re-do demo"
    # lqr_traj = {}
    # lqr_traj['target_x'] = np.load('tgt_x.npy')
    # lqr_traj['K'] = np.load('tgt_k.npy')

    for i in range(0,H-1):
        #print "step \t", i
        dx = ddp.compute_dx(lqr_traj['target_x'][i], x)
        u  = np.dot(lqr_traj['K'][i],dx)
        # print lqr_traj['K'][i]
        #print "u \n", u
        #ddp.right_arm.exec_torque_cmd(u)
        
        traj_xlist[i] = x.copy() 

        x = ddp.simulate_f(index=i,
                           x0=x,
                           delta_u0=u,
                           sim_time=dt)
        #rate.sleep()

    traj_xlist[H-1] = x
    np.save('tgt_x.npy', lqr_traj['target_x'])
    np.save('tgt_k.npy', lqr_traj['K'])

    #state trajectory to be followed
    targetXtraj = np.asarray(target_traj_x)
    #resulting Xtraj with optimal control inputs
    Xtraj = np.asarray(traj_xlist)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(ee_list[0][0], ee_list[0][1], ee_list[0][2],  linewidths=20, color='r', marker='*')
    #ax.scatter(ee_list[-1][0],ee_list[-1][1],ee_list[-1][2], linewidths=20, color='g', marker='*')
    ax.scatter(Xtraj[:,idx['xyz'][0]],Xtraj[:,idx['xyz'][1]],Xtraj[:,idx['xyz'][2]], color='b', marker='*')
    ax.plot(targetXtraj[:,idx['xyz'][0]],targetXtraj[:,idx['xyz'][1]],targetXtraj[:,idx['xyz'][2]], color='m')
    ax.grid()
    plt.show()

if __name__=="__main__":
    rospy.init_node('lfd_node')
    demo_data = load_demo_data(demo_idx=6)
    find_optimal_control_sequence(data=demo_data, dt=0.01)