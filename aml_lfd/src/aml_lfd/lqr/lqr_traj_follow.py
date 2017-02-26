import numpy as np
import scipy.linalg as sp_linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from aml_robot.baxter_robot import BaxterArm
import rospy

from aml_lfd.utilities.utilities import quat_mult, compute_exp, load_demo_data

class LQRTrajFollow():
    def __init__(self, robot_interface, idx, H, model, target_traj, reward):
        self.idx            = idx
        self.H              = H
        self.target_traj    = target_traj
        self.model          = model
        self.reward         = reward
        self.num_states     = None
        self.num_ctrls      = None
        self.num_links      = None
        self.configure()
        self.robot          = robot_interface
    
    def configure(self):
        self.num_states = len(self.reward['state_multipliers'])
        self.num_ctrls  = len(self.reward['input_multipliers'])
        self.num_links  = len(self.reward['input_multipliers'])
        self.A  = np.zeros((self.num_states, self.num_states))
        self.B  = np.zeros((self.num_states, self.num_ctrls))
        self.Q  = np.diag(self.reward['state_multipliers'])
        self.Qf = np.diag(self.reward['state_multipliers'])
        self.R  = np.diag(self.reward['input_multipliers'])
        self.Rf = np.diag(self.reward['input_multipliers'])
        self.Alist = [self.A for _ in range(self.H)]
        self.Blist = [self.B for _ in range(self.H)]
        self.Qlist = [self.Q for _ in range(self.H)]
        self.Rlist = [self.R for _ in range(self.H)]
        #K has same size as A
        self.Klist = [np.zeros((self.num_states, self.num_states)) for _ in range(self.H)]
        self.target_xlist = [np.zeros((self.num_states,1)) for _ in range(self.H)]
        self.target_ulist = [np.zeros((self.num_ctrls,1))  for _ in range(self.H)]

        self.dxlist = [np.zeros(self.num_states) for _ in range(self.H)]
        self.xlist  = [np.zeros(self.num_states) for _ in range(self.H)]
        self.ulist  = [np.zeros(self.num_ctrls)  for _ in range(self.H)]

    def compute_dx(self, xtarget, x):
        dx  = x - xtarget
        dx[self.idx['ori']] = quat_mult(lq=np.multiply(np.array([1., -1., -1., -1.]),xtarget[self.idx['ori']]), 
                                          rq=x[self.idx['ori']])
        dx[self.idx['ori'][0]] = 1
        return dx

    def err_simulate(self, dx0, u0, sim_time):
        # dx0 is a state relative to some nominal trajectory
        x0  = self.compose_dx(xnom0, dx0)
        # x1 = run simulator on x0, u0
        x1  = self.simulate_f(index, x0, u0, sim_time)
        # compute state relative to nominal trajectory at next step
        dx1 = self.compute_dx(xnom1, x1)
        return dx1

    def compose_dx(self, x0, dx):
        x1                = x0 + dx
        x0[self.idx['ori']] = x0[self.idx['ori']]/np.linalg.norm(x0[self.idx['ori']])
        quat0             = x0[self.idx['ori']]
        dq                = np.hstack([ dx[[self.idx['ori'][1:4]]], np.sqrt(1-np.linalg.norm(dx[[self.idx['ori'][1:4]]])**2) ])
        x1[self.idx['ori']] = quat_mult(quat0, dq)
        return x1

    def simulate_f(self, x0, u0, dt):
   
        x1    = np.zeros_like(x0)

        q     = x0[self.idx['q']]

        dq    = x0[self.idx['q_dot']]

        Mq    = self.robot.get_arm_inertia(joint_angles=q)
        
        ddq   = np.dot(np.linalg.inv(Mq), u0)
        
        dq   += ddq*dt

        q    += dt*dq

        x1[self.idx['q']]     = q

        x1[self.idx['q_dot']] = dq
       
        # J = self.robot.get_jacobian_from_joints(joint_angles=q)

        # ee_pose_dot = np.dot(J, dq)

        # x1[self.idx['xyz_dot']] = ee_pose_dot[0:3]

        # x1[self.idx['xyz']] = x0[self.idx['xyz']] + ee_pose_dot[0:3]*dt

        # x1[self.idx['ori']] = quat_mult(compute_exp(dt/2.*ee_pose_dot[3:6]), x0[self.idx['ori']])
    
        return x1

    def linearized_dynamics(self, x0, u0, dt):

        epsilon = np.ones(self.num_states)*.01
        #epsilon(end-4:end-1) = ones(1,4) 

        for i in range(0, self.num_states): #[0],epsilon.shape[1]
            delta       = np.zeros_like(x0)
            delta[i]    = epsilon[i]

            fx_t1m      = self.simulate_f( x0=x0 - delta, 
                                            u0=u0,  
                                            dt=dt)

            fx_t1p      = self.simulate_f(  x0=x0 + delta, 
                                            u0=u0,  
                                            dt=dt)

            self.A[:,i] = ((fx_t1p - fx_t1m)/epsilon[i]/2).reshape(-1)
        
        epsilon = np.ones(self.num_ctrls)

        for i in range(self.num_ctrls):
            delta       = np.zeros_like(u0)
            delta[i]    = epsilon[i]

            fx_t1m      = self.simulate_f(x0=x0, 
                                            u0=u0 - delta, 
                                            dt=dt)

            fx_t1p      = self.simulate_f( x0=x0, 
                                            u0=u0 + delta,  
                                            dt=dt)

            self.B[:,i] = ((fx_t1p - fx_t1m)/epsilon[i]/2).reshape(-1)



    def lqr(self, A, B, Q, R):
        P = np.matrix(sp_linalg.solve_continuous_are(A, B, Q, R)) #solve continous time ricatti equation
        K = np.matrix(sp_linalg.inv(R)*(B.T*P)) #compute the LQR gain
        eigVals, eigVecs = sp_linalg.eig(A-B*K)

        return K, eigVals

    def dlqr(self, A, B, Q, R):
        P = np.matrix(sp_linalg.solve_discrete_are(A, B, Q, R))#solve discrete time ricatti equation
        K = np.matrix(sp_linalg.inv(B.T*P*B+R)*(B.T*P*A))#compute the LQR gain
        eigVals, eigVecs = sp_linalg.eig(A-B*K)

        return K, eigVals

    def find_feedback_gains(self, dt):

        test_err_list = np.zeros((self.num_states,self.H))

        for t in range(self.H-1):
            self.linearized_dynamics(x0=self.target_traj['x'][t].T, 
                                     u0=self.target_traj['u'][t].T,
                                     dt=dt)

            self.Alist[t]        = self.A.copy()
            self.Blist[t]        = self.B.copy()

            test_err_list[:,t]  = self.target_traj['x'][t+1] - (np.dot(self.A, self.target_traj['x'][t]) + np.dot(self.B,self.target_traj['u'][t]))

            self.Qlist[t]        = self.Q*dt
            self.Rlist[t]        = self.R*dt
            self.Klist[t],_      = self.dlqr(self.A, self.B, self.Q*dt, self.R*dt)
            #self.Klist[i],_      = self.lqr(self.A, self.B, self.Q*self.dt, self.R*self.dt)
        lqr_traj = {}
        lqr_traj['K'] = self.Klist
        

        plt.figure(1)
        for i in range(self.num_states):
            plt.plot(test_err_list[i,:])
        plt.show()
        # print asdfkjahfkjadshf
        

        return lqr_traj

######     MAIN ########
def find_optimal_control_sequence(robot_interface, data, dt=0.01):

    num_ctrls  = 7 
    num_states = 7
    num_ee_pos = 3
    num_ee_ori = 4
    num_ee_omg = 3
    num_data_pts   = len(data)

    k = 0
    idx = {}
    #state variable
    idx['q']             = range(k,k+num_states);   k += num_states # 23-37 => generalized coordinates
    idx['q_dot']         = range(k,k+num_states);   k += num_states # 23-37 => generalized coordinates
    idx['xyz']           = range(k,k+num_ee_pos);   k += num_ee_pos   # 17-19 => end effecter positions
    idx['ori']           = range(k,k+num_ee_pos);   k += num_ee_ori   # 17-19 => end effecter positions
    idx['xyz_dot']       = range(k,k+num_ee_pos);   k += num_ee_pos   # 14-16 => end effecter velocities
    idx['omg']           = range(k,k+num_ee_omg);   k += num_ee_omg   # 20-22 => end-effecter angular velocities

    # non-state variables used as features in the model:
    idx['u']             = range(k,k+num_ctrls);    k += num_ctrls   


    #horizon length = length of the trajectory to be followed
    H = num_data_pts

                                   # np.zeros(num_ee_pos),
                                   # np.zeros(num_ee_ori),
                                   # np.zeros(num_ee_pos),
                                   # np.zeros(num_ee_omg)
    target_move_state = np.hstack([np.zeros(num_states), 
                                   np.zeros(num_states)])                              

    target_traj = {}

    #just pre allocating memory
    target_traj_x = [target_move_state for _ in range(H)]

    for i in range(H):
        # target_move_state[idx['xyz']]       = data[i]['ee_pos']
        # target_move_state[idx['ori']]       = data[i]['ee_ori']
        target_move_state[idx['q']]         = data[i]['position']
        target_move_state[idx['q_dot']]     = data[i]['velocity']

        target_traj_x[i]                    = target_move_state.copy()


    #state trajectory
    target_traj['x'] = target_traj_x
    #control trajectory
    target_traj['u'] = [np.random.random(num_ctrls)  for _ in range(H)]
    #time trajectory
    target_traj['t'] = np.ones(H)*dt


    q_mult            = 1.5
    q_dot_mult        = 1.5
    xyz_mult          = 1.  # weightage given for ee position
    ori_mult          = 0.1
    xyz_dot_mult      = 1.  # weightage given for ee velocity
    omg_mult          = 0.4
    u_mult            = 3.1

    reward={}
    #running cost for state 
     # xyz_mult       * np.ones(num_ee_pos),
     # ori_mult       * np.ones(num_ee_ori),
     # xyz_dot_mult   * np.ones(num_ee_pos),
     # omg_mult       * np.ones(num_ee_omg),
    reward['state_multipliers'] = np.hstack([q_mult         * np.ones(num_states),
                                             q_dot_mult     * np.ones(num_states)]).T
    #running cost for control signal
    reward['input_multipliers'] = np.ones(num_ctrls)*u_mult

    
    lqrtrajfollow   = LQRTrajFollow(robot_interface=robot_interface,
                                    idx=idx, 
                                    H=H, 
                                    model=data,
                                    target_traj=target_traj, 
                                    reward=reward)

    lqr_traj = lqrtrajfollow.find_feedback_gains(dt=dt)
    # np.save('lqr_Ks.npy', lqr_traj['K'])

    lqr_Ks = np.load('lqr_Ks.npy')
    
        # let's check how well we perform:
    # note: for any given system, we could run all the code we have in matlab,
    # and then just use the lines below "on-board" or in a C-code control loop

    x = target_traj_x[0] 
    #preallocating memory
    traj_xlist = [np.zeros_like(x) for _ in range(H)]
    
    rate = rospy.timer.Rate(demo_data[0]['sampling_rate'])
    print "Starting to re-do demo"
    # lqr_traj = {}
    # lqr_traj['target_x'] = np.load('tgt_x.npy')
    # lqr_traj['K'] = np.load('tgt_k.npy')

    xlist = np.zeros((lqrtrajfollow.num_states,H))
    x_err_list = np.zeros((lqrtrajfollow.num_states,H))
    ulist = np.zeros((lqrtrajfollow.num_ctrls,H))

    for i in range(0,H-1):
        #print "step \t", i
        # dx = ddp.compute_dx(lqr_traj['target_x'][i], x)
        # u  = np.asarray(np.dot(lqr_traj['K'][i], x))[0]
        u  = np.asarray(np.dot(lqr_Ks[i], x))[0]

        xlist[:,i] = x
        ulist[:,i] = u

        x_err_list[:,i] = target_traj_x[i] - x

        # lqrtrajfollow.robot.exec_torque_cmd(u)
        
        traj_xlist[i] = x.copy() 

        x = lqrtrajfollow.simulate_f(x0=x,
                           u0=u,
                           dt=dt)
        # rate.sleep()

    plt.figure(1)
    for i in range(lqrtrajfollow.num_states):
        plt.plot(x_err_list[i,:-380])
    
    plt.figure(2)
    for i in range(lqrtrajfollow.num_ctrls):
        plt.plot(ulist[i,:-380])

    plt.show()

    print adfhaskfjh

    #state trajectory to be followed
    targetXtraj = np.asarray(target_traj_x)
    #resulting Xtraj with optimal control inputs
    Xtraj = np.asarray(traj_xlist)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(ee_list[0][0], ee_list[0][1], ee_list[0][2],  linewidths=20, color='r', marker='*')
    #ax.scatter(ee_list[-1][0],ee_list[-1][1],ee_list[-1][2], linewidths=20, color='g', marker='*')
    # ax.scatter(Xtraj[:,idx['xyz'][0]],Xtraj[:,idx['xyz'][1]],Xtraj[:,idx['xyz'][2]], color='b', marker='*')
    # ax.plot(targetXtraj[:,idx['xyz'][0]],targetXtraj[:,idx['xyz'][1]],targetXtraj[:,idx['xyz'][2]], color='m')
    # ax.grid()
    # plt.show()

if __name__=="__main__":
    rospy.init_node('lfd_node')
    demo_data = load_demo_data(demo_idx=6)
    
    arm  = BaxterArm('right')

    arm.untuck_arm()

    find_optimal_control_sequence(robot_interface=arm, data=demo_data, dt=0.01)
