import numpy as np


class PROMPCtrl(object):
    """
    This class computes the feedback and feedforward gains of the a PROMP in
    closed loop. For more reference, see: https://link.springer.com/article/10.1007/s10514-017-9648-7
    """

    def __init__(self, traj_data, dt=0.005):

        """
        Constructor of the class:
        Args:
        Phi = Basis function
        PhiD = Derivative of basis function
        PhiDD = Second derivative of basis function
        mean_W = computed mean weights
        sigma_W = computed variance of weights
        """

        #system matrix
        self._A = None

        #control matrix
        self._B = None

        #time step
        self._dt = dt

        self._Phi   = traj_data['Phi']
        self._PhiD  = traj_data['PhiD']
        self._PhiDD = traj_data['PhiDD']

        self._mean_W   = traj_data['mean_W']
        self._sigma_W  = traj_data['sigma_W']

        #time steps
        self._time_steps = self._Phi.shape[1]

       
    def update_system_matrices(self, A, B):
        """
        Update the system matrices 
        this is for the purpose of adding time varying 
        system matrices
        Args:
        A = System matrix shape: [state_dim x state_dim]
        B = Control matrix shape: [state_dim x action_dim]
        """
        self._A = A
        self._B = B


    def get_basis(self, t):
        """
        This function creates a basis and Dbasis
        basis  = [Phi; PhiD]
        Dbasis = [PhiD; PhiDD]
        """
        return np.vstack([self._Phi[:, t], self._PhiD[:, t]]), np.vstack([self._PhiD[:, t], self._PhiDD[:, t]])



    def compute_gains(self, t, inv_thresh=1e-4):

        """
        the control command is assumed to be of type
        u = Kx + k + eps
        """
        #get the basis funtion at a time step
        basis, Dbasis = self.get_basis(t)

        if t < self._time_steps-1:
            basis_t_dt, _ = self.get_basis(t+1)
        else:
            basis_t_dt = np.zeros_like(basis)


        #part 1 equation 46
        B_pseudo = np.linalg.pinv(self._B, rcond=inv_thresh)

        #equation 12 for t
        Sigma_t =  np.dot(np.dot(basis, self._sigma_W), basis.T)

        #equation 12 for t+dt
        Sigma_t_dt = np.dot(np.dot(basis_t_dt, self._sigma_W), basis_t_dt.T)

        #Cross correlation between t, t+dt, Equation 49
        Ct = np.dot(np.dot(basis, self._sigma_W), basis_t_dt.T)

        #System noise Equation 51
        Sigma_s = (1./self._dt)* ( Sigma_t_dt - np.dot( np.dot( Ct.T, np.linalg.pinv(Sigma_t, rcond=inv_thresh) ), Ct) )

        #control noise Equation 52
        Sigma_u = np.dot(np.dot(B_pseudo, Sigma_s), B_pseudo.T)

        #part 2 equation 46
        tmp1 = np.dot(np.dot(Dbasis, self._sigma_W), basis.T)

        #part 3 equation 46
        tmp2 = np.dot(self._A, Sigma_t) + 0.5*Sigma_s

        #compute feedback gain; complete equation 46
        K = np.dot( np.dot(B_pseudo, (tmp1-tmp2) ), np.linalg.pinv(Sigma_t, rcond=inv_thresh))

        #part 1 equation 48
        tmp3 = np.dot(Dbasis, self._mean_W)

        #part 2 equation 48
        tmp4 = np.dot( (self._A + np.dot(self._B, K)), np.dot(basis, self._mean_W) )

        #compute feedforward gain; complete equation 48
        k = np.dot(B_pseudo, (tmp3-tmp4))

        return K, k, Sigma_u


    def compute_gain_traj(self):
        """
        This function is to compute the entire gain trajectory
        of a given state distribution
        """
        time_steps = self._Phi.shape[1]
        state_dim, action_dim = self._B.shape

        K_traj = np.zeros([time_steps, state_dim, state_dim])
        k_traj = np.zeros([time_steps, action_dim])
        Sigma_u_traj = np.zeros([time_steps, action_dim, action_dim])

        for t in range(time_steps):

            K_traj[t, :, :], k_traj[t, :], Sigma_u_traj[t, :, :] = self.compute_gains(t)

        return K_traj, k_traj, Sigma_u_traj


    def compute_control_cmd(self, t, state, sample=False):
        """
        This function is compute the specific control
        command at a time step t 
        Args: 
        t : time step
        state : state for which control command needs to be computed
        """

        K, k, Sigma_u = self.compute_gains(t)

        mean_u = np.dot(K, state) + k

        if sample:
            #get a weight sample from the weight distribution
            return np.random.multivariate_normal(mean_u, Sigma_u, 1).T,  Sigma_u

        else:

            return mean_u, Sigma_u


    def compute_ctrl_traj(self, state_list):
        """
        This function computes an entire
        control sequence for a given state list
        Args:
        state_list for which control has to be computed: shape [state_dim * time_steps]
        this assumes that len(state_list) = timesteps in the basis function
        """

        _, action_dim = self._B.shape

        ctrl_cmds_mean  =  np.zeros([self._time_steps, action_dim])
        ctrl_cmds_sigma =  np.zeros([self._time_steps, action_dim, action_dim])

        for t in range(self._time_steps-1):

            ctrl_cmds_mean[t, :], ctrl_cmds_sigma[t, :, :] = self.compute_control_cmd(t, state_list[t,:])

        return ctrl_cmds_mean, ctrl_cmds_sigma
