import numpy as np


class PROMPCtrl(object):
    """
    This class computes the feedback and feedforward gains of the a PROMP in
    closed loop. For more reference, see: https://link.springer.com/article/10.1007/s10514-017-9648-7
    The system we try to control is a linearised system of the form
    x_(t+1) = (I + At)x_t + Bt u_t
    """

    def __init__(self, promp_obj, A, B, D_mu=None, D_cov=None, dt=0.01):

        """
        Constructor of the class:
        Args:
        A = linearised system dynamics matrix
        B = linearised system control matrix
        D_mu = 
        D_cov= 
        promb_obj = Instatiation of the discrete promp class
        """
        #system matrix
        self._A = A

        #control matrix
        self._B = B

        #drift matrix mean
        if D_mu is None:
            self._D_mu = np.zeros(self._A.shape[0])
        else:
            self._D_mu = D_mu

        #derif matrix sigma
        if D_cov is None:
            self._D_cov = np.diag(np.zeros(self._B.shape[1]))
        else:
            self._D_cov = D_cov

        #time step
        self._dt = dt

        #the I matrix to compute next state
        self._I = np.eye(A.shape[0])

        self._promp = promp_obj


    def update_system_matrices(self, A, B, D_mu=None, D_cov=None):
        """
        Update the system matrices 
        this is for the purpose of adding time varying 
        system matrices
        """
        self._A = A
        self._B = B

        if D_mu is not None:
            self._D_mu = D_mu

        if D_cov is not None:
            self._D_cov = D_cov

    def compute_next_state(self, x, u):
        """
        Discretised dynamics function
        """

        x_nxt = np.dot( (self._I + self._A*self._dt), x ) + np.dot(self._B*self._dt, u) + self._D*self._dt


        return x_nxt


    def compute_gains(self, t, add_noise=True):
        """
        the control command is assumed to be of type
        u = Kx + k + eps
        """
        #system noise matrix
        cov_s = np.dot( np.dot(self._B, self._D_cov), self._B.T)

        B_pseudo = np.linalg.pinv(self._B)

        #get the basis function for the current time stamp
        d_basis, basis = self._promp.get_basis(t)
        traj_cov = self._promp.get_traj_cov()[t,t]

        tmp1 = np.dot(np.dot(d_basis, self._promp._sigma_W), basis.T)
        tmp2 = np.dot(self._A, traj_cov)

        print tmp1

        K = np.dot( np.dot(B_pseudo, (tmp1 - tmp2 - 0.5*self._D_cov)), np.linalg.inv(traj_cov) )

        tmp3 = self._A + np.dot(self._B, K)

        tmp4 = np.dot(d_basis, self._promp._mean_W)
        tmp5 = np.dot(basis, self._promp._mean_W)

        c = np.random.multivariate_normal(self._D_mu, self._D_cov, 1).T 

        k = np.dot(B_pseudo, (tmp4 - np.dot(tmp3, tmp5) )) - c

        return K, k



