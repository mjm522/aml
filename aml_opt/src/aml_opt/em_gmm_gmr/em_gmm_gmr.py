import numpy as np
import numpy.matlib

class Model:
    
    def __init__(self, data):
        
        self._nb_states = 3 #Number of Gaussians in the GMM
        self._nb_var = 2 #Dimension of the datapoints in the dataset
        self.dt = 0 #time step 
        self._priors = np.ones(self._nb_states)*0.001
        self._mu = np.zeros([self._nb_var, self._nb_states])
        self._sigma = np.zeros([self._nb_var, self._nb_var, self._nb_states])
        self.init_model(data)

    def init_model(self, data):

        diagRegularizationFactor = 1e-4

        DataAll = data.reshape(data.shape[0], data.shape[1])

        TimingSep = np.linspace( np.min(DataAll[0,:] ), np.max( DataAll[0,:] ), self._nb_states+1 )

        Mu = np.zeros([self._nb_var, self._nb_states])

        Sigma = np.zeros([self._nb_var, self._nb_var, self._nb_states])
        
        for i in range(self._nb_states):

            idtmp   = np.argwhere( (DataAll[0, :] >= TimingSep[i] ) &  ( DataAll[0,:] < TimingSep[i+1] ) ).flatten()
            Mu[:,i] = np.mean( DataAll[:,idtmp], 1 ).flatten()
            Sigma[:,:,i] = np.cov( DataAll[:,idtmp].T, rowvar=False ) + np.eye( DataAll.shape[0] )*diagRegularizationFactor
            self._priors[i] = len(idtmp)
        
        self._priors = self._priors / np.sum(self._priors)

        for i in range(self._nb_states):
            self._mu[:,i] = Mu[ :, i ]
            tmp = Sigma[:, :, i]
            self._sigma[:,:,i] = tmp[:, :]


class EM_GMM_GMR:

    def __init__(self, data):
        #Parameters of the EM algorithm
        self._nb_min_steps = 5 #Minimum number of iterations allowed
        self._nb_max_steps = 200 #Maximum number of iterations allowed
        self._max_diff_LL = 1e-4 #Likelihood increase threshold to stop the algorithm
        self._nb_data = data.shape[1]

        #diagRegularizationFactor = 1E-2;
        self.diagRegularizationFactor = 1e-4

        self.model= Model(data)
        self.data = data

    def run_em(self):

        LL = np.zeros(self._nb_max_steps)
        
        for itr in range(self._nb_max_steps):
            #E-step
            L, gamma, gamma0 = self.compute_gamma() #See 'computegamma' function below
            gamma2 = np.divide(gamma, np.matlib.repmat(np.sum(gamma,1), self._nb_data, 1).T)
            #M-step
            for i in range(self.model._nb_states):
                #Update Priors
                self.model._priors[i] = np.sum( np.sum( gamma[i,:]) ) / self._nb_data
                
                #Matricization/flattening of tensor
                DataMat = self.data[:,:]
                #Update Mu 
                self.model._mu[:,i] = np.dot(DataMat, gamma2[i,:].T)
                #Update Sigma (regularization term is optional) 
                DataTmp = DataMat - np.matlib.repmat(self.model._mu[:,i], self._nb_data, 1).T

                self.model._sigma[:,:,i] = DataTmp.dot( np.diag( gamma2[i,:]) ).dot( DataTmp.T ) + np.eye(self.model._nb_var) * self.diagRegularizationFactor
        
            #Compute average log-likelihood 
            LL[itr] = np.sum( np.log( np.sum(L,0) ) ) / L.shape[1]
            #Stop the algorithm if EM converged (small change of LL)
            if itr>self._nb_min_steps:
                if ( LL[itr]-LL[itr-1] < self._max_diff_LL ) or ( itr==self._nb_max_steps-1 ):
                    print('EM converged after ' +  str(itr) + 'iterations.')
                    return self.model
                
        print('The maximum number of ' + str(self._nb_max_steps) + ' EM iterations has been reached.')

        return self.model

    def compute_gamma(self):

        L = np.ones([self.model._nb_states, self._nb_data])
        gamma0 = np.zeros([self.model._nb_states, self._nb_data])
        
        DataMat = self.data[:,:] #Matricization/flattening of tensor
        
        for i in range(self.model._nb_states):
            gamma0[i,:] = np.dot( self.model._priors[i],  self.gauss_PDF(DataMat, self.model._mu[:,i], self.model._sigma[:,:,i]) )
            L[i,:] = np.multiply(L[i,:], np.squeeze(gamma0[i,:]).T)
    
        #Normalization
        gamma = np.divide(L, np.matlib.repmat( np.sum(L,0) + np.finfo(float).tiny, L.shape[0], 1) )

        return L, gamma, gamma0


    def gmm_products(self):
        """
        this function is not tested
        """

        #GMM products 
        for i in range(self.model._nb_states):
            SigmaTmp = np.zeros([self.model._nb_var, self.model._nb_var])
            MuTmp = np.zeros(self.model._nb_var)
            
            # for m in range(self.model._nb_frames):
            MuP = self.model._mu[:,i] 
            SigmaP = self.model._sigma[:,:,i]

            SigmaTmp = SigmaTmp + np.linalg.inv(SigmaP)
            MuTmp = MuTmp + np.dot( np.linalg.inv(SigmaP), MuP)
                
            r._sigma[:,:,i] = np.linalg.inv(SigmaTmp)
            r._mu[:,i] = np.dot( r._sigma[:,:,i], MuTmp)


    def gmr(self, data_in):
    
        if data_in.ndim == 1:
            data_in = data_in.reshape(1, len(data_in))
        
        nb_data = data_in.shape[1]
        in_ = range(data_in.shape[0])
        out = range(in_[-1]+1, self.model._nb_var)
        nb_var_out = len(out)

        # GMR to estimate attractor path and associated variations

        weights = np.zeros([self.model._nb_states, nb_data])
        data_out_mu = np.zeros([nb_var_out, nb_data])
        data_out_sigma = np.zeros([nb_var_out, nb_var_out, nb_data])

        #GMR
        for t in range(nb_data):

            #Compute activation weight
            for i in range(self.model._nb_states):
                tmp = self.model._sigma[:,in_,i]
                weights[i,t] = self.model._priors[i] * self.gauss_PDF(data_in[:,t], self.model._mu[in_, i], tmp[in_,:])

            weights[:,t][ weights[:,t] < 1e-5] = 0.
            weights[:,t] = weights[:,t]/np.sum(weights[:,t])
            #Evaluate the current target 
            data_mu_tmp = np.zeros(nb_var_out)
            data_sigma_tmp = np.zeros([nb_var_out, nb_var_out])

            for i in range(self.model._nb_states):
                
                #sigma[out,out,i]
                tmp1 = self.model._sigma[:, out, i]
                sig_out_out = tmp1[out,:]

                #sigma[out,in,i]
                tmp2 = self.model._sigma[:, in_, i]
                sig_out_in = tmp2[out, :]

                #sigma[in, in, i]
                tmp3 = self.model._sigma[:, in_, i]
                sig_in_in = tmp3[in_, :]

                #sigma[in, out, i]
                tmp4 = self.model._sigma[:, out, i]
                sig_in_out = tmp4[in_, :]

                tar_tmp = self.model._mu[out,i] + np.dot( np.dot( sig_out_in, np.linalg.inv(sig_in_in) ), (data_in[:,t]-self.model._mu[in_,i] ) )
                sigma_tmp = sig_out_out - np.dot( np.dot( sig_out_in, np.linalg.inv(sig_in_in) ), sig_in_out)

                data_mu_tmp += weights[i,t]*tar_tmp
                data_sigma_tmp += weights[i,t]*sigma_tmp 

            data_out_mu[:,t] = data_mu_tmp
            data_out_sigma[:,:,t] = data_sigma_tmp
            
        return data_out_mu, data_out_sigma

    def gauss_PDF(self, data, mu, sigma):
        """
        Likelihood of datapoint(s) to be generated by a Gaussian parameterized by center and covariance.
        %
        Inputs -----------------------------------------------------------------
        %   o Data:  D x N array representing N datapoints of D dimensions.
        %   o Mu:    D x 1 vector representing the center of the Gaussian.
        %   o Sigma: D x D array representing the covariance matrix of the Gaussian.
        Output -----------------------------------------------------------------
        %   o prob:  1 x N vector representing the likelihood of the N datapoints.     
        """

        if data.ndim == 1:
            data = data.reshape(1, len(data))

        nb_var, nb_data = data.shape

        data = data.T - np.matlib.repmat(mu.T, nb_data, 1)

        prob = np.sum( np.multiply( np.dot( data, np.linalg.pinv(sigma) ), data) , 1)

        return np.exp(-0.5*prob) / np.sqrt((2*np.pi)**nb_var * (np.abs(np.linalg.det(sigma))+np.finfo(float).tiny))
  