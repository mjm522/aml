import numpy as np
import itertools

class LinGaussPolicy():
    """
    Representation of pi(w|s)
    where w is parameter of the policy and s is the context
    We assume w ~ N(mu, sigma)
    our aim is to lean the distribution of w from samples collected
    """

    def __init__(self, w_dim, context_feature_dim, variance, 
                 initial_params, bounds=None, covariance_scale=1.0, gamma=0.0, random_state=None, transform=True):

        #initial guess of w
        self._w_init = initial_params

        #dimenstion of the w
        self._w_dim = w_dim

        self._context_dim = context_feature_dim

        self._covariance_scale = covariance_scale

        #regularization for the fitting 
        self._gamma = gamma

        self._random_state = random_state

        self._bounds = bounds

        self._variance = variance

        self._scaling = 1.

        self._inv_scaling = 1.

        self._scaling *= self._variance

        self._inv_scaling /= self._variance

        self._mean = self.inv_scale_w(self._w_init)

        self._transform = transform

        #function handle to feaurize the context infomration
        self.transform_context = self.featurize_context()

        # Create weight matrix and covariance matrix Sigma
        self._w = np.zeros((self._w_dim, self._context_dim))

        if self._mean is not None:
            # It is assumed that the last dimension of the context is a
            # constant bias dimension
            self._w[:, -1] = self._mean

        #sigma of the policy param w
        self._sigma = np.eye(self._w_dim) * self._covariance_scale


    def fit(self, S, B, weights):
        
        #array of contexts
        S = np.asarray(S)
        #array of w
        B = np.asarray(B)
        #solution to the dual problem
        weights = np.asarray(weights)

        #rescale the params
        B = self.inv_scale_w(B.T).T

        # Avoid that all but one weights become 0
        weights[weights == 0] = np.finfo(np.float).eps
        Z = (weights.sum() ** 2 - (weights ** 2).sum()) / weights.sum()

        nominator = np.zeros_like(self._sigma)
        for i in range(B.shape[0]):
            temp = B[i] - self._w.dot(S[i])
            nominator += weights[i] * np.outer(temp, temp)
        #equation 11 from paper
        self._sigma = nominator / Z

        if not np.isfinite(self._sigma).all():
            raise ValueError("Computed non-finite covariance matrix.")

        P = np.diag(weights)
        #equation 12 from paper
        self._w = np.linalg.pinv(S.T.dot(P).dot(S) + np.eye(S.shape[1]) *
                                self._gamma).dot(S.T).dot(P).dot(B).T

    def compute_w(self, context, explore=True, transform=True):

        if self._transform:
            context_feature = self.transform_context(context)
        else:
            context_feature = context

        if explore:
            w =  self._random_state.multivariate_normal(
                self._w.dot(context_feature), self._sigma, size=[1])[0]
        else:
            w = self._w.dot(context_feature)

        w = self.scale_w(w)

        if self._bounds is not None:
            np.clip(w, self._bounds[0, :], self._bounds[1, :], out=w)

        return w

    def scale_w(self, w):
        
        return np.multiply(self._scaling, w)

    def inv_scale_w(self, w):

        return np.multiply(self._inv_scaling, w)

    def featurize_context(self):

        def polynomial(context, n_degrees=2):
            # From sklearn.preprocessing.PolynomialFeatures
            # Find permutations/combinations which add to degree or less
            context = np.asarray(context)
            n_features = context.shape[0]
            powers = itertools.product(*(range(n_degrees + 1)
                                         for i in range(n_features)))
            powers = np.array([c for c in powers if 0 <= np.sum(c) <= n_degrees])
            # Sort so that the order of the powers makes sense
            i = np.lexsort(np.vstack([powers.T, powers.sum(axis=1)]))
            powers = powers[i][::-1]
            return (context ** powers).prod(-1)
        
        def quadratic(context):
            return polynomial(context, n_degrees=2)

        return quadratic