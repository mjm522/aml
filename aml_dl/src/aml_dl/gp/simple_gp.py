from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import functools as ft
import time

__author__ = 'ermanoarruda'


def kernel(a, b, width, scale):
    """
     GP Squared Exponential Kernel
    """

    sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)

    return scale * np.exp(-(1.0 / (2 * width)) * sqdist)


def kernel2(a, b, width, scale):
    """
     GP Squared Exponential Kernel version 2
    """
    d = a - b
    sqdist = d*d

    return scale * np.exp(-(1.0 / (2 * width)) * sqdist)


def func(X):
    return X * np.cos(X)


class GaussianProcess:
    def __init__(self, noise, width, scale, kernel_func, mu_func=lambda x: 0):
        self.noise = noise
        self.width = width
        self.scale = scale
        self.x = np.array([], np.float64).reshape(-1, 1)
        self.y = np.array([], np.float64).reshape(-1, 1)
        self.K = np.array([], np.float64)
        self.kernel = kernel_func
        self.mu = mu_func
        self.precomputed_params = False

    def update(self, x, y):
        x_new = np.array(x).reshape(-1, 1)
        y_new = np.array(y).reshape(-1, 1)
        self.x = np.append(self.x, x_new, axis=0)
        self.y = np.append(self.y, y_new, axis=0)
        # print "X:",self.x
        # print "Y:",self.y
        ### PRIOR COVARIANCE ####

        self.K = kernel(self.x, self.x, self.width, self.scale)
        self.L = np.linalg.cholesky(self.K + self.noise * np.eye(len(self.x)))
        # print "L: ", L
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

        # print "K:", self.K, len(self.K)

    def compute_covariance(self, x1, x2):
        cov = np.zeros((len(x1), len(x2)), np.float64)

        for i in range(0, len(x1)):
            for j in range(0, len(x2)):
                cov[i, j] = kernel2(x1[i], x2[j], self.width, self.scale)

        return cov


    #### POSTERIOR ####
    def predict3(self, x_star):

        k_star = self.compute_covariance(self.x, x_star)

        f_star = np.dot(k_star.T, self.alpha)  # mean
        v = np.linalg.solve(self.L, k_star)

        K_ss = self.compute_covariance(x_star,x_star) 
        # self.kernel(x_star, x_star, width, scale) 

        var = K_ss - np.dot(v.T , v) 

        # Positive Loglikelhood
        logp = -0.5 * np.dot(self.y.T, self.alpha) - sum(
            np.log(self.L[range(0, len(self.L)), range(0, len(self.L))])) - (len(self.L) / 2.0) * np.log(
            2 * np.pi)

        # print "LOGP:", (1.0/(1+np.exp(-logp)))
        # K_star = np.zeros((self.K.shape[0] + 1, self.K.shape[0] + 1));
        # print K_star.shape
        return (f_star, var, logp)



    #### POSTERIOR ####
    def predict2(self, x_star):
        predtmp = map(self.predict, x_star)
        pred = map(lambda x: x[0][0][0], predtmp)
        var = map(lambda x: x[1][0][0], predtmp)
        # Positive Loglikelhood
        logp = -0.5 * np.dot(self.y.T, self.alpha) - sum(
            np.log(self.L[range(0, len(self.L)), range(0, len(self.L))])) - (len(self.L) / 2.0) * np.log(
            2 * np.pi)
        return (pred, var, logp)

    # Sample prior from given set of x-values returning n_samples
    def sample_prior2(self, x, n_samples):
        K_prior = self.kernel(x, x, self.width, self.scale)
        L = np.linalg.cholesky(K_prior + self.noise * np.eye(len(x)))
        samples = self.mu(x) + np.dot(L, np.random.normal(size=(len(x), n_samples)))
        return samples

    def sample_prior(self, n_samples):
        L = np.linalg.cholesky(self.K + self.noise * np.eye(len(self.x)))
        samples = self.mu(self.x) + np.dot(L, np.random.normal(size=(len(self.x), n_samples)))
        return samples

    def sample_univariate(self, mu, sigma, n_points, n_samples):

        L = np.sqrt(sigma)
        samples = np.add(mu, L * np.random.normal(size=(n_points, n_samples)))
        # Negative log likelihood
        logp_pred = 0.5 * np.log(2 * np.pi * (sigma ** 2)) + ((samples - mu) ** 2) / (2 * sigma ** 2)

        return (samples, logp_pred)

    def sample_multivariate(self, mu, cov, n_points, n_samples):
        return np.random.multivariate_normal(mu.ravel(), cov,n_samples).T



def main():

    # gp = GaussianProcess(0.5, 0.5, 2)
    # gp.update([0, 1], [1, 2])
    # print gp.x
    # Xtest = np.linspace(-4 * np.pi, 4 * np.pi, 5).reshape(-1, 1)
    # print Xtest

    pl.ion()
    #np.random.seed(2)
    noise_level = 0.00000001;
    width = 0.5
    scale = 30


    gp = GaussianProcess(noise_level, width, scale, kernel)

    n = 100
    n2 = 100

    ### PRIOR OBSERVATIONS ####
    corruption_noise = 0.00001;
    Xtest = np.linspace(-4 * np.pi, 4 * np.pi, n).reshape(-1, 1)

    y = func(Xtest) + np.random.normal(size=(n, 1)) * corruption_noise

    #### TEST POINTS ####
    Xtest2 = np.linspace(-4 * np.pi, 4 * np.pi, n2).reshape(-1, 1)

    obs_prob = np.array([])
    # gp.update(Xtest2[0], y[0])
    for i in range(0, n):
        idx = np.random.uniform(1, n)
        # idx = i
        gp.update(Xtest[int(idx)], y[int(idx)])


        #### PREDICTIONS ####
        # pred, var, logp = gp.predict(Xtest2)
        pred, cov, logp = gp.predict3(Xtest2)
        std = np.sqrt(np.diagonal(cov))
        # print "cov: ", cov
        # print "diag: ",np.diagonal(cov)
        print "Marginal prob", 1 / (1 + np.exp(-logp))
        # print pred
        # print var
        obs_prob = np.append(obs_prob, 1 / (1 + np.exp(-logp)))
        raw_input("Press Enter to continue...")
        # (sample_posterior, logp_pred) = gp.sample_univariate(np.array(pred).reshape(-1, 1), np.array(var).reshape(-1, 1),
        #                                                      len(pred), 3)

        sample_posterior = gp.sample_multivariate(pred, cov, len(pred), 5)

        #### PREDICTIONS PLOTTING ####
        fig = pl.figure(0)
        pl.clf()
        pl.errorbar(Xtest2, pred, yerr=std, fmt='b-o', ecolor='g',linewidth=3.5)
        pl.plot(gp.x, gp.y, 'ro')
        f = func(Xtest2)
        pl.plot(Xtest2, f, 'dodgerblue', linewidth=2.5)
        pl.xlabel('input, x')
        pl.ylabel('output, f(x)')
        pl.figure(2)
        pl.plot(range(0,len(gp.x)), obs_prob)

        #### PRIOR SAMPLING PLOTTING ####
        f_prior = gp.sample_prior2(Xtest2, 4)

        pl.figure(3)
        pl.clf()
        pl.plot(Xtest2, f_prior)
        pl.xlabel('input, x')
        pl.ylabel('output, f(x)')

        #### POSTERIOR SAMPLING PLOTTING ####
        pl.figure(4)
        pl.clf()
        pl.errorbar(Xtest2, pred, yerr=std, fmt='b-', ecolor='g', elinewidth=0.25, linewidth=3.5)
        pl.plot(Xtest2, f, 'dodgerblue', linewidth=3.0)
        pl.plot(Xtest2, sample_posterior)
        pl.plot(gp.x, gp.y, 'ro')
        # pl.figure(5)
        # prob = (1 / (1 + np.exp(logp_pred)))
        # pl.plot(Xtest2, map(lambda x: x[0], prob))
        print "Error: ", np.linalg.norm(f - pred)
        print "Uncertainty ", np.linalg.norm(std)
        pl.draw()
        pl.show()
        time.sleep(1)
        # raw_input("Press Enter to continue...")

if __name__ == '__main__':

    main()