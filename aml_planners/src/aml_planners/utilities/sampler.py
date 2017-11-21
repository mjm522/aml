import random
import numpy as np

np.random.seed(0)


class Sampler(object):

    def __init__(self, model, start, goal, config):
        self._model   =  model
        self._cmd_min = config['cmd_min']
        self._cmd_max = config['cmd_max']
        self._max_iter = config['max_iter']
        self._N = config['N']
        self._weights = np.zeros(self._N)
        self._start = start
        self._goal =  goal


    def generate_samples(self):
        return np.random.uniform(self._cmd_min, self._cmd_max, self._N)


    def SIR(self):
        cmd_samples = self.generate_samples()
        
        print "cmd_samples before \n", cmd_samples

        for k in range(self._max_iter):
            resampled_cmds = []

            print "Iteration \t", k

            for i in range(self._N):
                prediction, sigma = self._model.predict(np.r_[self._start, cmd_samples[i]][None,:])
                true_dir = self._goal-self._start
                true_dir /= np.linalg.norm(true_dir)
                prediction = prediction.ravel()
                pred_dir = prediction[:2]/np.linalg.norm(prediction[:2])

                vec_dir = np.linalg.norm(pred_dir - true_dir) 
                self._weights[i] = np.exp(-10.*vec_dir)
                # self._weights[i] = np.exp(-10*np.linalg.norm(prediction-self._goal))

            #normalization
            self._weights /= np.sum(self._weights)

            #resampling with replacement
            index = int(random.random()*self._N)
            beta = 0.
            max_weight = np.max(self._weights)
            for i in range(self._N):
                beta += random.random()*2.*max_weight

                while beta > self._weights[index]:
                    beta -= self._weights[index]
                    index = (index+1) % self._N
                resampled_cmds.append(cmd_samples[index])

            cmd_samples =  resampled_cmds

        print "cmd_samples after \n", cmd_samples
