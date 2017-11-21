

class Cost(object):


    def __init__(self, config):
        """ Not implemented """

        pass


    def get(self, x, u, du, sigma, t):

        """ Not implemented """

        pass


    def __call__(self, x, u, du, sigma, t):

        return self.get(x, u, du, sigma, t)