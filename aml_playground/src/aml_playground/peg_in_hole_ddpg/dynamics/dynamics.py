


class Dynamics(object):

    def __init__(self, dt):

        self._dt = dt

    def dynamics(self, x, u):

        """ Non implemented """

        return x


    def __call__(self, x, u):

        return self.dynamics(x,u)



