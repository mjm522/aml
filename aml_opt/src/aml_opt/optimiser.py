import abc

class Optimiser(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, config=None):
        self._config = config