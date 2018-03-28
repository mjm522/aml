import logging

LEVELS = {'debug': logging.DEBUG,
          'info': logging.INFO,
          'warning': logging.WARNING,
          'error': logging.ERROR,
          'critical': logging.CRITICAL,
          'not_set': logging.NOTSET}


class AMLLogger(object):
    def __init__(self, name, level_name='debug'):
        self.configure(name, level_name)

    def configure(self, name, level_name='debug'):
        format = '[%(asctime)s][%(levelname)s] - %(name)s: %(message)s'
        # '%(levelname)s: %(message)s'
        level = LEVELS.get(level_name, logging.NOTSET)
        logging.basicConfig(format=format, level=level)

        self._logger = logging.getLogger(name)
        self._logger.propagate = False
        # otherwise handler additions are propagated to root logger resulting in double printing

        # creating console handler
        self._ch = logging.StreamHandler()
        self._ch.setLevel(level)
        # formatter = logging.Formatter('%(levelname)s: %(message)s')
        self._formatter = logging.Formatter(format)
        self._ch.setFormatter(self._formatter)

        # add handler only if it doesn't already have one
        if not self._logger.handlers:
            self._logger.addHandler(self._ch)

        self._logger.setLevel(level)

    def debug(self, msg):
        self._logger.debug(msg)

    def info(self, msg):
        self._logger.info(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def error(self, msg):
        self._logger.error(msg)

    def critical(self, msg):
        self._logger.critical(msg)

    def setLevel(self, level_name):
        level = LEVELS.get(level_name, logging.NOTSET)

        self._ch.setLevel(level)
        self._logger.setLevel(level)


class aml_logging(object):
    @classmethod
    def setup(cls, level_name='debug'):
        level = LEVELS.get(level_name, logging.NOTSET)
        print level, logging.NOTSET
        # logging.setLevel(level)
        logging.basicConfig(format='%(levelname)s: %(message)s', level=level)

    @classmethod
    def get_logger(cls, name, level_name='debug'):
        return AMLLogger(name, level_name)

    @classmethod
    def debug(cls, msg):
        logging.debug(msg)

    @classmethod
    def info(cls, msg):
        logging.info(msg)

    @classmethod
    def warning(cls, msg):
        logging.warning(msg)

    @classmethod
    def error(cls, msg):
        logging.error(msg)

    @classmethod
    def critical(cls, msg):
        logging.critical(msg)
