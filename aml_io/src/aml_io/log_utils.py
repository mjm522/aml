import logging

LEVELS = {'debug':    logging.DEBUG,
          'info':     logging.INFO,
          'warning':  logging.WARNING,
          'error':    logging.ERROR,
          'critical': logging.CRITICAL,
          'all':      logging.NOTSET}

class aml_logging(object):

    @classmethod
    def setup(cls, level_name = 'all'):
        
        level = LEVELS.get(level_name, logging.NOTSET)
        print level, logging.NOTSET
        # logging.setLevel(level)
        logging.basicConfig(format='%(levelname)s: %(message)s', level=level)

    # @classmethod
    # def get_logger(cls, name, level_name = 'info'):
    #     logger = logging.getLogger(name)
    #     level = LEVELS.get(level_name, logging.NOTSET)

    #     logger.setLevel(level)
    #     # formatter = logger.Formatter('%(levelname)s: %(message)s')
    #     # logger.setFormatter(formatter)

    #     return logger


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
