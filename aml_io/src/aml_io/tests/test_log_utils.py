import argparse
import numpy as np
from aml_io.log_utils import aml_logging

class TestLogger():

    def __init__(self):

        self._logger = aml_logging.get_logger(__name__)

        self._logger.debug("This is a debug message with array arguments:{}".format(np.random.randn(3),))
        self._logger.debug("This is a debug message with integer argument:%d"%(123,))
        self._logger.debug("This is a debug message with no arguments")

        self._logger.info("This is a info message with array arguments:{}".format(np.random.randn(3),))
        self._logger.info("This is a info message with integer argument:%d"%(123,))
        self._logger.info("This is a info message with no arguments")

        self._logger.warning("This is a warning message with array arguments:{}".format(np.random.randn(3),))
        self._logger.warning("This is a warning message with integer argument:%d"%(123,))
        self._logger.warning("This is a warning message with no arguments")

        self._logger.error("This is a error message with array arguments:{}".format(np.random.randn(3),))
        self._logger.error("This is a error message with integer argument:%d"%(123,))
        self._logger.error("This is a error message with no arguments")

        self._logger.critical("This is a critical message with array arguments:{}".format(np.random.randn(3),))
        self._logger.critical("This is a critical message with integer argument:%d"%(123,))
        self._logger.critical("This is a critical message with no arguments")


def main():

    # parser = argparse.ArgumentParser(description='Train and test forward model')
    # parser.add_argument('-l', '--logtype', type=str, default='debug' help='-l debug -l info, -l warning, -l critical, -l error')
    # args = parser.parse_args()
    # aml_logging.setup(args.logtype)
    # aml_logging.debug('This is a debug message')
    # aml_logging.info('This is an info message')
    # aml_logging.warning('This is a warning message')
    # aml_logging.error('This is an error message')
    # aml_logging.critical('This is a critical error message')

    tl = TestLogger()


if __name__ == '__main__':
    main()